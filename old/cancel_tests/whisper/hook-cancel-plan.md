# Hook-based Cancellation Prototype: Whisper

## Goal

Replicate the public API of `cancel_whisper/workers/cancelable_whisper_worker.py`
(the `align`, `refine`, `align_and_refine` methods) but:

1. Use the **PyTorch OpenAI Whisper** model (`stable_whisper.load_model()`),
   matching `pipeline/workers/whisper_worker.py` — not the CTranslate2-backed
   `load_faster_whisper()` from cancel_whisper.
2. Replace the monkey-patch on `model.encode()` with PyTorch's
   `register_forward_pre_hook()` on the encoder `nn.Module`.

The cancel granularity (per-encoder-pass, ~few-second worst case) is the
same as today; the mechanism is the cleaner stable PyTorch API.

## Why this is now possible (unlike cancel_whisper)

`stable_whisper.load_faster_whisper()` returns a wrapper around a
`ctranslate2.models.Whisper` — a C++ inference engine, **not** a
`torch.nn.Module`. It has no `register_forward_pre_hook`, so cancel_whisper
had to monkey-patch the Python-level `encode()` method.

`stable_whisper.load_model()` returns `whisper.model.Whisper` **directly**
(after running `modify_model()` on it to bind `align`/`refine`/etc. as
methods). There is no outer Python wrapper — the returned object IS the
`nn.Module`, with a real PyTorch encoder (`whisper.model.AudioEncoder`).
We hook the encoder directly:

```
whisper.model.Whisper  (nn.Module — returned directly by load_model())
  ├─ .encoder = whisper.model.AudioEncoder  (nn.Module)  ← hook target
  └─ .decoder = whisper.model.TextDecoder   (nn.Module)  ← not hooked
```

Note: `whisper.model.Whisper` does **not** have an `.encode()` method —
that method only exists on `faster_whisper.WhisperModel`, which is why
cancel_whisper's monkey-patch on `model.encode()` worked for
`load_faster_whisper()` but would not work here. The hook approach is
therefore not just "cleaner" — it intercepts at a fundamentally different
layer (`nn.Module.__call__` dispatch on `model.encoder`, not a Python
wrapper method).

**Verified encoder call paths (runtime-checked against installed stable_whisper):**
- `model.align(...)` → `Aligner.align` → `compute_timestamps` →
  `add_word_timestamps_stable` → `model.encoder(mel.unsqueeze(0))` at
  `stable_whisper/timing.py:60`. This is `nn.Module.__call__`, so
  `register_forward_pre_hook` fires.
- `model.refine(...)` → refinement `inference_func` at
  `stable_whisper/alignment.py:661` calls `model(mel_segments, tokens)`,
  which dispatches to `Whisper.forward` → `self.encoder(...)` — also via
  `nn.Module.__call__`. Hook fires.

## Why encoder only (not decoder)

You confirmed "few seconds at most" cancel latency is acceptable. The encoder
is the dominant GPU cost per alignment iteration; the decoder runs lighter
work between encoder calls. Hooking only the encoder gives:
- Cancel landing within one encoder pass (~0.5–3s of GPU work)
- One hook target instead of two
- Identical granularity to the current `encode()` patch

If decoder-granularity cancellation is needed later, add a second
`register_forward_pre_hook` on `model.model.decoder` — same mechanism, no
architectural change.

## Files to create

```
cancel_tests/
├── __init__.py
└── whisper/
    ├── __init__.py
    ├── hook-cancel-plan.md          ← this file
    ├── config.py                    ← WhisperModelConfig (adapted — see below)
    ├── whisper_worker.py            ← prototype worker (uses cancel_whisper
    │                                  align_and_refine contract with
    │                                  cancel_event.clear() between phases,
    │                                  NOT the pipeline no-clear contract)
    └── test_cancel_whisper.py       ← cancel-then-rerun proof
```

`config.py` is adapted from `cancel_whisper/config.py` with one change:
`compute_type` defaults to `""` (empty string) rather than `"int8"`, since
`stable_whisper.load_model()` doesn't accept a `compute_type` parameter
(int8/float16 quantization is a CTranslate2 feature used by
`load_faster_whisper()`). The field is kept in the dataclass only for
interface compatibility with cancel_whisper. This matches the pipeline's
convention.

The test script mirrors `cancel_whisper/test_cancel_whisper.py`.

## Data structures

### Exceptions

```python
class _Cancelled(Exception):
    """Internal — raised by the forward_pre_hook to unwind alignment."""

class AlignmentCancelledError(Exception):
    """Public — raised to caller after cleanup when cancelled."""
```

`_Cancelled` is renamed from `_CancelledInsideEncode` since it's now raised
from a PyTorch hook on the encoder, not from inside `encode()`. Public
`AlignmentCancelledError` keeps its name and semantics for drop-in
compatibility with cancel_whisper test code.

### Worker state

```python
class WhisperWorker:
    _config: WhisperModelConfig
    _model:                   # stable_whisper wrapper around whisper.model.Whisper
    _model_loaded: bool
    _encoder_module:          # cached ref to model.model.encoder (nn.Module)
    _audioloader_patched: bool
```

`_encoder_module` is a new field, populated at `load_model()` time, used so
hook installation in `align()`/`refine()` doesn't rewalk attributes.

## Methods

### `__init__(self, config=None)`
**Unchanged** from cancel_whisper. Initializes `_audioloader_patched=False`,
`_encoder_module=None`.

### `model_loaded` property
**Unchanged.**

### `load_model(self) -> None`
Mirrors `pipeline/workers/whisper_worker.py`:
1. Idempotent if already loaded
2. Resolve device (`auto` → cuda/cpu via torch)
3. `self._model = stable_whisper.load_model(model_source, device=device)`
4. **Cache encoder ref:** `self._encoder_module = self._model.encoder`
   (the returned model IS `whisper.model.Whisper`, so `.encoder` is a
   direct attribute — not `self._model.model.encoder`)
   - Validate `isinstance(self._encoder_module, torch.nn.Module)`; if not,
     log a clear warning referencing the actual attempted path
     (`self._model.encoder`). Cancellation will degrade to "wait for the
     current encode call to return" if the guard fails.
5. `self._patch_audioloader_stderr()` (kept verbatim from cancel_whisper)

### `_patch_audioloader_stderr(self)` *(unchanged from cancel_whisper)*
Class-level patch on `AudioLoader._audio_loading_process` to redirect FFmpeg
stderr to `/dev/null`. Independent of cancel mechanism. **No changes.**

### `_terminate_orphaned_audioloaders(self)` *(unchanged from cancel_whisper)*
Walks `gc.get_objects()` for `AudioLoader` instances with running FFmpeg
subprocesses and terminates them. Called from the cancel-handler in
`align`/`refine`. **No changes** — the cleanup need is identical regardless
of where the cancel exception originated.

### `align(self, vocal_path, lyrics_text, cancel_event=None)`

**Old (monkey-patch):**
```python
original_encode = model.encode
def cancelable_encode(*a, **kw):
    if cancel_event.is_set(): raise _CancelledInsideEncode()
    return original_encode(*a, **kw)
model.encode = cancelable_encode
try:
    return model.align(...)
finally:
    model.encode = original_encode
```

**New (hook):**
```python
if self._model is None:
    raise RuntimeError("Model not loaded — call load_model() first")

if cancel_event is None:
    return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)

if self._encoder_module is None:
    log.warning("Encoder module not cached — running without cancel check")
    return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)

encode_counter = [0]

def cancel_pre_hook(module, inputs):
    if cancel_event.is_set():
        log.info(f"Cancel detected before encode pass #{encode_counter[0] + 1} — aborting")
        raise _Cancelled()
    encode_counter[0] += 1  # counts passes STARTED, not completed — see note below

handle = self._encoder_module.register_forward_pre_hook(cancel_pre_hook)
try:
    return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)
except _Cancelled:
    log.info(f"Alignment cancelled after {encode_counter[0]} encode passes started — model still loaded")
    self._terminate_orphaned_audioloaders()
    _clear_gpu_state()
    raise AlignmentCancelledError(
        f"Alignment cancelled after {encode_counter[0]} encode passes started (model still loaded)"
    )
finally:
    try:
        handle.remove()
    except Exception:
        pass  # defensive: if stable-ts internals already removed the hook
    log.debug(f"Removed encoder hook ({encode_counter[0]} encode passes started)")
```

The `align_kwargs` block (`language`, `vad`, `vad_threshold`,
`suppress_silence`, `suppress_word_ts`, `only_voice_freq`) is unchanged —
extracted into a local dict for readability since it appears in two places.

**Guards added vs. cancel_whisper:**
- `if self._model is None: raise RuntimeError(...)` — matches the guard in
  cancel_whisper line 347 and pipeline line 303. Without it, the
  `cancel_event is None` branch would crash with `AttributeError` rather
  than a clear error.
- `try/except Exception: pass` around `handle.remove()` — PyTorch's hook
  bookkeeping can raise on double-removal. If it did, the outer
  `except _Cancelled` would never run and `AlignmentCancelledError` would
  be replaced by the remove-error. The monkey-patch version has the same
  theoretical risk but hook-remove is more likely to fail.

**Counter semantics note:** The pre-hook increments `encode_counter`
**before** the encoder forward runs, whereas cancel_whisper's monkey-patch
incremented **after** the forward returned successfully. This means the
reported count is +1 compared to cancel_whisper for the same cancel point
(it includes the pass that was about to start but never completed). This
is logging-only and intentional — a clean two-hook fix (pre for cancel,
post for counting) is possible but overkill for a debug counter. Log
messages say "passes started" to make the semantics explicit.

### `refine(self, vocal_path, result, cancel_event=None)`
Same hook pattern as `align()`, including the `self._model is None` guard
and the defensive `try/except` around `handle.remove()`. The encoder is
the same `nn.Module`, so the hook installation is identical — only the
underlying model call changes (`model.refine(...)` instead of
`model.align(...)`) and the kwargs differ (`steps`, `word_level`).

`_terminate_orphaned_audioloaders()` is called in the cancel handler; the
cancel_whisper comment notes refine() uses `prep_audio()` not `AudioLoader`
but the call is kept "for safety" — same here, no behavior change.

### `align_and_refine(self, vocal_path, lyrics_text, cancel_event=None)`
**Unchanged from cancel_whisper:**
```python
result = self.align(vocal_path, lyrics_text, cancel_event)
if result is None:
    return None
if cancel_event is not None:
    cancel_event.clear()                # standalone-test semantics:
                                        # don't propagate align cancel into refine
refined = self.refine(vocal_path, result, cancel_event)
return refined
```

Keeps cancel_whisper's `cancel_event.clear()` between phases — that's the
standalone-test contract (a single cancel only kills the in-flight phase).
Pipeline's variant *intentionally* propagates; this prototype mirrors
cancel_whisper as you specified.

### `unload_model(self)` *(unchanged)*
Adds one line: `self._encoder_module = None` alongside clearing `_model`.

### `_clear_gpu_state()`, `_clear_gpu_cache()` *(docstring updated)*
Module-level helpers. The cancel_whisper docstring references CTranslate2's
internal memory management — that's stale here. Updated docstring:

> After `_Cancelled` unwinds from the encoder hook, intermediate tensors
> from the interrupted forward pass should be freed as their stack frames
> are unwound by normal Python exception propagation. We clear the PyTorch
> CUDA cache as a safety net — `torch.cuda.empty_cache()` only releases
> unused cached allocator blocks, it does not free tensors still
> referenced by live frames.

## Flows

### Happy path (no cancel event)
1. `align(vocal, lyrics, cancel_event=None)` → no hook installed, runs
   straight through stable-ts → returns `WhisperResult`

### Happy path with cancel event but never set
1. Hook installed on encoder
2. Each encoder forward pass: `cancel_event.is_set()` returns False, counter
   increments, original forward runs
3. Stable-ts returns; `finally` removes hook
4. Returns `WhisperResult`
- Overhead: one `Event.is_set()` per encoder pass (microseconds)

### Cancel during alignment
1. Caller sets `cancel_event`
2. Next encoder pre_hook fires, sees event set, raises `_Cancelled`
3. Exception unwinds `Whisper.encoder.__call__ → encode() → inference_func →
   _compute_timestamps → Aligner while-loop → Aligner.align → model.align`
4. `align()` catches `_Cancelled`:
   - `_terminate_orphaned_audioloaders()` kills the FFmpeg subprocess that
     stable-ts's AudioLoader spawned (the normal cleanup at line 357 of
     `non_whisper/alignment.py` is skipped on exception)
   - `_clear_gpu_state()`
   - Raises `AlignmentCancelledError`
5. `finally` removes the hook
6. Model still loaded — caller can immediately call `align()` again

### Cancel during refinement
Same as alignment, but unwinds through `Refiner._refine`. No AudioLoader to
clean up (refine uses `prep_audio`), but the cleanup call is harmless.

### Cancel between phases of `align_and_refine` (standalone-test contract)
1. Cancel set during `align` → align raises `AlignmentCancelledError`
2. `align_and_refine` does NOT catch it → propagates to caller
3. The `cancel_event.clear()` line is only reached when `align` returns
   normally; on cancel, it's not executed. (Same as cancel_whisper today.)

## Test plan: `test_cancel_whisper.py`

Mirror `cancel_whisper/test_cancel_whisper.py`:

```
1. Parse args: vocal_path, lyrics_path, --cancel-after seconds, --phase {align,refine}
2. worker = WhisperWorker(); worker.load_model()
3. Print: "Whisper model loaded in Xs (PyTorch device=...)"
4. Phase A: cancel test
   a. cancel_event = threading.Event()
   b. threading.Timer(args.cancel_after, cancel_event.set).start()
   c. Try: result = worker.align(vocal, lyrics, cancel_event)
   d. Expect AlignmentCancelledError
   e. Log: "Cancelled after Xs"
5. Phase B: prove model survived
   a. Re-call worker.align(vocal, lyrics, cancel_event=None)
   b. Time it; assert result is non-empty
   c. Log: "Second align completed in Xs without model reload"
6. worker.unload_model()
```

Optional flag: `--phase refine` runs an initial `align` first (no cancel) to
get a `WhisperResult`, then the cancel timer races against `refine`.

INFO logs from the worker show the per-pass hook firing
(`"Cancel detected before encode pass #N"`).

## AudioLoader cleanup — porting back to pipeline

You asked whether the FFmpeg-stderr patch and orphaned-loader cleanup pose
problems when ported back into pipeline. Verdict: **no problems, port as-is.**

- `_patch_audioloader_stderr()` is a class-level monkey-patch on
  `AudioLoader._audio_loading_process`. It's idempotent (guarded by
  `_audioloader_patched`), runs once per process, and survives any number
  of `WhisperWorker` instances. Pipeline already does this in its own
  whisper_worker.py.
- `_terminate_orphaned_audioloaders()` walks GC for `AudioLoader` instances
  whose FFmpeg subprocess (`_process`/`_extra_process`) is still running and
  terminates them. It's bounded (only runs on cancel), safe (try/except around
  each kill), and only touches AudioLoader objects. No state leaks across
  pipeline jobs.

The only thing to revisit when porting back is `align_and_refine`'s
`cancel_event.clear()` — pipeline's version intentionally **doesn't** clear,
because the pipeline contract says a cancelled song is discarded entirely
and a cancel during align must propagate through refine. Standalone test
keeps the clear; pipeline drops it. This is a documented divergence, not
a bug.

## Risks / open questions

1. **Encoder attribute path** — `self._model.encoder` is the canonical path:
   `stable_whisper.load_model()` returns `whisper.model.Whisper` directly,
   which has `.encoder` as a direct attribute. If stable-ts ever wraps the
   model differently in a future version, the `isinstance(..., nn.Module)`
   check + warning at `load_model()` time surfaces the issue immediately
   rather than silently degrading.

2. **Hook timing inside no_grad / autocast contexts** — Whisper's encode
   runs under `torch.no_grad()` (and possibly autocast). PyTorch documents
   that `forward_pre_hook` raises propagate cleanly out of these contexts;
   confirmed by current monkey-patch behavior which raises from the same
   call frame.

3. **Token-level cancel granularity** — alignment's inner loop interleaves
   encoder + decoder passes per token batch (~100 tokens). The encoder hook
   fires once per batch. If a single batch ever balloons (very long word,
   pathological audio), cancel waits for that one batch. Acceptable per
   your "few seconds" tolerance.

4. **Multiple concurrent jobs on one worker** — not supported; the
   `encode_counter` and hook handle live in a single `align()` call's
   closure, but if a caller invoked `align()` twice concurrently on the
   same worker, both hooks would coexist on the encoder. Matches existing
   single-job-at-a-time contract; flagged here for porters.

   **Different failure mode vs. monkey-patch:** With hooks, concurrent
   calls install independent hooks that coexist on the encoder; each
   `handle.remove()` in `finally` removes only its own hook. With the
   monkey-patch approach, the second `model.encode = cancelable_encode`
   silently replaces the first, and the first call's `finally` restores
   the original method — leaving the second call unpatched. Neither
   approach supports concurrent use; the hook version fails more
   gracefully (both hooks run, both cancel checks work) while the
   monkey-patch version silently drops the second call's cancel check.

## Out of scope

- The `transcribe` / `transcribe_and_refine` methods from
  `pipeline/workers/whisper_worker.py` (you said "strictly replicate
  cancel_whisper")
- Cancellation inside the decoder (encoder hook is sufficient at the
  latency budget you confirmed)
- Switching back to `load_faster_whisper` (the whole point is to test
  hooks, which require PyTorch)
