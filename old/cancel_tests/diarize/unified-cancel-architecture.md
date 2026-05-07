# Unified Cancel Architecture for CancelableDiarizeWorker

## Goal

Allow a caller to cancel a pyannote `SpeakerDiarization` run at any time after
the worker starts, with a latency bound of "a few seconds" regardless of which
pipeline stage is currently executing. The loaded model must survive
cancellation so the next `diarize()` call does not pay the model-load cost.

## Problem with the current design

Today the worker monkey-patches `pipeline._segmentation.model.forward()` and
checks `cancel_event` before each forward pass. This only covers the
**segmentation** phase. In practice segmentation finishes in the first few
seconds; most wall time is spent in the **embedding** phase
(`wespeaker-voxceleb-resnet34-LM`) and a small amount in **clustering**.
Setting the cancel event during embedding has no effect — the run completes
normally.

## Pre-implementation probe

Before committing to the design, run a one-off probe script
(`probe_pipeline.py`) against a loaded `SpeakerDiarization` pipeline to
validate the assumptions this plan rests on. Expected runtime: ~30s on top
of one short diarization call.

### What to probe

1. **Embedding model is a `torch.nn.Module`.** Print:
   - `type(pipeline._embedding)` and whether it's `isinstance(..., nn.Module)`
   - `type(getattr(pipeline._embedding, "model_", None))` and same check
   - All `(qualified_name, class_name)` pairs from `pipeline.named_modules()`
     whose name contains `"embed"` or `"wespeaker"`
   - If none of those land on an `nn.Module`, Layer 2 degrades to
     segmentation-only and the plan needs a different embedding strategy
     (e.g. wrapping `pipeline._embedding.__call__`).

2. **`hook=` kwarg is accepted by this pyannote version.** Inspect
   `inspect.signature(pipeline.__call__)` for a `hook` parameter. We're on
   pyannote 4.0.4 which has it, but record it so future upgrades surface a
   regression.

3. **Pyannote doesn't swallow hook-raised exceptions.** Call
   `pipeline(short_audio.wav, hook=raising_hook)` where `raising_hook`
   unconditionally raises a sentinel exception on its first invocation.
   Confirm the sentinel reaches the caller (vs being caught and logged
   internally). If pyannote wraps hook calls in try/except:
   - Fallback plan: hook stores a flag, diarize() checks the flag after
     `pipeline()` returns and raises `DiarizationCancelledError` post-hoc.
     Layer 2 (PyTorch `forward_pre_hook`) still fires normally inside
     `nn.Module.__call__`, which is outside pyannote's control, so
     within-stage cancel still works.

4. **PyTorch `forward_pre_hook` raises propagate cleanly.** Register a hook
   on `pipeline._segmentation.model` that raises on first call, run a short
   diarization, confirm the exception reaches our `except` clause with the
   model still usable on a subsequent call (just don't raise next time).
   This is PyTorch's documented behavior (pre-hooks fire before forward,
   so no partial computation to unwind), but confirming against this
   specific pyannote stack eliminates a class of risk.

### Deliverable

A short `probe_pipeline.py` that prints a pass/fail table for the four
checks above plus a one-liner summary. If any check fails, the plan is
revised before implementation starts.

## Solution: two hook layers, one event, one exception

Cancel checks are installed at two complementary layers, both driving the same
`threading.Event` and raising the same internal `_Cancelled` exception.

| Layer | Mechanism                                   | Fires at                              | Latency bound          |
|-------|---------------------------------------------|---------------------------------------|------------------------|
| 1     | pyannote `hook=` callback on `pipeline(...)`| Stage boundaries + progress events    | End of current stage   |
| 2     | PyTorch `register_forward_pre_hook` on each inference `nn.Module` | Before every batch forward pass of segmentation and embedding models | One batch (~0.1–1s)    |

Layer 2 is what gives the "cancel within embedding" behavior we're missing
today. Layer 1 gives progress logging for free and covers any stage that
doesn't go through a hooked `nn.Module` (e.g. clustering).

Clustering is CPU-bound agglomerative work and doesn't call an `nn.Module`; if
cancel fires during clustering we wait for it to end (typically <1s).

## Data structures

### Exceptions

```python
class _Cancelled(Exception):
    """Internal. Raised by any cancel check to unwind the pipeline."""

class DiarizationCancelledError(Exception):
    """Public. Raised to caller after cleanup when cancelled."""
```

`_CancelledInsideForward` is renamed to `_Cancelled` since it is now raised
from both PyTorch hooks and the pyannote hook callback.

### Worker state (persistent)

```python
self._pipeline: Pipeline | None
self._model_loaded: bool
self._segmentation_model: nn.Module | None   # from pipeline._segmentation.model
self._embedding_model:    nn.Module | None   # from pipeline._embedding(.model)
```

### Per-call context (transient, reset each diarize())

```python
self._current_stage:  str | None              # last stage name from pyannote hook
self._batch_counter:  dict[str, int]          # {stage_name: forward_count}
self._hook_handles:   list[RemovableHandle]   # PyTorch hook handles to remove in finally
```

## Methods

### `load_model(self) -> None`
Same as today, with one added step:
1. `Pipeline.from_pretrained(...)`
2. `.to(device)` if not CPU
3. **Call `self._find_inference_models()`** — populates both model refs
4. Log classes: `"Segmentation: {cls}, Embedding: {cls}"`

### `_find_inference_models(self) -> None` *(replaces `_find_segmentation_model`)*

Single discovery point for both models. Strategy per model:

**Segmentation**
1. Try `pipeline._segmentation.model`
2. Fallback: walk `pipeline.named_modules()` for `nn.Module` whose qualified name contains `"segmentation"`
3. Validate `isinstance(found, torch.nn.Module)` (required by `register_forward_pre_hook`)

**Embedding**
1. Try `pipeline._embedding.model_` (pyannote wraps some embedders)
2. Try `pipeline._embedding` directly
3. Fallback: walk `named_modules()` for `nn.Module` whose name contains `"embedding"` or `"wespeaker"`
4. Validate `isinstance(..., torch.nn.Module)`

If either cannot be resolved, log a clear warning. The worker still functions
— that stage just degrades to Layer-1-only cancel granularity.

### `diarize(self, vocal_path, cancel_event=None) -> list[dict]`

```
1. Validate pipeline is loaded
2. Build kwargs from config (num_speakers, min_speakers, max_speakers)
3. Reset per-call state:
       self._current_stage = None
       self._batch_counter = {}
       self._hook_handles = []
4. Install hooks (always installs pyannote progress hook;
   installs PyTorch pre_hooks only when cancel_event is provided):
       pyannote_hook = self._install_hooks(cancel_event)
5. try:
       return self._run_diarization(vocal_path, hook=pyannote_hook, **kwargs)
   except _Cancelled:
       _clear_gpu_state()
       stage = self._current_stage or "pre-start"
       total_batches = sum(self._batch_counter.values())
       log.info(f"Diarization cancelled during stage={stage} "
                f"after {total_batches} forward passes — model still loaded")
       raise DiarizationCancelledError(
           f"Cancelled during stage={stage} after {total_batches} forward passes"
       )
   finally:
       self._remove_cancel_hooks()
```

Progress logging (Layer 1) is installed unconditionally so callers get
stage transitions for free. Within-stage cancel (Layer 2) is only installed
when `cancel_event` is supplied. No `cancel_event`-is-None shortcut —
hook overhead is negligible and progress logging is almost always wanted.

### `_install_hooks(self, cancel_event=None) -> Callable`

Installs progress logging (Layer 1, always) and cancel checks (Layer 2,
only when `cancel_event` is provided). Returns the pyannote-style callback
to pass as `hook=` to `pipeline(...)`.

**Layer 2 — PyTorch forward_pre_hook on each inference model (conditional)**

Only installed when `cancel_event is not None`. Without within-stage cancel
there's no reason to incur hook overhead on every batch.

```python
if cancel_event is not None:
    def make_pre_hook(stage_label: str):
        def pre_hook(module, inputs):
            if cancel_event.is_set():
                raise _Cancelled()
            self._batch_counter[stage_label] = self._batch_counter.get(stage_label, 0) + 1
        return pre_hook

    for model, label in [(self._segmentation_model, "segmentation"),
                         (self._embedding_model,    "embedding")]:
        if model is not None:
            handle = model.register_forward_pre_hook(make_pre_hook(label))
            self._hook_handles.append(handle)
```

`stage_label` here is the model we hooked, not the pyannote step name. It's
used only for diagnostic logging — pyannote's own step name is tracked
separately by Layer 1.

**Layer 1 — pyannote hook callback (always installed)**

Always installed when hooks are active so callers get stage-boundary
progress for free. The cancel check is guarded on `cancel_event` so the
same callback works with or without cancellation.

```python
def pyannote_hook(step_name, *args, completed=None, total=None, **kwargs):
    self._current_stage = step_name
    if cancel_event is not None and cancel_event.is_set():
        raise _Cancelled()
    if completed is not None and total:
        log.info(f"Progress: {step_name} {completed}/{total}")
    # else: stage-start events — don't spam the log

return pyannote_hook
```

If the probe (see "Pre-implementation probe") finds pyannote swallows hook
exceptions, the callback additionally sets `self._cancel_requested = True`
and `diarize()` checks that flag after `_run_diarization()` returns.

### `_remove_cancel_hooks(self) -> None`

```python
for handle in self._hook_handles:
    try:
        handle.remove()
    except Exception:
        pass
self._hook_handles = []
```

Deterministic cleanup; never raises. Called from the `finally` block so hooks
are removed on both success and cancellation.

### `_run_diarization(self, vocal_path, hook=None, **kwargs) -> list[dict]`

Same as today, with one signature change: accepts a `hook` kwarg and passes
it through to `self._pipeline(...)` when not None.

### `unload_model(self) -> None`

Adds one line: also set `self._embedding_model = None`. Otherwise unchanged.

## Flows

### Happy path, no cancel event
```
caller → diarize(audio, cancel_event=None)
      → _install_hooks(None)              [pyannote callback only, no pre_hooks]
      → _run_diarization(hook=pyannote_hook)
         • pyannote_hook fires at stage boundaries → logs progress
         • cancel check is a no-op (cancel_event is None)
      → returns turns
      finally: _remove_cancel_hooks()
```
Progress logging is on; no per-batch overhead.

### Happy path with cancel event available but never set
```
caller → diarize(audio, cancel_event)
      → _install_hooks(cancel_event)     [2 PyTorch handles + 1 callback]
      → _run_diarization(hook=pyannote_hook)
         • pyannote_hook fires at stage boundaries → logs progress
         • pre_hook fires per batch → increments counter, cancel check is False
      → returns turns
      finally: _remove_cancel_hooks()
```

### Cancel during segmentation (within-stage)
```
caller sets cancel_event at t=2s
  next segmentation batch → pre_hook → raises _Cancelled
  exception unwinds through pipeline.__call__
  diarize() catches _Cancelled:
    - _clear_gpu_state()
    - logs "Cancelled during stage=segmentation after N batches"
    - raises DiarizationCancelledError
  finally: hook handles removed
  model still loaded — caller can immediately call diarize() again
```

### Cancel during embedding (within-stage) *(the case that fails today)*
Same as segmentation, but `_Cancelled` is raised from the embedding model's
pre_hook. Without the second hook, today the cancel event is never seen.

### Cancel during clustering (stage-boundary)
Clustering is CPU-bound, does not call a hooked `nn.Module`. Cancel waits
until clustering finishes and the next pyannote hook event fires, then
unwinds via Layer 1. Worst case ≈ clustering duration (<1s typically).

### Cancel set before diarize() starts
First pre_hook or first pyannote hook call sees the event set → raises
immediately. Returns `DiarizationCancelledError` with `stage="pre-start"` or
the first stage name.

### Unexpected exception from pyannote
Propagates out. `finally` still removes hook handles — no leaked state.

## Logging

**INFO (default)**
- `"Diarization started on {file}"`
- `"Progress: {stage} {completed}/{total}"` (only when total is known)
- `"Diarization complete: N turns, M speakers"` on success
- `"Diarization cancelled during stage={stage} after {N} forward passes — model still loaded"` on cancel

**DEBUG**
- `"Found segmentation model at {path} ({cls})"`
- `"Found embedding model at {path} ({cls})"`
- `"Installed N forward_pre_hooks + pyannote callback"`
- `"Removed N hook handles"`

No per-batch logging at INFO — too noisy. The batch counter is only surfaced
on cancel and at DEBUG.

## Tests

### `test_cancel_diarize.py`
No structural change — already does load → cancel → rerun. Works for
segmentation-phase cancel today; with the new architecture, `--cancel-after`
values that land in the embedding phase (e.g. `--cancel-after 8` on a 45s
run) will actually cancel instead of silently completing.

### `run_test.py`
No required change. Optional enhancement: add `--cancel-stage
{segmentation,embedding}` that watches the pyannote hook callback and fires
the cancel event when the requested stage starts. This makes the embedding
cancel test deterministic regardless of audio length.

## Risks and open questions

1. **Embedding model discovery.** Pyannote may wrap the actual torch model
   in a non-`nn.Module` class (`PretrainedSpeakerEmbedding`). If so,
   `register_forward_pre_hook` will not exist on the wrapper. Mitigation:
   `_find_inference_models` drills one level (`.model_` or `.model`) and
   validates `isinstance(..., torch.nn.Module)`. If still not found, we fall
   back to Layer-1-only cancel for that stage (not ideal, but not broken).
   **Validated by the pre-implementation probe (check #1).**

2. **pyannote version skew / `hook=` kwarg.** The attribute paths
   (`_segmentation.model`, `_embedding`) and the `hook=` kwarg are private
   / version-dependent. Keeping all discovery in `_find_inference_models()`
   means there is exactly one place to adjust if pyannote restructures
   internals. **`hook=` availability is validated by the pre-implementation
   probe (check #2).**

3. **Pyannote swallowing hook exceptions.** If pyannote wraps the hook
   callback in a try/except internally, a `_Cancelled` raised from Layer 1
   would die silently. Mitigation: hook also sets `self._cancel_requested`
   as a flag; `diarize()` checks it after `_run_diarization()` returns.
   Layer 2 (PyTorch pre_hooks) is unaffected since those raise from inside
   `nn.Module.__call__`, below pyannote's layer. **Validated by the
   pre-implementation probe (check #3).**

4. **PyTorch hook exception unwinding.** `register_forward_pre_hook` is a
   documented API and pre-hooks fire before any forward computation, so
   there's no partial state to clean up. Diarization runs under
   `torch.no_grad()` so gradient tracking isn't involved either. The only
   residual is GPU memory from the previous (completed) forward, handled
   by `_clear_gpu_state()`. **Validated by the pre-implementation probe
   (check #4).**

5. **Concurrency.** The worker is not thread-safe for concurrent `diarize()`
   calls against the same instance — hook state lives on `self`. This
   matches the current design; flagged here in case a future caller tries
   to parallelize. If needed later, pass per-call context through closure
   instead of storing on `self`.

6. **Clustering cancel latency.** No hook fires during clustering. If
   clustering grows noticeably (many speakers, long audio), revisit by
   patching `AgglomerativeClustering.__call__` as a third layer. Not worth
   doing speculatively.

## Out of scope

- Subprocess-based cancellation (would lose the loaded model; rejected by
  the requirement that cancel preserves the model).
- Progress UI beyond INFO logging — caller can wrap the pyannote hook for
  richer progress reporting when integrating into the main app.
- Changes to `config.py`, `caption.py`, or the CLI flags in the test
  scripts.
