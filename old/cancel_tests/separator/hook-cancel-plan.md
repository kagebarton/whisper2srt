# Hook-based Cancellation Prototype: Separator

## Goal

Replicate `pipeline/workers/stem_worker.py` (which mirrors
`cancel_separator/workers/cancelable_stem_worker.py`) but replace the
monkey-patch on `model_run.forward()` with PyTorch's
`register_forward_pre_hook()`. Same subprocess + Pipe architecture, same
public API, same per-chunk cancel granularity — only the in-worker injection
mechanism changes.

## Why hooks instead of monkey-patching

`model_run` is a `torch.nn.Module` (Roformer). PyTorch exposes a stable,
documented hook API designed for exactly this purpose:

- `module.register_forward_pre_hook(fn)` returns a `RemovableHandle`
- `fn(module, inputs)` is called **before** every forward pass
- Raising inside `fn` propagates the exception through `nn.Module.__call__`
  and unwinds the demix loop the same way the monkey-patched forward does
- `handle.remove()` is the canonical cleanup — no more "what if the original
  forward got rebound during inference?" worry, no manual save/restore

The behavior is identical to the current monkey-patch; the API is just
cleaner, less surprising to read, and resilient to upstream changes (e.g. if
audio-separator ever swaps `model_run` mid-job, the hook still fires because
it's registered on the module instance, not patched on its dict).

## Architecture (unchanged)

```
Main process                              Subprocess
─────────────                              ──────────
StemWorker.start()  ──spawn──►            _worker_main()
                                            ├─ load Separator + model
                                            └─ loop on job_recv:
StemWorker.separate(wav, dir, evt)              ├─ recv (wav, dir)
  ├─ job_send.send((wav, dir))                  ├─ install hook
  ├─ start cancel_forwarder thread:             │   on model_run
  │   evt.wait() → cancel_send.send(1)          ├─ separator.separate()
  └─ result_recv.recv() → return paths          │   • per-chunk hook polls
                                                │     cancel_recv → may raise
                                                │     _CancelledInsideDemix
                                                ├─ remove hook (finally)
                                                └─ result_send.send(...)
```

The only thing that changes is what runs inside `_separate_with_cancel_check()`.

## Files to create

```
cancel_tests/
├── __init__.py
└── separator/
    ├── __init__.py
    ├── hook-cancel-plan.md          ← this file
    ├── stem_worker.py               ← the prototype worker
    └── run_test.py                  ← cancel-then-rerun proof
```

No `config.py`, `context.py`, or `orchestrator.py` — the prototype is
deliberately minimal. Module/folder layout is import-clean
(`from cancel_tests.separator.stem_worker import StemWorker`).

## Data structures

### Exceptions

```python
class WorkerCancelledError(Exception):
    """Public — raised to caller when separation was cancelled."""

class WorkerDiedError(Exception):
    """Public — raised when the subprocess dies during a job."""

class _CancelledInsideDemix(Exception):
    """Internal — raised by the forward_pre_hook to unwind the demix loop."""
```

Same names and semantics as `pipeline/workers/stem_worker.py`. Kept
identical so this prototype is a drop-in for testing without renaming.

**Keep `_CancelledInsideDemix` at module scope**, same as production
([stem_worker.py:275](../../pipeline/workers/stem_worker.py#L275)). The
`cancel_pre_hook` closure inside `_separate_with_cancel_check` captures
it from the enclosing module scope — do not nest it inside the function.

### Worker state

`StemWorker` (main-process side) — unchanged from pipeline:
- `_process: Process | None`
- `_job_send / _job_recv: Connection`
- `_result_send / _result_recv: Connection`
- `_cancel_send / _cancel_recv: Connection`
- `_temp_dir, _log_level, _model_dir, _model_name`

Subprocess side — same `Separator` instance held across jobs.

## Methods

### `StemWorker.__init__`, `.start()`, `.is_alive()`, `.separate()`, `.kill()`, `.stop()`
**Unchanged** from `pipeline/workers/stem_worker.py`. Same signatures, same
behavior. The cancel-event-to-Pipe forwarding thread (`_forward_cancel`) and
pipe draining (`_drain_cancel_pipe`) are kept verbatim.

**`model_name` must be a constructor arg forwarded to the subprocess** —
matches production ([stem_worker.py:72-118](../../pipeline/workers/stem_worker.py#L72-L118)):
`__init__(..., model_name: str = DEFAULT_MODEL_NAME)` stores
`self._model_name`, which is passed through `Process(args=(..., self._model_name))`
to `_worker_main(..., model_name=...)` which calls
`separator.load_model(model_filename=model_name)`. Do **not** adopt
`cancel_separator`'s hardcoded `MODEL_NAME` pattern — that breaks the
drop-in claim.

### `_worker_main(job_recv, result_send, cancel_recv, ...)` *(subprocess)*
**Unchanged.** Still loads the `Separator` once at startup, loops on
`job_recv`, dispatches to `_separate_with_cancel_check`, sends `("ok", ...)`,
`("cancelled",)`, or `("error", msg)` back.

### `_separate_with_cancel_check(audio_path, tmp_dir, separator, cancel_recv, log)` *(subprocess, **changed**)*

**Old (monkey-patch):**
```python
original_forward = model_run.forward
model_run.forward = cancelable_forward     # patch
try:
    separator.separate(...)
finally:
    model_run.forward = original_forward    # restore
```

**New (hook):**
```python
# Production also sets these before separate() — keep them
# (stem_worker.py:386-388). Without these, outputs land in the wrong dir.
separator.output_dir = str(tmp_dir)
if separator.model_instance:
    separator.model_instance.output_dir = str(tmp_dir)

def cancel_pre_hook(module, inputs):
    # Do NOT inspect or modify `inputs` — keeps this hook
    # order-independent with respect to any future pre-hooks
    # audio-separator might register.
    if cancel_recv.poll(0):
        try:
            cancel_recv.recv()              # consume the signal
        except (EOFError, OSError):
            pass
        log.info(f"Cancel detected before chunk #{chunk_counter[0] + 1} — aborting")
        raise _CancelledInsideDemix()
    chunk_counter[0] += 1                   # incremented BEFORE forward
                                            # (we ran the check, not the forward yet)

handle = model_run.register_forward_pre_hook(cancel_pre_hook)
try:
    output_paths = separator.separate(str(audio_path))
finally:
    # Hook-specific hardening: handle.remove() is a dict delete and
    # essentially cannot raise, but if it did while a _CancelledInsideDemix
    # was propagating, the cancel exception would be replaced and
    # _worker_main would fall through to the generic except branch and
    # send ("error", ...) instead of ("cancelled",).
    try:
        handle.remove()
    except Exception:
        pass
    log.debug(f"Removed hook (processed {chunk_counter[0]} chunks before exit)")
```

Notes:
- `chunk_counter` is a single-element list (closure mutability, same as
  current code).
- The counter increments inside the pre_hook before the forward runs —
  matches "this is the Nth attempted chunk", consistent with current
  monkey-patch counting.
- If `model_run` is `None`, fall back to `_run_separation_unpatched()`
  exactly as today.
- **Stem identification must be copied from
  [pipeline/workers/stem_worker.py:450-465](../../pipeline/workers/stem_worker.py#L450-L465),
  not from `cancel_separator/workers/cancelable_stem_worker.py`.** The
  production logic includes the `(other)` pattern (`is_other = "(other)" in lower`),
  required for non-karaoke MelBand Roformer models. The cancel_separator
  prototype omits it and will fail on those models.

### `_run_separation_unpatched(...)`
**Unchanged.** Fallback path used when `model_run` is unavailable.

### `_clear_gpu_state(separator, log)`, `_clear_gpu_cache()`, `_setup_worker_logger(level)`
**Unchanged.**

## Flows

### Happy path (no cancel event)
1. Caller → `separate(wav, dir, cancel_event=None)`
2. Main: `job_send.send((wav, dir))`, no cancel_forwarder thread spawned
3. Subprocess: registers pre_hook (it polls a never-signaled pipe — overhead
   is one `Connection.poll(0)` per chunk, negligible)
4. Demix loop runs to completion; pre_hook removed in `finally`
5. Result `("ok", vocal, instrumental)` sent back
6. Main returns `(Path(vocal), Path(instrumental))`

### Cancel during demix
1. Caller sets `cancel_event` mid-job
2. Main's `_forward_cancel` thread wakes from `cancel_event.wait()`,
   sends `1` on `cancel_send`
3. Subprocess: next chunk's pre_hook sees `cancel_recv.poll(0) is True`,
   consumes the signal, raises `_CancelledInsideDemix`
4. Exception unwinds `nn.Module.__call__ → forward → demix → separate`
5. `_worker_main` catches `_CancelledInsideDemix`, calls `_clear_gpu_state`,
   sends `("cancelled",)`
6. `finally` removes the hook, drains any leftover cancel signals
7. Main `separate()` raises `WorkerCancelledError`
8. **Model is still loaded in subprocess — next `separate()` runs immediately**

### Cancel set after job completed
1. Subprocess already sent `("ok", ...)`, hook already removed
2. `cancel_event.set()` triggers the forwarder thread, which sends `1`
3. Main's `finally` calls `_drain_cancel_pipe()` which consumes the stale `1`
4. Subprocess `finally` block also drains — belt-and-suspenders, both sides clean
5. Next job is unaffected

### Subprocess died during job
1. `result_recv.poll(0.5)` returns False repeatedly while subprocess gone
2. `proc.is_alive() is False` → `WorkerDiedError` raised
3. Main responsibility: caller decides whether to `start()` again

## Test plan: `run_test.py`

Mirror `cancel_separator/run_test.py` structure: a single script that
demonstrates one-shot cancellation and proves the model survives.

```
1. Parse args: input WAV path, --cancel-after seconds (default 3.0)
2. worker = StemWorker(temp_dir=...); worker.start()
3. Print: "loaded model in subprocess, ready"
4. Phase A: cancel test
   a. cancel_event = threading.Event()
   b. Schedule timer: threading.Timer(args.cancel_after, cancel_event.set)
   c. Try: worker.separate(input, output_a, cancel_event)
   d. Expect WorkerCancelledError; log time elapsed since job start
   e. If completed instead: log warning "increase --cancel-after"
5. Phase B: prove model survived
   a. Re-call worker.separate(input, output_b, cancel_event=None)
   b. Time it; assert output files exist
   c. Log: "second run completed in Xs without subprocess restart"
6. worker.stop()
```

INFO-level logging shows the per-chunk hook firing (the existing
`"Cancel detected before chunk #N"` line proves the hook ran).

## AudioLoader cleanup

Not applicable — audio-separator does not use stable-ts's `AudioLoader`.
The subprocess isolation already contains any FFmpeg processes audio-separator
itself spawns; the existing `_clear_gpu_state` covers separator-level cleanup.

## Risks / open questions

1. **Hook ordering with audio-separator's own hooks** — audio-separator
   does not register any user-facing hooks on `model_run` today, so our
   pre_hook fires alone. If they ever add one, PyTorch fires hooks in
   registration order; ours runs first because we register inside the
   prototype before `separator.separate()` runs.

2. **Per-chunk pipe poll overhead** — `Connection.poll(0)` is a syscall
   wrapped in cheap Python. Roformer chunks are ~3–6s of audio each →
   one poll per ~0.5–2s of GPU time. Overhead is irrelevant.

3. **Exception inside a no_grad context** — the demix loop runs under
   `torch.no_grad()`. Raising from a pre_hook inside no_grad is supported;
   no autograd state to unwind. The current monkey-patch already raises
   from this same location, so no change in behavior.

4. **Drop-in replacement for pipeline** — class names, exception types,
   and method signatures are kept identical to
   `pipeline/workers/stem_worker.py` so a port back consists of swapping
   the module file. No callers in `pipeline/` need to change.

5. **Logger name deviation** — production uses
   `"pipeline.workers.stem_worker"` as the logger name
   ([stem_worker.py:527](../../pipeline/workers/stem_worker.py#L527)).
   The prototype lives at `cancel_tests/separator/stem_worker.py`, so
   using the production logger name here would be confusing during
   prototype debugging. The prototype uses its own module-scoped logger
   name (`cancel_tests.separator.stem_worker`); on port-back, rename it
   to match production in the same commit that moves the file — no other
   changes needed for log-filter compatibility.

## Out of scope

- Going in-process (the subprocess design is intentional — model isolation,
  easier reload, mirrors pipeline)
- Changing the cancel_event → Pipe forwarding (Pipe is the right primitive
  for cross-process signaling)
- Multi-job batching, queuing, or progress reporting (none of the current
  prototypes do this; defer)
