# Review: Unified Cancel Architecture for CancelableDiarizeWorker

Reviewed against the current implementation in `workers/cancelable_diarize_worker.py`
and the existing model pre-download infrastructure in `pyann-models/`.

## Summary

The architecture proposes upgrading from a single monkey-patch on
`segmentation.model.forward()` to a two-layer hook system:

| Layer | Mechanism | Coverage | Latency bound |
|-------|-----------|----------|---------------|
| 1 | pyannote `hook=` callback on `pipeline(...)` | Stage boundaries | End of current stage |
| 2 | PyTorch `register_forward_pre_hook` on segmentation & embedding `nn.Module`s | Per batch | ~0.1–1s |

This correctly addresses the core problem: the current monkey-patch only covers
segmentation (~first few seconds), while most wall time is spent in embedding
where cancel events are invisible today.

## Issues Found

### 1. Embedding model may not be an `nn.Module` — **High risk**

Pyannote wraps the embedding model in `PretrainedSpeakerEmbedding`, which is not
an `nn.Module`. `register_forward_pre_hook` only exists on `nn.Module` subclasses.

The doc's fallback strategy (`_embedding.model_`, `_embedding.model`, walk
`named_modules()` for `"embedding"` or `"wespeaker"`) may still fail if the
wrapper doesn't expose the inner model through a consistent attribute path.

**Impact**: If discovery fails, Layer 2 silently degrades to segmentation-only —
the same gap that exists today. The worker still functions, but the primary
motivation for the architecture change is lost.

**Recommendation**: Probe the actual attribute structure of a loaded
`speaker-diarization-3.1` pipeline at runtime before implementing. Print
`dir(pipeline._embedding)` and `type(pipeline._embedding)` to confirm the
drill-down path. If `PretrainedSpeakerEmbedding` doesn't expose `.model_` or
`.model`, we need a different approach (e.g. walking `__dict__` or patching
the wrapper's `__call__` instead).

### 2. Exception raised from PyTorch forward_pre_hook may not unwind safely — **Medium risk**

The current monkey-patch replaces the entire `forward()` method, so the
exception unwinds from application code. `register_forward_pre_hook` executes
inside PyTorch's `__call__` → `forward()` dispatch (specifically inside
`_call_impl` in `nn.Module`). Raising an exception from a pre-hook interrupts
PyTorch's internal bookkeeping:

- Hook iteration may be mid-loop across registered hooks
- Gradient tracking state may be inconsistent
- Any context managers inside `__call__` may not exit cleanly

The doc calls `_clear_gpu_state()` (which calls `torch.cuda.empty_cache()`),
but this may not cover PyTorch-internal state beyond the GPU cache (e.g.
autograd graph nodes, gradient accumulator state).

**Impact**: Possible GPU state corruption after cancellation, requiring a
pipeline reload to recover.

**Recommendation**: Test empirically — cancel during embedding multiple times
in a row, then run a clean diarization. If results are consistent, the risk
is low. If results drift or CUDA errors appear, consider wrapping the hook
in a `try/finally` or falling back to monkey-patching the embedding model's
`forward()` directly (same pattern as current segmentation patch).

### 3. `_Cancelled` raised from pyannote hook callback may be swallowed — **Medium risk**

The doc assumes `raise _Cancelled()` inside the pyannote hook callback
propagates out of `pipeline.__call__()`. If pyannote wraps hook calls in a
`try/except Exception` (e.g. for robustness or logging), the cancel signal
dies silently and the pipeline continues running.

**Impact**: Layer 1 cancel doesn't work. Layer 2 still catches it at the next
batch, so the overall system degrades gracefully (batch-level latency instead
of stage-boundary latency), but clustering-phase cancel is lost entirely.

**Recommendation**: Read the pyannote-audio source for `SpeakerDiarization.__call__`
to verify hook callback exception propagation. If pyannote catches exceptions,
Layer 1 is unreliable and the doc should document this limitation explicitly.

### 4. `hook=` kwarg version assumption — **Low risk**

The `hook=` callback API was added in pyannote-audio 3.x. If someone runs with
an older version, `hook=` will either be silently ignored (kwargs swallowed)
or raise a `TypeError`. The doc doesn't mention a version check or graceful
fallback.

**Impact**: Layer 1 silently disabled on old pyannote versions. Layer 2 still
works, so cancel is still functional but with no progress logging and no
clustering-phase cancel.

**Recommendation**: Add a runtime check at `load_model()` time:
```python
sig = inspect.signature(self._pipeline)
if "hook" not in sig.parameters:
    logger.warning("pyannote version does not support hook= callback — Layer 1 cancel disabled")
    self._supports_hook = False
```

### 5. Progress logging lost when `cancel_event is None` — **Low risk, usability**

The doc says: "If cancel_event is None: run without hooks, zero overhead."
This also disables Layer 1 progress logging. A caller who wants progress but
not cancellation has no way to get it — they'd need to pass a dummy
`threading.Event` that never fires.

**Impact**: Minor UX gap. Not a correctness issue.

**Recommendation**: Consider separating the hook callback from the cancel
check. Allow passing a `progress_callback` independently of `cancel_event`,
so progress logging is available without cancellation. This is out of scope
for the current design but worth noting for the pipeline integration.

### 6. Clustering cancel latency accepted — **Low risk, acknowledged**

Clustering is CPU-bound agglomerative work with no `nn.Module` calls. Cancel
waits until clustering finishes and the next pyannote hook fires. The doc
estimates "<1s typically" and accepts this.

**Impact**: For long audio with many speakers, clustering could take longer.
No hard bound.

**Recommendation**: Acceptable for now. If clustering becomes a problem,
the doc already notes patching `AgglomerativeClustering.__call__` as a third
layer. Don't implement speculatively.

### 7. Thread safety of per-call state on `self` — **Low risk, acknowledged**

`self._current_stage`, `self._batch_counter`, `self._hook_handles` are set per
`diarize()` call. Concurrent `diarize()` calls on the same instance would
corrupt this state. The doc flags this in Risk #3.

**Recommendation**: Acceptable for now. If concurrency is needed later, move
per-call state into a closure (as the doc suggests). The current design
matches `cancel_whisper` which has the same constraint.

## Model Pre-Download — Already Working

The project already has a working pre-download mechanism:

- **`pyann-models/download_pyannote_model.py`**: Downloads all 4 sub-models
  (segmentation-3.0, speaker-diarization-3.1, speaker-diarization-community-1,
  wespeaker-voxceleb-resnet34-LM) by setting `HF_HOME` before import and
  calling `Pipeline.from_pretrained()` / `Model.from_pretrained()`.

- **Both test scripts** (`test_cancel_diarize.py`, `run_test.py`) already set
  `os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)` pointing at `pyann-models/`
  before any HF imports.

- **`_resolve_hf_token()`** in the worker already falls back to "relying on
  local HF cache (HF_HOME)" when no token is found, so cached models work
  without a token at runtime.

`Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")` reads from the
`HF_HOME` cache when the model is already there — no network access needed at
runtime. The pre-download strategy is already in place and functional.

## Recommended Implementation Order

1. **Probe first** (before any code changes): Load the pipeline, print
   `type(pipeline._embedding)`, `dir(pipeline._embedding)`, and check if the
   inner `nn.Module` is accessible. This resolves Issue #1 empirically.
2. **Verify hook exception propagation** (before any code changes): Check the
   pyannote source for `SpeakerDiarization.__call__` to confirm hook-raised
   exceptions propagate. This resolves Issue #3.
3. **Implement `_find_inference_models()`**: Replace `_find_segmentation_model()`.
4. **Implement `_install_cancel_hooks()` / `_remove_cancel_hooks()`**: Layer 2
   PyTorch hooks + Layer 1 pyannote callback.
5. **Rewrite `diarize()`**: Use hooks instead of monkey-patch.
6. **Test**: Cancel during segmentation, cancel during embedding, cancel
   during clustering, re-run after cancel.
