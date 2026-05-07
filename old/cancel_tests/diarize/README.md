# cancel-diarize: Mid-Diarization Cancellation Proof-of-Concept

This project demonstrates how to **cancel pyannote speaker diarization mid-computation without losing the loaded model**, using a two-layer hook architecture that covers both the segmentation and embedding phases. It follows the same cancellation patterns established in `cancel_separator` and `cancel_whisper`.

## The Problem

In a karaoke pipeline that uses pyannote for speaker diarization, `pipeline()` is a blocking call with **no cancellation API**:

1. The segmentation model processes audio in overlapping sliding-window chunks (~first few seconds)
2. The embedding model processes each detected speaker segment (**most wall time**)
3. Clustering groups embeddings into speakers (CPU-bound, typically <1s)
4. The pipeline is expensive to load (~10-30s on GPU, ~60s+ on CPU)
5. If the user cancels, the model stays loaded but the computation can't be interrupted

**Old behavior** (monkey-patch design): cancel only worked during segmentation. Setting the cancel event during embedding had no effect — the run completed normally.

**New behavior** (two-layer hooks): cancel works in all phases — segmentation, embedding, and clustering.

## The Solution: Two-Layer Hook Architecture

Cancel checks are installed at two complementary layers, both driving the same `threading.Event` and raising the same internal `_Cancelled` exception:

| Layer | Mechanism | Fires at | Latency bound |
|-------|-------------------------------------------|---------------------------------------|------------------------|
| 1 | pyannote `hook=` callback on `pipeline(...)`| Stage boundaries + progress events | End of current stage |
| 2 | PyTorch `register_forward_pre_hook` on each inference `nn.Module` | Before every batch forward pass of segmentation and embedding models | One batch (~0.1–1s) |

Layer 2 is what gives the "cancel within embedding" behavior. Layer 1 gives progress logging for free and covers any stage that doesn't go through a hooked `nn.Module` (e.g. clustering).

### Why hooks instead of monkey-patching?

The old monkey-patch design only covered `pipeline._segmentation.model.forward()`. Most wall time is in the embedding phase, where cancel was invisible. `register_forward_pre_hook` is PyTorch's documented API for intercepting forward calls — it works on any `nn.Module`, is removable via `handle.remove()`, and doesn't require method-swapping. This lets us hook both models cleanly.

## Architecture

```
CancelableDiarizeWorker (same process, like cancel_whisper)
├── Loads pyannote SpeakerDiarization pipeline at startup
├── _find_inference_models()
│   ├── Segmentation: pipeline._segmentation.model (nn.Module)
│   └── Embedding: pipeline._embedding.model_ or .model (nn.Module)
├── diarize(vocal_path, cancel_event)
│   ├── _install_hooks(cancel_event)
│   │   ├── Layer 2: register_forward_pre_hook on segmentation model
│   │   ├── Layer 2: register_forward_pre_hook on embedding model
│   │   └── Layer 1: pyannote hook= callback (always, for progress)
│   ├── _run_diarization(vocal_path, hook=pyannote_hook, **kwargs)
│   │   └── pipeline(vocal_path, hook=pyannote_hook)
│   │       ├── Segmentation: model.forward() → pre_hook fires per batch
│   │       ├── Embedding: model.forward() → pre_hook fires per batch
│   │       ├── Clustering: no nn.Module → Layer 1 hook fires at stage end
│   │       └── Progress: hook callback logs stage/progress
│   ├── On _Cancelled:
│   │   ├── _clear_gpu_state()
│   │   ├── Log stage + batch count
│   │   └── Raise DiarizationCancelledError
│   ├── On normal return:
│   │   └── Check _cancel_requested fallback flag
│   └── finally: _remove_cancel_hooks()
└── unload_model()
    └── Del pipeline, clear GPU cache
```

## Cancel Signal Flow

```
Main thread                          pyannote internals
───────────                          ──────────────────
threading.Event.set()
    │
    ▼
Layer 2 — PyTorch forward_pre_hook (per batch):
    def pre_hook(module, inputs):
        if cancel_event.is_set():
            raise _Cancelled()          ← exception thrown
        batch_counter[label] += 1

Layer 1 — pyannote hook callback (per stage):
    def pyannote_hook(step_name, ...):
        _current_stage = step_name
        if cancel_event.is_set():
            _cancel_requested = True    ← fallback flag
            raise _Cancelled()          ← exception thrown

Exception unwinds:
    pre_hook → nn.Module.__call__ → pyannote internals → pipeline()
    │
    ▼
Caught in CancelableDiarizeWorker.diarize():
    - _remove_cancel_hooks() (finally block)
    - _clear_gpu_state()
    - Raise DiarizationCancelledError
    - Model weights intact ✓
    - Ready for next job ✓
```

## Comparison with Other Prototypes

| Aspect | cancel_separator | cancel_whisper | cancel_diarize |
|--------|-----------------|----------------|----------------|
| **Target library** | audio-separator (Roformer) | stable-ts (faster-whisper) | pyannote (SpeakerDiarization) |
| **Process model** | Subprocess (multiprocessing.Process) | Same process | Same process |
| **Cancel signaling** | threading.Event → Pipe → subprocess | threading.Event directly | threading.Event directly |
| **Cancel mechanism** | Monkey-patch `model_run.forward()` | Monkey-patch `model.encode()` | `register_forward_pre_hook` + pyannote `hook=` callback |
| **Cancel coverage** | Segmentation model only | Encode passes only | Both segmentation AND embedding models + clustering stage boundary |
| **Cancel granularity** | Per-chunk (~10-15s audio, ~0.5-2s GPU) | Per-encode-pass (~1-3s GPU align, ~0.1-0.5s refine) | Per batch (~0.1-1s) + per stage boundary |
| **Cancel exception** | `_CancelledInsideDemix` | `_CancelledInsideEncode` | `_Cancelled` |
| **Propagated exception** | `WorkerCancelledError` | `AlignmentCancelledError` | `DiarizationCancelledError` |
| **Model weights survive?** | Yes (GPU class attributes) | Yes (CTranslate2 instance attributes) | Yes (pyannote pipeline attributes) |
| **Progress logging** | None built-in | None built-in | Free via Layer 1 pyannote hook |
| **Complexity** | Higher (subprocess + Pipe) | Lower (same process, Event only) | Lower (same process, Event + hooks) |
| **Caption output** | None | None | SRT + ASS from diarization turns |

## Files

| File | Purpose |
|------|---------|
| `config.py` | `DiarizeConfig` dataclass (HF token, device, speaker params, ASS styling) |
| `context.py` | `StageContext` (per-job state bag) + `CancelledError` |
| `caption.py` | SRT + ASS generation from diarization turn dicts |
| `stages/base.py` | `PipelineStage` protocol + `BaseStage` (delayed cancel default) |
| `stages/diarize_stage.py` | Diarization adapter stage (delegates to CancelableDiarizeWorker) |
| `workers/cancelable_diarize_worker.py` | **Core**: in-process worker with two-layer hook cancellation |
| `orchestrator.py` | Ties stages together with worker lifecycle management |
| `test_cancel_diarize.py` | Standalone test: cancel mid-diarization, verify model survives, re-run |
| `run_test.py` | Interactive + auto test runner with caption output (SRT/ASS/both) |
| `probe_pipeline.py` | Pre-implementation probe: validates 4 assumptions the architecture rests on |
| `unified-cancel-architecture.md` | Full design spec for the two-layer hook architecture |
| `README.md` | This file |

## Usage

### Quick test (standalone, no orchestrator)

```bash
# From the cancel_tests/diarize/ folder
python test_cancel_diarize.py /path/to/vocals.wav --cancel-after 5

# Cancel later to test embedding-phase cancel (previously impossible)
python test_cancel_diarize.py /path/to/vocals.wav --cancel-after 8

# With HF token file
python test_cancel_diarize.py /path/to/vocals.wav --hf-token ~/.cache/hf_token.txt

# Specify number of speakers
python test_cancel_diarize.py /path/to/vocals.wav --num-speakers 2
```

### Caption output test

```bash
# From the cancel_tests/diarize/ folder

# Generate both SRT and ASS from diarization
python run_test.py /path/to/vocals.wav both

# Generate only SRT
python run_test.py /path/to/vocals.wav srt

# Cancel then re-run, writing ASS output
python run_test.py /path/to/vocals.wav ass --auto --cancel-after 5

# Cancel deterministically when embedding stage starts
python run_test.py /path/to/vocals.wav both --auto --cancel-stage embedding
```

### Pre-implementation probe

```bash
# From the cancel_tests/diarize/ folder
# Validate architecture assumptions before relying on the two-layer design
python probe_pipeline.py /path/to/vocals.wav
```

### Output files

When `caption_format` is `srt` or `both`, writes:
- `{stem}.diarization.srt` — SRT with speaker labels and time ranges

When `caption_format` is `ass` or `both`, writes:
- `{stem}.diarization.ass` — ASS with speaker-labeled Dialogue events

Both files are placed next to the input vocal audio file.

Expected test output:
```
STEP 1: Loading pyannote diarization pipeline
Pipeline loaded in 15.2s
Segmentation: SegmentationModel, Embedding: WeSpeakerResNet

STEP 2: Running diarization (will cancel after 5s)
Progress: segmentation 5/12
✓ Diarization cancelled after 5.3s (as expected)
Cancelled during stage=segmentation after 5 forward passes

STEP 3: Verify model is still loaded after cancellation
Model is still loaded: True
✓ Model survived cancellation — no reload needed

STEP 4: Re-running diarization (model should still be loaded)
Progress: segmentation 12/12
Progress: embedding 24/24
✓ Second diarization completed in 8.7s
Speakers detected: ['SPEAKER_00', 'SPEAKER_01']
Turns: 42

✓✓✓ MODEL WAS STILL LOADED AFTER CANCELLATION — NO RELOAD NEEDED ✓✓✓
```

## Key Design Decisions

### 1. Same-process (like cancel_whisper, unlike cancel_separator)

pyannote runs in the same process as the caller, so we can use a `threading.Event` directly — no subprocess or Pipe bridging needed. This matches the cancel_whisper pattern.

### 2. Two hook layers instead of monkey-patching

The old monkey-patch design only covered segmentation. `register_forward_pre_hook` is PyTorch's documented API for intercepting forward calls — it works on any `nn.Module`, is removable via `handle.remove()`, and doesn't require method-swapping. This lets us hook both models cleanly.

### 3. Hook install/remove in try/finally

PyTorch hook handles are installed before `pipeline()` and removed in a `finally` block. This ensures:
- Hooks are always cleaned up, even on cancellation or error
- The model is left in its original state for the next job
- No leaked state between jobs

### 4. Layer 1 always installed (progress logging for free)

The pyannote `hook=` callback is always installed, even without a cancel event. The cancel check inside it is guarded on `cancel_event is not None`, so there's no overhead when cancellation isn't needed. This gives stage-boundary progress logging for free.

### 5. Exception unwinding preserves model weights

When `_Cancelled` is raised from a PyTorch pre_hook:
- Pre-hooks fire before `forward()` runs, so no partial computation to unwind
- The exception unwinds through pyannote internals → `pipeline()`
- The model weights are **attributes** on the pyannote pipeline — on GPU/CPU, not on the Python call stack
- `torch.cuda.empty_cache()` cleans up any remaining GPU intermediate state

### 6. Fallback flag for swallowed exceptions

If pyannote wraps the hook callback in try/except internally, `_Cancelled` from Layer 1 would die silently. The callback also sets `self._cancel_requested` as a flag, and `diarize()` checks it after `_run_diarization()` returns. Layer 2 (PyTorch pre_hooks) is unaffected since those raise from inside `nn.Module.__call__`, below pyannote's layer.

### 7. Model discovery with graceful degradation

`_find_inference_models()` tries multiple paths to find both the segmentation and embedding `nn.Module`s. If either can't be found, that stage degrades to Layer-1-only cancel granularity (stage boundaries only, not per-batch). The worker still functions — it just loses fine-grained cancel for that phase.

### 8. Caption generation from diarization output

Unlike the other prototypes (which don't produce caption files), this prototype includes caption generation. Diarization turns are converted to:
- **SRT**: Each turn becomes a subtitle entry with `[SPEAKER_00] (duration)` text
- **ASS**: Each turn becomes a Dialogue event with speaker label and lead-in/lead-out timing

This provides a usable output even without transcription — the speaker labels and timing information are useful for reviewing who speaks when.

## HuggingFace Token Setup

pyannote's speaker diarization model is gated — you need a HuggingFace token with access granted:

1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the license
2. Visit https://huggingface.co/pyannote/segmentation-3.0 and accept the license
3. Get a token from https://huggingface.co/settings/tokens
4. Set it one of:
   - `HF_TOKEN` environment variable
   - `--hf-token /path/to/token.txt` flag
   - `DiarizeConfig(hf_token_path=...)`

**Model cache location:** By default, pyannote models are cached at `<workspace_root>/models/` (HF_HOME is set to this location automatically when the test script runs). This is resolved from the script location, independent of working directory.

## Limitations

- **Not instant**: Cancellation takes effect at the next hook firing point. Layer 2 fires per batch (~0.1-1s), Layer 1 fires at stage boundaries (typically <1s for clustering). Within a single forward pass, cancel has no effect.
- **Clustering cancel latency**: Clustering is CPU-bound and doesn't call a hooked `nn.Module`. Cancel waits for clustering to finish and the next pyannote hook event fires. Typically <1s.
- **Model discovery heuristics**: The worker uses heuristics to find both models inside the pipeline. If pyannote changes the internal structure, `_find_inference_models()` may need updating.
- **No partial result**: Cancellation always produces a full discard. A partially-completed diarization could theoretically yield partial speaker labels, but pyannote doesn't expose intermediate results from within the loop.
- **Not thread-safe for concurrent calls**: Hook state lives on `self`. This matches the current design; only one `diarize()` call should be active at a time per worker instance.
