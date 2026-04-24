# cancel_separator: Mid-Separation Cancellation Proof-of-Concept

This project demonstrates how to **cancel audio-separator mid-separation without losing the loaded model**, using the stage-based pipeline architecture from the pipeline-robustness-fixes plan (A1).

## The Problem

In the current `pikaraoke` production code, `cancel_active()` for the stemming step is **delayed** — it sets a flag and waits for `separate()` to finish entirely. This is because:

1. audio-separator has **no cancellation API** — `separator.separate()` is a blocking call
2. Killing the `StemWorker` subprocess forces a model reload (~10-30s on GPU, ~60s+ on CPU)
3. The model is loaded once at worker startup and stays in GPU memory across jobs

So the current behavior: user clicks cancel → nothing visible happens until the entire separation finishes (could be 30-120s for a long song).

## The Solution: Per-Chunk Cancellation via Monkey-Patch

The Roformer model's `demix()` method processes audio in **chunks** in a loop:

```python
# audio_separator/separator/architectures/mdxc_separator.py:320
for i in tqdm(range(0, mix.shape[1], step)):
    part = mix[:, i : i + chunk_size]
    ...
    x = self.model_run(part.unsqueeze(0))[0]  # ← one forward pass per chunk
    ...
```

Each `self.model_run(...)` call goes through `nn.Module.__call__()` → `self.forward()`. By **monkey-patching `model_run.forward()`** to check a cancel signal before each call, we can abort separation between chunks while keeping the model weights intact on GPU.

Why patch `forward()` not `__call__()`?
- Python looks up dunder methods (`__call__`) on the **class**, not the instance
- Instance-level `__call__` patches are silently ignored
- `forward()` is a regular method — instance patches work correctly
- `nn.Module.__call__` internally calls `self.forward(...)`, so our patch runs at the exact same point

## Architecture

```
StageOrchestrator (main process)
├── StemSeparationStage (stateless adapter, reused per job)
│   └── CancelableStemWorker (persistent subprocess reference)
│       ├── Loads model at startup, stays loaded across jobs
│       ├── Receives (wav_path, output_dir) via Pipe
│       ├── Cancel signal via separate Pipe (main → worker)
│       ├── Before separate(): patches model_run.forward() with cancel check
│       ├── After separate()/cancel: restores original forward()
│       └── Model weights survive cancellation (GPU attributes, not stack locals)
├── FFmpegExtractStage (spawns ffmpeg, immediate cancel via Popen.kill)
├── FFmpegTranscodeStage (spawns ffmpeg, immediate cancel via Popen.kill)
└── Orchestrator thread (real threading.Thread)
    ├── Pulls next song from pending queue
    ├── Creates per-job StageContext (tmp_dir, artifacts, cancel_event)
    ├── Runs stages sequentially, checks cancel between stages
    └── On cancel: stage.cancel() → cleanup → next job (worker still alive)
```

## Cancel Signal Flow

```
Main process                     Worker subprocess
─────────────                    ─────────────────
threading.Event.set()
        │
        ▼
_forward_cancel thread
        │
        ▼
cancel Pipe.send(1)  ──────►   cancel_recv.poll(0)
                                    │
                                    ▼
                              patched forward():
                                if cancel_recv.poll(0):
                                    raise _CancelledInsideDemix
                                else:
                                    return original_forward(...)
                                    
                              Exception unwinds:
                                forward() → demix() → separate()
                                    │
                                    ▼
                              Caught in worker main loop:
                                - Clear GPU state
                                - Send ("cancelled",) on result Pipe
                                - Model weights intact ✓
                                - Ready for next job ✓
```

## Files

| File | Purpose |
|------|---------|
| `context.py` | `StageContext` (per-job state bag) + `CancelledError` |
| `stages/base.py` | `PipelineStage` protocol + `BaseStage` (delayed cancel default) |
| `stages/ffmpeg_extract.py` | Extract audio from video (immediate cancel: kill Popen) |
| `stages/ffmpeg_transcode.py` | Transcode WAV stems to M4A (immediate cancel: kill Popen) |
| `stages/stem_separation.py` | Stem separation adapter (delayed cancel: set flag, wait for chunk) |
| `workers/cancelable_stem_worker.py` | **Core**: persistent worker with per-chunk cancel via `forward()` patch |
| `orchestrator.py` | Ties stages together with worker lifecycle management |
| `run_test.py` | Interactive + auto test runner |
| `test_cancel_standalone.py` | Minimal standalone test (no pikaraoke imports needed) |

## Usage

### Quick test (recommended)

The standalone test requires only `audio-separator` to be installed:

```bash
# From the monkeypatch_cancel/cancel_separator/ folder

# Basic test: cancel after 5 seconds, then re-run to prove model survives
python test_cancel_standalone.py /path/to/song.mp4 --cancel-after 5

# Cancel sooner (if you have a fast GPU)
python test_cancel_standalone.py /path/to/song.mp4 --cancel-after 2
```

Models are resolved from `<workspace_root>/models/` (absolute path, independent of working directory).

Expected output:
```
STEP 1: Loading audio-separator model
Model loaded in 12.3s

STEP 2: Running separation (will cancel after 5s)
Patched model_run.forward() with cancel check
Waiting 5.0s before cancelling...
>>> SETTING CANCEL EVENT <<<
Cancel detected before chunk #3 — aborting!
✓ Separation cancelled between chunks (as expected)
Model is still loaded: True

STEP 3: Re-running separation (model should still be loaded)
✓ Second separation completed in 28.1s
✓✓✓ MODEL WAS STILL LOADED — NO RELOAD NEEDED ✓✓✓
```

### Interactive test (full pipeline)

```bash
# From the monkeypatch_cancel/cancel_separator/ folder
# Start the full stage pipeline with interactive controls
python run_test.py /path/to/song.mp4 --direct --cancel-after 5
```

### Auto test (multiple iterations)

```bash
# From the monkeypatch_cancel/cancel_separator/ folder
# Cancel and re-process 3 times to verify consistency
python run_test.py /path/to/song.mp4 --auto --cancel-after 5 --loop 3
```

## Key Design Decisions

### 1. Monkey-patch `forward()`, not `__call__()`

Python's method resolution for dunder methods always checks the class, not the instance. `model_run.__call__ = patched` is silently ignored. But `forward()` is a regular method — instance-level patches work. Since `nn.Module.__call__` calls `self.forward()`, our patch runs at exactly the right time.

### 2. Pipe-based cancel signaling between processes

The main process and worker subprocess can't share a `threading.Event` (they're separate processes). We bridge this with:
- A `cancel_send` / `cancel_recv` Pipe pair (multiprocessing.Pipe)
- A `_forward_cancel` daemon thread that watches the `threading.Event` and writes to the Pipe when set

This is clean, portable, and avoids shared-memory complications.

### 3. Patch-and-restore pattern

The forward() patch is applied **before** `separator.separate()` and removed in a **finally** block. This ensures:
- The original forward() is always restored, even on cancellation or error
- The model_run object is left in its original state for the next job
- No residual state from the patch leaks between jobs

### 4. Exception unwinding preserves model weights

When `_CancelledInsideDemix` is raised inside `forward()`:
- It unwinds through `demix()` → `separator.separate()` → `_separate_file()`
- The `with torch.no_grad():` context manager catches and re-raises properly
- The model weights are **class attributes** on `self.model_run` — they're on the GPU, not on the Python call stack
- Only the **intermediate tensors** from the current chunk are lost (they're stack locals that get garbage collected)
- `clear_gpu_cache()` and `clear_file_specific_paths()` clean up any remaining state

### 5. Worker lifecycle is manager-scoped, not stage-scoped

From the A1 architecture rules:
1. **Worker subprocess lifecycle is owned by the orchestrator, not by stages.** Stages hold a reference, never create/destroy.
2. **Worker lives outside the per-job StageContext.** Context is per-job; worker references are manager-scoped.
3. **Delayed cancel semantics are preserved.** The stage's `cancel()` sets a flag — never calls `worker.kill()`.
4. **Eager-restart-on-death stays in the orchestrator.** If the worker somehow dies, the orchestrator restarts it before the next job.

## How This Would Integrate Into Production

The production `StemWorker._stem_worker_main()` would add:
1. A `cancel_recv` Pipe argument
2. The `forward()` patching logic before calling `_separate_in_worker()`
3. A `("cancelled",)` result message type

The production `ProcessingManager._run_pipeline()` would change the STEMMING step to:
1. Pass `cancel_event` to `self._stem_worker.separate()`
2. Handle `WorkerCancelledError` → raise `_CancelledError()`

No other changes needed — the `cancel_active()` method already sets `state.cancelled = True`, which is exactly what triggers the `threading.Event` → Pipe → patched `forward()` chain.

## Limitations

- **Not instant**: Cancellation takes effect between chunks, not mid-inference. For a typical 3-minute song with the Roformer model, chunks are ~10-15 seconds of audio, so the worst-case delay is the time for one chunk's forward pass (~0.5-2s on GPU, ~2-5s on CPU).
- **Roformer-specific**: The injection point (`self.model_run(part.unsqueeze(0))[0]`) is specific to the Roformer demix path in `mdxc_separator.py`. Other architectures (MDX, VR, Demucs) have different loop structures that would need different injection points.
- **Monkey-patching is fragile**: If audio-separator changes the demix() loop structure, the patch may need updating. A proper cancellation API in audio-separator would be preferable long-term.
