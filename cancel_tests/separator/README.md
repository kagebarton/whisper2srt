# Separator Cancellation Prototype

This folder contains a **hook-based cancellation prototype** for the stem separation worker. It demonstrates per-chunk cancellation during audio separation using PyTorch's `register_forward_pre_hook()` instead of monkey-patching `model_run.forward()`.

## Overview

The prototype implements a `StemWorker` class that runs audio separation in a persistent subprocess. Unlike the production worker (in `pipeline/workers/stem_worker.py`), this version can cancel separation **between audio chunks** without killing the subprocess or reloading the model.

### Key Innovation: Forward Pre-Hooks

The Roformer model (`model_run`) is a `torch.nn.Module`. During demixing, it processes audio in chunks via `self.model_run(part.unsqueeze(0))[0]`. This prototype registers a **forward pre-hook** on `model_run` that:

1. Polls a multiprocessing Pipe for a cancel signal before each chunk
2. Raises `_CancelledInsideDemix` when cancellation is requested
3. The exception unwinds through the demix loop, aborting separation while keeping the model loaded in GPU memory

This is cleaner and more maintainable than monkey-patching `forward()` because it uses PyTorch's documented extension API with canonical cleanup via `RemovableHandle.remove()`.

## Files

| File | Purpose |
|------|---------|
| `stem_worker.py` | Main prototype: `StemWorker` class with hook-based cancellation |
| `run_test.py` | CLI test script demonstrating cancel-then-rerun proof |
| `hook-cancel-plan.md` | Original design specification and architecture |
| `hook-cancel-review.md` | Code review findings and corrections from production comparison |

## Model: `vocals_mel_band_roformer.ckpt`

**This prototype is specifically designed and tested with the `vocals_mel_band_roformer.ckpt` model located in the `audio-separator/models/` directory.**

### Model Details

- **Filename:** `vocals_mel_band_roformer.ckpt`
- **Path:** `<workspace_root>/audio-separator/models/vocals_mel_band_roformer.ckpt`
- **Config:** `vocals_mel_band_roformer.yaml` (same directory)
- **Type:** Mel-Band Roformer trained for vocal separation
- **Stems:** Produces `(vocals)` and `(other)` stems (non-karaoke variant)

### Why This Model Matters

The stem identification logic in `stem_worker.py` (lines 467–477) includes the `(other)` pattern check (`is_other = "(other)" in lower`) which is **required** for non-karaoke MelBand Roformer models. The prototype was verified against this specific model file; using karaoke variants or other architectures may require adjustments to stem matching.

## Running the Test

### Prerequisites

1. Python environment with `audio-separator` installed
2. `vocals_mel_band_roformer.ckpt` and its `.yaml` config in `audio-separator/models/`
3. A test audio file (WAV format recommended)

### Basic Usage

```bash
# From the workspace root:
python -m cancel_tests.separator.run_test path/to/input.wav

# With custom cancel timing (default: cancel after 3 seconds):
python -m cancel_tests.separator.run_test path/to/input.wav --cancel-after 5.0
```

### Custom Model Directory/Name

If your model isn't in the default location (`./audio-separator/models`) or you want to use a different filename:

```bash
python -m cancel_tests.separator.run_test input.wav \
  --model-dir /path/to/models \
  --model-name vocals_mel_band_roformer.ckpt
```

**Default values:** `--model-dir` defaults to `./audio-separator/models`, `--model-name` defaults to `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` (the karaoke variant). When using the `vocals_mel_band_roformer.ckpt` file, you **must** specify `--model-name` unless you change the default in `stem_worker.py`.

### What the Test Proves

**Phase A — Cancellation:**
- Starts the worker subprocess and waits for model load
- Schedules a cancellation after `--cancel-after` seconds
- Calls `worker.separate()` with a `threading.Event`
- Expects `WorkerCancelledError` to be raised
- Verifies the subprocess is still alive (model survived)

**Phase B — Re-run:**
- Immediately calls `worker.separate()` again without cancellation
- Proves the model is still loaded by measuring fast completion time
- Confirms no subprocess restart was needed

**Success criterion:** Phase B completes quickly and produces output files in the output directory (`vocal.wav` and `instrumental.wav`).

## Programmatic Usage

```python
from cancel_tests.separator.stem_worker import StemWorker, WorkerCancelledError
from pathlib import Path
import threading

worker = StemWorker(
    temp_dir="/tmp/separator",
    model_dir="./audio-separator/models",
    model_name="vocals_mel_band_roformer.ckpt",
    log_level=logging.INFO,
)
worker.start()

# Give model time to load
import time; time.sleep(5)

# Cancel after 2 seconds
cancel_event = threading.Event()
timer = threading.Timer(2.0, cancel_event.set)
timer.start()

try:
    vocal, instrumental = worker.separate(
        Path("input.wav"),
        Path("output/"),
        cancel_event=cancel_event
    )
except WorkerCancelledError:
    print("Separation cancelled — model still loaded")

# Re-run without cancellation
vocal, instrumental = worker.separate(Path("input.wav"), Path("output2/"))
print(f"Vocal: {vocal}, Instrumental: {instrumental}")

worker.stop()
```

## Configuration Notes

- **Output format:** Fixed to `wav` (line 53 in `stem_worker.py`)
- **Default model (code):** `DEFAULT_MODEL_NAME = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"` — **override this when using `vocals_mel_band_roformer.ckpt`**
- **Default model dir:** `DEFAULT_MODEL_DIR = "./audio-separator/models"` (relative to CWD)
- **Logger name:** `cancel_tests.separator.stem_worker` (differs from production's `pipeline.workers.stem_worker`)

## Architecture Reference

See `hook-cancel-plan.md` for the full design rationale and `hook-cancel-review.md` for the detailed review against the production implementation. Key distinction:

| Aspect | Production (`pipeline/workers/stem_worker.py`) | Prototype (`cancel_tests/separator/stem_worker.py`) |
|--------|-----------------------------------------------|----------------------------------------------------|
| Injection mechanism | Monkey-patch `model_run.forward` | `register_forward_pre_hook` on `model_run` |
| Cancellation granularity | Between full separations | Between audio chunks within a separation |
| Subprocess lifetime | One per session | One per session (unchanged) |
| Model persistence after cancel | No (model survives but cancellation only between jobs) | **Yes** — model stays loaded, next job runs immediately |

## Relation to `audio-separator/`

The `audio-separator/` folder (at the workspace root) contains the model files used by this prototype. The `vocals_mel_band_roformer.ckpt` file is the **recommended model** for testing this prototype because:

1. It is the non-karaoke MelBand Roformer variant (outputs `(vocals)` / `(other)`)
2. The prototype's stem identification logic explicitly handles the `(other)` stem label
3. It has been verified to work with the forward pre-hook injection pattern

When using this model, ensure both `.ckpt` and `.yaml` files are present in the `audio-separator/models/` directory or the path you specify via `--model-dir`.
