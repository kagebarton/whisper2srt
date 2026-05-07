# Whisper Cancellation Prototype

This folder contains a **hook-based cancellation prototype** for the Whisper alignment/refinement worker. It demonstrates per-encoder-pass cancellation during speech-to-text alignment using PyTorch's `register_forward_pre_hook()` on the Whisper encoder instead of monkey-patching a non-existent `model.encode()` method.

## Overview

The prototype implements a `WhisperWorker` class that runs forced alignment and transcription using `stable-whisper` (the PyTorch-based stable-ts variant). Unlike `cancel_whisper` (which uses `load_faster_whisper()` and monkey-patches `model.encode()`), this version can cancel alignment or transcription **between encoder passes** without killing the process or reloading the model — by hooking directly into the PyTorch encoder module.

### Key Innovation: Hooking the Real Encoder

`stable_whisper.load_model()` returns the raw `whisper.model.Whisper` `nn.Module` directly (unlike `load_faster_whisper()` which wraps a CTranslate2 engine). There is no `.encode()` method on this object. Instead, the encoder is accessed as `model.encoder` (an `AudioEncoder` `nn.Module`). The prototype registers a **forward pre-hook** on `model.encoder` that:

1. Checks `cancel_event.is_set()` before each encoder forward pass (~100 tokens per pass, ~0.5–3s GPU time)
2. Raises `_CancelledInsideEncoder` when cancellation is requested
3. The exception unwinds through the alignment/transcription/refinement loop, aborting work while keeping the model loaded

This approach is necessary because monkey-patching `.encode()` is impossible (the method doesn't exist) and undesirable (the pure-PyTorch path calls `model.encoder(...)` directly, not a wrapper).

The hook applies to **all three operations** — `align()`, `refine()`, and `transcribe()` — because they all use the same encoder under the hood.

### Why This Is Different from `cancel_whisper`

| Feature | `cancel_whisper` | This prototype |
|---------|-----------------|----------------|
| Backend | `stable_whisper.load_faster_whisper()` (CTranslate2) | `stable_whisper.load_model()` (pure PyTorch) |
| Monkey-patch target | `model.encode()` method | N/A — `model.encode()` doesn't exist |
| Injection method | Method monkey-patch | `register_forward_pre_hook` on `model.encoder` |
| Cancellation target | Per-encoder-call (faster-whisper wrapper) | Per-encoder-forward-pass (PyTorch `nn.Module.__call__`) |
| Supported operations | `align()`, `refine()` | `align()`, `refine()`, `transcribe()` |
| Model weights survival | Yes | Yes |

## Files

| File | Purpose |
|------|---------|
| `whisper_worker.py` | Main prototype: `WhisperWorker` class with encoder hook-based cancellation for `align()`, `refine()`, and `transcribe()` |
| `test_cancel_whisper.py` | CLI test script demonstrating cancel-then-rerun proof for align, refine, and transcribe |
| `config.py` | `WhisperModelConfig` — PyTorch model configuration with empty `compute_type` |
| `hook-cancel-plan.md` | Original design specification and architecture rationale |
| `hook-cancel-review.md` | Code review findings comparing against `cancel_whisper` and `pipeline` |

## Model Configuration

### Default Model

The prototype is configured to use a local PyTorch Whisper model:

```
model_path: "<workspace_root>/models/large-v3-turbo.pt"  (absolute path, resolved from script location)
device: "auto"  # uses CUDA if available, else CPU
```

This is the **large-v3-turbo** model (a fine-tuned variant of OpenAI's Whisper large-v3). You can override it via `--model-path` on the command line or by constructing a custom `WhisperModelConfig`.

### The `compute_type` Field

The config includes `compute_type: str = ""` — this is **unused** by `stable_whisper.load_model()` (PyTorch Whisper automatically uses FP16 on CUDA). The field exists only for interface compatibility with `cancel_whisper` (where `compute_type="int8"` controls CTranslate2 quantization).

### Alignment Options

All alignment parameters are exposed in `config.py`:

| Field | Default | Purpose |
|-------|---------|---------|
| `language` | `"en"` | Language code for Whisper |
| `vad` | `True` | Voice activity detection |
| `vad_threshold` | `0.25` | VAD sensitivity threshold |
| `suppress_silence` | `True` | Remove silent segments from output |
| `suppress_word_ts` | `True` | Suppress word-level timestamps on silent audio |
| `only_voice_freq` | `True` | Filter non-voice frequencies before VAD |

### Refinement Options

| Field | Default | Purpose |
|-------|---------|---------|
| `refine_steps` | `"s"` | Refine segment starts (`s`) and/or ends (`e`) — `"se"` does both |
| `refine_word_level` | `False` | Refine at word-level instead of segment-level |

## Running the Test

### Prerequisites

1. Python environment with `stable-ts` and `openai-whisper` installed
2. A Whisper `.pt` model file (by default at `<workspace_root>/models/large-v3-turbo.pt`)
3. A vocal stem WAV file (output from audio separation)
4. A plain-text lyrics file (one or more lines, matching the vocal content)

### Basic Usage

```bash
# From the cancel_tests/whisper/ folder:
python test_cancel_whisper.py vocals.wav lyrics.txt

# Cancel alignment after 5 seconds (default):
python test_cancel_whisper.py vocals.wav lyrics.txt

# Cancel sooner (faster cancellation test):
python test_cancel_whisper.py vocals.wav lyrics.txt --cancel-after 2

# Cancel during refinement instead of alignment:
python test_cancel_whisper.py vocals.wav lyrics.txt --phase refine
```

### Custom Model Path

```bash
# Use a different local .pt model:
python test_cancel_whisper.py vocals.wav lyrics.txt \
  --model-path /path/to/model.pt

# Use a model by short name (will auto-download if not cached):
python test_cancel_whisper.py vocals.wav lyrics.txt \
  --model-path turbo
```

The default model path (`<workspace_root>/models/large-v3-turbo.pt`) is resolved from the script location, independent of working directory.

### Phase Selection

```bash
# Cancel during alignment (default):
python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --phase align

# Cancel during refinement instead:
python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --phase refine

# Note: Transcription (transcribe) is ALWAYS tested with cancellation in steps 7-8,
# regardless of --phase setting. The flag only controls whether refine (steps 5-6)
# is cancelled or runs to completion.
```

**Full test flow (8 steps):**

| Step | Operation | Cancel? | Notes |
|------|-----------|---------|-------|
| 1 | Model load | No | One-time cost |
| 2 | Align | Yes (if `--phase align`) | Timer fires, expects `AlignmentCancelledError` |
| 3 | Model survival check | N/A | Verifies model still loaded |
| 4 | Re-run align | No | Confirms model works |
| 5 | Refine | Yes (if `--phase refine`) | Timer fires, expects cancellation |
| 6 | Re-run refine | No | Confirms model works |
| 7 | Transcribe | Yes (always) | Timer fires, expects cancellation |
| 8 | Re-run transcribe | No | Confirms model works |

**Summary:** The test proves the model survives cancellation in all three operations — `align()`, `refine()`, and `transcribe()` — and can be reused immediately without reloading.

### Device Selection

```bash
# Force CPU:
python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --device cpu

# Force CUDA:
python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --device cuda

# Auto-detect (default):
python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --device auto
```

### What the Test Proves

**Phase 1 — Model Load:**
- Loads the Whisper model via `stable_whisper.load_model()`
- Times the load; caches the encoder module reference

**Phase 2 — Align with Cancellation:**
- Starts a timer to set `cancel_event` after `--cancel-after` seconds
- Calls `worker.align(vocal, lyrics, cancel_event)`
- The encoder pre-hook detects the cancel and raises `_CancelledInsideEncoder`
- Worker catches it, terminates orphaned AudioLoader FFmpeg subprocesses, clears GPU state
- Expects `AlignmentCancelledError` to be raised to caller

**Phase 3 — Model Survival Check:**
- Verifies `worker.model_loaded` is still `True`
- Proves the model weights survived the cancellation

**Phase 4 — Re-run Alignment (No Cancel):**
- Calls `align()` again with `cancel_event=None`
- Times the run; should complete quickly (model already loaded)
- Produces a `WhisperResult` with segments and word timestamps

**Phase 5 — Refine with Cancellation (if `--phase refine`):**
- First runs a successful `align()` (no cancel) to get a result
- Then runs `refine()` with a fresh cancel timer
- Same cancellation mechanism (encoder pre-hook fires)
- Expects `AlignmentCancelledError`

**Phase 6 — Re-run Refinement:**
- Re-aligns to get a fresh result
- Runs `refine()` to completion
- Proves model survived refinement cancellation

**Phase 7 — Transcribe with Cancellation (always runs):**
- Runs `worker.transcribe(vocal, cancel_event)` with a fresh cancel timer
- The encoder pre-hook fires during transcription's encoder passes
- Expects `AlignmentCancelledError`

**Phase 8 — Re-run Transcription:**
- Runs `transcribe()` again to completion
- Proves model survived transcription cancellation
- Produces a full `WhisperResult` with all segments and words

**Success criteria:** Phases 4, 6, and 8 complete without errors and produce valid results. The log should show `"Cancel detected before encode pass #N"` messages during cancelled phases.

## Programmatic Usage

```python
from cancel_tests.whisper.whisper_worker import WhisperWorker, AlignmentCancelledError
from cancel_tests.whisper.config import WhisperModelConfig
from pathlib import Path
import threading
import time

# Configure
config = WhisperModelConfig(
    model_path="/path/to/model.pt",
    device="cuda",
    language="en",
)

# Load model once
worker = WhisperWorker(config)
worker.load_model()

# --- Align with cancellation ---
cancel_event = threading.Event()
timer = threading.Timer(3.0, cancel_event.set)
timer.start()

try:
    result = worker.align(
        vocal_path=Path("vocals.wav"),
        lyrics_text="Here are the lyrics...",
        cancel_event=cancel_event,
    )
except AlignmentCancelledError:
    print("Alignment cancelled — model still loaded")

# Re-align without cancellation
result = worker.align(Path("vocals.wav"), "Lyrics...", cancel_event=None)
print(f"Alignment produced {len(result.segments)} segments")

# --- Refine with cancellation ---
cancel_event2 = threading.Event()
timer2 = threading.Timer(2.0, cancel_event2.set)
timer2.start()

try:
    refined = worker.refine(Path("vocals.wav"), result, cancel_event2)
except AlignmentCancelledError:
    print("Refinement cancelled")

# --- Transcribe with cancellation ---
# Transcribe directly from audio without lyrics (no alignment needed)
cancel_event3 = threading.Event()
timer3 = threading.Timer(4.0, cancel_event3.set)
timer3.start()

try:
    transcript = worker.transcribe(Path("vocals.wav"), cancel_event3)
except AlignmentCancelledError:
    print("Transcription cancelled")

# Re-transcribe without cancellation
transcript = worker.transcribe(Path("vocals.wav"), cancel_event=None)
print(f"Transcription produced {len(transcript.segments)} segments")

# --- Cleanup ---
worker.unload_model()
```

## AudioLoader FFmpeg Cleanup

A critical detail: when cancellation occurs during alignment, the `Aligner`'s normal cleanup (`self.audio_loader.terminate()`) is **skipped** because the exception exits the while-loop early. The orphaned `AudioLoader` still holds an FFmpeg subprocess (`_process` or `_extra_process`) that keeps writing raw PCM to stdout. When Python's GC eventually calls `AudioLoader.__del__()` → `terminate()`, it kills FFmpeg mid-stream, producing harmless but noisy "Broken pipe" / "Error submitting a packet" stderr messages.

The prototype suppresses these through **two defenses**:

1. **Stderr redirection patch:** `_patch_audioloader_stderr()` monkey-patches `AudioLoader._audio_loading_process()` to pass `stderr=subprocess.DEVNULL` when launching FFmpeg. This ensures any muxer errors never reach the terminal.

2. **Orphan termination:** `_terminate_orphaned_audioloaders()` walks Python's garbage collector to find live `AudioLoader` instances with running FFmpeg subprocesses and explicitly terminates them in the `_CancelledInsideEncoder` handler.

Both mechanisms are idempotent and safe; they can be ported back to `pipeline/workers/whisper_worker.py` unchanged.

**Note:** Refinement uses `prep_audio()` not `AudioLoader`, so no FFmpeg cleanup is needed — but the termination call is kept for belt-and-suspenders safety.

## Technical Notes

### Encoder Call Path Verification

The hook interception strategy was validated against the installed `stable_whisper` source:

- **Alignment:** `model.align()` → `Aligner.align()` → `compute_timestamps()` → `add_word_timestamps_stable()` → `model.encoder(mel.unsqueeze(0))` at `stable_whisper/timing.py:60`. This calls `AudioEncoder.__call__` → `forward()`. The `register_forward_pre_hook` fires here.

- **Refinement:** `model.refine()` → `refinement.inference_func` at `stable_whisper/alignment.py:661` calls `model(mel_segments, tokens)`, which dispatches to `Whisper.forward()` → `self.encoder(...)`. Hook fires.

- **Transcription:** `model.transcribe()` → `transcribe_stable()` → `DecodingTask._get_audio_features()` → calls `model.encoder(features)` (see `stable_whisper/transcription.py`). Hook fires on every encoder pass during the decoding loop.

### Counter Semantics

The `encode_counter` increments **before** the encoder forward runs (inside the `pre_hook`). This differs from `cancel_whisper`'s monkey-patch which increments **after** successful completion. The reported count includes the pass that was about to start but never finished — this is **intentional** (documented in logs as "encode passes started"). The counter is debug-only; correctness is unaffected.

### Exception Names

- Internal exception: `_CancelledInsideEncoder` (raised by the pre-hook, unwinds through `nn.Module.__call__` → `encoder.forward()` → alignment/refinement)
- Public exception: `AlignmentCancelledError` (re-raised to the caller after cleanup)

This maintains API compatibility with `cancel_whisper` test code.

### `align_and_refine` Contract

This prototype follows **`cancel_whisper`'s standalone-test contract**: `cancel_event.clear()` is called between the align and refine phases so a single cancel only kills the in-flight phase. The pipeline's variant intentionally **does not** clear — a cancelled song is discarded entirely.

If you port this back to pipeline, remember to **drop** the `cancel_event.clear()` line to match the pipeline's multi-phase cancellation propagation semantics.

**Note on `transcribe()`:** There is no `transcribe_and_refine` or two-phase variant — `transcribe()` is standalone and already supports cancellation directly. If you need transcription followed by refinement, you would call `transcribe()` first (which returns a `WhisperResult`) and then call `refine()` separately (which works on an existing result from alignment — `refine()` expects alignment output, not raw transcription). For forced alignment use cases, `align()` + `refine()` is the standard two-phase workflow; `transcribe()` is for lyrics-free transcription.

## Configuration Reference

| Setting | Default | Source |
|---------|---------|--------|
| Model path | `<workspace_root>/models/large-v3-turbo.pt` | `config.py` (resolved from file location) |
| Device | `auto` | `config.py` |
| Language | `en` | `config.py` |
| VAD enabled | `True` | `config.py` |
| VAD threshold | `0.25` | `config.py` |
| Refine steps | `"s"` | `config.py` |
| Refine word-level | `False` | `config.py` |

All alignment parameters mirror `snippets/stable_align.py`.

## Architecture Comparison

| Aspect | `cancel_whisper` (faster-whisper) | This prototype (stable-whisper PyTorch) |
|--------|-----------------------------------|----------------------------------------|
| Backend | `load_faster_whisper()` | `load_model()` |
| Model type | `faster_whisper.WhisperModel` (CTranslate2 C++) | `whisper.model.Whisper` (pure `nn.Module`) |
| Cancellation target | `model.encode()` Python wrapper | `model.encoder` (`nn.Module`) |
| Injection method | Method monkey-patch | `register_forward_pre_hook` |
| Cancellation granularity | Per `encode()` call | Per `encoder.forward()` pass (~100 tokens) |
| Supported operations | `align()`, `refine()` | `align()`, `refine()`, `transcribe()` |
| GPU memory persistence | Yes (CTranslate2 internal) | Yes (PyTorch `nn.Parameter` attributes) |
| AudioLoader cleanup required | Yes | Yes (identical pattern) |

## Relation to Other Folders

- `cancel_whisper/` — the original faster-whisper-based prototype with monkey-patching; this prototype replicates its API but uses a fundamentally different injection mechanism
- `pipeline/workers/whisper_worker.py` — the production worker; the hook approach here can be ported back (with the `align_and_refine` contract adjustment)
- `audio-separator/` — sister prototype for vocal/instrumental separation (uses forward pre-hooks on Roformer)
- `whisper-model/` — workspace-local model storage; the default `model_path` points to `large-v3-turbo.pt` here
