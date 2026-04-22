# cancel-whisper: Mid-Alignment Cancellation Proof-of-Concept

This project demonstrates how to **cancel stable-ts `model.align()` and `model.refine()` mid-computation without losing the loaded model**, using a monkey-patch on `model.encode()`.

This is the stable-ts equivalent of `cancel_test/` (which solves the same problem for audio-separator).

## The Problem

In the current `snippets/stable_align.py`, `model.align()` and `model.refine()` are blocking calls with **no cancellation API**:

1. `model.align()` processes lyrics in batches of ~100 tokens (the `token_step` parameter), calling `model.encode()` once per batch
2. `model.refine()` iteratively mutes audio sections and calls `model.encode()` in a binary-search loop
3. The faster-whisper model is expensive to load (~10-30s on GPU, ~60s+ on CPU)
4. If the user cancels, the model stays loaded in memory but the current computation can't be interrupted

So the current behavior: user clicks cancel → nothing visible happens until the entire alignment finishes (could be 30-120s for a long song).

## The Solution: Per-Encode-Pass Cancellation via Monkey-Patch

Both `align()` and `refine()` call `model.encode()` (the expensive GPU forward pass) in their inner loops:

```python
# stable_whisper/non_whisper/alignment.py:672
output = self.inference_func(audio_segment, input_word_tokens)
# → inference_func calls model.encode(features) for faster-whisper

# stable_whisper/non_whisper/refinement.py:291
token_probs = self.inference_func(audio_segment, text_tokens)
# → inference_func calls model.encode(features) for faster-whisper
```

By **monkey-patching `model.encode()`** to check a `threading.Event` before each call, we can abort alignment between encoder passes while keeping the model weights intact.

### Why `model.encode()`?

- It's the single expensive GPU call shared by both `align()` and `refine()` — one patch covers both
- It's a regular method (not a dunder like `__call__`), so instance-level monkey-patching works correctly (same reason we patch `forward()` not `__call__()` in the audio-separator case)
- It's called once per iteration of the Aligner's `while` loop and once per iteration of the Refiner's inner `while` loop

### Why not a subprocess like audio-separator?

The audio-separator cancel_test uses a subprocess + Pipe bridging because the model is loaded in the worker subprocess — the main process can kill the worker without losing its own state.

For stable-ts, the model is loaded in the **same process** (same as audio-separator, actually), but there's no need for a subprocess because:
- `model.encode()` is a regular method we can patch in-process
- The exception unwinds cleanly through the call stack
- We can use `threading.Event` directly (no Pipe bridging needed)
- The model weights survive as instance attributes on the `WhisperModel`

## Architecture

```
CancelableWhisperWorker (same process)
├── Loads faster-whisper model at startup via stable_whisper.load_faster_whisper()
├── align(vocal_path, lyrics_text, cancel_event)
│   ├── Patches model.encode() with cancel check before each call
│   ├── Calls model.align(vocal_path, lyrics_text, ...)
│   │   └── Aligner while loop:
│   │       for each token batch:
│   │         inference_func(audio_segment, word_tokens)
│   │           → model.encode(features) ← CANCEL CHECK HERE
│   │           → model.add_word_timestamps(...)
│   ├── On cancel: _CancelledInsideEncode unwinds through call stack
│   ├── Catches exception, restores original encode(), returns None
│   └── Model weights intact ✓ (CTranslate2 model on GPU/CPU)
├── refine(vocal_path, result, cancel_event)
│   ├── Patches model.encode() with cancel check before each call
│   ├── Calls model.refine(vocal_path, result, ...)
│   │   └── Refiner._refine():
│   │       for each word group:
│   │         while not all finished:
│   │           get_prob(audio_segment, text_tokens)
│   │             → model.encode(features) ← CANCEL CHECK HERE
│   ├── On cancel: same unwind pattern
│   └── Model weights intact ✓
└── align_and_refine(vocal_path, lyrics_text, cancel_event)
    └── Convenience: calls align() then refine(), sharing cancel_event
```

## Cancel Signal Flow

```
Main thread                          stable-ts internals
───────────                          ───────────────────
threading.Event.set()
│
▼
cancelable_encode():
  if cancel_event.is_set():
    raise _CancelledInsideEncode     ← exception thrown instead of
  else:                                running the encoder
    return original_encode(...)

Exception unwinds:
encode() → inference_func() → _compute_timestamps()
→ while loop → Aligner.align() → model.align()
│
▼
Caught in CancelableWhisperWorker.align():
  - Restore original model.encode()
  - Clear GPU state (safety net)
  - Raise AlignmentCancelledError
  - Model weights intact ✓
  - Ready for next job ✓
```

## Comparison with audio-separator cancel_test

| Aspect | audio-separator (cancel_test) | stable-ts (cancel_whisper) |
|--------|-------------------------------|----------------------------|
| **Runs in subprocess?** | Yes (CancelableStemWorker) | No (same process) |
| **Cancel signaling** | threading.Event → Pipe → subprocess | threading.Event directly |
| **Patch target** | `model_run.forward()` | `model.encode()` |
| **Cancel granularity** | Per-chunk (~10-15s audio, ~0.5-2s GPU) | Per-token-batch (~5-10s audio, ~1-3s GPU) for align; per-binary-search-step (~0.1-0.5s GPU) for refine |
| **Model weights survive?** | Yes (GPU class attributes) | Yes (CTranslate2 instance attributes) |
| **Complexity** | Higher (subprocess + Pipe) | Lower (same process, Event only) |

## Files

| File | Purpose |
|------|---------|
| `config.py` | `WhisperModelConfig` dataclass (model path, device, compute type, alignment/refinement options) |
| `workers/cancelable_whisper_worker.py` | **Core**: `CancelableWhisperWorker` with per-encode-pass cancel via `model.encode()` patch |
| `test_cancel_whisper.py` | Standalone test: cancel mid-align, verify model survives, re-run to completion |
| `README.md` | This file |

## Usage

### Quick test (recommended)

The standalone test requires only `stable-ts` + `faster-whisper` — no other project imports needed:

```bash
# From the whisper2srt project root
cd cancel_whisper

# Basic test: cancel after 5 seconds, then re-run to prove model survives
python test_cancel_whisper.py /path/to/vocals.wav /path/to/lyrics.txt --cancel-after 5

# Cancel sooner (if you have a fast GPU)
python test_cancel_whisper.py /path/to/vocals.wav /path/to/lyrics.txt --cancel-after 2

# Custom model path
python test_cancel_whisper.py /path/to/vocals.wav /path/to/lyrics.txt --model-path /path/to/whisper-model
```

Expected output:
```
STEP 1: Loading whisper model
Model loaded in 12.3s

STEP 2: Running alignment (will cancel after 5s)
Cancel detected before encode pass #3 — aborting!
✓ Alignment cancelled after 5.2s (as expected)

STEP 3: Verify model is still loaded after cancellation
Model is still loaded: True
✓ Model survived cancellation — no reload needed

STEP 4: Re-running alignment (model should still be loaded)
✓ Second alignment completed in 28.1s

STEP 5: Running refinement (will cancel after 5s)
Cancel detected before refine pass #7 — aborting!
✓ Refinement cancelled after 5.1s (as expected)

STEP 6: Re-running refinement (model should still be loaded)
✓ Second alignment+refine completed in 32.4s

✓✓✓ MODEL WAS STILL LOADED AFTER CANCELLATION — NO RELOAD NEEDED ✓✓✓
```

## Key Design Decisions

### 1. Patch `model.encode()`, not `inference_func`

The `Aligner` and `Refiner` store `self.inference_func` which calls `model.encode()`. We could patch `inference_func` instead, but:
- `model.encode()` is a single patch point that covers **both** `align()` and `refine()`
- `inference_func` is a closure created at call time and buried inside the `Aligner`/`Refiner` — harder to access
- `model.encode()` is a well-known, stable API on `WhisperModel`

### 2. Same-process, no subprocess

Unlike the audio-separator cancel_test, we don't need a subprocess because:
- The cancel signal is a `threading.Event` — no Pipe bridging needed
- Exception-based cancellation is cleaner than process-killing
- The model weights are on the `WhisperModel` instance which lives in the same process

### 3. Patch-and-restore pattern

The `model.encode()` patch is applied **before** `model.align()`/`model.refine()` and removed in a **finally** block. This ensures:
- The original `model.encode()` is always restored, even on cancellation or error
- The model is left in its original state for the next job
- No residual state from the patch leaks between jobs

### 4. Exception unwinding preserves model weights

When `_CancelledInsideEncode` is raised inside the patched `encode()`:
- It unwinds through `inference_func()` → `_compute_timestamps()` → `while` loop → `Aligner.align()` → `model.align()`
- The `with torch.no_grad():` context manager (if present) catches and re-raises properly
- The model weights are **instance attributes** on the `WhisperModel` (`model.model` is a CTranslate2 model) — they're on the GPU/CPU, not on the Python call stack
- Only the **intermediate tensors** (audio segments, encoder outputs, probability matrices) are lost — they're stack locals that get garbage collected
- `torch.cuda.empty_cache()` cleans up any remaining GPU intermediate state

### 5. Separate cancel events for align and refine

`align_and_refine()` clears the cancel event between the two operations. If the caller wants to cancel both, they set the event again after `align()` returns. This gives the caller fine-grained control over which phase to cancel.

## AudioLoader FFmpeg Cleanup on Cancellation

When `_CancelledInsideEncode` unwinds through `Aligner.align()`, the normal cleanup path (`self.audio_loader.terminate()` at line 357 of `non_whisper/alignment.py`) is **skipped** — it's after the while loop, so exception-based exit never reaches it. This creates two problems:

1. **Orphaned FFmpeg subprocess**: The `AudioLoader` holds an FFmpeg subprocess (`self._process`) that keeps running and writing raw PCM to stdout until Python's GC calls `AudioLoader.__del__()`.
2. **Broken pipe stderr noise**: When the GC-terminated FFmpeg process is killed mid-stream, it writes muxer errors to stderr: `"Error submitting a packet to the muxer: Broken pipe"`, `"Error muxing a packet"`, etc. These are harmless but noisy.

The worker fixes both issues with a two-pronged approach:

### Fix 1: Monkey-patch `AudioLoader._audio_loading_process()` to redirect stderr

During `load_model()`, the worker patches `AudioLoader._audio_loading_process()` to pass `stderr=subprocess.DEVNULL` when launching the FFmpeg subprocess. The original code doesn't redirect stderr at all, so FFmpeg errors from a killed process go to the inherited parent stderr. With `DEVNULL`, these messages are silently discarded.

This is safe because:
- FFmpeg is launched with `-loglevel error` — only error-level messages reach stderr
- The only error messages during normal operation are the muxer errors from a killed process
- These errors don't indicate a real problem — they're just FFmpeg complaining about being terminated

### Fix 2: Explicitly terminate orphaned AudioLoaders

In the `_CancelledInsideEncode` handler, the worker walks `gc.get_objects()` to find any `AudioLoader` instances with still-running FFmpeg subprocesses and calls `terminate()` on them immediately. This ensures prompt cleanup instead of waiting for GC.

## Limitations

- **Not instant**: Cancellation takes effect between encoder passes, not mid-inference. For a typical 3-minute song with large-v3, the alignment loop has ~5-8 iterations of ~1-3s each, so worst-case delay is one encoder pass duration. Refinement has finer granularity (~0.1-0.5s per binary-search step).
- **faster-whisper specific**: The injection point (`model.encode()`) is specific to the faster-whisper backend. Vanilla Whisper and MLX Whisper have different `inference_func` implementations (they call `model.encoder()` instead of `model.encode()`). Those would need different patch points.
- **Monkey-patching is fragile**: If stable-ts changes the internal call structure, the patch may need updating. A proper cancellation API in stable-ts would be preferable long-term.
- **No partial result**: Unlike audio-separator (where partial separation is useless), a partially-completed alignment could theoretically yield partial timestamps. However, stable-ts doesn't expose intermediate results from within the loop, so cancellation always produces a full discard.
