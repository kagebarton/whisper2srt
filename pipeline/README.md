# Pipeline Prototype

A single-song processing pipeline that extracts audio, separates vocals/instrumentals, transcodes to AAC, and generates karaoke subtitles (ASS). Supports both **alignment mode** (align provided lyrics to audio) and **transcription mode** (auto-generate lyrics from audio). Includes **phase-targeted cancellation** — you can cancel a running job at a specific stage and both workers will remain loaded for the next run.

## Overview

This pipeline processes a song file through a series of stages:

1. **FFmpeg Extract** — Extracts audio from video to WAV (44.1kHz, 16-bit, stereo)
2. **Loudnorm Analyze** — Measures loudness (LUFS, dBTP, LU) using FFmpeg
3. **Stem Separation** — Separates vocals and instrumentals using audio-separator (Roformer model)
4. **FFmpeg Transcode** — Transcodes stem WAVs to M4A (AAC) output
5. **Lyric Align/Transcribe** — Aligns provided lyrics to vocal stem OR transcribes audio directly, generates karaoke ASS

## Architecture

```
run_pipeline.py (CLI entry, --phase / --cancel-after flags)
↓
PipelineOrchestrator (start/stop, run_one, run_one_async, join, cancel_active)
↓                                        ↓ CancelToken
┌─────────────────┬─────────────────┐
│ StemWorker │ WhisperWorker │
│ (subprocess) │ (in-process) │
└────────┬────────┴────────┬────────┘
│ │
┌────┴────┐ ┌────┴────┐
↓ ↓ ↓ ↓
Stages → Stages → Stages → Stages
```

Each stage opens an **activity** scope on the shared `CancelToken`, registering a phase-specific cancellation mechanism (`KillProcess` for FFmpeg, `SetEvent` for model workers). The orchestrator (or test driver) calls `cancel_active()` which signals whichever mechanism is currently active.

### Workers

| Worker | Purpose | Lifecycle | Cancel Mechanism |
|--------|---------|-----------|-----------------|
| `StemWorker` | Loads audio-separator model once, separates stems per job | Spawned at `start()`, terminated at `stop()` | `SetEvent` — sets the cancel pipe, interrupting per-chunk processing |
| `WhisperWorker` | Loads faster-whisper model once, aligns lyrics per job | Loaded at `start()`, unloaded at `stop()` | `SetEvent` — sets the threading.Event, interrupting between encoder passes |

Workers persist across pipeline runs (between `start()` and `stop()`) to avoid reloading models. After cancellation, both workers remain loaded — no reload needed for subsequent jobs.

### Stages

All stages implement `PipelineStage` protocol:

```python
class PipelineStage(Protocol):
    name: str
    def run(self, ctx: StageContext) -> None: ...
```

| Stage | Input Artifacts | Output Artifacts | Cancel Phase(s) |
|-------|-----------------|------------------|------------------|
| `FFmpegExtractStage` | `song_path` | `extracted_wav` | `EXTRACT` |
| `LoudnormAnalyzeStage` | `extracted_wav` | `loudnorm_input_i`, `loudnorm_input_tp`, `loudnorm_input_lra`, etc. | `LOUDNORM` |
| `StemSeparationStage` | `extracted_wav` | `vocal_wav`, `instrumental_wav` | `STEM_SEPARATION` |
| `FFmpegTranscodeStage` | `vocal_wav`, `instrumental_wav` | `vocal_m4a`, `nonvocal_m4a` | `TRANSCODE` |
| `LyricAlignStage` | `lyrics_path` (optional), `vocal_wav` | `ass_file`, `srt_file` | `ALIGN` / `TRANSCRIBE` / `REFINE` |

### Context

`StageContext` carries state between stages:

```python
@dataclass
class StageContext:
    song_path: Path          # input audio/video file
    tmp_dir: Path            # per-job temp directory
    config: PipelineConfig   # shared configuration
    artifacts: dict[str, Any]  # stage outputs
    cancel: Optional[CancelToken] = None  # None when run without cancel
```

When `ctx.cancel is None` (the default for `run_one` / normal CLI usage), stages take their existing fast paths with zero cancellation overhead.

### Cancellation Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| `Phase` | `context.py` | Enum — one value per stage or per model-call (`EXTRACT`, `LOUDNORM`, `STEM_SEPARATION`, `TRANSCODE`, `ALIGN`, `TRANSCRIBE`, `REFINE`) |
| `CancelToken` | `context.py` | Per-job cancellation state: `cancelled` flag, `phase`, `active` target, `threading.Event`, and a `threading.Lock` for atomic updates |
| `Cancellable` | `context.py` | Protocol — anything the orchestrator can cancel mid-flight |
| `KillProcess` | `context.py` | `Cancellable` that SIGKILLs a `subprocess.Popen` (used by FFmpeg stages) |
| `SetEvent` | `context.py` | `Cancellable` that sets a `threading.Event` (used by model-worker stages) |
| `PipelineCancelled` | `context.py` | Exception raised when a job is cancelled — carries the `Phase` at which cancel was detected |
| `run_ffmpeg()` | `stages/_ffmpeg_helpers.py` | Shared helper: runs an FFmpeg subprocess inside `cancel_token.activity(Phase, KillProcess(proc))`, with a defensive `is_cancelled()` check on non-zero exit |

**Activity lifecycle** — Each long-running operation opens a `cancel_token.activity(phase, target)` context manager. On entry, it atomically registers the phase + cancel target. On exit (normal, exception, or cancel), it atomically clears the active target. If `cancelled` is already `True` on entry, it raises `PipelineCancelled` immediately so work never starts. On exit, it only synthesises `PipelineCancelled` if the body completed without an exception (avoids masking the original error).

**Orphan-output prevention** — `ffmpeg_transcode` and `lyric_align` write outputs to `ctx.tmp_dir` first, then `shutil.move()` to final destinations after full success. This prevents partial/orphan files on cancel.

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- `stable-ts` — Whisper alignment with word-level timestamps
- `faster-whisper` — Optimized Whisper inference
- `audio-separator` — Vocal/instrumental separation
- `srt` — SRT subtitle parsing
- `lyricsgenius` — Lyrics fetching (optional)

### Command Line

```bash
# Alignment mode: align provided lyrics to audio
python -m pipeline.run_pipeline <audio_file> <lyrics_file>

# Transcription mode: auto-generate lyrics from audio
python -m pipeline.run_pipeline <audio_file>
```

Examples:
```bash
# Align lyrics (normal mode — no cancellation)
python -m pipeline.run_pipeline "song.mp4" "lyrics.txt"

# Auto-transcribe
python -m pipeline.run_pipeline "song.mp4"

# Cancel-test: fire cancel during stem_separation, 5s after it begins
python -m pipeline.run_pipeline "song.mp4" "lyrics.txt" \
    --phase stem_separation --cancel-after 5

# Cancel-test: fire cancel during the refine model call, 2s after it begins
python -m pipeline.run_pipeline "song.mp4" \
    --phase refine --cancel-after 2
```

**Normal mode** (no flags) works exactly as before — `ctx.cancel` is `None` and stages take their existing fast paths.

**Cancel-test mode** (`--phase` + `--cancel-after`, both required) runs the pipeline on a background thread, waits for the target phase to begin, sleeps `cancel-after` seconds, then fires `cancel_active()`. It verifies that `PipelineCancelled` is raised and that both workers remain loaded afterwards. Does **not** re-run the pipeline after cancel — matching production behaviour.

Available `--phase` values: `extract`, `loudnorm`, `stem_separation`, `transcode`, `align`, `transcribe`, `refine`.

### Alignment vs. Transcription Modes

| Mode | Lyrics File | Output | Use Case |
|------|-------------|--------|----------|
| **Alignment** | `.txt` or `.srt` | ASS + SRT (if `.txt` input) | Accurate karaoke with hand-written line breaks |
| **Transcription** | None | ASS + SRT | Quick auto-generated subtitles without lyric input |

**Alignment** pairs provided lyrics to Whisper-aligned word boundaries, giving you human-curated line breaks and spelling. **Transcription** lets Whisper segment the audio based on pauses and voice activity, useful when you don't have pre-written lyrics but want quick subtitles.

### Output Structure

Given input `songs/artist/title.mp4`:

```
songs/artist/
├── title.mp4
├── vocal/
│   └── title---vocal.m4a
├── nonvocal/
│   └── title---nonvocal.m4a
├── karaoke/
│   └── title.ass                    (alignment + transcription)
└── subtitles/
    └── title.srt                    (alignment + transcription)
```

**Alignment mode** — SRT only generated if input lyrics were `.txt` (`.srt` input discards original timings)  
**Transcription mode** — both ASS and SRT always generated

## Orchestrator API

The `PipelineOrchestrator` provides both synchronous and asynchronous execution paths:

### Worker Lifecycle (idempotent)

| Method | Description |
|--------|-------------|
| `start()` | Spawn `StemWorker` subprocess + load Whisper model. No-op if already started. |
| `stop()` | Terminate `StemWorker` + unload Whisper model. No-op if already stopped. |
| `stem_worker` | Read-only property — access the `StemWorker` instance |
| `whisper_worker` | Read-only property — access the `WhisperWorker` instance |

### Synchronous Execution

| Method | Description |
|--------|-------------|
| `run(song_path, lyrics_path=None)` | Back-compat: `start()` → `run_one()` → `stop()` |
| `run_one(song_path, lyrics_path=None)` | Run one song synchronously. `ctx.cancel` is `None` — no cancellation. Workers must already be started. |

### Asynchronous Execution + Cancellation

| Method | Description |
|--------|-------------|
| `run_one_async(song_path, lyrics_path=None) → CancelToken` | Start the pipeline on a background thread. Returns a **fresh** `CancelToken` (fresh `threading.Event`) — never reused across jobs. Workers must already be started. |
| `join(timeout=None) → StageContext` | Wait for the pipeline thread to finish. Re-raises whatever exception the thread caught (`PipelineCancelled`, `FileNotFoundError`, `RuntimeError`, etc.). |
| `cancel_active()` | Request cancellation of the currently-running pipeline. Delegates to `CancelToken.cancel()`. |

**Typical async/cancel usage:**

```python
orchestrator.start()
try:
    token = orchestrator.run_one_async(song_path, lyrics_path)
    # ... later, maybe from a timer or UI callback:
    orchestrator.cancel_active()
    try:
        orchestrator.join()
    except PipelineCancelled as e:
        print(f"Cancelled at {e.phase}")
finally:
    orchestrator.stop()
```

## Configuration

All settings are defined in `PipelineConfig` (see `config.py` or construct programmatically):

### Model Paths
- `whisper_model_path` — Path to faster-whisper model
- `separator_model_dir` — Directory containing audio-separator models
- `separator_model_name` — Specific model checkpoint to load

### Device/Compute
- `whisper_device` — "cpu", "cuda", or "auto"
- `whisper_compute_type` — "float16", "int8", "int8_float16", "float32"

### Loudnorm Targets
- `loudnorm_target_i` — Target integrated loudness in LUFS (default: -24.0)
- `loudnorm_target_tp` — Target true peak in dBTP (default: -2.0)
- `loudnorm_target_lra` — Target loudness range in LU (default: 7.0)

### Whisper Alignment & Transcription
Settings apply to both alignment and transcription modes:
- `whisper_language` — Language code (default: "en")
- `whisper_vad` — Enable voice activity detection (default: True)
- `whisper_vad_threshold` — VAD threshold (default: 0.25)
- `whisper_suppress_silence` — Suppress silent timestamps (default: True)
- `whisper_suppress_word_ts` — Suppress word timestamps (default: True)
- `whisper_only_voice_freq` — Filter to voice frequencies only (default: True)
- `whisper_refine_steps` — "s" (start), "e" (end), or "se" (both) — applied after alignment/transcription
- `whisper_refine_word_level` — Refine at word level (default: False)
- `whisper_regroup` — Regroup algorithm for Whisper output (default: "dm_q4")

### ASS Styling
- `font_name`, `font_size` — Font settings
- `primary_color`, `secondary_color` — Karaoke fill colors
- `outline_color`, `back_color` — Border and shadow colors
- `outline_width`, `shadow_offset` — Border thickness
- `margin_left`, `margin_right`, `margin_vertical` — Positioning

### Karaoke Timing (centiseconds)
- `line_lead_in_cs` — Lead-in before line (default: 80)
- `line_lead_out_cs` — Lead-out after line (default: 20)
- `first_word_nudge_cs` — Adjustment for first word (default: 0)

### FFmpeg Settings
- `aac_quality` — AAC VBR quality (default: "2", ~128 kbps)
- `ffmpeg_threads` — Thread count for FFmpeg (default: "4")

## File Structure

```
pipeline/
├── __init__.py
├── run_pipeline.py          # CLI entry point (--phase / --cancel-after flags)
├── orchestrator.py           # PipelineOrchestrator (start/stop/run_one/run_one_async/join/cancel_active)
├── context.py                # StageContext, Phase, CancelToken, Cancellable, KillProcess, SetEvent, PipelineCancelled
├── config.py                 # PipelineConfig dataclass
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── stages/
│   ├── __init__.py
│   ├── base.py               # PipelineStage protocol + BaseStage
│   ├── _ffmpeg_helpers.py    # run_ffmpeg() helper (cancel-aware subprocess wrapper)
│   ├── ffmpeg_extract.py     # Extract audio from video
│   ├── loudnorm_analyze.py   # Loudness analysis
│   ├── stem_separation.py    # Vocal/instrumental separation
│   ├── ffmpeg_transcode.py   # Transcode to AAC
│   └── lyric_align.py        # Lyric alignment + ASS generation
└── workers/
    ├── __init__.py
    ├── stem_worker.py        # StemWorker (subprocess)
└── whisper_worker.py     # WhisperWorker (in-process)
```

## Design Notes

### Synchronous Default, Async Opt-In

The pipeline runs synchronously by default — `run_one()` is a thin wrapper over `run_one_async()` + `join()`. When `ctx.cancel` is `None` (the default for normal CLI and `run_one` usage), stages take their existing fast paths with zero cancellation overhead. Asynchronous execution with cancellation is opt-in via `run_one_async()`.

### Phase-Targeted Cancellation

Each long-running operation registers itself as an **activity** on the `CancelToken` with a specific `Phase` and a `Cancellable` target. When `cancel_active()` is called:

1. `cancelled` is set to `True` (under the lock, **before** calling `target.cancel()`)
2. The active `Cancellable` is invoked — `KillProcess` SIGKILLs the FFmpeg subprocess, `SetEvent` sets the worker's threading.Event
3. The interrupted operation exits its `activity()` scope, which sees `cancelled=True` and raises `PipelineCancelled`

This design means cancellation is always **phase-aware** — you know exactly where the pipeline was when it was cancelled. Phases are granular enough to target individual model calls within `LyricAlignStage` (`ALIGN`, `TRANSCRIBE`, `REFINE`).

### Worker Survival After Cancel

Both workers remain loaded after cancellation. `StemWorker`'s subprocess stays alive (it drains stale cancel-pipe signals on the next `separate()` call). `WhisperWorker`'s model remains loaded. The pipeline does **not** re-run after cancel — the caller decides whether to start a new job.

### No Per-Stage Cleanup

The orchestrator owns a per-job temp directory and deletes it at the end (even on cancel). Stages do not clean up intermediate outputs, making reordering stages safer. Disk usage per song is bounded (~200-400 MB for a typical song).

### Alignment & Transcription

- **Alignment**: Provided lyrics are paired to Whisper-aligned word boundaries by word count, giving you human-curated line breaks and spelling.
- **Transcription**: Whisper segments the audio based on pauses and voice activity; each segment becomes a karaoke line. Useful for quick subtitles without lyric input but may produce uneven line breaks.

Both modes run through the same ASS/SRT generation pipeline with identical word-level karaoke timing.

## License

See project root for license information.
