# Pipeline Prototype

A synchronous, single-song processing pipeline that extracts audio, separates vocals/instrumentals, transcodes to AAC, and generates karaoke subtitles (ASS). Supports both **alignment mode** (align provided lyrics to audio) and **transcription mode** (auto-generate lyrics from audio).

## Overview

This pipeline processes a song file through a series of stages:

1. **FFmpeg Extract** — Extracts audio from video to WAV (44.1kHz, 16-bit, stereo)
2. **Loudnorm Analyze** — Measures loudness (LUFS, dBTP, LU) using FFmpeg
3. **Stem Separation** — Separates vocals and instrumentals using audio-separator (Roformer model)
4. **FFmpeg Transcode** — Transcodes stem WAVs to M4A (AAC) output
5. **Lyric Align/Transcribe** — Aligns provided lyrics to vocal stem OR transcribes audio directly, generates karaoke ASS

## Architecture

```
run_pipeline.py (CLI entry)
    ↓
PipelineOrchestrator (manages workers + stages)
    ↓
┌─────────────────┬─────────────────┐
│   StemWorker    │ WhisperWorker   │
│   (subprocess)  │  (in-process)   │
└────────┬────────┴────────┬────────┘
         │                 │
    ┌────┴────┐      ┌────┴────┐
    ↓         ↓      ↓         ↓
Stages → Stages → Stages → Stages
```

### Workers

| Worker | Purpose | Lifecycle |
|--------|---------|-----------|
| `StemWorker` | Loads audio-separator model once, separates stems per job | Spawned at pipeline start, terminated at end |
| `WhisperWorker` | Loads faster-whisper model once, aligns lyrics per job | Loaded at pipeline start, unloaded at end |

Workers persist across the pipeline run to avoid reloading models between stages.

### Stages

All stages implement `PipelineStage` protocol:

```python
class PipelineStage(Protocol):
    name: str
    def run(self, ctx: StageContext) -> None: ...
```

| Stage | Input Artifacts | Output Artifacts |
|-------|-----------------|------------------|
| `FFmpegExtractStage` | `song_path` | `extracted_wav` |
| `LoudnormAnalyzeStage` | `extracted_wav` | `loudnorm_input_i`, `loudnorm_input_tp`, `loudnorm_input_lra`, etc. |
| `StemSeparationStage` | `extracted_wav` | `vocal_wav`, `instrumental_wav` |
| `FFmpegTranscodeStage` | `vocal_wav`, `instrumental_wav` | `vocal_m4a`, `nonvocal_m4a` |
| `LyricAlignStage` | `lyrics_path` (optional), `vocal_wav` | `ass_file`, `srt_file` |

### Context

`StageContext` carries state between stages:

```python
@dataclass
class StageContext:
    song_path: Path          # input audio/video file
    tmp_dir: Path            # per-job temp directory
    config: PipelineConfig   # shared configuration
    artifacts: dict[str, Any]  # stage outputs
```

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
# Align lyrics
python -m pipeline.run_pipeline "song.mp4" "lyrics.txt"

# Auto-transcribe
python -m pipeline.run_pipeline "song.mp4"
```

**Note:** The pipeline takes only positional arguments — no CLI flags. To customize behavior, edit `config.py` or construct a `PipelineConfig` programmatically.

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
├── run_pipeline.py          # CLI entry point
├── orchestrator.py          # PipelineOrchestrator
├── context.py               # StageContext dataclass
├── config.py                # PipelineConfig dataclass (to be created)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── stages/
│   ├── __init__.py
│   ├── base.py              # PipelineStage protocol + BaseStage
│   ├── ffmpeg_extract.py    # Extract audio from video
│   ├── loudnorm_analyze.py  # Loudness analysis
│   ├── stem_separation.py   # Vocal/instrumental separation
│   ├── ffmpeg_transcode.py  # Transcode to AAC
│   └── lyric_align.py       # Lyric alignment + ASS generation
└── workers/
    ├── __init__.py
    ├── stem_worker.py       # StemWorker (subprocess)
    └── whisper_worker.py    # WhisperWorker (in-process)
```

## Design Notes

### Synchronous Execution

The pipeline runs synchronously in the main thread — no job queue, no orchestrator thread, no cancel flag. This is a deliberate simplification for the prototype.

### No Per-Stage Cleanup

The orchestrator owns a per-job temp directory and deletes it at the end. Stages do not clean up intermediate outputs, making reordering stages safer. Disk usage per song is bounded (~200-400 MB for a typical song).

### Worker Persistence

Both workers are started before the first stage and stopped after the last stage. Model weights stay loaded across all jobs in a pipeline run.

### Alignment & Transcription

- **Alignment**: Provided lyrics are paired to Whisper-aligned word boundaries by word count, giving you human-curated line breaks and spelling.
- **Transcription**: Whisper segments the audio based on pauses and voice activity; each segment becomes a karaoke line. Useful for quick subtitles without lyric input but may produce uneven line breaks.

Both modes run through the same ASS/SRT generation pipeline with identical word-level karaoke timing.

### Cancellation Support (Worker Internals)

While the prototype pipeline is synchronous, the workers retain cancellation infrastructure from the parent project:

- **StemWorker**: Patches `model_run.forward()` to check a cancel pipe between chunks
- **WhisperWorker**: Patches `model.encode()` to check a threading.Event between encoder passes

These are currently disabled (cancel_event=None) in the pipeline orchestrator.

## License

See project root for license information.
