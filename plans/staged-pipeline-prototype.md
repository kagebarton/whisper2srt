# Plan: Staged Processing Manager Prototype

## Overview

Create a new `pipeline/` folder (sibling to `cancel_separator/` and `cancel_whisper/`) that implements a staged processing manager for a single song. The prototype runs the full processing pipeline — extract, loudnorm analyze, separate, transcode, lyric alignment + ASS/SRT generation — incorporating the cancellation infrastructure from both existing prototypes and ensuring each heavy model is loaded exactly once and reused across the pipeline lifetime.

## Folder Structure

```
pipeline/
├── __init__.py
├── config.py              # PipelineConfig dataclass (all paths, settings, styling)
├── context.py             # Per-job StageContext (simplified)
├── orchestrator.py        # Single-song, synchronous orchestrator
├── stages/
│   ├── __init__.py
│   ├── base.py            # PipelineStage protocol + BaseStage
│   ├── ffmpeg_extract.py  # FFmpeg extract audio → WAV
│   ├── loudnorm_analyze.py# NEW: FFmpeg loudnorm 1st pass, capture target_offset
│   ├── stem_separation.py # Stem separation via StemWorker
│   ├── ffmpeg_transcode.py# FFmpeg transcode WAV stems → M4A
│   └── lyric_align.py     # NEW: Lyric alignment + ASS/SRT generation via WhisperWorker
├── workers/
│   ├── __init__.py
│   ├── stem_worker.py               # Copied from cancel_separator/ (with changes)
│   └── whisper_worker.py            # Copied from cancel_whisper/ (with changes)
└── run_pipeline.py         # CLI entry point
```

## Pipeline Stage Order

```
1. ffmpeg_extract       — Extract audio from video → WAV
2. loudnorm_analyze     — Run loudnorm 1st pass on original extracted WAV, capture target_offset
3. stem_separation      — Separate vocal/instrumental WAV stems
4. ffmpeg_transcode     — Transcode WAV stems → M4A
5. lyric_align          — Align lyrics to vocal stem WAV, generate .ass (+ .srt if .txt input)
```

Only stage 1 runs before loudnorm_analyze, so loudnorm only depends on `extracted_wav` — moving it up allows loudnorm to fail fast before the expensive stem separation starts.

## Key Changes from `cancel_separator/`

### 1. Strip multi-cycle features

- Remove the job queue from the orchestrator — it processes exactly one song per invocation
- Remove `_stop_event`, pending queue, and orchestrator thread — the pipeline runs synchronously in the main thread
- Remove `cancel()` from `PipelineStage` protocol and `BaseStage` (no mid-pipeline cancellation at the orchestrator level)
- Remove `cancelled: threading.Event` and `CancelledError` from `StageContext`

### 2. Simplified orchestrator

`orchestrator.py`:
- Takes a list of stages, a `StemWorker`, and a `WhisperWorker`
- `run(song_path, lyrics_path) -> StageContext`: creates context, seeds input artifacts, runs stages sequentially, returns the context
- Manages both worker lifecycles: `stem_worker.start()` and `whisper_worker.load_model()` before pipeline; `stem_worker.stop()` and `whisper_worker.unload_model()` after pipeline
- Creates and cleans up intermediate temp dir under the configured `intermediate_dir`

### 3. Model loading: once per model, reused across stages

Both heavy models are loaded exactly once by the orchestrator before the pipeline starts, and injected into the stages that need them:

| Model | Worker | Loaded by orchestrator | Used by stage |
|-------|--------|----------------------|---------------|
| audio-separator (Roformer) | `StemWorker` | `stem_worker.start()` (spawns subprocess, model loads inside) | `StemSeparationStage` |
| faster-whisper (CTranslate2) | `WhisperWorker` | `whisper_worker.load_model()` (in-process) | `LyricAlignStage` |

Stages receive worker references at construction time (injected, not owned). They never create/destroy workers. The orchestrator owns the full lifecycle.

### 4. No per-stage cleanup

The orchestrator owns the temp dir and deletes it at the end. Stages do not clean up their intermediate outputs — keeping everything around until the end makes reordering stages safer and simplifies stage logic. Disk usage per song is bounded (one extracted WAV + two stem WAVs ≈ 200-400 MB for a typical song) and the temp dir is removed in the orchestrator's `finally` block.

## Config (`config.py`)

```python
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # --- Model paths ---
    whisper_model_path: str = "/home/ken/whisper2srt/whisper-model/large-v3"
    separator_model_dir: str = "./audio-separator/models"
    separator_model_name: str = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"

    # --- Device/compute ---
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"

    # --- Intermediate files directory ---
    intermediate_dir: str = ""  # Empty = system temp dir

    # --- Loudnorm targets ---
    loudnorm_target_i: float = -24.0   # Target integrated loudness (LUFS)
    loudnorm_target_tp: float = -2.0   # Target true peak (dBTP)
    loudnorm_target_lra: float = 7.0   # Target loudness range (LU)

    # --- Whisper alignment options ---
    whisper_language: str = "en"
    whisper_vad: bool = True
    whisper_vad_threshold: float = 0.25
    whisper_suppress_silence: bool = True
    whisper_suppress_word_ts: bool = True
    whisper_only_voice_freq: bool = True
    whisper_refine_steps: str = "s"       # 's' = refine starts, 'e' = ends, 'se' = both
    whisper_refine_word_level: bool = False

    # --- ASS styling ---
    font_name: str = "Arial"
    font_size: int = 60
    primary_color: str = "&H00D7FF&"      # Soft Yellow
    secondary_color: str = "&H00FFFFFF"   # White (not yet sung)
    outline_color: str = "&H00000000"     # Black outline
    back_color: str = "&H80000000&"       # Translucent shadow
    outline_width: int = 3
    shadow_offset: int = 2
    margin_left: int = 50
    margin_right: int = 50
    margin_vertical: int = 150

    # --- Karaoke timing (centiseconds) ---
    line_lead_in_cs: int = 80
    line_lead_out_cs: int = 20
    first_word_nudge_cs: int = 0

    # --- FFmpeg transcoding ---
    aac_quality: str = "2"                # ≈ 128 kbps VBR AAC
    ffmpeg_threads: str = "4"
```

All paths, device/compute, styling, and timing values are **config entries only** — there are no CLI flags. To change values, edit `config.py` or construct a `PipelineConfig` with overrides and pass it to the orchestrator programmatically.

## Context (`context.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig


@dataclass
class StageContext:
    """Per-job context passed forward through the pipeline stages."""

    song_path: Path                      # input audio/video file
    tmp_dir: Path                        # per-job temp directory
    config: PipelineConfig               # shared config reference
    artifacts: dict[str, Any] = field(default_factory=dict)
```

No `cancelled` event, no `CancelledError`, no lyrics fields on the context itself — all per-stage inputs and outputs live in `artifacts`.

### `artifacts` schema

Keys are populated by the orchestrator and each stage. Typing is loose (`dict[str, Any]`) because values are a mix of `Path`, `str`, and `float`. The schema below is the contract stages assume:

| Key | Type | Set by | Consumed by |
|---|---|---|---|
| `lyrics_path` | `Path` | orchestrator (before pipeline) | `lyric_align` |
| `extracted_wav` | `Path` | `ffmpeg_extract` | `loudnorm_analyze`, `stem_separation` |
| `loudnorm_input_i` | `float` | `loudnorm_analyze` | (future) |
| `loudnorm_input_tp` | `float` | `loudnorm_analyze` | (future) |
| `loudnorm_input_lra` | `float` | `loudnorm_analyze` | (future) |
| `loudnorm_input_thresh` | `float` | `loudnorm_analyze` | (future) |
| `loudnorm_target_offset` | `float` | `loudnorm_analyze` | (future) |
| `loudnorm_type` | `str` | `loudnorm_analyze` | (future) |
| `vocal_wav` | `Path` | `stem_separation` | `ffmpeg_transcode`, `lyric_align` |
| `instrumental_wav` | `Path` | `stem_separation` | `ffmpeg_transcode` |
| `vocal_m4a` | `Path` | `ffmpeg_transcode` | (output) |
| `nonvocal_m4a` | `Path` | `ffmpeg_transcode` | (output) |
| `lyrics_format` | `str` (`"txt"` or `"srt"`) | `lyric_align` | `lyric_align` (internal) |
| `lyrics_text` | `str` | `lyric_align` | `lyric_align` (internal) |
| `ass_file` | `Path` | `lyric_align` | (output) |
| `srt_file` | `Path` | `lyric_align` | (output, only if input was `.txt`) |

## Stage Protocol (`stages/base.py`)

```python
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from pipeline.context import StageContext

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStage(Protocol):
    name: str

    def run(self, ctx: StageContext) -> None:
        """Execute this stage, mutating ctx.artifacts as needed."""
        ...


class BaseStage:
    """Base class providing the name attribute; subclasses override run()."""

    name: str = "base"

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError
```

No `cancel()` method. The protocol is narrower than `cancel_separator/stages/base.py` because the prototype does not expose cancellation at the orchestrator level.

## Stage 1: `ffmpeg_extract.py` (ported from `cancel_separator/`)

**Purpose:** Run ffmpeg to extract audio from the source video/audio file into a 44.1 kHz stereo 16-bit WAV inside the temp dir.

**Class:**
```python
class FFmpegExtractStage(BaseStage):
    name = "ffmpeg_extract"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> None: ...
```

(The stage now needs config to read `ffmpeg_threads`. Previously the plan removed `__init__` entirely; it's reintroduced solely for this.)

**Flow:**
1. Compute `wav_out = ctx.tmp_dir / f"{ctx.song_path.stem}_input.wav"`.
2. Build command:
   ```
   ffmpeg -y -threads <ffmpeg_threads> -i <song_path> -vn -ac 2 -ar 44100 -sample_fmt s16 <wav_out>
   ```
3. Run via `subprocess.run(..., stdout=DEVNULL, stderr=DEVNULL)` (blocking, no cancellation handling).
4. If returncode != 0 → raise `RuntimeError("ffmpeg extract failed")`.
5. Set `ctx.artifacts["extracted_wav"] = wav_out`.

**Note on input format assumption:** Subsequent stages (loudnorm, stem separation) assume `extracted_wav` is 44.1 kHz stereo 16-bit, matching what this stage produces. The pipeline always runs extract first, so this holds in practice. If stages are ever run standalone with a pre-existing WAV, format differences could skew measurements — not a concern for the prototype.

**Changes from `cancel_separator/stages/ffmpeg_extract.py`:**
- Remove `self._proc` field and the `cancel()` method.
- Replace `subprocess.Popen(...).wait()` with `subprocess.run(...)`.
- Remove `CancelledError` handling.
- Rename artifact key from `wav_in` → `extracted_wav` (clearer in the multi-WAV pipeline).
- Take `PipelineConfig` in `__init__` to read `ffmpeg_threads` (added for consistency with the other ffmpeg stages).

## Stage 2: `loudnorm_analyze.py` (NEW)

**Purpose:** Run ffmpeg loudnorm 1st pass (analysis only, no output file) on the full extracted WAV and capture the measurement JSON for later use.

**Class:**
```python
import json
import re
import subprocess

class LoudnormAnalyzeStage(BaseStage):
    name = "loudnorm_analyze"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> None: ...
```

**Command:**
```
ffmpeg -hide_banner -nostats -threads <ffmpeg_threads> -i <extracted_wav>
       -af loudnorm=I=<target_i>:TP=<target_tp>:LRA=<target_lra>:print_format=json
       -f null -
```

`-threads` is threaded through from `self._config.ffmpeg_threads` for consistency with `FFmpegExtractStage` and `FFmpegTranscodeStage`.

**Flow:**
1. Read `extracted_wav = ctx.artifacts["extracted_wav"]`; raise `RuntimeError` if missing.
2. Build the ffmpeg command using `self._config.loudnorm_target_i/tp/lra` and `self._config.ffmpeg_threads`.
3. Run via `subprocess.run(..., capture_output=True, text=True)`. Loudnorm writes the stats JSON to stderr.
4. Parse stderr by walking backward from the end (safer than a regex — ffmpeg may emit other `{...}` fragments in log lines earlier in stderr):
   - Split stderr on `\n`, iterate from the last line toward the first.
   - Find the line ending with `}` (the JSON closer), then continue backward collecting lines until you hit the line starting with `{`.
   - Join those lines and `json.loads(...)` the result.
   - If no balanced `{ ... }` block is found, raise the "no JSON output" error below.
5. Store the following fields in `ctx.artifacts` (all floats except `loudnorm_type`):
   - `loudnorm_input_i` ← `input_i`
   - `loudnorm_input_tp` ← `input_tp`
   - `loudnorm_input_lra` ← `input_lra`
   - `loudnorm_input_thresh` ← `input_thresh`
   - `loudnorm_target_offset` ← `target_offset` (the key value for a future 2nd pass)
   - `loudnorm_type` ← `normalization_type` (`"linear"` or `"dynamic"`)
6. Log a summary line: `[loudnorm_analyze] I=<input_i> LUFS TP=<input_tp> dBTP LRA=<input_lra> LU offset=<target_offset> dB type=<normalization_type>`.

**Errors:**
- `returncode != 0` → `RuntimeError(f"ffmpeg loudnorm failed: {stderr[-500:]}")`.
- No JSON block found → `RuntimeError("loudnorm did not produce JSON output")`.
- JSON parse error → re-raise with added context about which field failed.

**Why on the extracted WAV, not the vocal stem:**
The full mix carries the loudness the listener actually hears. Analyzing the vocal stem in isolation would measure stem-only loudness, which is not what we want to normalize against.

## Stage 3: `stem_separation.py` (ported from `cancel_separator/`)

**Purpose:** Delegate to the injected `StemWorker` to separate `extracted_wav` into vocal and instrumental stem WAVs.

**Class:**
```python
class StemSeparationStage(BaseStage):
    name = "stem_separation"

    def __init__(self, stem_worker: StemWorker) -> None:
        self._worker = stem_worker

    def run(self, ctx: StageContext) -> None: ...
```

**Flow:**
1. Read `extracted_wav = ctx.artifacts["extracted_wav"]`; raise `RuntimeError` if missing.
2. If `not self._worker.is_alive()` → `self._worker.start()` (defensive; orchestrator normally starts it).
3. Call:
   ```python
   vocal_wav, instrumental_wav = self._worker.separate(
       wav_path=extracted_wav,
       output_dir=ctx.tmp_dir,
       cancel_event=None,
   )
   ```
4. Set `ctx.artifacts["vocal_wav"] = vocal_wav` and `ctx.artifacts["instrumental_wav"] = instrumental_wav`.

**Changes from `cancel_separator/stages/stem_separation.py`:**
- Remove `cancel()` method.
- Remove `WorkerCancelledError`/`CancelledError` handling (cancellation not exposed).
- Remove the `WorkerDiedError` branch's `ctx.check_cancelled()` call; simply re-raise as `RuntimeError`.
- Rename input artifact key from `wav_in` → `extracted_wav`.

## Stage 4: `ffmpeg_transcode.py` (ported from `cancel_separator/`)

**Purpose:** Transcode both stem WAVs to M4A (AAC) files alongside the original song, in `vocal/` and `nonvocal/` subdirectories.

**Class:**
```python
class FFmpegTranscodeStage(BaseStage):
    name = "ffmpeg_transcode"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> None: ...
    def _transcode(self, wav_path: Path, output_path: Path) -> None: ...
```

**Output locations:**
- `<song_dir>/vocal/<song_stem>---vocal.m4a`
- `<song_dir>/nonvocal/<song_stem>---nonvocal.m4a`

(Directories created with `mkdir(exist_ok=True)`. Naming matches the pikaraoke convention used by the mixer.)

**Flow:**
1. Read `vocal_wav` and `instrumental_wav` from `ctx.artifacts`.
2. Create `<song_dir>/vocal/` and `<song_dir>/nonvocal/`.
3. Call `_transcode(vocal_wav, vocal_out)` and `_transcode(instrumental_wav, nonvocal_out)`.
4. Set `ctx.artifacts["vocal_m4a"] = vocal_out` and `ctx.artifacts["nonvocal_m4a"] = nonvocal_out`.

**`_transcode` command:**
```
ffmpeg -y -threads <ffmpeg_threads> -i <wav_path> -c:a aac -q:a <aac_quality> <output_path>
```

Run via `subprocess.run(..., stdout=DEVNULL, stderr=DEVNULL)`. If returncode != 0 → `RuntimeError`.

**Changes from `cancel_separator/stages/ffmpeg_transcode.py`:**
- Remove `self._proc` field and `cancel()` method.
- Read `aac_quality` and `ffmpeg_threads` from `self._config` instead of module-level constants.
- Replace `Popen().wait()` with `subprocess.run(...)`.
- Remove `ctx.check_cancelled()` calls.

## Stage 5: `lyric_align.py` (NEW)

**Purpose:** Read the lyrics file, align it to the vocal stem via the injected `WhisperWorker`, generate the karaoke `.ass` file, and (if lyrics input was `.txt`) a segment-level `.srt` file.

**Class:**
```python
class LyricAlignStage(BaseStage):
    name = "lyric_align"

    def __init__(self, whisper_worker: WhisperWorker, config: PipelineConfig) -> None:
        self._worker = whisper_worker
        self._config = config

    def run(self, ctx: StageContext) -> None: ...

    # --- Helpers ---
    def _load_lyrics(self, lyrics_path: Path) -> tuple[str, str]:
        """Return (lyrics_text, lyrics_format) where format is 'txt' or 'srt'."""
        ...

    def _extract_words(self, result) -> list[dict]:
        """Flatten WhisperResult into [{word, start, end, is_segment_first}, ...]."""
        ...

    def _match_words_to_lines(self, words: list[dict], lines: list[str]) -> list[dict]:
        """Assign aligned words to lyrics lines by count."""
        ...

    def _generate_ass(self, line_objects: list[dict]) -> str:
        """Build .ass content from line objects using config styling/timing."""
        ...

    def _generate_srt(self, result) -> str:
        """Build .srt from segment-level timestamps."""
        ...
```

**Flow:**
1. Read `lyrics_path = ctx.artifacts["lyrics_path"]` and `vocal_wav = ctx.artifacts["vocal_wav"]`.
2. `lyrics_text, lyrics_format = self._load_lyrics(lyrics_path)`:
   - If `lyrics_path.suffix.lower() == ".srt"`: parse with `srt.parse(...)`, concatenate `sub.content` with newlines, `lyrics_format = "srt"`. **The original `.srt` timestamps are discarded** — only the text content is kept. stable-ts re-aligns from scratch against the vocal stem, so the produced `.ass` reflects stable-ts's segmentation, not the input `.srt`'s. If the user provided an `.srt`, they already have one and we do not write a new `.srt` output (see step 6), so the original file is preserved on disk.
   - Else: read raw text, `lyrics_format = "txt"`.
   - Store `ctx.artifacts["lyrics_text"] = lyrics_text` and `ctx.artifacts["lyrics_format"] = lyrics_format`.
3. Run alignment:
   ```python
   result = self._worker.align_and_refine(
       vocal_path=vocal_wav,
       lyrics_text=lyrics_text,
       cancel_event=None,
   )
   ```
4. Determine output paths relative to the original song:
   - `ass_out = ctx.song_path.parent / "karaoke" / f"{ctx.song_path.stem}.ass"`
   - `srt_out = ctx.song_path.parent / "subtitles" / f"{ctx.song_path.stem}.srt"` (only if lyrics_format == "txt")
   - Create `karaoke/` and (if needed) `subtitles/` with `mkdir(exist_ok=True)`.
5. Generate ASS:
   - `words = self._extract_words(result)`
   - Split lyrics_text into non-empty stripped lines.
   - `line_objects = self._match_words_to_lines(words, lines)`
   - `ass_content = self._generate_ass(line_objects)`
   - Write `ass_out` as UTF-8.
   - `ctx.artifacts["ass_file"] = ass_out`.
6. If `lyrics_format == "txt"`:
   - `srt_content = self._generate_srt(result)`
   - Write `srt_out` as UTF-8.
   - `ctx.artifacts["srt_file"] = srt_out`.

### `_extract_words(result)`

Ported from `snippets/stable_align.py:extract_words_from_alignment`. Iterates `result.segments`, then each segment's `.words`; builds:
```python
[{
    "word": word.word.strip(),
    "start": word.start,
    "end": word.end,
    "is_segment_first": i == 0,
} for ...]
```

### `_match_words_to_lines(words, lines)`

Ported from `snippets/stable_align.py:match_words_to_lines`. Splits each lyrics line on whitespace, takes `len(split)` words from the aligned list in order, builds:
```python
{
    "text": line,
    "words": [{word, start, end, is_segment_first}, ...],
    "start": first_word_start,
    "end": last_word_end,
}
```

Note: this is a count-based pairing — it assumes the lyrics file has the same word count and order as what stable-ts aligned. This matches the existing `snippets/stable_align.py` behavior.

### `_generate_ass(line_objects)`

Ported from `snippets/stable_align.py:generate_enhanced_karaoke_ass` + `generate_ass_header`. Key differences from the snippet:
- All styling values (`FONT_NAME`, `FONT_SIZE`, `PRIMARY_COLOR`, etc.) come from `self._config` instead of module-level constants.
- All timing values (`LINE_LEAD_IN_CS`, `LINE_LEAD_OUT_CS`, `FIRST_WORD_NUDGE_CS`) come from `self._config`.

The karaoke structure is preserved:
1. Event window: `[first_word_start - lead_in_cs/100, last_word_end + lead_out_cs/100]`.
2. Silent cursor tag `{\k<gap_cs>}` burns through pre-word gaps without sweeping color.
3. Fill sweep tag `{\kf<dur_cs>}word` sweeps color over each word.
4. Optional `first_word_nudge_cs` pushes a segment's first word back when it lands within ~50 ms of the expected lead-in gap (prevents clipping).

`seconds_to_ass_time(seconds)` helper: `f"{h}:{m:02d}:{s:02d}.{cs:02d}"` — identical to the snippet.

### `_generate_srt(result)`

Uses the `srt` library on segment-level timestamps:
```python
import srt
import datetime

def _generate_srt(self, result) -> str:
    subtitles = [
        srt.Subtitle(
            index=i,
            start=datetime.timedelta(seconds=segment.start),
            end=datetime.timedelta(seconds=segment.end),
            content=segment.text.strip(),
        )
        for i, segment in enumerate(result.segments, start=1)
    ]
    return srt.compose(subtitles)
```

Only generated when the input lyrics was `.txt` — if the user provided an `.srt`, they already have one and we don't overwrite it.

## Workers

### `workers/stem_worker.py`

Copied from `cancel_separator/workers/cancelable_stem_worker.py` with the following changes (see implementation step 5 for the full list): rename class, rewrite `cancel_test` imports, update subprocess logger name, and add a `model_name` constructor parameter. Public interface used by the stage:

```python
class StemWorker:
    def __init__(
        self,
        temp_dir: str = "",
        log_level: int = logging.INFO,
        model_dir: str = ...,
        model_name: str = DEFAULT_MODEL_NAME,   # NEW — was a module-level constant
    ) -> None
    def start(self) -> None
    def is_alive(self) -> bool
    def separate(self, wav_path: Path, output_dir: Path, cancel_event: threading.Event | None = None) -> tuple[Path, Path]
    def stop(self) -> None
    def kill(self) -> None
```

**`model_name` wiring:** The current source hardcodes `MODEL_NAME` as a module constant and passes it to `separator.load_model(model_filename=MODEL_NAME)` inside the subprocess entrypoint. The rewrite turns this into:
1. Constructor accepts `model_name` and stores it on `self._model_name`.
2. `model_name` is forwarded to the subprocess (either through the `Process(args=...)` tuple or through the first message on the job pipe).
3. Inside the subprocess, `separator.load_model(model_filename=<forwarded name>)` uses the forwarded value.

`run_pipeline.py` passes `model_name=cfg.separator_model_name` when constructing the worker, so changing the config field actually takes effect.

The stage calls `separate(..., cancel_event=None)` since the prototype doesn't expose cancellation. The subprocess + Pipe + `model_run.forward()` monkey-patch infrastructure is preserved unchanged so it can be wired to a cancel flag later.

### `workers/whisper_worker.py`

Copied from `cancel_whisper/workers/cancelable_whisper_worker.py` with **one behavioral change** (see below). Public interface used by the stage:

```python
class WhisperWorker:
    def __init__(self, config: WhisperModelConfig | None = None) -> None
    @property
    def model_loaded(self) -> bool
    def load_model(self) -> None
    def align(self, vocal_path: Path, lyrics_text: str, cancel_event: threading.Event | None = None) -> WhisperResult
    def refine(self, vocal_path: Path, result, cancel_event: threading.Event | None = None) -> WhisperResult
    def align_and_refine(self, vocal_path: Path, lyrics_text: str, cancel_event: threading.Event | None = None) -> WhisperResult
    def unload_model(self) -> None
```

The stage calls `align_and_refine(..., cancel_event=None)`. The `model.encode()` monkey-patch and AudioLoader FFmpeg stderr suppression are preserved. `WhisperModelConfig` is constructed from `PipelineConfig` fields by the orchestrator.

**Behavioral change from `cancel_whisper/` version:** Remove the `cancel_event.clear()` call between `align()` and `refine()` in `align_and_refine()`. In the pipeline, a cancelled song is discarded entirely — no partial outputs are kept and no further stages run — so a cancel signal during align must propagate through refine rather than being swallowed. Delete lines 517–521 of the original:

```python
# DELETE this block:
if cancel_event is not None:
    cancel_event.clear()
```

The method becomes a straight `align` → (if not cancelled) `refine` → return, with a single shared event driving both.

## Orchestrator (`orchestrator.py`)

```python
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from pipeline.config import PipelineConfig
from pipeline.context import StageContext
from pipeline.stages.base import PipelineStage
from pipeline.workers.stem_worker import StemWorker
from pipeline.workers.whisper_worker import WhisperWorker

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        stages: Sequence[PipelineStage],
        stem_worker: StemWorker,
        whisper_worker: WhisperWorker,
        config: PipelineConfig,
    ) -> None:
        self._stages = list(stages)
        self._stem_worker = stem_worker
        self._whisper_worker = whisper_worker
        self._config = config

    def run(self, song_path: Path, lyrics_path: Path) -> StageContext:
        self._validate_inputs(song_path, lyrics_path)
        self._start_workers()
        try:
            return self._run_pipeline(song_path, lyrics_path)
        finally:
            self._stop_workers()

    # --- Internals ---

    def _validate_inputs(self, song_path: Path, lyrics_path: Path) -> None:
        if not song_path.exists():
            raise FileNotFoundError(f"Song file not found: {song_path}")
        if not lyrics_path.exists():
            raise FileNotFoundError(f"Lyrics file not found: {lyrics_path}")
        if lyrics_path.suffix.lower() not in (".txt", ".srt"):
            raise ValueError(f"Lyrics must be .txt or .srt, got: {lyrics_path.suffix}")

    def _start_workers(self) -> None:
        logger.info("Starting stem worker...")
        self._stem_worker.start()
        logger.info("Loading whisper model...")
        self._whisper_worker.load_model()

    def _stop_workers(self) -> None:
        logger.info("Stopping stem worker...")
        try:
            self._stem_worker.stop()
        except Exception as e:
            logger.warning(f"stem_worker.stop() failed: {e}")
        logger.info("Unloading whisper model...")
        try:
            self._whisper_worker.unload_model()
        except Exception as e:
            logger.warning(f"whisper_worker.unload_model() failed: {e}")

    def _run_pipeline(self, song_path: Path, lyrics_path: Path) -> StageContext:
        tmp_parent = self._config.intermediate_dir or None
        tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_", dir=tmp_parent))
        logger.info(f"Temp dir: {tmp_dir}")

        ctx = StageContext(
            song_path=song_path,
            tmp_dir=tmp_dir,
            config=self._config,
        )
        ctx.artifacts["lyrics_path"] = lyrics_path

        try:
            for stage in self._stages:
                logger.info(f"[pipeline] ▶ {stage.name}")
                stage.run(ctx)
                logger.info(f"[pipeline] ✓ {stage.name}")
            return ctx
        finally:
            logger.debug(f"[pipeline] Cleanup tmp: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
```

**Key properties:**
- Synchronous, single-threaded — no job queue, no orchestrator thread, no cancel flag.
- Worker lifecycle is tied to a single `run()` call: workers start before the first stage, stop after the last (or on exception).
- The temp dir is always cleaned up in `finally`, even if a stage raises.
- If a stage raises, the exception propagates out of `run()` unchanged — the caller decides how to report it. Workers are still torn down because the outer `finally` wraps the entire pipeline.

## CLI Entry Point (`run_pipeline.py`)

```python
import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.stages.ffmpeg_extract import FFmpegExtractStage
from pipeline.stages.ffmpeg_transcode import FFmpegTranscodeStage
from pipeline.stages.loudnorm_analyze import LoudnormAnalyzeStage
from pipeline.stages.lyric_align import LyricAlignStage
from pipeline.stages.stem_separation import StemSeparationStage
from pipeline.workers.stem_worker import StemWorker
from pipeline.workers.whisper_worker import WhisperWorker
from cancel_whisper.config import WhisperModelConfig   # reuse existing dataclass


def build_whisper_config(cfg: PipelineConfig) -> WhisperModelConfig:
    return WhisperModelConfig(
        model_path=cfg.whisper_model_path,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        language=cfg.whisper_language,
        vad=cfg.whisper_vad,
        vad_threshold=cfg.whisper_vad_threshold,
        suppress_silence=cfg.whisper_suppress_silence,
        suppress_word_ts=cfg.whisper_suppress_word_ts,
        only_voice_freq=cfg.whisper_only_voice_freq,
        refine_steps=cfg.whisper_refine_steps,
        refine_word_level=cfg.whisper_refine_word_level,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Staged pipeline prototype.")
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("lyrics_file", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig()

    stem_worker = StemWorker(
        temp_dir=cfg.intermediate_dir,
        model_dir=cfg.separator_model_dir,
        model_name=cfg.separator_model_name,
    )
    whisper_worker = WhisperWorker(build_whisper_config(cfg))

    stages = [
        FFmpegExtractStage(cfg),
        LoudnormAnalyzeStage(cfg),
        StemSeparationStage(stem_worker),
        FFmpegTranscodeStage(cfg),
        LyricAlignStage(whisper_worker, cfg),
    ]

    orchestrator = PipelineOrchestrator(stages, stem_worker, whisper_worker, cfg)

    try:
        ctx = orchestrator.run(args.audio_file, args.lyrics_file)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return 1

    print("\n=== Pipeline complete ===")
    print(f"Vocal M4A:     {ctx.artifacts.get('vocal_m4a')}")
    print(f"Nonvocal M4A:  {ctx.artifacts.get('nonvocal_m4a')}")
    print(f"ASS:           {ctx.artifacts.get('ass_file')}")
    if "srt_file" in ctx.artifacts:
        print(f"SRT:           {ctx.artifacts['srt_file']}")
    print(f"Loudnorm I:    {ctx.artifacts.get('loudnorm_input_i')} LUFS")
    print(f"Loudnorm TP:   {ctx.artifacts.get('loudnorm_input_tp')} dBTP")
    print(f"Loudnorm LRA:  {ctx.artifacts.get('loudnorm_input_lra')} LU")
    print(f"Target offset: {ctx.artifacts.get('loudnorm_target_offset')} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## End-to-End Data Flow

```
run_pipeline.py args
  │
  ▼
Orchestrator.run(song_path, lyrics_path)
  │
  ├─► stem_worker.start()          ← subprocess spawns, audio-separator model loads
  ├─► whisper_worker.load_model()  ← in-process, faster-whisper model loads
  │
  ├─► tmp_dir = mkdtemp(...)
  │
  ├─► ctx = StageContext(song_path, tmp_dir, config)
  ├─► ctx.artifacts["lyrics_path"] = lyrics_path
  │
  ├─► [1] FFmpegExtractStage.run(ctx)
  │        ffmpeg -i song → tmp_dir/<stem>_input.wav
  │        ctx.artifacts["extracted_wav"] = <path>
  │
  ├─► [2] LoudnormAnalyzeStage.run(ctx)
  │        ffmpeg -i extracted_wav -af loudnorm=...print_format=json -f null -
  │        parse stderr JSON
  │        ctx.artifacts["loudnorm_input_i"]       = float
  │        ctx.artifacts["loudnorm_input_tp"]      = float
  │        ctx.artifacts["loudnorm_input_lra"]     = float
  │        ctx.artifacts["loudnorm_input_thresh"]  = float
  │        ctx.artifacts["loudnorm_target_offset"] = float
  │        ctx.artifacts["loudnorm_type"]          = str
  │
  ├─► [3] StemSeparationStage.run(ctx)
  │        stem_worker.separate(extracted_wav, tmp_dir) via subprocess IPC
  │        ctx.artifacts["vocal_wav"]        = tmp_dir/…_(Vocals).wav
  │        ctx.artifacts["instrumental_wav"] = tmp_dir/…_(Instrumental).wav
  │
  ├─► [4] FFmpegTranscodeStage.run(ctx)
  │        ffmpeg -i vocal_wav -c:a aac → <song_dir>/vocal/<stem>---vocal.m4a
  │        ffmpeg -i instrumental_wav -c:a aac → <song_dir>/nonvocal/<stem>---nonvocal.m4a
  │        ctx.artifacts["vocal_m4a"]    = <path>
  │        ctx.artifacts["nonvocal_m4a"] = <path>
  │
  ├─► [5] LyricAlignStage.run(ctx)
  │        read lyrics_path → (lyrics_text, lyrics_format)
  │        ctx.artifacts["lyrics_text"]   = str
  │        ctx.artifacts["lyrics_format"] = "txt"|"srt"
  │        result = whisper_worker.align_and_refine(vocal_wav, lyrics_text)
  │        ass_content = _generate_ass(_match_words_to_lines(_extract_words(result), lines))
  │        write <song_dir>/karaoke/<stem>.ass
  │        ctx.artifacts["ass_file"] = <path>
  │        if lyrics_format == "txt":
  │            srt_content = _generate_srt(result)
  │            write <song_dir>/subtitles/<stem>.srt
  │            ctx.artifacts["srt_file"] = <path>
  │
  ├─► shutil.rmtree(tmp_dir)       ← intermediate WAVs discarded
  ├─► stem_worker.stop()           ← subprocess exits, model unloads
  ├─► whisper_worker.unload_model()← model freed from GPU
  │
  ▼
return ctx
```

## Output Files (produced alongside the input song)

```
<song_dir>/
├── <song_stem>.<ext>              ← original (unchanged)
├── karaoke/
│   └── <song_stem>.ass            ← NEW
├── subtitles/
│   └── <song_stem>.srt            ← NEW (only if lyrics input was .txt)
├── vocal/
│   └── <song_stem>---vocal.m4a    ← NEW
└── nonvocal/
    └── <song_stem>---nonvocal.m4a ← NEW
```

## Prerequisites (must happen before Implementation Steps)

These are pre-existing gaps in the surrounding code that the pipeline depends on. Resolve them first, or the pipeline will fail at import/install time.

### P1. Create `cancel_whisper/config.py`

`cancel_whisper/workers/cancelable_whisper_worker.py` (line 82) imports `from cancel_whisper.config import WhisperModelConfig`, but the file does not exist — the whole `cancel_whisper` package is broken at import time today. Create it with the following dataclass (fields inferred from the worker's usage):

```python
# cancel_whisper/config.py
from dataclasses import dataclass


@dataclass
class WhisperModelConfig:
    model_path: str = ""
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    vad: bool = True
    vad_threshold: float = 0.25
    suppress_silence: bool = True
    suppress_word_ts: bool = True
    only_voice_freq: bool = True
    refine_steps: str = "s"
    refine_word_level: bool = False
```

The pipeline reuses this dataclass rather than redefining it — `run_pipeline.build_whisper_config()` constructs one from `PipelineConfig`.

### P2. Stem-worker source is in `cancel_separator/`, and its internal imports still use `cancel_test`

The folder was renamed `cancel_test/` → `cancel_separator/`, but every file inside still has `from cancel_test.context import ...` etc. When copying files into `pipeline/`, the copy step is also a rewrite step — all `cancel_test` references in the source become `pipeline` (or the matching `pipeline.workers.*` / `pipeline.stages.*` path). This also covers the hardcoded subprocess logger name `logging.getLogger("cancel_test.worker")` at line 519 of `cancelable_stem_worker.py` — it becomes `logging.getLogger("pipeline.workers.stem_worker")`.

### P3. Add missing dependencies to `requirements.txt`

Current file has only `stable-ts`, `faster-whisper`, `lyricsgenius`, `#torch`. The pipeline also needs:

- `srt` — used by `lyric_align.py` for both `.srt` input parsing and segment-level `.srt` output
- `audio-separator` — used by the stem worker

Both must be added before `run_pipeline.py` can run.

## Implementation Steps

1. **Create folder structure** — `pipeline/`, `pipeline/stages/`, `pipeline/workers/` with `__init__.py` files
2. **Create `config.py`** — `PipelineConfig` dataclass with all settings
3. **Create `context.py`** — simplified `StageContext` (no cancelled event, no lyrics fields on context)
4. **Create `stages/base.py`** — `PipelineStage` protocol with just `name` + `run()`, `BaseStage` class
5. **Copy `workers/stem_worker.py`** — from `cancel_separator/workers/cancelable_stem_worker.py`. Rename class `CancelableStemWorker` → `StemWorker`. Also carry over the error classes `WorkerCancelledError` and `WorkerDiedError` (names unchanged — the stage's `except WorkerDiedError` depends on them). Rewrite all `cancel_test.*` imports (the source still uses the old package name). Update the subprocess logger name from `cancel_test.worker` → `pipeline.workers.stem_worker`. Add a `model_name` constructor parameter (see "Workers" section) so `PipelineConfig.separator_model_name` actually takes effect.
6. **Copy `workers/whisper_worker.py`** — from `cancel_whisper/workers/cancelable_whisper_worker.py`. Rename class `CancelableWhisperWorker` → `WhisperWorker`. Carry over `AlignmentCancelledError` (raised internally — module must be self-consistent even though the stage doesn't catch it). Apply the `align_and_refine` cancel-event change described in the "Workers" section.
7. **Port `stages/ffmpeg_extract.py`** — remove cancel logic, rename artifact to `extracted_wav`. Keep an `__init__(self, config: PipelineConfig)` so the stage can read `ffmpeg_threads`.
8. **Create `stages/loudnorm_analyze.py`** — new stage: ffmpeg loudnorm 1st pass, parse JSON, store stats
9. **Port `stages/stem_separation.py`** — remove cancel logic, keep worker delegation
10. **Port `stages/ffmpeg_transcode.py`** — remove cancel logic, read AAC params from config. Update `_transcode` signature to `(self, wav_path: Path, output_path: Path)` — drop the `ctx` argument that was only used for `ctx.check_cancelled()`.
11. **Create `stages/lyric_align.py`** — lyrics loading, whisper align_and_refine, ASS generation (port from snippets), SRT generation
12. **Create `orchestrator.py`** — synchronous pipeline runner, owns both worker lifecycles
13. **Create `run_pipeline.py`** — CLI entry point; takes only `<audio_file> <lyrics_file>` positional args

## Dependencies

Already in `requirements.txt`:
- `stable-ts`
- `faster-whisper`

Must be added (see Prerequisite P3):
- `srt` — for `.srt` parsing and generation in `lyric_align.py`
- `audio-separator` — for the stem worker

## Notes

- `StemWorker` is copied with mechanical changes only (import rewrites, logger name, new `model_name` param); its cancellation infrastructure is preserved intact. `WhisperWorker` is copied with one behavioral change: `align_and_refine` no longer clears the caller's `cancel_event` between phases, so a single cancel propagates through both `align` and `refine`. The prototype doesn't expose cancel controls, but `separate(..., cancel_event=<event>)` and `align_and_refine(..., cancel_event=<event>)` can be wired up later by passing a shared event — cancelling a song discards all intermediate outputs and skips remaining stages, which matches the pipeline's "cancel = drop the song" semantics.
- Both models are loaded once by the orchestrator before any stage runs, and torn down after all stages complete (or on exception). Stages never create or destroy workers.
- ASS generation in `lyric_align.py` ports `snippets/stable_align.py` but reads all styling/timing parameters from `PipelineConfig` instead of module-level constants.
- `loudnorm_analyze` only stores measurement values — it does not apply normalization. A future stage can use `loudnorm_target_offset` for a 2nd-pass correction if desired.
- The A1 architecture (pikaraoke `pipeline-robustness-fixes.md`) is the reference model. Differences: single-song vs multi-song queue, no cancellation, in-process whisper worker vs subprocess. Everything else — stage protocol, worker injection, artifacts dict, orchestrator ownership — matches A1.
