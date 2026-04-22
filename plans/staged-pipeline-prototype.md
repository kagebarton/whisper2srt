# Plan: Staged Processing Manager Prototype

## Overview

Create a new `pipeline/` folder (sibling to `cancel_test/`) that implements a staged processing manager for a single song. The prototype mocks the processing pipeline and adds a new lyric alignment + ASS/SRT generation stage based on `snippets/stable_align.py`.

## Folder Structure

```
pipeline/
├── __init__.py
├── config.py              # Configuration dataclass (model paths, intermediate dir, ASS styling, etc.)
├── context.py             # Per-job StageContext (adapted from cancel_test)
├── orchestrator.py        # Simplified orchestrator — single song, no queue, no cancellation, no worker subprocess
├── stages/
│   ├── __init__.py
│   ├── base.py            # PipelineStage protocol + BaseStage (from cancel_test, minus cancel logic)
│   ├── ffmpeg_extract.py  # FFmpeg extract audio → WAV (from cancel_test)
│   ├── stem_separation.py # Stem separation via CancelableStemWorker (from cancel_test)
│   ├── ffmpeg_transcode.py# FFmpeg transcode WAV → M4A (from cancel_test)
│   └── lyric_align.py     # NEW: Lyric alignment + ASS/SRT generation via stable-ts
├── workers/
│   ├── __init__.py
│   └── cancelable_stem_worker.py  # From cancel_test (unchanged)
└── run_pipeline.py         # CLI entry point
```

## Key Changes from `cancel_test/`

### 1. Strip multi-cycle features
- Remove the job queue from the orchestrator — it processes exactly one song per invocation
- Remove cancellation logic from all stages (no `cancel()` method, no `CancelledError` checking)
- Remove `_stop_event`, pending queue, and orchestrator thread — the pipeline runs synchronously in the main thread
- Remove `cancel()` from `PipelineStage` protocol and `BaseStage`

### 2. Simplified orchestrator
`orchestrator.py`:
- Takes a list of stages + a stem worker
- `run(song_path, lyrics_path) -> StageContext`: creates context, runs stages sequentially, returns the context
- Manages stem worker lifecycle (`start()` before pipeline, `stop()` after)
- Creates and cleans up intermediate temp dir under the configured intermediate directory

### 3. New stage: `lyric_align.py`

**Responsibilities:**
1. Read the lyrics file (`.txt` or `.srt`)
2. If `.srt`: use the `srt` library to parse and extract plain text
3. If `.txt`: keep the text as-is
4. Load a stable-ts model and run `model.align()` on the vocal stem against the lyrics text
5. Run `model.refine()` on the alignment result
6. Extract word-level timestamps and match to lyrics lines
7. Generate `.ass` file with karaoke highlighting (ported from `snippets/stable_align.py`)
8. If the input lyrics was `.txt` (not `.srt`): also generate an `.srt` file from the alignment result
9. Write output files alongside the original song (same directory as `song_path`)

**Input artifacts from previous stages:**
- `ctx.artifacts["vocal_wav"]` — the vocal stem WAV from stem separation

**New artifacts added to context:**
- `ctx.artifacts["ass_file"]` — Path to the generated `.ass` file
- `ctx.artifacts["srt_file"]` — Path to the generated `.srt` file (only when lyrics input was `.txt`)

**Output file locations:**
- `.ass` file: `<song_dir>/<song_stem>.ass`
- `.srt` file (if generated): `<song_dir>/<song_stem>.srt`

### 4. Configuration (`config.py`)

```python
@dataclass
class PipelineConfig:
    # Model paths
    whisper_model_path: str = "/home/ken/whisper2srt/whisper-model/large-v3"
    separator_model_dir: str = "./audio-separator/models"
    separator_model_name: str = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"

    # Device/compute
    device: str = "auto"
    compute_type: str = "int8"

    # Intermediate files directory
    intermediate_dir: str = ""  # Empty = system temp dir

    # ASS styling (from snippets/stable_align.py)
    font_name: str = "Arial"
    font_size: int = 60
    primary_color: str = "&H00D7FF&"
    secondary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000&"
    outline_width: int = 3
    shadow_offset: int = 2
    margin_left: int = 50
    margin_right: int = 50
    margin_vertical: int = 150

    # Karaoke timing (centiseconds)
    line_lead_in_cs: int = 80
    line_lead_out_cs: int = 20
    first_word_nudge_cs: int = 0
```

All model paths, the intermediate directory, device/compute settings, and styling parameters are **config entries only** — there are no CLI flags for them. To change values, users edit the defaults in `config.py` or construct a `PipelineConfig` with custom values and pass it to the orchestrator programmatically.

The `intermediate_dir` config entry controls where the pipeline's temporary working directory is created. When set to a non-empty path, `tempfile.mkdtemp()` uses it as the `dir` argument, so all intermediate WAVs, stems, etc. land there. When empty (default), the system temp dir is used.

### 5. CLI entry point (`run_pipeline.py`)

```
python run_pipeline.py <audio_file> <lyrics_file>
```

- Validates that both input files exist
- Loads config from `PipelineConfig` defaults (all paths and settings are config entries, not CLI arguments)
- Builds the pipeline stages
- Runs the orchestrator
- Prints summary of outputs

The CLI takes only the two required positional arguments (audio file and lyrics file). All configuration — intermediate directory, model paths, device settings, ASS styling — lives in `PipelineConfig`. To change settings, users edit the defaults in `config.py` or construct a `PipelineConfig` programmatically.

### 6. SRT generation (when lyrics input is `.txt`)

When the lyrics file is a `.txt`, the alignment result already contains segment-level and word-level timestamps. We generate an SRT file using the `srt` library:

```python
import srt

def generate_srt_from_result(result):
    subtitles = []
    for i, segment in enumerate(result.segments, start=1):
        sub = srt.Subtitle(
            index=i,
            start=datetime.timedelta(seconds=segment.start),
            end=datetime.timedelta(seconds=segment.end),
            content=segment.text.strip(),
        )
        subtitles.append(sub)
    return srt.compose(subtitles)
```

When the lyrics input is `.srt`, we skip SRT generation (the user already has one).

### 7. Context changes

`StageContext` adds:
- `config: PipelineConfig` — shared config reference
- `lyrics_path: Path` — path to the lyrics file
- `lyrics_text: str | None` — populated by lyric_align stage after reading/converting
- `lyrics_format: str` — `"txt"` or `"srt"`, detected from file extension

The `cancelled` event and `CancelledError` are removed.

## Implementation Steps

1. **Create folder structure** — `pipeline/`, `pipeline/stages/`, `pipeline/workers/` with `__init__.py` files
2. **Port `config.py`** — new file with `PipelineConfig` dataclass
3. **Port `context.py`** — simplified `StageContext` (remove cancelled, add config/lyrics fields)
4. **Port `stages/base.py`** — `PipelineStage` protocol with just `name` + `run()`, `BaseStage` base class
5. **Port `stages/ffmpeg_extract.py`** — remove cancel logic, keep the core extraction
6. **Port `stages/ffmpeg_transcode.py`** — remove cancel logic, keep the core transcoding
7. **Port `stages/stem_separation.py`** — remove cancel logic, keep the worker delegation
8. **Copy `workers/cancelable_stem_worker.py`** — unchanged from cancel_test
9. **Create `stages/lyric_align.py`** — new stage with:
   - Lyrics loading (txt/srt detection, srt→txt conversion)
   - stable-ts model loading and alignment
   - ASS generation (ported from `snippets/stable_align.py`)
   - SRT generation (when lyrics input is `.txt`)
10. **Create `orchestrator.py`** — simplified synchronous pipeline runner
11. **Create `run_pipeline.py`** — CLI entry point; takes only `<audio_file> <lyrics_file>` positional args; all other settings come from `PipelineConfig`

## Dependencies

Already in `requirements.txt`:
- `stable-ts`
- `faster-whisper`
- `srt`
- `audio-separator`

## Notes

- The `CancelableStemWorker` is copied as-is since it's needed for stem separation. The cancellation mechanism within the worker is preserved (it's useful infrastructure even though the orchestrator doesn't expose cancel controls).
- The `lyric_align.py` stage runs stable-ts `model.align()` and `model.refine()` with the same parameters as `snippets/stable_align.py` (VAD enabled, English language, etc.)
- ASS styling and karaoke timing parameters are all pulled from `PipelineConfig` defaults (matching `snippets/stable_align.py` values)
