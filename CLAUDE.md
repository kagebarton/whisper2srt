# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A karaoke mixer web app (`mpv/`) that controls MPV playback of a video mixed with separate vocal and instrumental stems. Features real-time pitch shifting, volume control, subtitle support, and a file browser UI.

## Communication style

Use very concise explanations. Avoid lengthy summaries or over-explanation. Be direct about what was changed and why.

## Development environment

This project runs in a conda environment named `pik`. Activate it before working:

```bash
conda activate pik
```

## Plans

Name plan files with a short, descriptive kebab-case filename that reflects the task
(e.g., `subtitle-delay-cleanup.md`, `auth-refactor.md`), not the auto-generated random
name Claude Code assigns by default.

## Running the app

```bash
cd mpv
pip install -r requirements.txt   # flask>=3.0
python app.py
# open http://localhost:5000
```

Requires `mpv` on `$PATH`.

## Architecture

```
Browser UI  ──HTTP/JSON──►  Flask (app.py)  ──libmpv bindings──►  mpv in-process
```

**`mpv/app.py`** — the entire backend. Owns `MpvController` (wraps libmpv). Key responsibilities:
- Launches MPV in-process via python-mpv bindings (three audio tracks: video + vocal + nonvocal, mixed via `lavfi-complex`)
- Event-driven architecture: property observers for time-pos, duration, idle-active, osd dimensions
- Volume and pitch changes rebuild `lavfi-complex` and send live via `controller.set_lavfi_complex()`
- Pitch converted: semitones (±6) → multiplier `2^(st/12)` → `rubberband` filter on both stems

**`mpv/templates/index.html`** — single-page UI with file browser, seek bar, vocal volume slider, pitch slider, subtitle delay control.

### lavfi-complex filter
The filter mixes two audio tracks:
- `aid2` = vocal stem → `volume@vocalvol` → `rubberband@vocalrb` (formant=preserved)
- `aid3` = nonvocal stem → `volume@nonvocalvol` → `rubberband@nonvocalrb` (formant=shifted)
- Both → `amix=inputs=2:normalize=0` → `[ao]`

The entire filter string is rebuilt and re-sent to MPV on every volume or pitch change.

### Subtitle handling
Browser maintains two subtitle data caches:
- `_assData`/`_srtData` (pending): next file selected in browser
- `_activeAssData`/`_activeSrtData` (active): currently-playing file

UI buttons read from active cache, preventing file-selection from corrupting live UI state.

### State
A single `state` dict in `app.py` holds all runtime state: file paths, vocal volume, semitones, playing flag, position, duration, subtitle delay.

## Planning guidance

When creating implementation plans (via `/plan` or `EnterPlanMode`), use descriptive names that summarize the task. Examples:
- "add-qr-overlay-to-mixer"
- "fix-mpv-reuse-and-polling"
- "refactor-filter-complex-builder"

This helps keep conversations focused and provides clear context.

## Background context (other scripts in repo)

The root-level scripts are the pipeline that produces the files the mixer plays:
1. `snippets/separate.py` — splits a video into `---vocal.m4a` + `---nonvocal.m4a` using `audio-separator`
2. `whisper2srt_genius.py` / `whisper2srt_transcription.py` — transcribe or align lyrics to `.srt`
3. `snippets/stable_align.py` — generates `.ass` karaoke files with progressive word highlighting (`\kf` tags)
