# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A karaoke mixer web app (`mpv/`) that controls MPV playback of a video mixed with separate vocal and instrumental stems. Features real-time pitch shifting, volume control, subtitle support, and a file browser UI.

## Development environment

This project runs in a conda environment named `avtest`. Activate it before working:

```bash
conda activate avtest
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
Browser UI  ──HTTP/JSON──►  Flask (app.py)  ──UNIX socket──►  mpv process
```

**`mpv/app.py`** — the entire backend. Key responsibilities:
- Launches MPV as a subprocess with three tracks: video + vocal `.m4a` + nonvocal `.m4a`, mixed via `--lavfi-complex`
- Communicates with MPV via newline-delimited JSON over a UNIX socket (`/tmp/mpv-socket`)
- A background thread polls `time-pos` and `duration` from MPV every 500ms
- Volume and pitch changes rebuild the entire `lavfi-complex` filter string and push it live via `set_property`
- Pitch is converted from semitones (±6) to a multiplier via `2^(st/12)` and applied via `rubberband` filter on both stems

**`mpv/templates/index.html`** — single-page UI with file browser, seek bar, vocal volume slider, pitch slider, subtitle delay control.

### lavfi-complex filter
The filter mixes two audio tracks:
- `aid2` = vocal stem → `volume@vocalvol` → `rubberband@vocalrb` (formant=preserved)
- `aid3` = nonvocal stem → `volume@nonvocalvol` → `rubberband@nonvocalrb` (formant=shifted)
- Both → `amix=inputs=2:normalize=0` → `[ao]`

The entire filter string is rebuilt and re-sent to MPV on every volume or pitch change.

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
