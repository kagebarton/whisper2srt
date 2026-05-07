# MPV Mixer – Project Context

## Overview

This is a prototype playback framework to replace the one in the parent project; it is designed around all the features that would apply to one song with all contexts (song and system) mocked up via config entries.

**MPV Mixer** is a Flask-based web application that provides a browser UI for launching and controlling an [mpv](https://mpv.io/) media player instance. Its primary purpose is **karaoke-style audio mixing**: it plays a video alongside separate vocal and non-vocal audio tracks, with real-time controls for vocal volume, pitch shifting (±6 semitones), subtitle delay, and seeking.

The app targets Linux (developed on `/home/ken/whisper2srt/`) and communicates with mpv via a UNIX domain socket using mpv's JSON IPC protocol.

## Architecture

```
Browser UI  ◄──HTTP/JSON──►  Flask app (app.py)  ◄──UNIX socket──►  mpv process
  - seek/vol/pitch              - launches mpv subprocess
  - polls /api/status           - rebuilds lavfi-complex on vol/pitch change
    every 500 ms                - polls time-pos via socket
```

### Key Components

| File | Description |
|---|---|
| `app.py` | Flask server with REST API endpoints, mpv process management, and IPC socket communication |
| `templates/index.html` | Single-page web UI with file browser, transport controls, sliders, and a modal file picker |
| `requirements.txt` | Python dependencies (Flask >= 3.0) |
| `qrcode.png` | QR code image overlaid on video playback via mpv's `lavfi-complex` filter |

### Audio Filter Chain

MPV is launched with a `--lavfi-complex` that:
1. Overlays `qrcode.png` on the video
2. Applies per-track **volume** and **rubberband pitch shifting** to vocal and non-vocal stems
3. Mixes both stems via `amix`
4. Uses different rubberband presets for voice (formant-preserved) vs. instruments (formant-shifted)

## Tech Stack

- **Backend:** Python 3.8+, Flask >= 3.0
- **Frontend:** Vanilla HTML/CSS/JavaScript (no framework)
- **Media Engine:** mpv (external, must be on `$PATH`)
- **IPC:** UNIX domain socket (`/tmp/mpv-socket`) with JSON protocol

## Running the Application

> **Note:** This project runs inside the **`avtest`** conda environment. Activate it first:
> ```bash
> conda activate avtest
> ```

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (listens on 0.0.0.0:5000)
python app.py
```

Then open **http://localhost:5000** in a browser.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Render the main UI |
| POST | `/api/files` | Set video, vocal, non-vocal, and subtitle file paths |
| POST | `/api/play` | Launch mpv with the selected files |
| POST | `/api/pause` | Toggle pause |
| POST | `/api/stop` | Quit mpv and reset state |
| POST | `/api/exit` | Shut down the Flask server |
| POST | `/api/seek` | Seek to an absolute position (seconds) |
| POST | `/api/volume` | Set vocal volume (0.0 – 2.0) |
| POST | `/api/pitch` | Set pitch shift in semitones (-6 to +6) |
| POST | `/api/sub_delay` | Set subtitle delay in seconds (±5s) |
| GET | `/api/status` | Poll playback state (position, duration, volume, pitch, etc.) |
| GET | `/api/browse?path=...` | Server-side directory listing for the file browser modal |

## Default File Paths

The app defaults to files in `/home/ken/whisper2srt/song/`:
- Video: `selfish.mp4`
- Vocal: `selfish---vocal.m4a`
- Non-vocal: `selfish---nonvocal.m4a`
- Subtitle: `selfish---vocal.ass`

These are hardcoded in `app.py` and should be updated if the song changes.

## Key Implementation Details

- **mpv lifecycle:** `launch_mpv()` spawns a subprocess; `stop_mpv()` sends a `quit` command via IPC and cleans up the socket file.
- **Position polling:** A daemon thread queries `time-pos` and `duration` every 500 ms. If mpv exits, it auto-resets the playing state.
- **Live filter rebuild:** Volume and pitch changes rebuild the entire `lavfi-complex` string and send it as a `set_property` command to mpv.
- **Thread safety:** Global `state` dict is shared between Flask request handlers and the polling thread. No explicit locking is used (relies on GIL for simple assignments).
- **Shutdown:** The `/api/exit` endpoint attempts Werkzeug's `shutdown` callable, falling back to `os.kill(os.getpid(), signal.SIGINT)`.

## Development Conventions

- Single-file Flask app (`app.py`) – no blueprints or modularization.
- Vanilla JS in the template – no build step, no npm/Node.js dependencies.
- Dark theme UI with CSS custom properties for theming.
- Flask runs in `debug=True` during development.

## Plans

Implementation plans are stored in `/home/ken/.claude/plans/`. When asked to understand or implement a plan, look for it in that directory.
