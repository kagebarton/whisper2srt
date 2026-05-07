# MPV Mixer – Web Prototype

A Flask-based web UI that launches and controls an MPV instance for mixing vocal and non-vocal audio tracks with real-time pitch adjustment.

## Prerequisites

- **Python 3.8+**
- **mpv** installed and on your `$PATH`
- **socat** is NOT required (communication is done via Python sockets)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

## Usage

1. **Select files** — paste or browse for the video, vocal `.m4a`, and non-vocal `.m4a` paths.
   - Due to browser security, full local paths must be entered manually (e.g. `/home/ken/whisper2srt/video.mp4`).
2. Hit **▶ Play** to launch MPV with the three tracks mixed via `--lavfi-complex`.
3. Adjust **Vocal Volume** and **Pitch (±6 semitones)** — changes are applied live by rebuilding the filter complex and sending it to MPV via IPC.
4. Use the **seek bar** to jump to any position (seeks on release; position updates every 500 ms).
5. **⏹ Stop** quits MPV and resets the UI.

## Architecture

```
┌─────────────┐      HTTP / JSON       ┌───────────┐    UNIX socket    ┌──────┐
│  Browser UI │ ◄────────────────────► │ Flask app │ ◄───────────────► │ mpv  │
└─────────────┘                        └───────────┘                   └──────┘
  - seek/vol/pitch                      - rebuilds lavfi-complex
  - polls /api/status                     on vol/pitch change
    every 500 ms                        - polls time-pos via socket
```

## MPV command (reference)

```
mpv video.mp4 \
    --audio-file=vocal.m4a \
    --audio-file=nonvocal.m4a \
    --lavfi-complex="[aid2]volume@vocalvol=1.0[vocal];[aid3]volume@nonvocalvol=1.0[nonvocal];[vocal][nonvocal]amix=inputs=2:normalize=0[mixed];[mixed]rubberband@rb=pitch=1.0:formant=preserved[ao]" \
    --input-ipc-server=/tmp/mpv-socket
```

Volume and pitch changes are sent as:

```
echo '{"command":["set_property","lavfi-complex","..."]}' | socat - /tmp/mpv-socket
```

(Implemented in Python via `socket.AF_UNIX`.)
