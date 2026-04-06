import os
import json
import signal
import subprocess
import socket
import threading
import time
import logging
from pathlib import Path
import qrcode
from PIL import Image, ImageFont
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SHOW_QR = True   # Show QR code + URL overlay pair (top-left)
OVERLAY_MARGIN_TOP = 0   # px from top edge in ASS 1920x1080 space

# Common ASS style tags applied to every overlay text (inside the {…} block).
# Examples: \bord2\shad1\fnDejaVu Sans\b1
# Leave empty ("") for no extra styling.
OVERLAY_STYLE = "\\bord3\\shad2\\3c&H000000&\\4c&H000000&\\4a&H80&"

# Overlay text colors (ASS hex BB GG RR format)
URL_COLOR        = "&HFFFFFF&"   # URL text (top-left)
NOWPLAYING_COLOR = "&H507FFF&"   # "Now Playing: <title>"
TIMECODE_COLOR   = "&HAAD5FF&"   # "1:23 / 3:45 | Transpose: +2 | Vocals: 80%"
UPNEXT_COLOR     = "&HB48246&"   # "Up Next: <title>"

# ── OSD ID constants ───────────────────────────────────────────────────────────
OSD_URL        = 1   # URL text, paired with QR bitmap, top-left
OSD_NOWPLAYING = 2   # "Now Playing: <title>" line, top-right
OSD_TIMECODE   = 3   # "1:23 / 3:45 | Transpose: +2 | Vocals: 80%" line, top-right
OSD_UPNEXT     = 4   # "Up Next: <title>", top-right

# ── State ──────────────────────────────────────────────────────────────────────
IPC_SOCKET = "/tmp/mpv-socket"
NONVOCAL_VOL = 1.0  # fixed
QR_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrcode.png")
PLACEHOLDER_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placeholder.png")
LOGO_JPG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.jpg")

SONG_DIR = "/home/ken/whisper2srt/song"
DEFAULT_VIDEO    = os.path.join(SONG_DIR, "selfish.mp4")

def derive_companion_paths(video_path):
    """Given a video path, return derived companion file paths."""
    p = Path(video_path)
    stem = p.stem.split("---")[0]   # e.g. "selfish" from "selfish.mp4"
    base = p.parent / stem
    return {
        "vocal":    str(base) + "---vocal.m4a",
        "nonvocal": str(base) + "---nonvocal.m4a",
        "ass":      str(base) + ".ass",
        "srt":      str(base) + ".srt",
    }

def semitones_to_pitch(st):
    """Convert semitone offset to MPV rubberband pitch multiplier."""
    return 2 ** (st / 12)

state = {
    "video_path": DEFAULT_VIDEO,
    "vocal_path": None,
    "nonvocal_path": None,
    "subtitle_path": None,
    "subtitle_delay": 0.0,       # seconds, positive = delay, negative = advance
    "vocal_volume": 1.0,
    "semitones": 0,            # -3 to +3, mapped to pitch via 2^(st/12)
    "playing": False,
    "duration": 0.0,
    "position": 0.0,
    "playing_video_path": None,   # tracks what is currently loaded in MPV
}

mpv_proc = None
poll_thread = None
poll_stop = threading.Event()

# Persistent socket for overlay commands (overlay-add and osd-overlay are client-scoped;
# MPV removes them when the connection that created them closes).
_overlay_sock = None
_overlay_sock_lock = threading.Lock()


def _get_overlay_sock():
    """Return the persistent overlay socket, creating it if needed."""
    global _overlay_sock
    if _overlay_sock is not None:
        return _overlay_sock
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(IPC_SOCKET)
        _overlay_sock = s
        return s
    except OSError:
        return None


def send_overlay_command(cmd_dict):
    """Send an overlay/OSD command over the persistent socket (with one reconnect retry)."""
    global _overlay_sock
    with _overlay_sock_lock:
        for _ in range(2):
            s = _get_overlay_sock()
            if s is None:
                return
            try:
                payload = json.dumps(cmd_dict).encode() + b'\n'
                s.sendall(payload)
                s.settimeout(1.0)
                # Read until we get a command response (skip event notifications)
                buf = b""
                for _ in range(10):
                    try:
                        buf += s.recv(4096)
                    except socket.timeout:
                        break
                    # Parse each newline-delimited JSON message
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                            if "error" in msg or "request_id" in msg:
                                return  # got the command response
                        except json.JSONDecodeError:
                            pass
                return
            except OSError:
                try:
                    s.close()
                except OSError:
                    pass
                _overlay_sock = None


def send_osd(overlay_id, data, res_x=1920, res_y=1080):
    """Send an ASS event string to a specific OSD overlay slot."""
    send_overlay_command({
        "command": {"name": "osd-overlay", "id": overlay_id,
                    "format": "ass-events", "data": data,
                    "res_x": res_x, "res_y": res_y}
    })


def clear_osd(overlay_id, res_x=1920, res_y=1080):
    """Clear a specific OSD overlay slot."""
    send_overlay_command({
        "command": {"name": "osd-overlay", "id": overlay_id,
                    "format": "none", "data": "",
                    "res_x": res_x, "res_y": res_y}
    })


def close_overlay_sock():
    """Close the persistent overlay socket on shutdown."""
    global _overlay_sock
    with _overlay_sock_lock:
        if _overlay_sock is not None:
            try:
                _overlay_sock.close()
            except OSError:
                pass
            _overlay_sock = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_filter_complex(vocal_vol, pitch):
    """Rebuild the lavfi-complex string with per-track rubberband settings."""
    return (
        # Vocal stem: volume + pitch shift (formant-preserved for voice)
        f'[aid2]volume@vocalvol={vocal_vol},'
        f'rubberband@vocalrb=pitch={pitch}'
        f':window=long'
        f':pitchq=quality'
        f':transients=crisp'
        f':detector=compound'
        f':formant=preserved'
        f':channels=together'
        f'[vocal];'
        # Non-vocal stem: fixed volume + pitch shift (formant-shifted for instruments)
        f'[aid3]volume@nonvocalvol={NONVOCAL_VOL},'
        f'rubberband@nonvocalrb=pitch={pitch}'
        f':window=standard'
        f':pitchq=quality'
        f':transients=crisp'
        f':detector=compound'
        f':formant=shifted'
        f'[nonvocal];'
        f'[vocal][nonvocal]amix=inputs=2:normalize=0[ao]'
    )


def send_mpv_command(cmd_dict):
    """Send a JSON command to the running MPV instance via IPC socket."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(IPC_SOCKET)
            s.sendall(json.dumps(cmd_dict).encode() + b'\n')
            # read response (best-effort)
            s.settimeout(0.5)
            try:
                s.recv(4096)
            except socket.timeout:
                pass
    except (ConnectionRefusedError, OSError):
        pass


def send_mpv_query(cmd_dict):
    """Send a JSON command to MPV and return the parsed response (or None on failure)."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(IPC_SOCKET)
            s.sendall(json.dumps(cmd_dict).encode() + b'\n')
            s.settimeout(1.0)
            resp = s.recv(4096).decode()
            return json.loads(resp)
    except (ConnectionRefusedError, OSError, json.JSONDecodeError, socket.timeout):
        return None


def load_placeholder():
    """Load the placeholder PNG into the running MPV instance."""
    send_mpv_command({"command": ["loadfile", PLACEHOLDER_PNG]})


def end_song():
    """Teardown after a song ends: clear filter, show placeholder, reset state."""
    send_mpv_command({"command": ["set_property", "lavfi-complex", ""]})
    load_placeholder()
    reset_state_defaults()
    clear_osd(OSD_NOWPLAYING)
    clear_osd(OSD_TIMECODE)
    # poll_position keeps running in idle state to handle resize events;
    # it will also call send_qr_overlay/send_url_overlay/send_upnext_overlay
    # on the next cycle via the resize-detection branch (last_osd_w/h reset on load).


def reset_state_defaults():
    """Reset all playback state to defaults."""
    state["playing"] = False
    state["position"] = 0.0
    state["duration"] = 0.0
    state["vocal_volume"] = 1.0
    state["semitones"] = 0
    state["subtitle_delay"] = 0.0


def start_mpv():
    """Start a persistent MPV instance in idle mode. No files loaded yet."""
    global mpv_proc
    cmd = [
        "mpv",
        "--idle",
        "--force-window",
        "--image-display-duration=inf",
        f"--input-ipc-server={IPC_SOCKET}",
        "--no-terminal",
        "--osd-margin-x=0",
        "--osd-margin-y=0",
    ]
    mpv_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # give MPV a moment to create the socket
    time.sleep(0.5)
    load_placeholder()
    ensure_qr_png()
    # Start the poll thread immediately so it catches the window settling to its
    # final size and resends overlays via the resize-detection branch.
    global poll_thread
    poll_stop.clear()
    poll_thread = threading.Thread(target=poll_position, daemon=True)
    poll_thread.start()


def quit_mpv():
    """Tell MPV to quit and clean up the socket file."""
    global mpv_proc
    close_overlay_sock()
    if mpv_proc is not None:
        send_mpv_command({"command": ["quit"]})
        try:
            mpv_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            mpv_proc.kill()
        mpv_proc = None
    # remove stale socket if it exists
    if os.path.exists(IPC_SOCKET):
        try:
            os.unlink(IPC_SOCKET)
        except OSError:
            pass


def get_server_url():
    """Discover the local IP address and return the server URL."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "localhost"
    return f"http://{ip}:5000"


def get_filename_prefix(path):
    """Strip the filename to the part before '---' (e.g. 'selfish---vocal.m4a' -> 'selfish')."""
    if not path:
        return ""
    return Path(path).stem.split("---")[0]


def ensure_qr_png():
    """Generate qrcode.png with the current server URL."""
    url = get_server_url()
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(QR_CODE)


def _overlay_font_size(screen_h=None):
    """Calculate the font size used for QR URL text and Now Playing title line.
    Based on QR overlay sizing: max(120, screen_h // 6) for QR height,
    then font = qr_h // 4.5."""
    if screen_h is None:
        screen_h = 1080
    qr_h = max(120, screen_h // 6)
    return qr_h // 3


def send_qr_overlay():
    """Send the QR code bitmap as a persistent overlay via overlay-add (no URL text)."""
    if not SHOW_QR:
        return
    # Query OSD height for sizing
    resp = send_mpv_query({"command": ["get_property", "osd-height"]})
    screen_h = resp.get("data") if resp and resp.get("data") is not None else 1080
    qr_h = max(120, screen_h // 6)

    # Open QR code image and resize
    qr_img = Image.open(QR_CODE).convert("RGBA").resize((qr_h, qr_h))

    # Convert RGBA -> BGRA (swap R and B channels) for mpv overlay format
    r, g, b, a = qr_img.split()
    bgra = Image.merge("RGBA", (b, g, r, a))
    bgra_bytes = bgra.tobytes()

    # Write raw BGRA bytes to temp file
    overlay_path = "/tmp/qr_overlay.bgra"
    with open(overlay_path, "wb") as f:
        f.write(bgra_bytes)

    # Send overlay-add command (fmt="bgra" is required between offset and w)
    send_overlay_command({
        "command": ["overlay-add", 0, 0, 0, overlay_path, 0, "bgra", qr_h, qr_h, qr_h * 4]
    })


def send_url_overlay():
    """Send the server URL text as an OSD overlay, positioned to the right of the QR image."""
    if not SHOW_QR:
        clear_osd(OSD_URL)
        return
    # Query screen height for sizing (same as QR uses)
    resp = send_mpv_query({"command": ["get_property", "osd-height"]})
    screen_h = resp.get("data") if resp and resp.get("data") is not None else 1080
    resp_w = send_mpv_query({"command": ["get_property", "osd-width"]})
    screen_w = resp_w.get("data") if resp_w and resp_w.get("data") is not None else 1920

    qr_h = max(120, screen_h // 6)
    font_size = _overlay_font_size(screen_h)

    # Try DejaVuSans font, fall back to default
    font = ImageFont.load_default(size=font_size)
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/TTF/DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            pass

    url_text = get_server_url()
    # Position to the right of QR (with 5px gap), vertically aligned to top.
    # Convert actual pixel coordinates to ASS 1920x1080 space.
    x = (qr_h + 10) * 1920 / screen_w
    y = OVERLAY_MARGIN_TOP
    # \an7 = top-left anchor (same top-edge semantics as \an9 for Now Playing)
    data = f"{{\\an7\\pos({x},{y})\\fs{font_size}{OVERLAY_STYLE}\\c{URL_COLOR}}}{url_text}"
    send_osd(OSD_URL, data)


def _fmt_time(seconds):
    """Format seconds as m:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _osd_screen_h():
    """Query MPV osd-height, defaulting to 1080 on failure."""
    resp = send_mpv_query({"command": ["get_property", "osd-height"]})
    return resp.get("data") if resp and resp.get("data") is not None else 1080


def send_nowplaying_overlay():
    """Send 'Now Playing: <title>' line via osd-overlay (OSD_NOWPLAYING, ID 2).
    Displays the title only; clears if not playing."""
    if not state["playing"]:
        clear_osd(OSD_NOWPLAYING)
        return
    name = get_filename_prefix(state.get("playing_video_path", ""))
    fs = _overlay_font_size(_osd_screen_h())
    data = f"{{\\an9\\pos(1920,{OVERLAY_MARGIN_TOP})\\fs{fs}{OVERLAY_STYLE}\\c{NOWPLAYING_COLOR}}}Now Playing: {name}"
    send_osd(OSD_NOWPLAYING, data)


def send_timecode_overlay():
    """Send elapsed/total time + transpose + vocals via osd-overlay (OSD_TIMECODE, ID 3).
    Clears when not playing."""
    if not state["playing"]:
        clear_osd(OSD_TIMECODE)
        return
    fs = _overlay_font_size(_osd_screen_h())
    # Stack below Now Playing
    y = OVERLAY_MARGIN_TOP + int(fs * 0.9)

    elapsed = _fmt_time(state["position"])
    total = _fmt_time(state["duration"])
    st = state["semitones"]
    st_str = f"+{st}st" if st > 0 else f"{st}st"
    vol_pct = int(state["vocal_volume"] * 100)

    data = f"{{\\an9\\pos(1920,{y})\\fs{fs-15}{OVERLAY_STYLE}\\c{TIMECODE_COLOR}}}{elapsed} / {total} | Pitch: {st_str} | Vocals: {vol_pct}%"
    send_osd(OSD_TIMECODE, data)


def send_upnext_overlay():
    """Send 'Up Next: <title>' via osd-overlay (OSD_UPNEXT, ID 4).
    Displays when next path differs from playing path; clears otherwise."""
    next_path = state["video_path"]
    show_up_next = next_path and next_path != state.get("playing_video_path")
    if not show_up_next:
        clear_osd(OSD_UPNEXT)
        return
    next_name = get_filename_prefix(next_path)
    fs = _overlay_font_size(_osd_screen_h())
    # Stack below Now Playing + Timecode lines
    y = OVERLAY_MARGIN_TOP + int(fs * 0.9) + int((fs - 10) * 1.05)
    data = f"{{\\an9\\pos(1920,{y})\\fs{fs-10}{OVERLAY_STYLE}\\c{UPNEXT_COLOR}}}Up Next: {next_name}"
    send_osd(OSD_UPNEXT, data)


def poll_position():
    """Periodically query MPV for time-pos and duration. Checks idle-active for song-end detection.
    Also monitors osd-width/osd-height changes and re-positions overlays on resize."""
    last_osd_w = None
    last_osd_h = None
    while not poll_stop.is_set():
        # Check if MPV went idle (song ended naturally) — only act if we were playing
        idle_resp = send_mpv_query({"command": ["get_property", "idle-active"]})
        if idle_resp and idle_resp.get("data") is True:
            if state["playing"]:
                end_song()
            # Always reset OSD tracking so resize detection fires after file change
            last_osd_w = None
            last_osd_h = None

        resp = send_mpv_query({"command": ["get_property", "time-pos"]})
        if resp and resp.get("data") is not None:
            state["position"] = float(resp["data"])

        resp = send_mpv_query({"command": ["get_property", "duration"]})
        if resp and resp.get("data") is not None:
            state["duration"] = float(resp["data"])

        # Update elapsed time overlay every cycle
        if state["playing"]:
            send_timecode_overlay()

        # Detect window resize via OSD dimension change
        resp = send_mpv_query({"command": ["get_property", "osd-width"]})
        cur_w = resp.get("data") if resp and resp.get("data") is not None else None
        resp = send_mpv_query({"command": ["get_property", "osd-height"]})
        cur_h = resp.get("data") if resp and resp.get("data") is not None else None

        if cur_w is not None and cur_h is not None:
            if cur_w != last_osd_w or cur_h != last_osd_h:
                last_osd_w, last_osd_h = cur_w, cur_h
                send_qr_overlay()
                send_url_overlay()
                if state["playing"]:
                    send_nowplaying_overlay()
                    send_timecode_overlay()
                send_upnext_overlay()

        poll_stop.wait(0.5)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", state=state)


@app.route("/api/files", methods=["POST"])
def set_files():
    """Receive file paths from the UI."""
    data = request.json
    state["video_path"] = data.get("video")
    state["vocal_path"] = data.get("vocal")
    state["nonvocal_path"] = data.get("nonvocal")
    state["subtitle_path"] = data.get("subtitle")
    send_upnext_overlay()
    return jsonify({"ok": True})


@app.route("/api/play", methods=["POST"])
def play():
    if not all([state["video_path"], state["vocal_path"], state["nonvocal_path"]]):
        return jsonify({"ok": False, "error": "Missing file selection"}), 400
    if state["playing"]:
        return jsonify({"ok": True})  # already playing

    reset_state_defaults()

    # Load video
    send_mpv_command({"command": ["loadfile", state["video_path"]]})

    # Wait for duration > 0 (max 5s)
    for _ in range(25):
        resp = send_mpv_query({"command": ["get_property", "duration"]})
        if resp and resp.get("data") is not None and float(resp["data"]) > 0:
            break
        time.sleep(0.2)

    # Add audio tracks
    send_mpv_command({"command": ["audio-add", state["vocal_path"]]})
    send_mpv_command({"command": ["audio-add", state["nonvocal_path"]]})
    time.sleep(0.2)

    # Set filter complex with default vol=1.0 and pitch=1.0
    fc = build_filter_complex(1.0, semitones_to_pitch(0))
    send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})

    # Add subtitles if available
    if state["subtitle_path"]:
        send_mpv_command({"command": ["sub-add", state["subtitle_path"]]})

    state["playing"] = True
    state["playing_video_path"] = state["video_path"]
    send_nowplaying_overlay()
    send_timecode_overlay()
    send_upnext_overlay()
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def pause():
    if not state["playing"]:
        return jsonify({"ok": True})
    send_mpv_command({"command": ["cycle", "pause"]})
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop():
    send_mpv_command({"command": ["stop"]})
    time.sleep(0.1)   # let MPV process the stop before loading placeholder
    end_song()
    return jsonify({"ok": True})


@app.route("/api/exit", methods=["POST"])
def exit_app():
    """Stop playback and shut down the Flask server."""
    quit_mpv()
    poll_stop.set()
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    else:
        # Fallback for Werkzeug >= 2.1 (shutdown removed)
        os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"ok": True})


@app.route("/api/seek", methods=["POST"])
def seek():
    pos = request.json.get("position", 0)
    send_mpv_command({"command": ["seek", float(pos), "absolute"]})
    state["position"] = float(pos)
    return jsonify({"ok": True})


@app.route("/api/volume", methods=["POST"])
def set_volume():
    vol_pct = request.json.get("volume", 100)
    state["vocal_volume"] = float(vol_pct) / 100.0
    if state["playing"]:
        fc = build_filter_complex(state["vocal_volume"], semitones_to_pitch(state["semitones"]))
        send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})
    send_timecode_overlay()
    return jsonify({"ok": True})


@app.route("/api/pitch", methods=["POST"])
def set_pitch():
    st = request.json.get("semitones", 0)
    state["semitones"] = int(st)
    if state["playing"]:
        fc = build_filter_complex(state["vocal_volume"], semitones_to_pitch(state["semitones"]))
        send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})
    send_timecode_overlay()
    return jsonify({"ok": True})


@app.route("/api/sub_delay", methods=["POST"])
def set_sub_delay():
    """Set subtitle delay in seconds (positive = delay, negative = advance)."""
    delay = request.json.get("delay", 0)
    state["subtitle_delay"] = float(delay)
    if state["playing"]:
        send_mpv_command({"command": ["set_property", "sub-delay", float(delay)]})
    return jsonify({"ok": True})


# ── Suppress /api/status from Werkzeug request log ────────────────────────────
class _StatusFilter(logging.Filter):
    def filter(self, record):
        return '/api/status' not in record.getMessage()

logging.getLogger('werkzeug').addFilter(_StatusFilter())


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "playing": state["playing"],
        "position": state["position"],
        "duration": state["duration"],
        "vocal_volume": state["vocal_volume"],
        "semitones": state["semitones"],
        "subtitle_delay": state["subtitle_delay"],
    })


@app.route("/api/browse", methods=["GET"])
def browse():
    """Server-side file browser. Returns directory listing for the given path."""
    dir_path = request.args.get("path", os.path.expanduser("~"))
    dir_path = os.path.abspath(dir_path)

    if not os.path.isdir(dir_path):
        return jsonify({"error": f"Not a directory: {dir_path}"}), 400

    entries = []
    try:
        for name in sorted(os.listdir(dir_path), key=lambda n: (not os.path.isdir(os.path.join(dir_path, n)), n.lower())):
            full = os.path.join(dir_path, name)
            is_dir = os.path.isdir(full)
            entries.append({
                "name": name,
                "is_dir": is_dir,
                "path": full,
            })
    except PermissionError:
        return jsonify({"error": f"Permission denied: {dir_path}"}), 403

    parent = os.path.dirname(dir_path) if dir_path != "/" else None

    return jsonify({
        "current": dir_path,
        "parent": parent,
        "entries": entries,
    })


@app.route("/api/derive_files", methods=["GET"])
def derive_files():
    video = request.args.get("video", "")
    if not video:
        return jsonify({"error": "no video path"}), 400
    paths = derive_companion_paths(video)
    return jsonify({
        k: {"path": v, "exists": os.path.exists(v)}
        for k, v in paths.items()
    })


if __name__ == "__main__":
    _debug = True
    # With debug=True the Werkzeug reloader spawns a parent watcher + child worker.
    # Only start MPV in the child (WERKZEUG_RUN_MAIN=true) to avoid two windows.
    # With debug=False there is only one process, so always start.
    if not _debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_mpv()
    try:
        app.run(host="0.0.0.0", port=5000, debug=_debug)
    finally:
        quit_mpv()
