import os
import json
import signal
import subprocess
import socket
import threading
import time
from pathlib import Path
import qrcode
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── State ──────────────────────────────────────────────────────────────────────
IPC_SOCKET = "/tmp/mpv-socket"
NONVOCAL_VOL = 1.0  # fixed
QR_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrcode.png")
PLACEHOLDER_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placeholder.png")
LOGO_JPG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.jpg")

SONG_DIR = "/home/ken/whisper2srt/song"
DEFAULT_VIDEO    = os.path.join(SONG_DIR, "selfish.mp4")
DEFAULT_VOCAL    = os.path.join(SONG_DIR, "selfish---vocal.m4a")
DEFAULT_NONVOCAL = os.path.join(SONG_DIR, "selfish---nonvocal.m4a")
DEFAULT_SUBTITLE = os.path.join(SONG_DIR, "selfish---vocal.ass")

def semitones_to_pitch(st):
    """Convert semitone offset to MPV rubberband pitch multiplier."""
    return 2 ** (st / 12)

state = {
    "video_path": DEFAULT_VIDEO,
    "vocal_path": DEFAULT_VOCAL,
    "nonvocal_path": DEFAULT_NONVOCAL,
    "subtitle_path": DEFAULT_SUBTITLE,
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


def generate_placeholder():
    """Create a placeholder PNG with logo.jpg centered on a black background."""
    logo = Image.open(LOGO_JPG)
    # Use 1280x720 canvas, scale logo to fit within it with padding
    canvas_w, canvas_h = 1280, 720
    # Scale logo to fit within 80% of canvas, preserving aspect ratio
    max_w, max_h = int(canvas_w * 0.8), int(canvas_h * 0.8)
    ratio = min(max_w / logo.width, max_h / logo.height)
    new_w, new_h = int(logo.width * ratio), int(logo.height * ratio)
    logo = logo.resize((new_w, new_h), Image.LANCZOS)

    img = Image.new("RGB", (canvas_w, canvas_h), "black")
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    img.paste(logo, (x, y))
    img.save(PLACEHOLDER_PNG)


def load_placeholder():
    """Load the placeholder PNG into the running MPV instance."""
    send_mpv_command({"command": ["loadfile", PLACEHOLDER_PNG]})


def end_song():
    """Teardown after a song ends: clear filter, show placeholder, reset state."""
    send_mpv_command({"command": ["set_property", "lavfi-complex", ""]})
    load_placeholder()
    reset_state_defaults()
    send_qr_overlay()
    send_nowplaying_overlay()
    poll_stop.set()


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
    ]
    mpv_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # give MPV a moment to create the socket
    time.sleep(0.5)
    load_placeholder()
    ensure_qr_png()
    send_qr_overlay()
    send_nowplaying_overlay()


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


def send_qr_overlay():
    """Send the QR code + URL text as a persistent overlay via overlay-add."""
    # Query OSD height for sizing
    resp = send_mpv_query({"command": ["get_property", "osd-height"]})
    screen_h = resp.get("data") if resp and resp.get("data") is not None else 1080
    qr_h = max(120, screen_h // 6)

    # Open QR code image and resize
    qr_img = Image.open(QR_CODE).convert("RGBA").resize((qr_h, qr_h))

    # Render URL text beside it
    url_text = get_server_url()
    font = ImageFont.load_default(size=qr_h // 4.5)
    # Try DejaVuSans fallback
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/TTF/DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(font_path, qr_h // 4.5)
            break
        except (OSError, IOError):
            pass

    draw = ImageDraw.Draw(qr_img)
    bbox = draw.textbbox((0, 0), url_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Compose combined RGBA image: QR on left, transparent background, URL text top-aligned to QR
    gap = qr_h // 8
    combined_w = qr_h + gap + text_w
    combined_h = qr_h
    combined = Image.new("RGBA", (combined_w, combined_h), (0, 0, 0, 0))
    combined.paste(qr_img, (0, 0))

    # Draw text top-aligned with the QR code
    text_draw = ImageDraw.Draw(combined)
    text_draw.text((qr_h + gap, 0), url_text, fill=(255, 255, 255, 255), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0, 255))

    # Convert RGBA -> BGRA (swap R and B channels) for mpv overlay format
    r, g, b, a = combined.split()
    bgra = Image.merge("RGBA", (b, g, r, a))
    bgra_bytes = bgra.tobytes()

    # Write raw BGRA bytes to temp file
    overlay_path = "/tmp/qr_overlay.bgra"
    with open(overlay_path, "wb") as f:
        f.write(bgra_bytes)

    # Send overlay-add command (fmt="bgra" is required between offset and w)
    send_overlay_command({
        "command": ["overlay-add", 0, 0, 0, overlay_path, 0, "bgra", combined_w, combined_h, combined_w * 4]
    })


def _fmt_time(seconds):
    """Format seconds as m:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def send_nowplaying_overlay():
    """Send Now Playing / Up Next display via osd-overlay (ASS events, ID 1).
    Uses command-as-object format: {"command": {"name": "osd-overlay", ...}}"""
    next_path = state["video_path"]
    show_up_next = next_path and next_path != state.get("playing_video_path")
    next_name = get_filename_prefix(next_path) if show_up_next else ""

    if not state["playing"]:
        if show_up_next:
            data = f"{{\\an9\\fs24\\bord2\\c&HFFFFFF&}}Up Next: {next_name}"
            send_overlay_command({
                "command": {"name": "osd-overlay", "id": 1, "format": "ass-events",
                            "data": data, "res_x": 1920, "res_y": 1080}
            })
        else:
            send_overlay_command({
                "command": {"name": "osd-overlay", "id": 1, "format": "none",
                            "data": "", "res_x": 1920, "res_y": 1080}
            })
        return

    name = get_filename_prefix(state.get("playing_video_path", ""))
    elapsed = _fmt_time(state["position"])
    total = _fmt_time(state["duration"])
    st = state["semitones"]
    st_str = f"+{st}st" if st > 0 else f"{st}st"
    vol_pct = int(state["vocal_volume"] * 100)

    lines = [
        f"{{\\an9\\fs30\\bord2\\c&HFFFFFF&}}Now Playing: {name}",
        f"{{\\an9\\fs22\\bord1\\c&HCCCCCC&}}{elapsed} / {total}  |  Transpose: {st_str}  |  Vocals: {vol_pct}%",
        #f"{{\\an9\\fs22\\bord1\\c&HCCCCCC&}}Transpose: {st_str}  |  Vocals: {vol_pct}%",
    ]
    if show_up_next:
        lines.append(f"{{\\an9\\fs24\\bord2\\c&HFFFFFF&}}Up Next: {next_name}")

    data = "\n".join(lines)
    send_overlay_command({
        "command": {"name": "osd-overlay", "id": 1, "format": "ass-events",
                    "data": data, "res_x": 1920, "res_y": 1080}
    })


def poll_position():
    """Periodically query MPV for time-pos and duration. Checks idle-active for song-end detection.
    Also monitors osd-width/osd-height changes and re-positions overlays on resize."""
    last_osd_w = None
    last_osd_h = None
    while not poll_stop.is_set():
        # Check if MPV went idle (song ended naturally)
        idle_resp = send_mpv_query({"command": ["get_property", "idle-active"]})
        if idle_resp and idle_resp.get("data") is True:
            end_song()
            break

        resp = send_mpv_query({"command": ["get_property", "time-pos"]})
        if resp and resp.get("data") is not None:
            state["position"] = float(resp["data"])

        resp = send_mpv_query({"command": ["get_property", "duration"]})
        if resp and resp.get("data") is not None:
            state["duration"] = float(resp["data"])

        # Update elapsed time overlay every cycle
        if state["playing"]:
            send_nowplaying_overlay()

        # Detect window resize via OSD dimension change
        resp = send_mpv_query({"command": ["get_property", "osd-width"]})
        cur_w = resp.get("data") if resp and resp.get("data") is not None else None
        resp = send_mpv_query({"command": ["get_property", "osd-height"]})
        cur_h = resp.get("data") if resp and resp.get("data") is not None else None

        if cur_w is not None and cur_h is not None:
            if cur_w != last_osd_w or cur_h != last_osd_h:
                last_osd_w, last_osd_h = cur_w, cur_h
                if state["playing"]:
                    send_qr_overlay()
                    send_nowplaying_overlay()

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
    send_nowplaying_overlay()
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
    poll_stop.clear()
    global poll_thread
    if poll_thread is not None:
        poll_thread.join(timeout=1)
    poll_thread = threading.Thread(target=poll_position, daemon=True)
    poll_thread.start()
    send_qr_overlay()
    send_nowplaying_overlay()
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
    send_nowplaying_overlay()
    return jsonify({"ok": True})


@app.route("/api/pitch", methods=["POST"])
def set_pitch():
    st = request.json.get("semitones", 0)
    state["semitones"] = int(st)
    if state["playing"]:
        fc = build_filter_complex(state["vocal_volume"], semitones_to_pitch(state["semitones"]))
        send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})
    send_nowplaying_overlay()
    return jsonify({"ok": True})


@app.route("/api/sub_delay", methods=["POST"])
def set_sub_delay():
    """Set subtitle delay in seconds (positive = delay, negative = advance)."""
    delay = request.json.get("delay", 0)
    state["subtitle_delay"] = float(delay)
    if state["playing"]:
        send_mpv_command({"command": ["set_property", "sub-delay", float(delay)]})
    return jsonify({"ok": True})


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


if __name__ == "__main__":
    generate_placeholder()
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
