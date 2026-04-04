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
NOWPLAYING_OVERLAY_PATH = "/tmp/nowplaying_overlay.bgra"

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
                print(f"[OVERLAY TX] {payload[:200]}")
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
                        print(f"[OVERLAY RX] {line}")
                        try:
                            msg = json.loads(line)
                            if "error" in msg or "request_id" in msg:
                                return  # got the command response
                        except json.JSONDecodeError:
                            pass
                print(f"[OVERLAY RX] no command response found, buf={buf[:200]}")
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
    """Create a 1920x1080 black PNG with 'PROTOTYPE' text and the localhost URL."""
    img = Image.new("RGB", (1280, 720), "black")
    draw = ImageDraw.Draw(img)
    font_large = ImageFont.load_default(size=80)
    font_small = ImageFont.load_default(size=36)

    text_main = "PROTOTYPE"
    text_sub = "http://localhost:5000"

    bbox_main = draw.textbbox((0, 0), text_main, font=font_large)
    text_w = bbox_main[2] - bbox_main[0]
    draw.text(((1280 - text_w) / 2, 400), text_main, fill="white", font=font_large)

    bbox_sub = draw.textbbox((0, 0), text_sub, font=font_small)
    text_w_sub = bbox_sub[2] - bbox_sub[0]
    draw.text(((1280 - text_w_sub) / 2, 520), text_sub, fill="white", font=font_small)

    img.save(PLACEHOLDER_PNG)


def load_placeholder():
    """Load the placeholder PNG into the running MPV instance."""
    send_mpv_command({"command": ["loadfile", PLACEHOLDER_PNG]})


def end_song():
    """Teardown after a song ends: clear filter, show placeholder, reset state."""
    send_mpv_command({"command": ["set_property", "lavfi-complex", ""]})
    load_placeholder()
    reset_state_defaults()
    hide_qr_overlay()
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


def hide_qr_overlay():
    """Remove the QR overlay."""
    send_overlay_command({"command": ["overlay-remove", 0]})


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


def send_nowplaying_overlay():
    """Render Now Playing / Up Next as a bitmap overlay (ID 1, upper-right)."""
    if not state["playing"]:
        send_overlay_command({"command": ["overlay-remove", 1]})
        return

    # Gather text
    name = get_filename_prefix(state.get("playing_video_path", ""))
    st = state["semitones"]
    st_str = f"+{st}st" if st > 0 else f"{st}st"
    vol_pct = int(state["vocal_volume"] * 100)

    next_path = state["video_path"]
    show_up_next = next_path and next_path != state.get("playing_video_path")
    next_name = get_filename_prefix(next_path) if show_up_next else ""

    lines = [
        (f"Now Playing: {name}", 30, (255, 255, 255, 255)),
        (f"Transpose: {st_str}  |  Vocals: {vol_pct}%", 22, (204, 204, 204, 255)),
    ]
    if show_up_next:
        lines.append((f"Up Next: {next_name}", 24, (255, 255, 255, 255)))

    # Query screen dimensions for positioning
    resp = send_mpv_query({"command": ["get_property", "osd-width"]})
    screen_w = resp.get("data") if resp and resp.get("data") else 1920
    resp = send_mpv_query({"command": ["get_property", "osd-height"]})
    screen_h = resp.get("data") if resp and resp.get("data") else 1080

    # Font cache
    font_cache = {}
    def get_font(size):
        if size not in font_cache:
            for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                              "/usr/share/fonts/TTF/DejaVuSans.ttf"]:
                try:
                    font_cache[size] = ImageFont.truetype(font_path, size)
                    break
                except (OSError, IOError):
                    pass
            else:
                font_cache[size] = ImageFont.load_default(size=size)
        return font_cache[size]

    # Measure all lines
    margin = 0
    line_spacing = 6
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)

    line_metrics = []
    max_w = 0
    total_h = 0
    for text, size, color in lines:
        font = get_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_metrics.append((text, font, color, w, h))
        max_w = max(max_w, w)
        total_h += h + line_spacing
    total_h -= line_spacing

    img_w = max_w + margin * 2
    img_h = total_h + margin * 2

    # Render text with black outline for readability
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(img)
    y = margin
    for text, font, color, w, h in line_metrics:
        x = img_w - margin - w
        # Draw black outline
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                text_draw.text((x + dx, y + dy), text, fill=(0, 0, 0, 255), font=font)
        # Draw main text
        text_draw.text((x, y), text, fill=color, font=font)
        y += h + line_spacing

    # Convert RGBA -> BGRA
    r, g, b, a = img.split()
    bgra = Image.merge("RGBA", (b, g, r, a))

    with open(NOWPLAYING_OVERLAY_PATH, "wb") as f:
        f.write(bgra.tobytes())

    # Position: upper-right corner
    overlay_x = screen_w - img_w - 0
    overlay_y = 0

    send_overlay_command({
        "command": ["overlay-add", 1, overlay_x, overlay_y,
                    NOWPLAYING_OVERLAY_PATH, 0, "bgra", img_w, img_h, img_w * 4]
    })


def poll_position():
    """Periodically query MPV for time-pos and duration. Checks idle-active for song-end detection."""
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
