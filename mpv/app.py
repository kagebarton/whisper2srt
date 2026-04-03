import os
import json
import signal
import subprocess
import socket
import threading
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── State ──────────────────────────────────────────────────────────────────────
IPC_SOCKET = "/tmp/mpv-socket"
NONVOCAL_VOL = 1.0  # fixed
QR_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrcode.png")
PLACEHOLDER_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placeholder.png")

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
    "semitones": 0,            # -6 to +6, mapped to pitch via 2^(st/12)
    "playing": False,
    "duration": 0.0,
    "position": 0.0,
}

mpv_proc = None
poll_thread = None
poll_stop = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_filter_complex(vocal_vol, pitch):
    """Rebuild the lavfi-complex string with per-track rubberband settings and QR overlay."""
    return (
        # QR code overlay: loop the single-frame PNG and overlay on video
        f'movie={QR_CODE},loop=loop=-1:size=1[qr];'
        f'[vid1][qr]overlay=0:0[vo];'
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
    draw.text(((1920 - text_w) / 2, 400), text_main, fill="white", font=font_large)

    bbox_sub = draw.textbbox((0, 0), text_sub, font=font_small)
    text_w_sub = bbox_sub[2] - bbox_sub[0]
    draw.text(((1920 - text_w_sub) / 2, 520), text_sub, fill="white", font=font_small)

    img.save(PLACEHOLDER_PNG)


def load_placeholder():
    """Load the placeholder PNG into the running MPV instance."""
    send_mpv_command({"command": ["loadfile", PLACEHOLDER_PNG]})


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
        f"--input-ipc-server={IPC_SOCKET}",
        "--no-terminal",
    ]
    mpv_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # give MPV a moment to create the socket
    time.sleep(0.5)
    load_placeholder()


def quit_mpv():
    """Tell MPV to quit and clean up the socket file."""
    global mpv_proc
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


def poll_position():
    """Periodically query MPV for time-pos and duration. Checks idle-active for song-end detection."""
    while not poll_stop.is_set():
        # Check if MPV went idle (song ended naturally)
        idle_resp = send_mpv_query({"command": ["get_property", "idle-active"]})
        if idle_resp and idle_resp.get("data") is True:
            reset_state_defaults()
            load_placeholder()
            poll_stop.set()
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
    poll_stop.clear()
    global poll_thread
    if poll_thread is not None:
        poll_thread.join(timeout=1)
    poll_thread = threading.Thread(target=poll_position, daemon=True)
    poll_thread.start()
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
    time.sleep(0.1)
    load_placeholder()
    reset_state_defaults()
    poll_stop.set()
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
    vol = request.json.get("volume", 1.0)
    state["vocal_volume"] = float(vol)
    if state["playing"]:
        fc = build_filter_complex(state["vocal_volume"], semitones_to_pitch(state["semitones"]))
        send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})
    return jsonify({"ok": True})


@app.route("/api/pitch", methods=["POST"])
def set_pitch():
    st = request.json.get("semitones", 0)
    state["semitones"] = int(st)
    if state["playing"]:
        fc = build_filter_complex(state["vocal_volume"], semitones_to_pitch(state["semitones"]))
        send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})
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
