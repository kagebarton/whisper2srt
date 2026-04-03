import os
import json
import signal
import subprocess
import socket
import threading
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── State ──────────────────────────────────────────────────────────────────────
IPC_SOCKET = "/tmp/mpv-socket"
NONVOCAL_VOL = 1.0  # fixed
QR_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrcode.png")

def semitones_to_pitch(st):
    """Convert semitone offset to MPV rubberband pitch multiplier."""
    return 2 ** (st / 12)

state = {
    "video_path": None,
    "vocal_path": None,
    "nonvocal_path": None,
    "subtitle_path": None,
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


def launch_mpv():
    """Start MPV with the three tracks and the initial filter complex."""
    global mpv_proc
    cmd = [
        "mpv",
        state["video_path"],
        f'--audio-file={state["vocal_path"]}',
        f'--audio-file={state["nonvocal_path"]}',
        f'--lavfi-complex={build_filter_complex(state["vocal_volume"], semitones_to_pitch(state["semitones"]))}',
        f'--input-ipc-server={IPC_SOCKET}',
        "--no-terminal",
    ]
    if state["subtitle_path"]:
        cmd.insert(2, f'--sub-file={state["subtitle_path"]}')
    mpv_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # give MPV a moment to create the socket
    time.sleep(0.5)


def stop_mpv():
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
    """Periodically query MPV for time-pos and duration. Clears playing state if MPV exits."""
    while not poll_stop.is_set():
        # If MPV exited naturally (song ended), reset state
        if mpv_proc is not None and mpv_proc.poll() is not None:
            state["playing"] = False
            state["position"] = 0.0
            state["duration"] = 0.0
            poll_stop.set()
            break

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(IPC_SOCKET)
                s.sendall(json.dumps({
                    "command": ["get_property", "time-pos"]
                }).encode() + b'\n')
                s.settimeout(0.3)
                resp = s.recv(4096).decode()
            data = json.loads(resp)
            if data.get("data") is not None:
                state["position"] = float(data["data"])

            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(IPC_SOCKET)
                s.sendall(json.dumps({
                    "command": ["get_property", "duration"]
                }).encode() + b'\n')
                s.settimeout(0.3)
                resp = s.recv(4096).decode()
            data = json.loads(resp)
            if data.get("data") is not None:
                state["duration"] = float(data["data"])
        except Exception:
            pass
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

    launch_mpv()
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
    stop_mpv()
    state["playing"] = False
    state["position"] = 0.0
    state["duration"] = 0.0
    poll_stop.set()
    return jsonify({"ok": True})


@app.route("/api/exit", methods=["POST"])
def exit_app():
    """Stop MPV and shut down the Flask server."""
    stop_mpv()
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
    app.run(host="0.0.0.0", port=5000, debug=True)
