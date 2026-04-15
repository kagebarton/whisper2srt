import os
import re
import signal
import subprocess
import socket
import threading
import time
import logging
from datetime import datetime
from pathlib import Path
import qrcode
from PIL import Image, ImageFont
from flask import Flask, render_template, request, jsonify
import zmq

from mpv_controller import MpvController

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SHOW_QR = True   # Show QR code + URL overlay pair (top-left)
SHOW_CLOCK = True  # Show clock overlay (bottom-left)
OVERLAY_MARGIN_TOP = 0   # px from top edge in ASS 1920x1080 space
OVERLAY_MARGIN_BOTTOM = 0   # px from bottom edge in ASS 1920x1080 space

# Common ASS style tags applied to every overlay text (inside the {…} block).
# Examples: \bord2\shad1\fnDejaVu Sans\b1
# Leave empty ("") for no extra styling.
OVERLAY_STYLE = "\\bord3\\shad2\\3c&H000000&\\4c&H000000&\\4a&H80&"

# Overlay text colors (ASS hex BB GG RR format)
URL_COLOR        = "&HFFFFFF&"   # URL text (top-left)
NOWPLAYING_COLOR = "&H507FFF&"   # "Now Playing: <title>"
TIMECODE_COLOR   = "&HAAD5FF&"   # "1:23 / 3:45 | Transpose: +2 | Vocals: 80%"
UPNEXT_COLOR     = "&HB48246&"   # "Up Next: <title>"
CLOCK_COLOR      = "&HFFFFFF&"   # Clock (bottom-left)

# ── Volume normalization ───────────────────────────────────────────────────────
NORMALIZATION_ENABLED = False
NORMALIZATION_DB = -40.0

# ── SRT subtitle style ─────────────────────────────────────────────────────────
SRT_STYLE = {
    "sub-ass-override":  "no",
    "sub-font":          "Arial",
    "sub-font-size":     40,
    "sub-color":         "#FFD700",
    "sub-border-color":  "#000000",
    "sub-border-size":   3,
    "sub-shadow-offset": 2,
    "sub-shadow-color":  "#80000000",
    "sub-bold":          "no",
    "sub-margin-y":      100,
}

# ── OSD ID constants ───────────────────────────────────────────────────────────
OSD_URL        = 1   # URL text, paired with QR bitmap, top-left
OSD_NOWPLAYING = 2   # "Now Playing: <title>" line, top-right
OSD_TIMECODE   = 3   # "1:23 / 3:45 | Transpose: +2 | Vocals: 80%" line, top-right
OSD_UPNEXT     = 4   # "Up Next: <title>", top-right
OSD_CLOCK      = 5   # Clock, bottom-left

# ── State ──────────────────────────────────────────────────────────────────────
NONVOCAL_VOL = 1.0  # fixed
QR_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qrcode.png")
PLACEHOLDER_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placeholder.png")
LOGO_JPG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.jpg")

SONG_DIR = "/home/ken/whisper2srt/song"
DEFAULT_VIDEO    = os.path.join(SONG_DIR, "selfish.mp4")

# ── Per-song defaults (applied on every song load, reset after playback) ──────
DEFAULT_SRT_SUB_DELAY = -0.8
DEFAULT_VOCAL_VOLUME  = 0.4
DEFAULT_STARTING_PITCH = -1

def derive_companion_paths(video_path):
    """Given a video path, return derived companion file paths."""
    p = Path(video_path)
    stem = p.stem.split("---")[0]
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
    "subtitle_mode":      "off",
    "srt_delay":          DEFAULT_SRT_SUB_DELAY,
    "subtitle_available": {"ass": None, "srt": None},
    "vocal_volume": DEFAULT_VOCAL_VOLUME,
    "semitones": DEFAULT_STARTING_PITCH,
    "playing": False,
    "duration": 0.0,
    "position": 0.0,
    "playing_video_path": None,
    "dual_stem": False,
}

controller: MpvController | None = None

# ── Clock thread ───────────────────────────────────────────────────────────────
_clock_stop = threading.Event()

def _run_clock():
    """Send the clock overlay immediately, then every 30 s."""
    send_clock_overlay()
    while not _clock_stop.wait(30):
        send_clock_overlay()

def _start_clock_thread():
    _clock_stop.clear()
    t = threading.Thread(target=_run_clock, daemon=True)
    t.start()

def _stop_clock_thread():
    _clock_stop.set()


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_filter(vocal_vol, pitch, dual_stem):
    """Rebuild the lavfi-complex string with per-track rubberband settings."""
    norm_vol = f"{NORMALIZATION_DB}dB" if NORMALIZATION_ENABLED else None

    if dual_stem:
        # ZMQ bind allows live volume changes without filter rebuild (Phase 4
        # gate: once af_command is verified to reach lavfi-complex filters,
        # remove the azmq clause here and replace the ZMQ block in /api/volume
        # with: controller.af_command("vocalvol", "volume", str(vocal_volume)))
        zmq_bind = ",azmq=bind_address=tcp\\\\://127.0.0.1\\\\:5556"
        if norm_vol:
            amix_out = f'[vocal][nonvocal]amix=inputs=2:normalize=0[mixed];[mixed]volume={norm_vol}[ao]'
        else:
            amix_out = f'[vocal][nonvocal]amix=inputs=2:normalize=0[ao]'
        return (
            f'[aid2]volume@vocalvol={vocal_vol}{zmq_bind},'
            f'rubberband@vocalrb=pitch={pitch}'
            f':pitchq=quality'
            f':transients=crisp'
            f':detector=compound'
            f':phase=laminar'
            f':window=long'
            f':formant=preserved'
            f':channels=together'
            f':smoothing=off'
            f'[vocal];'
            f'[aid3]volume@nonvocalvol={NONVOCAL_VOL},'
            f'rubberband@nonvocalrb=pitch={pitch}'
            f':pitchq=quality'
            f':transients=crisp'
            f':detector=percussive'
            f':phase=laminar'
            f':window=short'
            f':formant=shifted'
            f':channels=apart'
            f':smoothing=off'
            f'[nonvocal];'
            f'{amix_out}'
        )
    else:
        if norm_vol:
            stem = f'[aid1]volume={norm_vol}[pre];[pre]rubberband@rb=pitch={pitch}'
        else:
            stem = f'[aid1]rubberband@rb=pitch={pitch}'
        rest = (
            f':pitchq=quality'
            f':transients=crisp'
            f':detector=compound'
            f':phase=laminar'
            f':window=long'
            f':formant=preserved'
            f':channels=together'
            f':smoothing=off'
            f'[ao]'
        )
        return stem + rest


def apply_srt_style():
    """Push SRT_STYLE sub-* properties to the running MPV instance."""
    controller.apply_srt_style(SRT_STYLE)


def load_placeholder():
    """Load the placeholder PNG into the running MPV instance."""
    controller.load_placeholder(PLACEHOLDER_PNG)


def end_song():
    """Teardown after a song ends: clear filter, show placeholder, reset state."""
    # Set playing=False FIRST so _on_idle_active re-entry during placeholder
    # load sees playing=False and bails immediately.
    state["playing"] = False
    controller.clear_lavfi_complex()
    load_placeholder()
    reset_state_defaults()
    controller.clear_osd(OSD_NOWPLAYING)
    controller.clear_osd(OSD_TIMECODE)


def reset_state_defaults():
    """Reset all playback state to defaults."""
    state["playing"] = False
    state["position"] = 0.0
    state["duration"] = 0.0
    state["vocal_volume"] = DEFAULT_VOCAL_VOLUME
    state["semitones"] = DEFAULT_STARTING_PITCH
    state["subtitle_mode"]      = "off"
    state["srt_delay"]          = DEFAULT_SRT_SUB_DELAY  # Invariant 1: reset per-song
    state["subtitle_available"] = {"ass": None, "srt": None}
    state["dual_stem"] = False


def _apply_subtitle_mode(mode, *, skip_remove=False):
    """Atomic: update state, remove current sub, add new, apply delay.
    Invariant: sub_delay is 0.0 for any mode != 'srt'.

    Caller must ensure mpv has a video loaded. Does NOT gate on
    state['playing'] — /api/play calls this while 'playing' is still False
    to keep _on_idle_active from firing on_song_end during sub_add.
    /api/subtitle gates the call externally when nothing is playing.

    Args:
        skip_remove: When True, skip the sub_remove() call.  Used on
            initial play where mpv may have auto-loaded a matching subtitle;
            calling sub-remove on that auto-loaded track while lavfi-complex
            is active triggers a libmpv segfault.  sub_add(..., "select")
            alone correctly replaces the active subtitle without the crash.
    """
    state["subtitle_mode"] = mode
    path = None
    if   mode == "karaoke": path = state["subtitle_available"]["ass"]
    elif mode == "srt":     path = state["subtitle_available"]["srt"]
    if not skip_remove:
        controller.sub_remove()
    if path:
        controller.sub_add(path, "select")
        if mode == "srt":
            controller.apply_srt_style(SRT_STYLE)
            controller.set_sub_delay(state["srt_delay"])
        else:
            controller.set_sub_delay(0.0)  # Invariant 2: no delay for ASS
    else:
        controller.set_sub_delay(0.0)


def start_mpv():
    """Start the libmpv instance and initial UI state."""
    global controller
    # Preserve the WERKZEUG_RUN_MAIN guard at the call site (__main__ block)
    # so the Werkzeug reloader only instantiates one MPV window per process.
    controller = MpvController(
        state=state,
        on_song_end=end_song,
        on_resize=refresh_overlays,
        on_tick=tick_overlays,
    )
    controller.start()
    load_placeholder()
    apply_srt_style()
    ensure_qr_png()
    # QR/URL/Up-Next overlays are sent once the window settles and the
    # osd-width/osd-height observers fire refresh_overlays().
    _start_clock_thread()


def quit_mpv():
    """Shut down the libmpv instance and clock thread."""
    _stop_clock_thread()
    if controller is not None:
        controller.quit()


def refresh_overlays():
    """Refresh all persistent overlays (called on window resize and file load)."""
    send_qr_overlay()
    send_url_overlay()
    if state["playing"]:
        send_nowplaying_overlay()
        send_timecode_overlay()
    send_upnext_overlay()


def tick_overlays():
    """Refresh playback-state overlays (called at ~2 Hz via time-pos observer)."""
    if state["playing"]:
        send_timecode_overlay()


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
    """Strip the filename to the part before '---'."""
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
    """Calculate font size for overlay text based on screen height."""
    if screen_h is None:
        screen_h = 1080
    qr_h = max(120, screen_h // 6)
    return qr_h // 3


def send_osd(overlay_id, data, res_x=1920, res_y=1080):
    """Send an ASS event string to a specific OSD overlay slot."""
    controller.osd_overlay(overlay_id, data, res_x, res_y)


def clear_osd(overlay_id, res_x=1920, res_y=1080):
    """Clear a specific OSD overlay slot."""
    controller.clear_osd(overlay_id, res_x, res_y)


def send_qr_overlay():
    """Send the QR code bitmap as a persistent overlay via overlay-add."""
    if not SHOW_QR:
        return
    _, screen_h = controller.osd_size
    qr_h = max(120, screen_h // 6)

    qr_img = Image.open(QR_CODE).convert("RGBA").resize((qr_h, qr_h))
    r, g, b, a = qr_img.split()
    bgra = Image.merge("RGBA", (b, g, r, a))
    bgra_bytes = bgra.tobytes()

    overlay_path = "/tmp/qr_overlay.bgra"
    with open(overlay_path, "wb") as f:
        f.write(bgra_bytes)

    controller.overlay_add(0, 0, 0, overlay_path, 0, "bgra", qr_h, qr_h, qr_h * 4)


def send_url_overlay():
    """Send the server URL text as an OSD overlay, to the right of the QR image."""
    if not SHOW_QR:
        clear_osd(OSD_URL)
        return
    screen_w, screen_h = controller.osd_size

    qr_h = max(120, screen_h // 6)
    font_size = _overlay_font_size(screen_h)

    font = ImageFont.load_default(size=font_size)
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/TTF/DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            pass

    url_text = get_server_url()
    x = (qr_h + 10) * 1920 / screen_w
    y = OVERLAY_MARGIN_TOP
    data = f"{{\\an7\\pos({x},{y})\\fs{font_size}{OVERLAY_STYLE}\\c{URL_COLOR}}}{url_text}"
    send_osd(OSD_URL, data)


def _fmt_time(seconds):
    """Format seconds as m:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _osd_screen_h():
    """Return the current OSD height, defaulting to 1080."""
    _, h = controller.osd_size
    return h


def send_nowplaying_overlay():
    """Send 'Now Playing: <title>' via osd-overlay. Clears if not playing."""
    if not state["playing"]:
        clear_osd(OSD_NOWPLAYING)
        return
    name = get_filename_prefix(state.get("playing_video_path", ""))
    fs = _overlay_font_size(_osd_screen_h())
    data = f"{{\\an9\\pos(1920,{OVERLAY_MARGIN_TOP})\\fs{fs}{OVERLAY_STYLE}\\c{NOWPLAYING_COLOR}}}Now Playing: {name}"
    send_osd(OSD_NOWPLAYING, data)


def send_timecode_overlay():
    """Send elapsed/total time + transpose + vocals. Clears when not playing."""
    if not state["playing"]:
        clear_osd(OSD_TIMECODE)
        return
    fs = _overlay_font_size(_osd_screen_h())
    y = OVERLAY_MARGIN_TOP + int(fs * 0.9)

    elapsed = _fmt_time(state["position"])
    total = _fmt_time(state["duration"])
    st = state["semitones"]
    st_str = f"+{st}st" if st > 0 else f"{st}st"
    vol_pct = int(state["vocal_volume"] * 100)

    data = f"{{\\an9\\pos(1920,{y})\\fs{fs-15}{OVERLAY_STYLE}\\c{TIMECODE_COLOR}}}{elapsed} / {total} | Pitch: {st_str} | Vocals: {vol_pct}%"
    send_osd(OSD_TIMECODE, data)


def send_upnext_overlay():
    """Send 'Up Next: <title>' when next path differs from playing. Clears otherwise."""
    next_path = state["video_path"]
    show_up_next = next_path and next_path != state.get("playing_video_path")
    if not show_up_next:
        clear_osd(OSD_UPNEXT)
        return
    next_name = get_filename_prefix(next_path)
    fs = _overlay_font_size(_osd_screen_h())
    y = OVERLAY_MARGIN_TOP + int(fs * 0.9) + int((fs - 10) * 1.05)
    data = f"{{\\an9\\pos(1920,{y})\\fs{fs-10}{OVERLAY_STYLE}\\c{UPNEXT_COLOR}}}Up Next: {next_name}"
    send_osd(OSD_UPNEXT, data)


def send_clock_overlay():
    """Send current time as an OSD overlay (bottom-left). Always shows when SHOW_CLOCK."""
    if not SHOW_CLOCK:
        clear_osd(OSD_CLOCK)
        return
    fs = _overlay_font_size(_osd_screen_h())
    y = 1080 - OVERLAY_MARGIN_BOTTOM
    now = datetime.now()
    clock_text = now.strftime("%I:%M %p").lstrip("0")
    data = f"{{\\an1\\pos(0,{y})\\fs{fs}{OVERLAY_STYLE}\\c{CLOCK_COLOR}}}{clock_text}"
    send_osd(OSD_CLOCK, data)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", state=state,
                           default_vocal_vol_pct=int(DEFAULT_VOCAL_VOLUME * 100),
                           default_sub_delay_ms=int(DEFAULT_SRT_SUB_DELAY * 1000),
                           default_pitch_st=DEFAULT_STARTING_PITCH)


@app.route("/api/files", methods=["POST"])
def set_files():
    """Receive file paths from the UI."""
    data = request.json
    state["video_path"] = data.get("video")
    state["vocal_path"] = data.get("vocal")
    state["nonvocal_path"] = data.get("nonvocal")
    send_upnext_overlay()
    return jsonify({"ok": True})


@app.route("/api/play", methods=["POST"])
def play():
    if not state["video_path"]:
        return jsonify({"ok": False, "error": "Missing video path"}), 400
    if state["playing"]:
        return jsonify({"ok": True})

    reset_state_defaults()

    controller.load_video(state["video_path"])

    # Block until the duration observer fires or we time out (5 s)
    controller.duration_ready.wait(5.0)

    dual = bool(state["vocal_path"] and state["nonvocal_path"]
                and os.path.exists(state["vocal_path"])
                and os.path.exists(state["nonvocal_path"]))
    state["dual_stem"] = dual

    if dual:
        controller.add_audio(state["vocal_path"])
        controller.add_audio(state["nonvocal_path"])
        time.sleep(0.2)

    fc = build_filter(state["vocal_volume"], semitones_to_pitch(state["semitones"]), dual)
    controller.set_lavfi_complex(fc)

    companions = derive_companion_paths(state["video_path"])
    state["subtitle_available"] = {
        "ass": companions["ass"] if Path(companions["ass"]).exists() else None,
        "srt": companions["srt"] if Path(companions["srt"]).exists() else None,
    }
    if   state["subtitle_available"]["ass"]: default_mode = "karaoke"
    elif state["subtitle_available"]["srt"]: default_mode = "srt"
    else:                                    default_mode = "off"
    _apply_subtitle_mode(default_mode, skip_remove=True)

    # Set playing=True LAST so _on_idle_active cannot fire on_song_end
    # during any of the subtitle/audio setup above.
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
    controller.toggle_pause()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop():
    controller.stop()
    time.sleep(0.1)   # let libmpv finish the stop before loading placeholder
    end_song()
    return jsonify({"ok": True})


@app.route("/api/exit", methods=["POST"])
def exit_app():
    """Stop playback and shut down the Flask server."""
    quit_mpv()
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    else:
        os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"ok": True})


@app.route("/api/seek", methods=["POST"])
def seek():
    pos = request.json.get("position", 0)
    controller.seek_absolute(float(pos))
    state["position"] = float(pos)
    return jsonify({"ok": True})


@app.route("/api/volume", methods=["POST"])
def set_volume():
    vol_pct = request.json.get("volume", 100)
    state["vocal_volume"] = float(vol_pct) / 100.0
    if state["playing"] and state["dual_stem"]:
        # Phase 4 gate: once controller.af_command("vocalvol", "volume", ...)
        # is verified to reach lavfi-complex filters without a rebuild blip,
        # replace this ZMQ block with:
        #   controller.af_command("vocalvol", "volume", str(state["vocal_volume"]))
        # and remove the azmq= clause from build_filter().
        try:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 2000)
            sock.connect("tcp://127.0.0.1:5556")
            sock.send_string(f"volume@vocalvol volume {state['vocal_volume']}")
            sock.recv()
        except zmq.ZMQError:
            pass
    send_timecode_overlay()
    return jsonify({"ok": True})


@app.route("/api/pitch", methods=["POST"])
def set_pitch():
    st = request.json.get("semitones", 0)
    state["semitones"] = int(st)
    if state["playing"]:
        fc = build_filter(state["vocal_volume"], semitones_to_pitch(state["semitones"]), state["dual_stem"])
        controller.set_lavfi_complex(fc)
    send_timecode_overlay()
    return jsonify({"ok": True})


@app.route("/api/subtitle", methods=["POST"])
def api_subtitle():
    """Atomic endpoint for subtitle mode and/or SRT delay changes."""
    data = request.get_json() or {}
    if "srt_delay" in data:
        state["srt_delay"] = float(data["srt_delay"])
    if "mode" in data:
        if state["playing"]:
            _apply_subtitle_mode(data["mode"])
        else:
            state["subtitle_mode"] = data["mode"]
    elif state["subtitle_mode"] == "srt" and state["playing"]:
        # delay-only update while in SRT mode
        controller.set_sub_delay(state["srt_delay"])
    return jsonify({"ok": True})


# ── Suppress /api/status from Werkzeug request log ────────────────────────────
class _StatusFilter(logging.Filter):
    def filter(self, record):
        return '/api/status' not in record.getMessage()

logging.getLogger('werkzeug').addFilter(_StatusFilter())


def get_master_volume_pct():
    """Query the system default sink volume as a 0–100 integer."""
    try:
        r = subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                           capture_output=True, text=True, timeout=1)
        m = re.search(r'(\d+(?:\.\d+)?)', r.stdout)
        return round(float(m.group(1)) * 100) if m else 100
    except Exception:
        return 100


@app.route("/api/master_volume", methods=["POST"])
def set_master_volume():
    pct = int(request.json.get("volume", 100))
    pct = max(0, min(100, pct))
    subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct / 100:.2f}"],
                   check=False, timeout=2)
    return jsonify({"ok": True})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "playing":            state["playing"],
        "position":           state["position"],
        "duration":           state["duration"],
        "vocal_volume":       state["vocal_volume"],
        "semitones":          state["semitones"],
        "subtitle_mode":      state["subtitle_mode"],
        "srt_delay":          state["srt_delay"],
        "subtitle_available": state["subtitle_available"],
        "dual_stem":          state["dual_stem"],
        "master_volume":      get_master_volume_pct(),
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
