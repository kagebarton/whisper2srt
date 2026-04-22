"""
MpvController — wraps python-mpv (libmpv) for the karaoke mixer.

All public methods are safe to call from Flask request threads.
Observer callbacks run on python-mpv's event-dispatch thread; they may
call player.command() / set properties safely (libmpv serialises these
internally), but must NOT call blocking primitives (wait_for_property,
wait_for_event) — that would deadlock the event loop.
"""

import threading
import time
import logging

import mpv
import zmq as _zmq

_zmq_ctx = _zmq.Context()

log = logging.getLogger(__name__)


def _safe(method):
    """Decorator: catch and log exceptions from mpv calls; return None on failure.

    Applied to every public method except start/quit (where failure is fatal
    and should propagate). Observer callbacks wrap their own bodies the same
    way so a transient libmpv hiccup does not kill the dispatch thread.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except mpv.ShutdownError:
            return None
        except Exception:
            log.exception("MpvController.%s failed", method.__name__)
            return None
    wrapper.__name__ = method.__name__
    return wrapper


class MpvController:
    """
    Owns the libmpv instance and mediates all player interactions.

    Constructor callbacks:
      on_song_end() — called when idle-active flips True while playing
      on_resize()   — called when osd dimensions change (coalesced)
      on_tick()     — called at ~2 Hz while time-pos is advancing
    """

    def __init__(self, state: dict, on_song_end, on_resize, on_tick):
        self._state = state
        self._on_song_end = on_song_end
        self._on_resize = on_resize
        self._on_tick = on_tick
        self._lock = threading.RLock()       # guards filter rebuilds
        self._last_tick_emit = 0.0           # throttles time-pos → overlay refresh
        self._current_osd_dim = [None, None] # [w, h] updated per observer callback
        self._fired_osd_dim = (None, None)   # last (w, h) pair we fired refresh for
        self._duration_ready = threading.Event()
        self._player: mpv.MPV | None = None

    @property
    def duration_ready(self) -> threading.Event:
        """Event set when duration > 0 is known; cleared on each load_video()."""
        return self._duration_ready

    @property
    def osd_size(self) -> tuple[int, int]:
        """Current (width, height) of the mpv OSD, defaulting to 1920×1080.

        Returns the last values received via the osd-width/osd-height observers
        rather than querying the player, so it is safe to call at any time.
        """
        w, h = self._current_osd_dim
        return (int(w) if w else 1920, int(h) if h else 1080)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Create the libmpv instance and register all property observers."""
        self._player = mpv.MPV(
            idle=True,
            force_window=True,
            image_display_duration="inf",
            osd_margin_x=0,
            osd_margin_y=0,
            terminal=False,
            input_vo_keyboard=True,
        )
        p = self._player
        p.observe_property("time-pos",    self._on_time_pos)
        p.observe_property("duration",    self._on_duration)
        p.observe_property("idle-active", self._on_idle_active)
        p.observe_property("osd-width",   self._on_osd_dim)
        p.observe_property("osd-height",  self._on_osd_dim)

        @p.on_key_press('f')
        def _toggle_fullscreen():
            p.fullscreen = not p.fullscreen

    def quit(self) -> None:
        """Shut down the libmpv instance cleanly."""
        p = self._player
        if p is not None:
            self._player = None
            try:
                p.quit(0)
                p.wait_for_shutdown()
            except Exception:
                pass

    # ── Observer callbacks ────────────────────────────────────────────────────

    def _on_time_pos(self, _name, value):
        try:
            if value is None:
                return
            self._state["position"] = float(value)
            now = time.monotonic()
            if now - self._last_tick_emit >= 0.5:
                self._last_tick_emit = now
                self._on_tick()
        except Exception:
            log.exception("_on_time_pos error")

    def _on_duration(self, _name, value):
        try:
            if value is not None and float(value) > 0:
                self._state["duration"] = float(value)
                self._duration_ready.set()
        except Exception:
            log.exception("_on_duration error")

    def _on_idle_active(self, _name, value):
        # state["playing"] is set to False as the FIRST line of end_song(),
        # so re-entry during placeholder load sees playing=False and bails.
        try:
            if value is True and self._state["playing"]:
                self._on_song_end()
        except Exception:
            log.exception("_on_idle_active error")

    def _on_osd_dim(self, name, value):
        # Each observer delivers one dimension at a time. Accumulate both into
        # _current_osd_dim, then fire refresh only when the complete (w, h)
        # pair differs from the last pair we fired for — coalescing the two
        # back-to-back callbacks a resize produces into one refresh call.
        try:
            if value is None:
                return
            if "width" in name:
                self._current_osd_dim[0] = value
            else:
                self._current_osd_dim[1] = value
            w, h = self._current_osd_dim
            if w is None or h is None:
                return
            pair = (w, h)
            if pair == self._fired_osd_dim:
                return
            self._fired_osd_dim = pair
            self._on_resize()
        except Exception:
            log.exception("_on_osd_dim error")

    # ── Playback ──────────────────────────────────────────────────────────────

    @_safe
    def load_video(self, path: str) -> None:
        """Load a video. Clears duration_ready so /api/play can await it."""
        self._duration_ready.clear()
        self._fired_osd_dim = (None, None)   # force overlay refresh on next resize
        self._player.loadfile(path)

    @_safe
    def load_placeholder(self, path: str) -> None:
        self._player.loadfile(path)

    @_safe
    def add_audio(self, path: str) -> None:
        self._player.audio_add(path)

    @_safe
    def seek_absolute(self, seconds: float) -> None:
        self._player.command("seek", seconds, "absolute")

    @_safe
    def toggle_pause(self) -> None:
        self._player.cycle("pause")

    @_safe
    def stop(self) -> None:
        self._player.stop()

    # ── Filter graph ──────────────────────────────────────────────────────────

    @_safe
    def set_lavfi_complex(self, fc: str) -> None:
        with self._lock:
            self._player.lavfi_complex = fc

    @_safe
    def clear_lavfi_complex(self) -> None:
        with self._lock:
            self._player.lavfi_complex = ""

    # ── Subtitles ─────────────────────────────────────────────────────────────

    @_safe
    def sub_add(self, path: str, flag: str = "select") -> None:
        self._player.sub_add(path, flag)

    @_safe
    def set_sid(self, value: str | int) -> None:
        # Prefer sid="no" over sub-remove: sub-remove segfaults libmpv while
        # lavfi-complex is active (known libmpv bug).
        self._player.sid = value

    @_safe
    def set_sub_delay(self, seconds: float) -> None:
        self._player.sub_delay = float(seconds)

    @_safe
    def apply_srt_style(self, style: dict) -> None:
        # python-mpv maps p.prop_name → "prop-name" (underscore → hyphen).
        for prop, val in style.items():
            setattr(self._player, prop.replace("-", "_"), val)

    # ── Overlays ──────────────────────────────────────────────────────────────

    @_safe
    def osd_overlay(self, overlay_id: int, data: str,
                    res_x: int = 1920, res_y: int = 1080) -> None:
        """Send an ASS-events OSD overlay via the named-args command form."""
        self._player.command(
            "osd-overlay",
            id=overlay_id,
            format="ass-events",
            data=data,
            res_x=res_x,
            res_y=res_y,
        )

    @_safe
    def clear_osd(self, overlay_id: int,
                  res_x: int = 1920, res_y: int = 1080) -> None:
        """Clear an OSD overlay slot."""
        self._player.command(
            "osd-overlay",
            id=overlay_id,
            format="none",
            data="",
            res_x=res_x,
            res_y=res_y,
        )

    @_safe
    def overlay_add(self, overlay_id: int, x: int, y: int, path: str,
                    offset: int, fmt: str, w: int, h: int, stride: int) -> None:
        """Add a raw BGRA bitmap overlay (positional args form)."""
        self._player.command(
            "overlay-add",
            overlay_id, x, y, path, offset, fmt, w, h, stride,
        )

    @_safe
    def overlay_remove(self, overlay_id: int) -> None:
        self._player.command("overlay-remove", overlay_id)

    # ── Live filter parameters (Phase 4 gate) ─────────────────────────────────

    @_safe
    def af_command(self, label: str, cmd: str, argument: str) -> None:
        """Issue a live af-command to a named filter label.

        Note: af_command does NOT reach filters inside lavfi-complex.
        Use zmq_af_command() for those.
        """
        self._player.af_command(label, cmd, argument)

    def zmq_af_command(self, label: str, cmd: str, value: str,
                       address: str = "tcp://127.0.0.1:5556") -> None:
        """Send a live parameter change to a lavfi-complex filter via ZMQ.

        A fresh socket is created per call so a receive timeout never leaves
        the REQ socket in a stuck state. The context is module-level and reused.
        """
        sock = _zmq_ctx.socket(_zmq.REQ)
        try:
            sock.setsockopt(_zmq.RCVTIMEO, 2000)
            sock.connect(address)
            sock.send_string(f"{label} {cmd} {value}")
            sock.recv()
        except _zmq.ZMQError:
            pass
        finally:
            sock.close()
