# python-mpv Migration Plan

Replace the hand-rolled JSON/UNIX-socket IPC layer in `mpv/app.py` with the
[`python-mpv`](https://github.com/jaseg/python-mpv) library (libmpv bindings).
Goal: simpler architecture, event-driven state updates, and removal of the
persistent-socket workaround currently required for overlay commands.

## 1. Motivation

Today the app drives MPV through three parallel IPC channels:

1. **Ephemeral UNIX socket** per command — `send_mpv_command` / `send_mpv_query`
   open, write, read, close on every call (app.py:268–294).
2. **Persistent UNIX socket** for overlays — `_overlay_sock` is held open
   because `osd-overlay` and `overlay-add` are client-scoped and disappear when
   their originating client disconnects (app.py:120–207).
3. **ZMQ REQ socket** on `tcp://127.0.0.1:5556` — pushes live `volume@vocalvol`
   changes into the running `lavfi-complex` filter graph without rebuilding it
   (app.py:711–726, plus `azmq=bind_address=…` in `build_filter`).

And a **500 ms polling thread** (`poll_position`, app.py:554–600) issues
`time-pos`, `duration`, `idle-active`, `osd-width`, `osd-height` queries every
cycle and drives both progress updates and window-resize overlay refresh.

With `python-mpv` all three IPC paths collapse into one in-process `MPV()`
object, and the polling thread is replaced by property observers that fire only
when values actually change.

## 2. Dependency & environment

- Add `python-mpv>=1.0.5` to `mpv/requirements.txt`.
- `python-mpv` dlopens `libmpv.so.2`; verify on the `avtest` conda env:
  ```bash
  conda activate avtest
  python -c "import mpv; print(mpv.MPV().mpv_version)"
  ```
- `libmpv` must match the mpv binary on `$PATH` closely enough to support
  `lavfi-complex`, `rubberband`, and `overlay-add`. No functional change
  expected — same libmpv the CLI is built against.

## 3. Architecture: before vs. after

### Before
```
Flask route ──► send_mpv_command ──► open socket ──► mpv
                                              ◄── close socket
poll_position (thread, 2 Hz)
    └── send_mpv_query × N  ──► state dict ──► overlays
Overlay fns ──► send_overlay_command ──► persistent _overlay_sock ──► mpv
/api/volume ──► ZMQ REQ :5556 ──► azmq filter ──► vocalvol
```

### After
```
Flask route ──► MpvController.* (direct attr/command calls) ──► libmpv (in-proc)
MPV property observers ──fire on change──► MpvController._on_*  ──► state + overlays
Overlay fns ──► MpvController.osd_overlay / overlay_add ──► libmpv
/api/volume ──► MpvController.set_vocal_volume (command_node or filter rebuild)
```

One in-process client, one lock, zero sockets we manage by hand.

## 4. New module: `mpv/mpv_controller.py`

All libmpv interaction moves behind a single class so `app.py` only sees
high-level calls. This keeps Flask routes thin and gives us one place to add
thread-safety, error handling, and logging.

### 4.1 Class shape

```python
# mpv/mpv_controller.py
import threading
import mpv

class MpvController:
    """Owns the libmpv instance and mediates all player interactions.

    All public methods are safe to call from Flask request threads. Property
    observer callbacks run on libmpv's event thread — they must not call back
    into blocking libmpv commands; instead they update `state` and schedule
    overlay refreshes via a small in-thread queue or by calling the
    (already thread-safe) overlay helpers.
    """

    def __init__(self, state: dict, on_song_end, on_resize, on_tick):
        self._state = state
        self._on_song_end = on_song_end     # called when idle-active flips True mid-song
        self._on_resize = on_resize         # called on osd-dimensions change
        self._on_tick = on_tick             # called on time-pos change (throttled)
        self._lock = threading.RLock()      # guards filter rebuilds
        self._last_tick_emit = 0.0          # for throttling time-pos → overlay refresh
        self._last_osd_dim = (None, None)   # coalesces back-to-back resize fires
        self._duration_ready = threading.Event()   # set by _on_duration, awaited by /api/play
        self._player: mpv.MPV | None = None

    @property
    def duration_ready(self) -> threading.Event:
        return self._duration_ready

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None: ...
    def quit(self) -> None: ...

    # ── playback ──────────────────────────────────────────────────────────
    def load_video(self, path: str) -> None:
        """Load a video and clear duration_ready so /api/play can wait on it."""
        ...
    def add_audio(self, path: str) -> None: ...
    def load_placeholder(self, path: str) -> None: ...
    def seek_absolute(self, seconds: float) -> None: ...
    def toggle_pause(self) -> None: ...
    def stop(self) -> None: ...

    # ── filter graph ──────────────────────────────────────────────────────
    def set_lavfi_complex(self, fc: str) -> None: ...
    def clear_lavfi_complex(self) -> None: ...

    # ── subtitles ─────────────────────────────────────────────────────────
    def sub_add(self, path: str) -> None: ...
    def sub_remove(self) -> None: ...
    def set_sub_delay(self, seconds: float) -> None: ...
    def apply_srt_style(self, style: dict) -> None: ...

    # ── overlays ──────────────────────────────────────────────────────────
    def osd_overlay(self, overlay_id: int, data: str,
                    res_x: int = 1920, res_y: int = 1080) -> None: ...
    def clear_osd(self, overlay_id: int,
                  res_x: int = 1920, res_y: int = 1080) -> None: ...
    def overlay_add(self, overlay_id: int, x: int, y: int,
                    path: str, offset: int, fmt: str,
                    w: int, h: int, stride: int) -> None: ...
    def overlay_remove(self, overlay_id: int) -> None: ...

    # ── live parameters ──────────────────────────────────────────────────
    def set_filter_param(self, label: str, key: str, value: str) -> None:
        """Live-update a named filter param without rebuilding lavfi-complex.
        Replaces the ZMQ REQ path for volume@vocalvol."""

    # ── queries exposed for Flask ─────────────────────────────────────────
    @property
    def osd_size(self) -> tuple[int, int]: ...
```

### 4.2 `start()` body

```python
def start(self):
    self._player = mpv.MPV(
        idle=True,
        force_window=True,
        image_display_duration="inf",
        osd_margin_x=0,
        osd_margin_y=0,
        # no input-ipc-server — no external clients any more.
    )
    p = self._player

    # Property observers replace poll_position entirely.
    p.observe_property("time-pos",     self._on_time_pos)
    p.observe_property("duration",     self._on_duration)
    p.observe_property("idle-active",  self._on_idle_active)
    p.observe_property("osd-width",    self._on_osd_dim)
    p.observe_property("osd-height",   self._on_osd_dim)
```

### 4.3 Property observer callbacks

Run on `python-mpv`'s observer-dispatch thread. Keep them short: mutate `state`,
coalesce duplicates, then fan out to the overlay helpers.

```python
def _on_time_pos(self, _name, value):
    if value is None:
        return
    self._state["position"] = float(value)
    # Throttle: overlay refresh no more than ~2 Hz (matches current cadence)
    now = time.monotonic()
    if now - self._last_tick_emit >= 0.5:
        self._last_tick_emit = now
        self._on_tick()

def _on_duration(self, _name, value):
    if value is not None and float(value) > 0:
        self._state["duration"] = float(value)
        self._duration_ready.set()    # unblocks /api/play — see §8.1

def _on_idle_active(self, _name, value):
    # state["playing"] is reset to False as the FIRST line of end_song(),
    # so re-entry during placeholder load sees playing=False and bails.
    if value is True and self._state["playing"]:
        self._on_song_end()

def _on_osd_dim(self, _name, _value):
    # Both osd-width and osd-height fire on every resize. Compare against
    # last-seen pair and skip if unchanged so the second callback is a no-op.
    w = self._player["osd-width"]
    h = self._player["osd-height"]
    if (w, h) == self._last_osd_dim or w is None or h is None:
        return
    self._last_osd_dim = (w, h)
    self._on_resize()
```

Thread-safety rule: calling `player.command(...)` or assigning to a property
from inside these callbacks is safe — `python-mpv` dispatches commands
asynchronously via libmpv's own queue. What is **not** safe is calling any
blocking primitive (`wait_for_property`, `wait_for_event`) from inside an
observer, because observers run on the thread that delivers those events and
would deadlock waiting on themselves. None of our callbacks do this; keep it
that way.

### 4.4 Command methods (thin wrappers)

`python-mpv` exposes MPV commands as Python methods, properties as attributes.
The mapping is mechanical:

| Current call                                                   | Replacement |
|----------------------------------------------------------------|-------------|
| `send_mpv_command({"command": ["loadfile", path]})`            | `p.loadfile(path)` |
| `send_mpv_command({"command": ["audio-add", path]})`           | `p.audio_add(path)` |
| `send_mpv_command({"command": ["sub-add", path, "select"]})`   | `p.sub_add(path, "select")` |
| `send_mpv_command({"command": ["sub-remove"]})`                | `p.sub_remove()` |
| `send_mpv_command({"command": ["seek", pos, "absolute"]})`     | `p.seek(pos, "absolute")` |
| `send_mpv_command({"command": ["cycle", "pause"]})`             | `p.cycle("pause")` |
| `send_mpv_command({"command": ["stop"]})`                      | `p.stop()` |
| `send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})` | `p["lavfi-complex"] = fc` |
| `send_mpv_command({"command": ["set_property", "sub-delay", d]})` | `p["sub-delay"] = d` |
| `send_mpv_query({"command": ["get_property", "duration"]})`    | `p.duration` (may be None until demuxer parses) |
| `send_mpv_query({"command": ["get_property", "osd-height"]})`  | `p["osd-height"]` |

### 4.5 Overlay commands

Both `osd-overlay` and `overlay-add` are issued with `p.command(...)`:

```python
def osd_overlay(self, overlay_id, data, res_x=1920, res_y=1080):
    self._player.command(
        "osd-overlay",
        overlay_id,
        "ass-events",
        data,
        res_x, res_y,
        0,       # z
        "no",    # hidden
        "no",    # compute-bounds
    )

def clear_osd(self, overlay_id, res_x=1920, res_y=1080):
    self._player.command(
        "osd-overlay",
        overlay_id,
        "none",
        "",
        res_x, res_y,
        0, "no", "no",
    )

def overlay_add(self, overlay_id, x, y, path, offset, fmt, w, h, stride):
    self._player.command(
        "overlay-add",
        overlay_id, x, y, path, offset, fmt, w, h, stride,
    )
```

The client-scoping problem goes away: there is exactly one libmpv client, and
it lives for the whole process lifetime. `_overlay_sock`, `_get_overlay_sock`,
`send_overlay_command`, and `close_overlay_sock` are all deleted.

## 5. Eliminating the ZMQ volume path

Today `/api/volume` pushes through `azmq` so vocal volume can change without
rebuilding the filter graph (which would cause an audible re-init blip).

`python-mpv` exposes the same libmpv `af-command` / `vf-command` entry points
MPV uses internally. For libavfilter filter graphs embedded via
`lavfi-complex`, the command is routed via `command` + filter label:

```python
def set_filter_param(self, label: str, key: str, value: str):
    # Equivalent to: af-command <label> <key> <value>
    self._player.command("af-command", label, key, value)
```

Then `/api/volume` becomes:

```python
controller.set_filter_param("vocalvol", "volume", str(state["vocal_volume"]))
```

**Hard gate** — Phase 4 does not touch `/api/volume` until a prototype has
verified two things against a running dual-stem song:

1. The exact `python-mpv` invocation for `af-command`. The plan guesses
   `p.command("af-command", "vocalvol", "volume", "0.5")`, but `python-mpv`
   may expose it as `p.af_command(...)` instead. Both should be tried; the
   one that returns without raising is the one to use.
2. That `af-command` reaches filters embedded inside `lavfi-complex` (not
   just top-level `--af` filters). If it does not, keep `azmq` + ZMQ for this
   one path; the rest of the migration stands on its own.

No ZMQ code is deleted until the prototype passes.

## 6. Replacing `poll_position`

`poll_position` currently does four jobs. Each one moves to an observer:

| Job                                      | New mechanism |
|------------------------------------------|---------------|
| Update `state["position"]` at 2 Hz       | `observe_property("time-pos")` + throttle to 0.5 s |
| Update `state["duration"]`               | `observe_property("duration")` (fires once per load) |
| Song-end detection via `idle-active`     | `observe_property("idle-active")` |
| Resize detection via osd-width/height    | `observe_property("osd-width")` + `observe_property("osd-height")` |
| Refresh timecode overlay every cycle     | Inside `_on_time_pos` throttle branch |
| Refresh clock overlay every cycle        | Separate 1 Hz `threading.Timer` loop — the clock ticks on wall time, not player state, so it does not belong on a property observer |

Net result: `poll_thread`, `poll_stop`, and `poll_position` are deleted. A
small `clock_thread` replaces the wall-clock refresh (≈10 lines).

## 7. `state` dict (unchanged)

`state` keeps its current shape (app.py:101–114). Ownership moves: the
controller writes `position`, `duration`, and reads `playing`, `dual_stem`,
`semitones`, `vocal_volume` as needed. Flask routes keep writing the user-
intent fields (`video_path`, `subtitle_path`, etc.).

One addition: the controller holds a reference to `state` so observer
callbacks can mutate it without a circular import.

## 8. Data flow walkthroughs

### 8.1 Play a song

```
POST /api/play
  └─ reset_state_defaults()
  └─ controller.load_video(state["video_path"])
        └─ self._duration_ready.clear()
        └─ p.loadfile(path)
        (duration observer fires when demuxer is ready → state["duration"]
         and controller._duration_ready.set())
  └─ controller.duration_ready.wait(timeout=5.0)   ← blocks until observer fires
  └─ dual = bool(...)
  └─ if dual: controller.add_audio(vocal); controller.add_audio(nonvocal)
  └─ controller.set_lavfi_complex(build_filter(...))
  └─ if subtitle: controller.sub_add(path); controller.set_sub_delay(...)
  └─ state["playing"] = True
  └─ send_nowplaying_overlay(); send_timecode_overlay(); send_upnext_overlay()
```

### 8.2 Vocal volume change (dual-stem)

```
POST /api/volume
  └─ state["vocal_volume"] = v
  └─ controller.set_filter_param("vocalvol", "volume", str(v))   # was ZMQ
  └─ send_timecode_overlay()
```

### 8.3 Pitch change

```
POST /api/pitch
  └─ state["semitones"] = st
  └─ if state["playing"]:
       controller.set_lavfi_complex(build_filter(...))  # full rebuild, same as today
  └─ send_timecode_overlay()
```

Filter rebuild on pitch change stays — rubberband has no live `af-command`
for pitch shift. Volume is the only filter knob that benefits from
live-update, and that is specifically why `azmq` is there today.

### 8.4 Song ends naturally

```
libmpv: idle-active → True
  └─ observer _on_idle_active
       └─ if not state["playing"]: return          # re-entry guard
       └─ controller._on_song_end()                # callback supplied by app.py = end_song
             └─ state["playing"] = False           # FIRST, before any mpv commands,
             │                                      so any re-entry of _on_idle_active
             │                                      during the steps below is a no-op
             └─ controller.clear_lavfi_complex()
             └─ controller.load_placeholder(...)
             └─ reset_state_defaults()             # clears remaining state fields
             └─ clear_osd(OSD_NOWPLAYING); clear_osd(OSD_TIMECODE)
```

The `/api/stop` path keeps the existing `time.sleep(0.1)` between
`controller.stop()` and `end_song()`. It is there to let libmpv finish its
stop transition before the placeholder `loadfile` races in. A cleaner
alternative (`wait_for_event('idle')`) exists but has its own timeout
semantics; the 100 ms sleep is known to work and costs nothing, so keep it.

### 8.5 Window resize

```
libmpv: osd-width / osd-height change
  └─ observer _on_osd_dim
       └─ controller._on_resize()         # callback = refresh_overlays in app.py
             └─ send_qr_overlay()
             └─ send_url_overlay()
             └─ (if playing) send_nowplaying_overlay(); send_timecode_overlay()
             └─ send_upnext_overlay()
```

Resize is edge-triggered now, not polled — free CPU savings.

## 9. Thread-safety notes

- `python-mpv`'s `MPV.command()` and property setters dispatch via libmpv's
  internal queue and are safe to call from any thread, **including from
  inside observer callbacks**. Observer fan-out to overlay helpers is fine.
- The one thing that **will** deadlock is calling a blocking primitive —
  `player.wait_for_property`, `player.wait_for_event` — from inside an
  observer callback. Observers run on the same dispatch thread those
  primitives wait on, so they'd wait for themselves. None of the callbacks
  in this plan do this; keep it that way.
- Overlay helpers (`send_qr_overlay` etc.) are called from Flask request
  threads and from observer callbacks. The `_overlay_sock_lock` is gone;
  libmpv serializes internally.
- `build_filter` is pure; no locking needed.
- `state` dict mutations are still racy in theory (Flask threads + observer
  thread). They are racy today too and in practice have not mattered — dict
  operations on single keys are atomic under CPython's GIL. Flag as known,
  do not add locks unless a bug appears.

### 9.1 Error handling in `MpvController`

`python-mpv` raises exceptions on command failures (invalid filter graph,
missing file, shutdown race) where the old socket path silently swallowed
`ConnectionRefusedError` / `OSError`. Do not let those exceptions bubble up
into Flask routes — preserve the resilient-controller behavior the socket
path accidentally gave us.

Wrap public `MpvController` methods in a decorator:

```python
def _safe(method):
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except mpv.ShutdownError:
            return None
        except Exception:
            log.exception("mpv command failed: %s", method.__name__)
            return None
    return wrapper
```

Apply to every public method except `start` / `quit` (where failures are
fatal and should propagate). Observer callbacks wrap their own bodies the
same way so a transient libmpv hiccup does not kill the dispatch thread.

## 10. Files touched

- **new** `mpv/mpv_controller.py` — the class above (~250 lines).
- **modified** `mpv/app.py`:
  - Delete: `IPC_SOCKET`, `_overlay_sock*`, `_get_overlay_sock`,
    `send_overlay_command`, `close_overlay_sock`, `send_mpv_command`,
    `send_mpv_query`, `poll_position`, `poll_thread`, `poll_stop`,
    `start_mpv` (body), `quit_mpv` (body), the ZMQ import & block in
    `set_volume`.
  - Rewrite: every call site of the deleted functions to go through a
    module-level `controller: MpvController`.
  - Add: `clock_thread` (1 Hz wall-clock refresh for `send_clock_overlay`).
  - Keep: `build_filter`, all overlay formatting helpers, all Flask routes'
    outer shape, `state` dict.
- **modified** `mpv/requirements.txt`: add `python-mpv`. Optionally remove
  `pyzmq` if Phase 3 proves `af-command` works.
- **unchanged** `mpv/templates/index.html`, `qrcode.png`, placeholder assets.

## 11. Migration phases

Each phase leaves the app functional so we can stop if something goes wrong.

### Phase 1 — introduce `MpvController`, keep socket path alive
1. Write `mpv_controller.py` with `start`, `quit`, `load_video`, `add_audio`,
   `load_placeholder`, `seek_absolute`, `toggle_pause`, `stop`,
   `set_lavfi_complex`, `sub_add`, `sub_remove`, `set_sub_delay`,
   `apply_srt_style`.
2. In `app.py`, instantiate `controller = MpvController(state, ...)` in
   `start_mpv()` alongside the existing `subprocess.Popen`. Do **not** delete
   the socket path yet. **Preserve the existing `WERKZEUG_RUN_MAIN=true`
   guard around controller startup** — instantiating `MPV()` in both the
   reloader parent and child would spawn two player windows and fight over
   the display. The guard is working code today; do not remove or relocate
   it during this phase.
3. Route all command-sending Flask endpoints (`/api/play`, `/api/pause`,
   `/api/stop`, `/api/seek`, `/api/pitch`, `/api/sub_delay`, `/api/sub_mode`)
   through `controller`. Keep overlays + polling on the old socket path.
4. Smoke test: load + play + pause + seek + pitch + stop. Confirm identical
   behavior. This proves libmpv works in-process in the `avtest` env.

### Phase 2 — move overlays to the controller
1. Add `osd_overlay`, `clear_osd`, `overlay_add` to the controller.
2. Replace every call site in `send_qr_overlay`, `send_url_overlay`,
   `send_nowplaying_overlay`, `send_timecode_overlay`, `send_upnext_overlay`,
   `send_clock_overlay`.
3. Delete `_overlay_sock`, `_get_overlay_sock`, `send_overlay_command`,
   `send_osd`, `clear_osd` socket helpers, `close_overlay_sock`.
4. Smoke test: overlays visible on launch, on resize, during playback, after
   stop. Check that QR, URL, Now Playing, Timecode, Up Next, and Clock all
   render.

### Phase 3 — replace poll thread with observers
1. Add observer methods to `MpvController`; wire the three callbacks
   (`on_song_end=end_song`, `on_resize=refresh_overlays`, `on_tick=tick`).
2. Extract the overlay-refresh-on-resize block from `poll_position` into
   `refresh_overlays()` in `app.py`.
3. Extract clock refresh into `clock_thread` (1 Hz `threading.Timer` or
   `threading.Event().wait(1.0)` loop).
4. Delete `poll_position`, `poll_thread`, `poll_stop` and the legacy
   `send_mpv_command` / `send_mpv_query`.
5. Smoke test: song progress bar advances smoothly, natural song-end triggers
   placeholder, resizing the MPV window re-lays out overlays.

### Phase 4 — collapse ZMQ (conditional)
1. Prototype `p.command("af-command", "vocalvol", "volume", "0.5")` against a
   running dual-stem song.
2. If the volume change takes effect without an audible re-init blip,
   replace `/api/volume` ZMQ block with `controller.set_filter_param(...)`,
   drop `azmq=bind_address=…` from `build_filter` (dual-stem branch), remove
   `pyzmq` dependency and `import zmq`.
3. If it does not work, leave `/api/volume` and `azmq` alone. The ZMQ path is
   isolated to one route and one filter string; it is acceptable tech debt.

### Phase 5 — cleanup
1. Delete `IPC_SOCKET` constant and all references.
2. Confirm `requirements.txt` and imports are tidy.
3. Leave the `WERKZEUG_RUN_MAIN` guard in place. The in-proc model has the
   same parent/child split the subprocess model had; removing the guard
   would spawn two MPV windows under `debug=True`.

## 12. Testing checklist

After each phase, run through these manually against a real song:

- [ ] Launch app, MPV window shows placeholder with QR + URL overlay.
- [ ] Clock overlay updates every minute (bottom-left).
- [ ] Pick a dual-stem song, Play. Video + both stems mix, subtitles load.
- [ ] Pitch slider ±6 steps — audio pitches without truncation, subtitles
      unaffected.
- [ ] Vocal volume slider 0–100 — no audible re-init blip (Phase 4 gate).
- [ ] Subtitle delay slider updates live.
- [ ] Seek bar scrubs forward and backward.
- [ ] Pause / resume toggles correctly.
- [ ] Stop returns to placeholder with QR visible and overlays cleared.
- [ ] Let a song play through to the end — `idle-active` path triggers
      `end_song` without manual Stop.
- [ ] Resize the MPV window — QR, URL, Now Playing, Timecode, Up Next, Clock
      reposition to the new dimensions.
- [ ] Switch subtitle mode mid-playback (SRT ↔ ASS).
- [ ] Exit via `/api/exit` — process shuts down cleanly, no orphaned mpv.

## 13. Risks & rollbacks

| Risk | Mitigation |
|------|------------|
| libmpv version in `avtest` env is too old for some command | Pin or upgrade `mpv` / `libmpv2` in conda env; test via Phase 1 smoke test before deleting old path |
| Observer callback blocks or deadlocks | Keep callbacks to state mutation + function dispatch; no `wait_for_property` inside observers |
| `af-command` does not reach `lavfi-complex` filters | Keep ZMQ path (Phase 4 is conditional) |
| Overlay client-scoping semantics differ under libmpv | Verified above — single client, lifetime = process; fallback is to re-send all overlays on `file-loaded` event, which is cheap |
| Werkzeug reloader spawns two libmpv instances | Keep the existing `WERKZEUG_RUN_MAIN` guard around controller startup |
| Loss of the `/tmp/mpv-socket` file breaks external tooling that relied on it | There is no such tooling in this repo; document the removal in the commit message |

Rollback at any phase: revert `app.py` and delete `mpv_controller.py`. The
legacy socket code is untouched until Phase 2–3 deletions; keep those in
separate commits so a single `git revert` restores the old path.

## 14. Line-count expectation

Rough estimate of the diff, for sanity-checking scope:

- Delete from `app.py`: ~180 lines (socket helpers + poll thread + ZMQ block).
- Add in `app.py`: ~40 lines (controller wiring, refresh_overlays, clock
  thread, callback plumbing).
- New `mpv_controller.py`: ~250 lines.

Net: roughly +110 lines, but behind a clean boundary and with one external
dependency instead of three ad-hoc IPC mechanisms.
