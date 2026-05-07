# Bug: `sub-remove` + `lavfi-complex` = libmpv Segfault on Initial Play

**Date:** 2026-04-15
**Commit that introduced the bug:** `5a5467a` ("Centralized sub management WIP")
**Commit where it worked:** `84910fd` ("Centralized UI state management")

---

## Symptom

Playing a song whose companion subtitle file exists (e.g. `selfish.ass`) causes both the
Flask server **and** the mpv window to exit immediately after play begins, with no error
message or log output. Exit code is **139** (SIGSEGV — segmentation fault in libmpv).

---

## Root Cause

The centralization commit introduced `_apply_subtitle_mode()`, which always calls
`controller.sub_remove()` before `controller.sub_add()`:

```python
# app.py — _apply_subtitle_mode() as introduced in 5a5467a
controller.sub_remove()   # ← always runs
if path:
    controller.sub_add(path, "select")
```

This is called from `/api/play` while `lavfi-complex` is already active (it is set
a few lines earlier). At that point mpv has **auto-loaded** the companion subtitle
(mpv matches `selfish.ass` to `selfish.mp4` by filename). Calling `sub-remove` on
that auto-loaded track **while `lavfi-complex` is active** triggers a segfault inside
libmpv, killing both the media engine and the Python process.

The old code (pre-centralization) never called `sub-remove` during play; it called
`sub_add` directly, which safely replaces the active track without removing it first:

```python
# Old /api/play (84910fd) — worked fine
if state["subtitle_path"]:
    controller.sub_add(state["subtitle_path"], "select")
    ...
```

### Why `sub-remove` crashes here specifically

| Scenario | `lavfi-complex` active? | `sub-remove` caller | Crash? |
|---|---|---|---|
| Initial play (auto-loaded sub) | ✅ Yes | `_apply_subtitle_mode` | ✅ **YES** |
| Mode switch during playback | ✅ Yes | `_apply_subtitle_mode` | ❌ No |
| No subtitle companion exists | ✅ Yes | `_apply_subtitle_mode` | ❌ No |
| Idle / no filter loaded | ❌ No | any | ❌ No |

The crash is specific to the combination of:
1. The subtitle being **auto-loaded by mpv** (via filename matching), not by our own `sub_add`.
2. `sub-remove` being called on that auto-loaded track while `lavfi-complex` is running.

When we load the subtitle ourselves first (via `sub_add`) and then later call `sub-remove`
during a user-initiated mode switch, libmpv does not crash. This suggests it is a libmpv
internal state issue with auto-loaded vs. manually added subtitle tracks under `lavfi-complex`.

### Why there is no error message

- `@_safe` catches `mpv.ShutdownError` and generic `Exception`, but a segfault in
  libmpv's C layer terminates the process before any Python exception can propagate.
- Flask's Werkzeug logger also terminates, so nothing is printed to the console.

---

## Reproduction

```python
import mpv, time, threading

p = mpv.MPV(idle=True, terminal=False, force_window=False)

dur_ready = threading.Event()
p.observe_property('duration', lambda _n, v: dur_ready.set() if v and float(v) > 0 else None)

p.loadfile('/path/to/selfish.mp4')   # selfish.ass exists alongside it
dur_ready.wait(5)
p.audio_add('/path/to/selfish---vocal.m4a')
p.audio_add('/path/to/selfish---nonvocal.m4a')
time.sleep(0.2)

p.lavfi_complex = "...full dual-stem filter string..."
time.sleep(0.3)

# ← CRASH: sub-remove on auto-loaded track while lavfi-complex is active
p.command('sub-remove')
p.sub_add('/path/to/selfish.ass', 'select')  # never reached
```

Verified exit code 139 (SIGSEGV) on `pik` conda env, mpv package from repo.

---

## Fix

Added a `skip_remove` keyword argument to `_apply_subtitle_mode()`. The `/api/play` call
site passes `skip_remove=True`; the `/api/subtitle` endpoint (user mode switches during
active playback) uses the default `skip_remove=False`.

```python
def _apply_subtitle_mode(mode, *, skip_remove=False):
    """...
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
    if not skip_remove:          # ← guard
        controller.sub_remove()
    if path:
        controller.sub_add(path, "select")
        ...
```

And in `/api/play`:
```python
_apply_subtitle_mode(default_mode, skip_remove=True)
```

### Why not always skip `sub_remove`?

Mode switches from the `/api/subtitle` endpoint **do** need `sub_remove()` to evict the
currently-active manually-loaded subtitle before swapping in a different file. Skipping
it there would leave orphaned subtitle tracks. The `skip_remove` flag is intentionally
scoped only to the initial-play code path.

---

## Notes / Future Investigation

- This is almost certainly a libmpv bug, not a python-mpv bug. The `sub-remove` command
  itself succeeds (returns OK, no exception), but something in libmpv's filter-graph
  teardown for the auto-loaded track corrupts state when `lavfi-complex` is live.
- If a future libmpv version fixes this, the `skip_remove=True` call site can be reverted
  to a plain `_apply_subtitle_mode(default_mode)` with no behavioral difference, since
  `sub_add(..., "select")` still works correctly to replace any pre-existing track.
- Single-stem playback (no `lavfi-complex`) was not tested as a crash-reproducer, but
  the default song (`selfish`) always uses dual-stem, so this is the relevant path.
