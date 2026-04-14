# python-mpv Migration — Potential Issues

## Critical

### 1. Observer Callback Command Re-entrancy

**Location**: Section 4.3 & 9 (Thread-safety notes)

The plan states observer callbacks (`_on_song_end`, `_on_resize`, `_on_tick`) call back into `app.py` functions that issue overlay commands, claiming this is "safe because `python-mpv` serializes `command()` internally."

- **Contradiction**: Section 4.3 says "do not issue commands from inside these callbacks" but then immediately says the `app.py` callbacks do exactly that.
- **Real risk**: If `_on_resize()` triggers 5 overlay commands synchronously from the libmpv event thread, and a Flask route simultaneously issues a command, there could be re-entrant lock contention or event-loop starvation depending on `python-mpv`'s internal implementation.
- **Missing**: The plan should clarify whether `python-mpv`'s thread-safety model uses a re-entrant lock or a command queue, and verify the event thread can safely call `command()`.

### 2. `end_song()` Recursion Guard

**Location**: Section 8.4 (Song ends naturally)

`end_song()` calls `load_placeholder()`, which triggers a new file load. If `idle-active` fires again during that load (before `state["playing"]` is reset to `False`), you could get a recursive `end_song()` → `load_placeholder()` loop.

- **Mitigation needed**: Explicitly document and enforce the `state["playing"]` guard in `_on_idle_active`. Consider adding a "song-ending in progress" flag if the state reset isn't atomic enough.

### 3. `stop()` Race with `end_song()`

**Location**: Section 8.4 vs current `app.py:683`

Current code includes:
```python
send_mpv_command({"command": ["stop"]})
time.sleep(0.1)   # let MPV process the stop before loading placeholder
end_song()
```

The plan's `end_song()` flow doesn't include the `time.sleep(0.1)` delay. Without it, `load_placeholder()` may race with `stop()` and fail to load.

- **Missing**: The controller's `stop()` method should either block until mpv acknowledges the stop, or `end_song()` should include a small delay/sync point.

---

## High

### 4. `_on_osd_dim` Duplicate Calls

**Location**: Section 4.3

```python
def _on_osd_dim(self, _name, _value):
    # Coalesce width/height bursts — resize typically fires both.
    self._on_resize()
```

Both `osd-width` and `osd-height` observers fire `_on_resize()` independently, potentially causing duplicate overlay refreshes in rapid succession.

- **Missing**: No debounce/coalescing mechanism. The current polling approach naturally coalesces because it checks both dimensions in one cycle. Here, two back-to-back `_on_resize()` calls could waste CPU and cause visual flicker.
- **Mitigation**: Add a short debounce window (e.g., 100ms) using `threading.Timer` or track a "resize pending" timestamp and skip if already fired recently.

### 5. `af-command` Syntax Unverified

**Location**: Section 5

The plan acknowledges this is conditional, but the command syntax is guessed:
```python
self._player.command("af-command", label, key, value)
```

- **Missing**: No reference to `python-mpv` documentation or mpv's `af-command` IPC spec. The actual mpv JSON IPC uses `{"command": ["af-command", "vocalvol", "volume", "0.5"]}`, but whether `python-mpv` exposes this as `.command("af-command", ...)` or `.af_command(...)` or something else is unverified.
- **Action**: Run a quick experiment in Phase 4 before committing to removing ZMQ.

---

## Medium

### 6. Duration Polling Race in `/api/play`

**Location**: Section 8.1

```
└─ wait for state["duration"] > 0 (bounded, 5 s)  ← now just polls state, not MPV
```

The plan doesn't specify what happens if the `duration` observer hasn't fired yet after 5 seconds. The current code polls MPV directly every 200ms. The new approach polls a `state` dict that's only updated by an async observer callback.

- **Risk**: If the observer fires slightly after the 5s timeout, `state["duration"]` gets updated but `/api/play` has already given up — song loads but framework thinks it failed.
- **Mitigation**: Increase the timeout slightly (e.g., 8s) or add a retry mechanism. Alternatively, use a `threading.Event` that the observer sets when duration is available.

### 7. `apply_srt_style` Batching

**Location**: Section 4.1

```python
def apply_srt_style(self, style: dict) -> None: ...
```

Current implementation sends each `sub-*` property as a separate command. With libmpv, these become individual property assignments (`p["sub-font"] = "Arial"`, etc.).

- **Risk**: mpv may apply them atomically or in sequence, potentially causing intermediate malformed states (e.g., a font size change before the font name change takes effect).
- **Mitigation**: Verify whether mpv batches property changes or if there's a `set_property_many` equivalent. If not, test for visual glitches.

### 8. Werkzeug Reloader Guard

**Location**: Section 11, Phase 5

`python-mpv` creates an in-process `MPV()` instance. If the Werkzeug reloader spawns a parent+child, and the parent also instantiates `MPV()`, you get **two mpv windows** and potential libmpv initialization conflicts.

- The `WERKZEUG_RUN_MAIN=true` guard is critical, but the plan treats it as an afterthought.
- **Action**: Make this a first-class concern in Phase 1, not a cleanup item in Phase 5. Verify the guard works before building on top of it.

---

## Low

### 9. BGRA Overlay File Cleanup

**Location**: Section 4.5

Current code writes raw BGRA bytes to `/tmp/qr_overlay.bgra`. The plan doesn't address:
- Whether libmpv can read from the same tmp path when running in-process (should be fine, but unmentioned).
- File cleanup — `/tmp/qr_overlay.bgra` is never deleted in current code, and the plan doesn't change this.
- **Action**: Minor cleanup — add `os.unlink(overlay_path)` after `overlay_add` succeeds, or at least document it as known tech debt.

### 10. Phase Numbering Inconsistency

**Location**: Section 5 vs Section 11

Section 5 references "Phase 3" for the `af-command` experiment, but Section 11 labels it **Phase 4**.

- **Action**: Fix the cross-reference before execution begins.

### 11. No Error Handling Strategy

**Location**: Throughout

The plan doesn't specify what happens when `libmpv` calls fail (e.g., file not found, invalid filter graph). Current socket code silently catches `ConnectionRefusedError`/`OSError` and continues.

- **Action**: The controller should at least log exceptions from `python-mpv` methods. Consider adding a simple try/except + `logging.exception` wrapper around public methods.

---

## Summary Matrix

| Priority | # | Issue |
|----------|---|-------|
| **Critical** | 1 | Observer callback command re-entrancy |
| **Critical** | 2 | `end_song()` recursion guard |
| **Critical** | 3 | `stop()` race with `end_song()` |
| **High** | 4 | `_on_osd_dim` duplicate calls |
| **High** | 5 | `af-command` syntax unverified |
| **Medium** | 6 | Duration polling race in `/api/play` |
| **Medium** | 7 | `apply_srt_style` batching |
| **Medium** | 8 | Werkzeug reloader guard |
| **Low** | 9 | BGRA overlay file cleanup |
| **Low** | 10 | Phase numbering inconsistency |
| **Low** | 11 | No error handling strategy |
