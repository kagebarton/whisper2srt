# Subtitle State Architecture Refactor

## Context

The subtitle feature has been a persistent source of bugs: delay not reapplied when switching modes, delay lost between songs, races when rapidly switching mode, and inconsistent UI state. These are symptoms of a fragile architecture, not isolated bugs:

1. **Delay is stored once with a cache sidecar.** The `subDelay` / `srtDelayCache` dance in [index.html:850-855](../templates/index.html#L850-L855) tries to restore the SRT delay when returning to SRT mode. Any poll, race, or double-switch can corrupt it.
2. **Two un-awaited parallel fetches** in `setSubMode` ([index.html:863-872](../templates/index.html#L863-L872)) — `/api/sub_mode` and `/api/sub_delay` can land out of order, and polling fires every 500ms between them.
3. **Polling syncs `subDelay` unconditionally** ([index.html:1016](../templates/index.html#L1016)), opening a race window during mode changes.
4. **Split ownership.** Both client and server track subtitle path/delay, with inconsistent reset semantics (`reset_state_defaults` resets `subtitle_delay` but not `subtitle_path` — see [app.py:210-218](../app.py#L210-L218)).

**Outcome:** consolidate subtitle state into a cohesive slice, make the server the sole authority, collapse mode+delay into one atomic endpoint, and have the client mirror server state with a sync-id guard for in-flight mutations.

### Invariants (must hold after refactor)

1. **SRT delay resets to `DEFAULT_SRT_SUB_DELAY` on every song start.** No carry-over between songs.
2. **ASS (karaoke) mode never applies a delay.** mpv's `sub_delay` is forced to `0.0` whenever `mode != 'srt'`, including karaoke and off. ASS files have word-level `\kf` timing and a delay would desync them.
3. **`srt_delay` is sticky across mode switches *within* a song.** Karaoke → SRT restores the user's last SRT delay for that song.

This plan **preserves** the recent UI refactor (`S.pending`/`S.active` split, `update()`/`render()`, `locks`). Only the subtitle slice is changed.

---

## Data Structures

### Server `state` dict ([app.py:93-106](../app.py#L93-L106))

**Add:**
```python
state["subtitle_mode"]      = 'off'          # 'karaoke' | 'srt' | 'off'
state["srt_delay"]          = DEFAULT_SRT_SUB_DELAY  # sticky across mode switches within a song
state["subtitle_available"] = {'ass': None, 'srt': None}  # derived on /api/play
```

**Remove:**
```python
state["subtitle_path"]   # replaced — path is resolved server-side from mode+available
state["subtitle_delay"]  # replaced by srt_delay (only SRT has a meaningful delay)
```

### Client `S` store ([index.html:475-525](../templates/index.html#L475-L525))

**Add:**
```js
S.active.subtitle = {
  mode: 'off',                          // mirrored from server
  available: { ass: null, srt: null },  // mirrored from server
  srtDelay: 0,                          // seconds; sticky across mode switches
};
S.subtitleMutation = { id: 0, committed: 0 };  // sync-id guard vs polling
```

**Remove:**
```js
S.active.subMode               // → S.active.subtitle.mode
S.active.companions.ass/srt    // → S.active.subtitle.available
S.controls.subDelay            // → S.active.subtitle.srtDelay
S.controls.srtDelayCache       // not needed — delay is per-mode now
S.pending.subtitle             // no longer sent to server
S.pending.subMode              // unused after this refactor
```

`S.pending.companions.{ass,srt}` **stays** — used for the file-browser preview, independent of playback subtitle state.

---

## Backend Changes (`mpv/app.py`)

### New: `POST /api/subtitle`

Single atomic endpoint for mode and/or delay changes. Idempotent.

```python
@app.route('/api/subtitle', methods=['POST'])
def api_subtitle():
    data = request.get_json() or {}
    if 'srt_delay' in data:
        state['srt_delay'] = float(data['srt_delay'])
    if 'mode' in data:
        _apply_subtitle_mode(data['mode'])
    elif state['subtitle_mode'] == 'srt' and state['playing']:
        # delay-only update while in SRT
        controller.set_sub_delay(state['srt_delay'])
    return jsonify({'ok': True})

def _apply_subtitle_mode(mode):
    """Atomic: update state, remove current sub, add new, apply delay.
    Invariant: sub_delay is 0.0 for any mode != 'srt'."""
    state['subtitle_mode'] = mode
    if not state['playing']:
        return
    path = None
    if   mode == 'karaoke': path = state['subtitle_available']['ass']
    elif mode == 'srt':     path = state['subtitle_available']['srt']
    controller.sub_remove()
    if path:
        controller.sub_add(path, 'select')
        if mode == 'srt':
            controller.apply_srt_style(SRT_STYLE)
            controller.set_sub_delay(state['srt_delay'])
        else:
            controller.set_sub_delay(0.0)
    else:
        controller.set_sub_delay(0.0)
```

Reuses existing `MpvController` methods at [mpv_controller.py:208-231](../mpv_controller.py#L208-L231): `sub_add`, `sub_remove` (already safe when no sub loaded), `set_sub_delay`, `apply_srt_style`. No controller changes needed.

### Modify: `/api/play` (~app.py:445-480)

After video loads and `reset_state_defaults()` runs:

```python
# Derive companions for the active video
companions = derive_companion_paths(state['video_path'])  # existing helper, app.py:77
state['subtitle_available'] = {
    'ass': companions['ass'] if Path(companions['ass']).exists() else None,
    'srt': companions['srt'] if Path(companions['srt']).exists() else None,
}
# Pick default mode
if   state['subtitle_available']['ass']: default_mode = 'karaoke'
elif state['subtitle_available']['srt']: default_mode = 'srt'
else:                                    default_mode = 'off'
_apply_subtitle_mode(default_mode)
```

Remove the old block that reads `state['subtitle_path']` and manually calls `sub_add`/`set_sub_delay`.

### Modify: `reset_state_defaults()` ([app.py:210-218](../app.py#L210-L218))

```python
state['subtitle_mode']      = 'off'
state['srt_delay']          = DEFAULT_SRT_SUB_DELAY   # ← Invariant 1: per-song reset
state['subtitle_available'] = {'ass': None, 'srt': None}
# remove: state['subtitle_delay'] = ...
```

### Modify: `/api/status` response (~app.py:620)

```python
return jsonify({
    ...
    'subtitle_mode':      state['subtitle_mode'],
    'srt_delay':          state['srt_delay'],
    'subtitle_available': state['subtitle_available'],
    # remove: 'subtitle_delay'
})
```

### Modify: `/api/files` (~app.py:435-445)

Remove the line that writes `state['subtitle_path']` from request body. `/api/files` now only receives `{video, vocal, nonvocal}`; subtitles are resolved server-side on play.

### Remove endpoints

- `DELETE /api/sub_mode` — replaced by `/api/subtitle`
- `DELETE /api/sub_delay` — folded into `/api/subtitle`

### Keep unchanged

- `GET /api/derive_files` — still used by the browser preview pane for the **pending** (not-yet-playing) file.
- `derive_companion_paths()` ([app.py:77-87](../app.py#L77-L87)) — reused in the new `/api/play` logic.

---

## Frontend Changes (`mpv/templates/index.html`)

### `render()` ([index.html:555-613](../templates/index.html#L555-L613))

Replace the subtitle section:

```js
// Subtitle mode buttons
const sub = S.active.subtitle;
$('subBtnKaraoke').disabled = !sub.available.ass;
$('subBtnSrt').disabled     = !sub.available.srt;
['karaoke', 'srt', 'off'].forEach(m => {
  const key = m.charAt(0).toUpperCase() + m.slice(1);
  $('subBtn' + key).classList.toggle('active', m === sub.mode);
});

// Sub delay row — only in SRT mode
$('subDelayRow').style.display = sub.mode === 'srt' ? '' : 'none';
if (!S.locks.subDelay) {
  const ms = Math.round(sub.srtDelay * 1000);
  $('subDelaySlider').value    = ms;
  $('subDelayVal').textContent = fmtDelay(ms);
}
```

### `setSubMode(mode)` ([index.html:847-873](../templates/index.html#L847-L873)) — rewritten

```js
function setSubMode(mode) {
  const myId = ++S.subtitleMutation.id;
  update(draft => { draft.active.subtitle.mode = mode; });
  fetch('/api/subtitle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  }).finally(() => { S.subtitleMutation.committed = myId; });
}
```

Cache/restore logic gone. One fetch. Sync-id advances.

### Subtitle delay slider handlers ([index.html:941-968](../templates/index.html#L941-L968))

```js
subDelaySlider.addEventListener('input', () => {
  S.locks.subDelay = true;
  const s = parseInt(subDelaySlider.value) / 1000;
  update(draft => { draft.active.subtitle.srtDelay = s; });
});
subDelaySlider.addEventListener('change', () => {
  const myId = ++S.subtitleMutation.id;
  fetch('/api/subtitle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ srt_delay: S.active.subtitle.srtDelay }),
  }).finally(() => {
    S.locks.subDelay = false;
    S.subtitleMutation.committed = myId;
  });
});

function stepSubDelay(deltaMs) {  // debounced variant, same body shape
  const min = parseInt(subDelaySlider.min);
  const max = parseInt(subDelaySlider.max);
  const curMs = Math.round(S.active.subtitle.srtDelay * 1000);
  const nextMs = Math.max(min, Math.min(max, curMs + deltaMs));
  S.locks.subDelay = true;
  update(draft => { draft.active.subtitle.srtDelay = nextMs / 1000; });
  debounce(() => {
    const myId = ++S.subtitleMutation.id;
    fetch('/api/subtitle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ srt_delay: S.active.subtitle.srtDelay }),
    }).finally(() => {
      S.locks.subDelay = false;
      S.subtitleMutation.committed = myId;
    });
  }, 300, '_sub');
}
```

### Polling ([index.html:1000-1019](../templates/index.html#L1000-L1019))

```js
update(draft => {
  if (draft.active.playing && !d.playing) {
    resetState(draft);
    stopPolling();
    return;
  }
  draft.transport.position = d.position || 0;
  draft.transport.duration = d.duration || 0;
  if (!S.locks.masterVol) draft.system.masterVolume  = d.master_volume ?? 100;
  if (!S.locks.vocalVol)  draft.controls.vocalVolume = Math.round(d.vocal_volume * 100);

  // Subtitle: only sync when no in-flight mutation
  if (S.subtitleMutation.id === S.subtitleMutation.committed) {
    draft.active.subtitle.mode      = d.subtitle_mode;
    draft.active.subtitle.available = d.subtitle_available;
    if (!S.locks.subDelay) draft.active.subtitle.srtDelay = d.srt_delay;
  }
});
```

### `togglePlayPause()` play branch (~index.html:804)

Remove all the "freeze pending subtitle into active" code. Just handle video/dualStem:

```js
update(draft => {
  draft.active.playing   = true;
  draft.active.videoPath = draft.pending.video;
  draft.active.dualStem  = !!(draft.pending.vocal && draft.pending.nonvocal);
  draft.controls.vocalVolume = DEFAULT_VOCAL_VOL_PCT;
  draft.controls.pitch       = DEFAULT_PITCH_SEMITONES;
  // subtitle state will be populated by the next /api/status poll
});
```

The first poll after `/api/play` returns fills in `subtitle.mode`, `subtitle.available`, and `subtitle.srtDelay` from the server. Sync-id guard is `0===0` at this point, so sync proceeds.

### `resetState()` ([index.html:623-638](../templates/index.html#L623-L638))

```js
draft.active.subtitle = {
  mode: 'off',
  available: { ass: null, srt: null },
  srtDelay: DEFAULT_SUB_DELAY_MS / 1000,
};
S.subtitleMutation.id = 0;
S.subtitleMutation.committed = 0;
// remove: old subMode / companions.ass / companions.srt / srtDelayCache lines
```

### `sendFiles()` ([index.html:651-662](../templates/index.html#L651-L662))

Remove `subtitle` field from payload:
```js
body: JSON.stringify({
  video:    S.pending.video,
  vocal:    S.pending.vocal,
  nonvocal: S.pending.nonvocal,
}),
```

### `updateCompanionFiles()` ([index.html:663-687](../templates/index.html#L663-L687))

Simplify: populate only `S.pending.companions` for the browser preview. Remove `draft.pending.subMode` and `draft.pending.subtitle` writes — those fields no longer exist.

### Initialization ([index.html:640-648](../templates/index.html#L640-L648))

No structural change, but the initial `/api/status` fetch now also picks up `subtitle_mode`/`srt_delay`/`subtitle_available` if playback is already active on page load. Either reuse the polling path by calling `startPolling()` immediately (which already runs `/api/status`), or extend the existing one-shot status fetch to populate subtitle fields.

---

## Files to Modify

| File | Nature of changes |
|------|-------------------|
| [mpv/app.py](../app.py) | Add `/api/subtitle` + `_apply_subtitle_mode()`, remove `/api/sub_mode` and `/api/sub_delay`, update `state` dict defaults, update `/api/play` and `/api/status`, update `reset_state_defaults()`, strip `subtitle` from `/api/files` |
| [mpv/templates/index.html](../templates/index.html) | Replace subtitle slice of `S`, rewrite `setSubMode` and delay handlers, update `render()`, `resetState()`, polling sync, `togglePlayPause`, `sendFiles`, `updateCompanionFiles` |
| [mpv/mpv_controller.py](../mpv_controller.py) | **No changes.** Existing `sub_add` / `sub_remove` / `set_sub_delay` / `apply_srt_style` are sufficient. |

---

## Verification

1. `conda activate pik && cd mpv && python app.py`
2. Open `http://localhost:5000` in a browser on the same machine as the MPV window.
3. Test matrix (watch MPV window *and* UI state simultaneously):

   **Defaults / new song:**
   - [ ] Select a song with both ASS+SRT → play → mode defaults to karaoke, delay row hidden
   - [ ] Select a song with only SRT → play → mode defaults to SRT, delay row visible, default delay applied
   - [ ] Select a song with only ASS → play → mode karaoke
   - [ ] Select a song with neither → both mode buttons disabled, Off selected

   **Mode switching (the main bug targets):**
   - [ ] Karaoke → SRT → delay slider shows default, SRT renders
   - [ ] Adjust SRT delay to e.g. +1.5s → MPV timing visibly shifts
   - [ ] SRT → karaoke → delay row hidden, MPV shows ASS with zero delay
   - [ ] Karaoke → SRT → **delay slider shows +1.5s again, MPV reapplies it** ← broken case today
   - [ ] SRT → off → nothing displayed
   - [ ] Off → SRT → delay restored

   **Invariant checks:**
   - [ ] In karaoke mode, MPV `sub-delay` property is exactly `0.0` (verify via mpv IPC or logging)
   - [ ] On every new song start, `srt_delay` returns to `DEFAULT_SRT_SUB_DELAY` regardless of previous song's value

   **Races:**
   - [ ] Click karaoke/srt/off rapidly 5× in ~1s → UI ends on the last clicked mode, MPV matches, no stale state
   - [ ] Drag subDelay slider while another mode button is clicked → no slider jumping

   **Song lifecycle:**
   - [ ] Stop mid-song → all subtitle state resets to off, delay back to default
   - [ ] Play new song after stop → delay starts at default, not previous song's value
   - [ ] Natural song end (MPV reaches EOF) → poll detects stop, state resets

4. Reload browser mid-playback — initial status fetch should populate `subtitle.mode`/`available`/`srtDelay` correctly, UI matches MPV.

## Out of Scope

- vocal/nonvocal path handling in `/api/files` (still client-sent; could also be server-derived, but not this refactor)
- Multi-song queue / auto-advance behavior
- Changes to `MpvController` or the pipeline scripts at repo root
