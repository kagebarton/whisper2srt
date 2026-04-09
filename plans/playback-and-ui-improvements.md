# Plan: Playback and UI Improvements

## Context

Four improvements to the karaoke mixer:
1. **Stem fallback** — When one or both stems are missing, fall back to the video's native audio track with rubberband pitch shifting instead of refusing to play.
2. **Conditional vocal volume** — Hide the vocal volume control when not in dual-stem mode.
3. **3-segment subtitle selector** — Replace the implicit ASS-preference logic with an explicit Karaoke / Subtitles / Off segmented control.
4. **Conditional sub delay** — Show the sub delay slider only when the Subtitles (.srt) mode is selected; hide it for Karaoke and Off.

---

## Files to Modify

- `mpv/app.py`
- `mpv/templates/index.html`

---

## app.py Changes

### 1. Add `dual_stem` to state dict (line ~64)

```python
"dual_stem": False,   # True when both vocal + nonvocal stems are loaded
```

Also reset it in `reset_state_defaults()`.

### 2. Rename / extend `build_filter_complex()` (line 174)

Rename to `build_filter(vocal_vol, pitch, dual_stem)`. Branch on mode:

**Dual-stem** (unchanged current behavior):
```
[aid2]volume@vocalvol={vocal_vol},rubberband@vocalrb=pitch={pitch}
  :window=long:pitchq=quality:transients=crisp:detector=compound
  :formant=preserved:channels=together[vocal];
[aid3]volume@nonvocalvol=1.0,rubberband@nonvocalrb=pitch={pitch}
  :window=standard:pitchq=quality:transients=crisp:detector=compound
  :formant=shifted[nonvocal];
[vocal][nonvocal]amix=inputs=2:normalize=0[ao]
```

**Single-stem** (new — best overall settings for mixed song audio):
```
[aid1]rubberband@rb=pitch={pitch}
  :window=long:pitchq=quality:transients=crisp:detector=compound
  :formant=preserved:channels=together[ao]
```

Update the three call sites (`/api/play`, `/api/volume`, `/api/pitch`) to pass `state["dual_stem"]`.

### 3. Relax the play guard (line 529)

Change:
```python
if not all([state["video_path"], state["vocal_path"], state["nonvocal_path"]]):
    return jsonify({"ok": False, "error": "Missing file selection"}), 400
```
To:
```python
if not state["video_path"]:
    return jsonify({"ok": False, "error": "Missing video path"}), 400
```

### 4. Branch `/api/play` on stem availability (lines 546-553)

```python
dual = bool(state["vocal_path"] and state["nonvocal_path"]
            and os.path.exists(state["vocal_path"])
            and os.path.exists(state["nonvocal_path"]))
state["dual_stem"] = dual

if dual:
    send_mpv_command({"command": ["audio-add", state["vocal_path"]]})
    send_mpv_command({"command": ["audio-add", state["nonvocal_path"]]})
    time.sleep(0.2)

fc = build_filter(1.0, semitones_to_pitch(0), dual)
send_mpv_command({"command": ["set_property", "lavfi-complex", fc]})
```

### 5. Include `dual_stem` in `/api/status` response (line 667)

Add `"dual_stem": state["dual_stem"]` to the status JSON so the UI can react to it.

---

## index.html Changes

### 1. Add `id` to vocal volume row (line 375)

```html
<div class="slider-row" id="vocalVolRow">
```

### 2. Show/hide vocal volume based on derive_files response (JS, line ~481)

In `updateCompanionFiles()`, after setting `files.vocal` / `files.nonvocal`:

```javascript
const dualStem = vocal.exists && nonvocal.exists;
document.getElementById('vocalVolRow').style.display = dualStem ? '' : 'none';
```

Also hide on page load if the server state has `dual_stem: false` (can check via inline template var or initial fetch).

### 3. Replace subtitle companion rows + wire-up with 3-segment control

**Remove** the current `chkAss` / `chkSrt` companion rows (they're read-only status indicators; Vocal and Nonvocal rows are sufficient status).

**Add** a segmented control between the Pitch row and the Sub Delay row:

```html
<!-- Subtitle mode selector -->
<div class="seg-row" id="subModeRow">
  <label>Subtitles</label>
  <div class="seg-ctrl">
    <button id="subBtnKaraoke" class="seg-btn" onclick="setSubMode('karaoke')">Karaoke</button>
    <button id="subBtnSrt"     class="seg-btn" onclick="setSubMode('srt')">Subtitles</button>
    <button id="subBtnOff"     class="seg-btn active" onclick="setSubMode('off')">Off</button>
  </div>
</div>
```

Add minimal CSS for `.seg-ctrl`, `.seg-btn`, `.seg-btn.active`, `.seg-btn:disabled`.

### 4. JS: `setSubMode(mode)` and `updateSubControl(ass, srt)`

```javascript
// Called by updateCompanionFiles after derive_files response
function updateSubControl(assData, srtData) {
  document.getElementById('subBtnKaraoke').disabled = !assData.exists;
  document.getElementById('subBtnSrt').disabled     = !srtData.exists;

  // Determine default: karaoke > srt > off
  let defaultMode = 'off';
  if (assData.exists) defaultMode = 'karaoke';
  else if (srtData.exists) defaultMode = 'srt';
  setSubMode(defaultMode, assData, srtData);
}

function setSubMode(mode, assData, srtData) {
  // assData/srtData can be cached from last derive_files call
  ['karaoke','srt','off'].forEach(m => {
    document.getElementById('subBtn' + m.charAt(0).toUpperCase() + m.slice(1))
      .classList.toggle('active', m === mode);
  });
  files.subtitle = mode === 'karaoke' ? assData?.path
                 : mode === 'srt'     ? srtData?.path
                 : null;
  // Show/hide sub delay
  document.getElementById('subDelayRow').style.display = mode === 'srt' ? '' : 'none';
  sendFiles();
}
```

Cache `_assData` / `_srtData` at module scope so the segment buttons can call `setSubMode` without re-fetching.

### 5. Add `id="subDelayRow"` to delay slider div (line 393)

```html
<div class="slider-row" id="subDelayRow">
```

Initial display is controlled by the server's `state.subtitle_path` on page load. Since the default subtitle priority is ASS (karaoke) first, the delay row should be hidden initially unless the active sub mode is SRT. Use Jinja: `style="{{ '' if state.subtitle_path and state.subtitle_path.endswith('.srt') else 'display:none' }}"`.

---

## Verification

1. **No stems** — select a plain `.mp4` with no companion files → Play works, pitch slider shifts pitch, vocal volume row is hidden, subtitle control shows all three buttons disabled except Off.
2. **Dual stems** — select a video with `---vocal.m4a` / `---nonvocal.m4a` → vocal volume row visible, pitch shifts both stems.
3. **Subtitle selection** — with both `.ass` and `.srt` present: Karaoke selected by default; switching to Subtitles reloads `.srt` as subtitle path; switching to Off hides sub delay row.
4. **Sub delay visibility** — only visible when Subtitles (.srt) is active; hidden for Karaoke and Off.
5. **Pitch in single-stem mode** — pitch slider applies rubberband to `[aid1]`.
