# Centralized UI State Management

*Produced by Claude Opus 4 (Thinking) — April 2026*

Replace the scattered global variables in `index.html` with a single state store, a reactive render loop, and clean pending/active separation.

## Problem

Current state is spread across ~12 independent globals with no enforced relationships:

```
playing, seeking, files, _assData, _srtData, _activeAssData, _activeSrtData,
_srtDelay, adjustingVocalVol, adjustingSubDelay, adjustingMasterVol, state
```

This causes **settings leakage** because:

1. `files.vocal` / `files.nonvocal` are shared between pending and active — selecting a new video while playing immediately changes what `/api/volume` shows, and `vocalVolRow` visibility leaks
2. Subtitle data has a pending/active split (`_assData` → `_activeAssData`) but it was bolted on after the fact; the same pattern isn't applied to other controls
3. DOM updates happen from 4+ disconnected paths (event handlers, polling, play/stop, `_updateSubModeUI`), each responsible for remembering which elements to touch
4. Each slider has its own independent "user is dragging" boolean — easy to forget or mishandle

## Proposed Architecture

### Single State Store

```js
const S = {
  // ── What the user has selected for the NEXT play ──
  pending: {
    video: null,         // string path
    vocal: null,
    nonvocal: null,
    subtitle: null,      // resolved path (from subMode + companions)
    subMode: 'off',      // 'karaoke' | 'srt' | 'off'  (default chosen by companion availability)
    companions: {
      vocal:    { path: null, exists: false },
      nonvocal: { path: null, exists: false },
      ass:      { path: null, exists: false },
      srt:      { path: null, exists: false },
    },
  },

  // ── Snapshot of what is currently playing ──
  active: {
    playing: false,
    videoPath: null,
    dualStem: false,
    subMode: 'off',
    companions: {        // frozen copy of pending.companions at play time
      ass: { path: null, exists: false },
      srt: { path: null, exists: false },
    },
  },

  // ── Per-song controls (reset to defaults on each play) ──
  controls: {
    vocalVolume: DEFAULT_VOCAL_VOL_PCT,   // 0–100
    pitch: DEFAULT_PITCH_SEMITONES,       // integer semitones
    subDelay: DEFAULT_SUB_DELAY_MS,       // ms
    srtDelayCache: DEFAULT_SUB_DELAY_MS,  // stashed when leaving SRT mode
  },

  // ── System-wide (persists across songs) ──
  system: {
    masterVolume: 100,
  },

  // ── Playback transport ──
  transport: {
    position: 0,
    duration: 0,
  },

  // ── UI interaction locks (polling skips these while true) ──
  locks: {
    seeking: false,
    vocalVol: false,
    subDelay: false,
    masterVol: false,
  },
};
```

### Reactive Render

One `render()` function that reads `S` and updates every DOM element. Idempotent — safe to call from anywhere, anytime.

```js
function render() {
  const $ = id => document.getElementById(id);

  // ── Transport ──
  $('playPauseBtn').textContent = S.active.playing ? '⏸ Pause' : '▶ Play';

  // ── Seek ──
  if (!S.locks.seeking && S.transport.duration > 0) {
    $('seekBar').value = (S.transport.position / S.transport.duration) * 100;
  }
  $('timeElapsed').textContent = fmtTime(S.transport.position);
  $('timeTotal').textContent = fmtTime(S.transport.duration);

  // ── Vocal volume (only visible when active song is dual-stem) ──
  $('vocalVolRow').style.display = S.active.dualStem ? '' : 'none';
  if (!S.locks.vocalVol) {
    $('vocalVol').value = S.controls.vocalVolume;
    $('vocalVolVal').textContent = `${S.controls.vocalVolume}%`;
  }

  // ── Pitch ──
  $('pitchSlider').value = S.controls.pitch;
  $('pitchVal').textContent = SEMITONE_LABELS[String(S.controls.pitch)]
    || `${S.controls.pitch > 0 ? '+' : ''}${S.controls.pitch} st`;

  // ── Master volume ──
  if (!S.locks.masterVol) {
    $('masterVol').value = S.system.masterVolume;
    $('masterVolVal').textContent = `${S.system.masterVolume}%`;
  }

  // ── Subtitle mode ──
  const activeAss = S.active.companions.ass;
  const activeSrt = S.active.companions.srt;
  $('subBtnKaraoke').disabled = !activeAss?.exists;
  $('subBtnSrt').disabled     = !activeSrt?.exists;
  ['karaoke', 'srt', 'off'].forEach(m => {
    const key = m.charAt(0).toUpperCase() + m.slice(1);
    $('subBtn' + key).classList.toggle('active', m === S.active.subMode);
  });

  // ── Sub delay (only visible in SRT mode) ──
  $('subDelayRow').style.display = S.active.subMode === 'srt' ? '' : 'none';
  if (!S.locks.subDelay) {
    $('subDelaySlider').value = S.controls.subDelay;
    $('subDelayVal').textContent = fmtDelay(S.controls.subDelay);
  }

  // ── Companion file checkboxes (always reflect pending) ──
  const c = S.pending.companions;
  $('chkVocal').checked    = c.vocal.exists;
  $('lblVocal').textContent    = c.vocal.exists ? basename(c.vocal.path) : '—';
  $('chkNonvocal').checked = c.nonvocal.exists;
  $('lblNonvocal').textContent = c.nonvocal.exists ? basename(c.nonvocal.path) : '—';
  $('chkAss').checked      = c.ass.exists;
  $('lblAss').textContent      = c.ass.exists ? basename(c.ass.path) : '—';
  $('chkSrt').checked      = c.srt.exists;
  $('lblSrt').textContent      = c.srt.exists ? basename(c.srt.path) : '—';
}
```

### Mutate + Render Pattern

Every state change goes through a helper that mutates and re-renders:

```js
function update(mutator) {
  mutator(S);
  render();
}
```

Example — polling callback becomes:

```js
function onPollData(d) {
  update(S => {
    if (S.active.playing && !d.playing) {
      // Server says stopped — reset
      resetState(S);
      return;
    }
    S.transport.position = d.position || 0;
    S.transport.duration = d.duration || 0;
    S.system.masterVolume = d.master_volume ?? 100;
    S.controls.vocalVolume = Math.round(d.vocal_volume * 100);
    S.controls.subDelay = Math.round(d.subtitle_delay * 1000);
  });
}
```

No more `syncVocalVol()`, `syncSubDelay()`, `syncMasterVol()` — the lock-aware skip is inside `render()`.

---

## Data Flow Walkthroughs

### Select a new video (while another may be playing)

```
User picks file in browser
  └─ update(S => {
       S.pending.video = path;
     })
  └─ fetch /api/derive_files
  └─ update(S => {
       S.pending.companions = { vocal, nonvocal, ass, srt };
       S.pending.vocal    = vocal.exists    ? vocal.path    : null;
       S.pending.nonvocal = nonvocal.exists ? nonvocal.path : null;
       // Auto-select best subtitle mode
       S.pending.subMode = ass.exists ? 'karaoke' : srt.exists ? 'srt' : 'off';
       S.pending.subtitle = resolveSubtitlePath(S.pending);
     })
  └─ sendFiles()     // sends S.pending.{video, vocal, nonvocal, subtitle}
```

> **Key**: Nothing in `S.active` changes. The playing song's UI (vocal vol row visibility,
> subtitle buttons, sub delay row) is untouched because `render()` reads from `S.active`,
> not `S.pending`.

### Play

```
POST /api/play → success
  └─ update(S => {
       // Freeze pending into active
       S.active.playing = true;
       S.active.videoPath = S.pending.video;
       S.active.dualStem = !!(S.pending.vocal && S.pending.nonvocal);
       S.active.companions.ass = { ...S.pending.companions.ass };
       S.active.companions.srt = { ...S.pending.companions.srt };
       S.active.subMode = S.pending.subMode;

       // Reset per-song controls
       S.controls.vocalVolume = DEFAULT_VOCAL_VOL_PCT;
       S.controls.pitch = DEFAULT_PITCH_SEMITONES;
       S.controls.subDelay = DEFAULT_SUB_DELAY_MS;
       S.controls.srtDelayCache = DEFAULT_SUB_DELAY_MS;
     })
  └─ startPolling()
```

### Stop / Song End

```
update(S => resetState(S))

function resetState(S) {
  S.active.playing = false;
  S.active.videoPath = null;
  S.active.dualStem = false;
  S.active.subMode = 'off';
  S.active.companions = { ass: { path: null, exists: false }, srt: { path: null, exists: false } };
  S.controls.vocalVolume = DEFAULT_VOCAL_VOL_PCT;
  S.controls.pitch = DEFAULT_PITCH_SEMITONES;
  S.controls.subDelay = DEFAULT_SUB_DELAY_MS;
  S.controls.srtDelayCache = DEFAULT_SUB_DELAY_MS;
  S.transport.position = 0;
  S.transport.duration = 0;
  // S.system.masterVolume is NOT reset
  // S.pending is NOT reset — keeps the queued-up next song
}
```

### Slider interaction (unified pattern)

```js
// All sliders follow the same lock → update → unlock pattern.
// Example: vocal volume

vocalVol.addEventListener('input', () => {
  S.locks.vocalVol = true;
  update(S => { S.controls.vocalVolume = parseInt(vocalVol.value); });
});

vocalVol.addEventListener('change', () => {
  api.setVolume(parseInt(vocalVol.value)).finally(() => {
    S.locks.vocalVol = false;
  });
});
```

### Subtitle mode switch (during playback)

```js
function setSubMode(mode) {
  update(S => {
    // Cache/restore SRT delay
    if (S.active.subMode === 'srt' && mode !== 'srt') {
      S.controls.srtDelayCache = S.controls.subDelay;
    }
    if (mode === 'srt' && S.active.subMode !== 'srt') {
      S.controls.subDelay = S.controls.srtDelayCache;
    }
    if (mode !== 'srt') {
      S.controls.subDelay = 0;
    }
    S.active.subMode = mode;
  });

  const path = mode === 'karaoke' ? S.active.companions.ass?.path
             : mode === 'srt'     ? S.active.companions.srt?.path
             : null;
  api.setSubMode(path);
  api.setSubDelay(S.controls.subDelay / 1000);
}
```

---

## Files Changed

### `templates/index.html` (MODIFY)

The HTML structure stays the same. All JavaScript is restructured:

1. **Replace** all global variables with single `S` object
2. **Add** `render()` function (~45 lines) that is the sole DOM-writer
3. **Add** `update(mutator)` — mutate-then-render helper
4. **Add** `resetState(S)` — centralized reset
5. **Simplify** all event handlers to `update()` + `fetch` calls
6. **Delete** `syncVocalVol()`, `syncSubDelay()`, `syncMasterVol()`, `_updateSubModeUI()`, `_updateSubModeState()`, `_resetSongControls()`, `resetUI()`
7. **Consolidate** polling callback into `onPollData()` → `update()`

### No backend changes

`app.py` and `mpv_controller.py` are unchanged. The API contract stays the same.

---

## What This Fixes

| Bug | Why it happens now | Why it's fixed |
|-----|-------------------|----------------|
| Vocal vol row appears/disappears when browsing files mid-song | `vocalVolRow` visibility reads `files.vocal`/`files.nonvocal` which are pending | `render()` reads `S.active.dualStem`, which only changes on play |
| Subtitle buttons enable/disable when browsing files mid-song | `updateSubControl()` touches button disabled state from pending data | `render()` reads `S.active.companions`, frozen at play time |
| Sub delay row flickers on file selection | `updateSubControl()` calls `_updateSubModeState()` which changes `files.subtitle` | Pending subtitle mode lives in `S.pending.subMode`, render reads `S.active.subMode` |
| Polling overwrites slider mid-drag | Each slider has its own `adjusting*` flag, easy to miss | Unified `S.locks.*` checked in one place inside `render()` |
| Controls don't reset on natural song end | `resetUI()` is only called from `stopPlayback()`, not from the polling "server says stopped" path | `resetState(S)` is called from both paths via `update()` |

## Verification

- Select a new video while a song is playing → vocal volume row, subtitle buttons, and sub delay row should not change
- Let a song end naturally → all controls reset to defaults
- Stop a song → same reset behavior
- Drag a slider while polling is active → no visual fighting
- Switch subtitle mode mid-playback → delay row and delay value behave correctly
- Browse files, select a different song, then play → pending state correctly becomes active
