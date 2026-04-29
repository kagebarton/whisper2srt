# Genius Diarize — Efficiency & Robustness Improvements

## Summary

The `genius_diarize` module is well-structured and already has solid foundations — the
NW alignment, section carry-forward, and contraction normalization are all good
design choices. Below are the most impactful improvements.

---

## Efficiency

### 1. Banded Needleman–Wunsch Alignment (High Impact)

**File:** `genius_diarize/word_extraction.py` — `match_words_to_lines`

The NW DP is **O(m·n)** where `m` = lyric tokens and `n` = whisper tokens. For a
4-minute song this is ~500×500 = 250K cells, each calling `_score()` which may run
`_levenshtein()` (O(k²) per token pair).

**Observation:** Lyrics and whisper produce tokens in monotonically increasing
order; the optimal alignment path rarely deviates more than ±10–15 tokens from the
diagonal. **Banding** reduces the DP to **O(m·w)** where `w` = band width (e.g., 20).

```python
BAND_WIDTH = 15  # tunable; 10–20 covers all realistic misalignment

def _needleman_wunsch_banded(lyric_norms, whisper_norms, band=BAND_WIDTH):
    m, n = len(lyric_norms), len(whisper_norms)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize first row/col within band only
    for i in range(1, min(band + 1, m + 1)):
        dp[i][0] = i * _GAP
    for j in range(1, min(band + 1, n + 1)):
        dp[0][j] = j * _GAP
    for i in range(1, m + 1):
        lo = max(1, i - band)
        hi = min(n, i + band)
        for j in range(lo, hi + 1):
            match = (dp[i - 1][j - 1] + _score(lyric_norms[i - 1], whisper_norms[j - 1])
                     if j > lo else None)  # dp[i-1][j-1] valid only in band
            # ...
```

**Expected speedup:** ~25× for a 500×500 song (250K → 10K cells).

### 2. Cache `_score` During Alignment (Medium Impact)

The same (lyric_tok, whisper_tok) pair appears in multiple DP cells. Add a simple
dict cache scoped to one alignment call:

```python
def _needleman_wunsch(lyric_norms, whisper_norms):
    score_cache = {}
    def cached_score(a, b):
        key = (a, b)
        if key not in score_cache:
            score_cache[key] = _score(a, b)
        return score_cache[key]
    # use cached_score in place of _score inside the DP
```

An `@functools.lru_cache` on `_score` itself would persist across calls, which is
wasteful (only one call per run). Scoped dict is cleaner.

### 3. Early-Exit in `_levenshtein` (Low Impact)

`_levenshtein` is only called to answer "distance ≤ 1?". Instead of computing the
full distance, return 2 as soon as any row's minimum exceeds 1:

```python
def _levenshtein_at_most_one(a, b):
    if abs(len(a) - len(b)) > 1:
        return False  # can't be ≤ 1
    # compute DP, but if any row has min > 1, return False early
```

This avoids completing the full DP for mismatched tokens.

### 4. (Minor) Contraction Normalization Direction

`_CONTRACTIONS` maps `"dont"` → `"do not"`, but if whisper says `"do" "not"`
(separate tokens) and lyrics say `"don't"` (one token), alignment fails because
the single lyric token tries to match two whisper tokens individually. A
pre-processing step that **expands** known contracted tokens **in the lyrics**
(before tokenization) would give `"do" "not"` on both sides, improving match
accuracy. This is correctness more than speed, but it reduces gaps/wasted DP
cells.

---

## Robustness

### 5. Section-Type Fuzzy Carry-Forward (Medium Impact)

**File:** `genius_diarize/genius.py` — `parse_genius_sections`

Currently `[Verse 2]` (bare, no attribution) does **not** inherit from
`[Verse 1: Brian]` because the section names differ. But in practice, all
`Verse*` sections often feature the same singer.

**Fix:** Strip trailing digits when looking up `section_history`:

```python
import re
_SEC_NUM_RE = re.compile(r'\s+\d+$')

def _section_base(name):
    return _SEC_NUM_RE.sub('', name).strip()

# In parse_genius_sections, for bare headers:
current_groups = (section_history.get(current_section)
                  or section_history.get(_section_base(current_section)))
```

This handles `Verse 1` → `Verse 2`, `Chorus 1` → `Chorus 2`, etc., while still
respecting exact name matches first.

### 6. Skip-Section Headers (Medium Impact)

Headers like `[Instrumental]`, `[Guitar Solo]`, `[Interlude]`, `[Spoken]` should
not reset `current_section` or `current_groups` — subsequent lyrical lines should
inherit the last real section's attribution:

```python
_SKIP_SECTIONS = {"Instrumental", "Guitar Solo", "Interlude", "Spoken", "Intro"}
# In the header branch:
if current_section in _SKIP_SECTIONS:
    continue  # don't change section or groups
```

### 7. Overflow Color Cycling (Low-Medium Impact)

**File:** `genius_diarize/caption.py` — `_generate_styles`

When there are more dominant speakers than `speaker_colors` entries, all overflow
speakers reuse the **last** color. This makes them visually indistinguishable.
Cycling through the palette is more useful:

```python
# Instead of:
cfg.speaker_colors[-1]
# Use:
cfg.speaker_colors[color_idx % len(cfg.speaker_colors)]
```

### 8. Output File Naming Edge Case

**File:** `genius_diarize/run.py`, lines 146–147

```python
srt_out = vocal_path.with_suffix(".diarized.srt")
```

`Path.with_suffix` replaces only the **last** suffix. If the vocal file is
`song.vocals.wav`, the output is `song.vocals.diarized.srt` — probably fine.
But if the vocal file has no suffix, `.with_suffix` appends to the **last** path
component, which could produce odd results for paths ending in `/`. Consider:

```python
stem = vocal_path.stem
srt_out = vocal_path.with_name(f"{stem}.diarized.srt")
ass_out = vocal_path.with_name(f"{stem}.diarized.ass")
```

This is more predictable regardless of source file naming.

### 9. Remove `sys.path.insert` Antipattern (Low Impact)

**File:** `genius_diarize/run.py`, line 22

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

This is fragile and breaks when `genius_diarize` is installed as a package. The
project should be run with `PYTHONPATH` or installed via `pip install -e .`.

### 10. Style Name Sanitization Fallback (Low Impact)

**File:** `genius_diarize/caption.py` — `_safe_style_name`

If a speaker label consists entirely of non-word characters (e.g., a Chinese name
in CJK characters), `\W` regex matches **all** of them, producing an empty string
→ style name `Karaoke_`. Add a fallback:

```python
def _safe_style_name(label):
    safe = _UNSAFE_CHAR_RE.sub('_', label)
    return safe.strip('_') or 'speaker'
```

### 11. Better Error Handling in `run.py` (Low Impact)

`align_and_refine` exceptions (model OOM, corrupted audio, whisper failure) are
caught only by the `finally` for model unloading. Adding explicit exception
handling with user-friendly messages would improve the experience:

```python
try:
    result = whisper_worker.align_and_refine(vocal_path, plain_text)
except FileNotFoundError:  # model file moved/deleted
    logger.error("Whisper model not found at configured path")
    sys.exit(1)
except RuntimeError as e:  # OOM, device errors
    logger.error(f"Alignment failed: {e}")
    sys.exit(1)
```

### 12. SRT Lyrics Input Guard (Low Impact)

If a user accidentally passes an `.srt` file as the lyrics input, the Genius
parser will treat timestamp lines (`[00:00:01,000]`) as section headers, producing
garbage output. A simple guard in `parse_genius_sections` or `run.py` could detect
and reject SRT content:

```python
if line.startswith("[") and re.match(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\]", line):
    # looks like an SRT timestamp, not a Genius header
    raise ValueError("Input appears to be SRT, not Genius-format lyrics")
```

---

## Not Worth Changing (Already Good)

| Area | Why |
|------|-----|
| Module-level regex compilation | ✅ Correct — compiled once |
| `finally: unload_model()` | ✅ Ensures GPU memory freed even on error |
| `_INLINE_PAREN_RE.sub` for align_text | ✅ Good heuristic for whisper alignment |
| Section carry-forward via `section_history` | ✅ Elegant — handles exact-name bare repeats |
| `dominant_speaker` dedup in ASS styles | ✅ Correctly shares color slot for solo+duet |
| `first_word_nudge_cs` timing tweak | ✅ Helps with segment-gap visual artifact |
| `AudioLoader` stderr patch in `WhisperWorker` | ✅ Thorough — no broken-pipe noise |