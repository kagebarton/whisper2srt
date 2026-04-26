# Genius Diarize Prototype

Use Genius.com section-header annotations as the **sole source of truth** for speaker attribution. No pyannote, no overlap detection, no cluster mapping — just parse headers and assign one speaker label per section under a conservative attribution rule.

## 1. Goal & scope

- **Input**: vocal stem audio + a Genius-formatted lyrics file (`.txt`).
- **Output**: `.diarized.srt` + `.diarized.ass` captions with speakers assigned per header.
- **Attribution rule**: For each lyric line, look at its section header.
  - **Single group, single name** (`[Verse 1: Brian]`): assign "Brian" to all words.
  - **Single group, named duet** (`[Bridge: Kevin & AJ]`): assign the pair "Kevin & AJ" to all words.
  - **Single group "All"** (`[Chorus: All]`) OR **multiple groups** (`[Chorus: Nick, All]`, `[Bridge: Brian, AJ]`, `[Verse: Brian & AJ, Nick]`): leave unlabeled (`speaker=None` → default/white color in ASS). Multi-group headers are treated as ensemble because line-to-group alignment within a section is unknown — guessing the dominant voice per line would be wrong as often as right.
  - **No attribution** (`[Verse 1]`, no `:`): unlabeled.
- **Fallback**: if no header has single-name or single-duet attribution, output is solo (all unlabeled).
- **Scope**: Genius headers only. No pyannote, no overlap detection, no cluster mapping.

### Out of scope (v1)
- Fetching lyrics from Genius API (user provides `.txt` with headers).
- Per-line attribution within multi-group sections (treated as ensemble).
- Fine-grained parenthetical attribution (`(ad-lib)` etc. inherit section's attribution).
- De-duplication of variant header names (`Kevin` vs `Kevon`).

---

## 2. Genius header format

**Grammar**:
```
HEADER     ::= "[" SECTION (":" ATTRIBUTION)? "]"
SECTION    ::= "Verse" | "Verse 1" | "Chorus" | "Bridge" | ... (free text)
ATTRIBUTION::= GROUP ("," GROUP)*
GROUP      ::= "All" | NAME ("&" NAME)*
NAME       ::= <free text up to "," / "&" / "]">
```

**Examples**:
- `[Verse 1: Brian]` — 1 group, single name → assign "Brian"
- `[Bridge: Kevin & AJ]` — 1 group, named pair → assign "Kevin & AJ"
- `[Chorus: All]` — 1 group, "All" → unlabeled
- `[Chorus: Nick, All]` — 2 groups → unlabeled
- `[Bridge: Brian, AJ]` — 2 groups → unlabeled
- `[Verse: Brian & AJ, Nick]` — 2 groups → unlabeled
- `[Pre-Chorus]` — no `:` → unlabeled

**Attribution semantics** — applied to the **set of groups** in the header:

| Group count | First group is `All`? | First group has `&`? | Result |
|-------------|----------------------|----------------------|--------|
| 0 (no `:`)  | —                    | —                    | unlabeled |
| 1           | yes                  | —                    | unlabeled (ensemble) |
| 1           | no                   | no                   | labeled with name (`speaker_label="Brian"`) |
| 1           | no                   | yes                  | labeled with pair (`speaker_label="Kevin & AJ"`) |
| ≥2          | —                    | —                    | unlabeled (ensemble) |

`is_ensemble` is True for any unlabeled-by-rule case. Caption code only inspects `speaker_label is None`; `is_ensemble` is internal/diagnostic.

---

## 3. Data structures

### `GeniusLine` (returned by `parse_genius_sections`)

```python
{
    "text": "You are my fire",            # Lyric text (parenthetical-stripped, see §6)
    "section": "Verse 1",                 # Section header name (debug/logging)
    "speaker_label": "Brian" | "Kevin & AJ" | None,
                                          # None for ensemble or no-attribution
    "dominant_speaker": "Brian" | "Kevin" | None,
                                          # First name in first group; None for ensemble
    "is_ensemble": False                  # True iff the rule resolves to unlabeled
}
```

**Semantics**:
- `text` — lyric text after fully-parenthesized-line stripping; matches the plain text fed to Whisper.
- `section` — header name, for debugging and future features (e.g., `Verse 3: 'Brian' assigned to 4 lines`).
- `speaker_label` — `None` for ensemble/no-attribution. Otherwise the full display label (single name or `"A & B"`).
- `dominant_speaker` — first individual name from the first group (e.g., `"Kevin"` from `"Kevin & AJ"`). Used for ASS color palette keying so a Kevin solo and a Kevin & AJ duet share one color slot. `None` whenever `speaker_label is None`.
- `is_ensemble` — derived flag, True whenever `speaker_label is None`.

### `line_objects` after speaker assignment

Each line and word carry both the display label (`speaker`) and the style key (`dominant_speaker`):

```python
# Single-name section
{
    "text": "You are my fire",
    "words": [
        { "word": "You", "start": 1.2, "end": 1.4,
          "speaker": "Brian", "dominant_speaker": "Brian", "is_segment_first": True },
        ...
    ],
    "speaker": "Brian",
    "dominant_speaker": "Brian",
    "start": 1.2, "end": 3.1
}

# Named-duet section
{
    "text": "Tell me why",
    "words": [
        { "word": "Tell", "start": 2.1, "end": 2.3,
          "speaker": "Kevin & AJ", "dominant_speaker": "Kevin", "is_segment_first": True },
        ...
    ],
    "speaker": "Kevin & AJ",
    "dominant_speaker": "Kevin",
    "start": 2.1, "end": 3.5
}

# Ensemble / no-attribution section
{
    "text": "All together now",
    "words": [
        { "word": "All", "start": 4.0, "end": 4.2,
          "speaker": None, "dominant_speaker": None, "is_segment_first": True },
        ...
    ],
    "speaker": None,
    "dominant_speaker": None,
    "start": 4.0, "end": 5.2
}
```

---

## 4. Workflow

1. **Load audio**: vocal stem at `vocal_path`.
2. **Load & parse lyrics — single source of truth**:
   - Read `.txt`.
   - `genius_lines, plain_text = load_genius_lyrics(lyrics_text)`.
   - `plain_text` is the `\n`-joined `text` field of `genius_lines`. Both outputs come from the same parse, guaranteeing 1:1 line correspondence.
3. **Detect mode**: `mode = genius_singer_mode(genius_lines)` — `"multi"` if any `speaker_label` is non-None, else `"solo"`.
4. **Whisper align**:
   - `result = whisper_worker.align_and_refine(vocal_path, plain_text)`.
   - `words = extract_words(result)`.
   - `lyrics_lines = [g["text"] for g in genius_lines]`.
   - `line_objects = match_words_to_lines(words, lyrics_lines)`.
   - `reset_segment_first_flags(line_objects)`.
   - **Guard**: `assert len(line_objects) == len(genius_lines)` to catch silent misalignment.
5. **Assign speakers** (multi mode only): copy `speaker_label` and `dominant_speaker` from each `genius_line` onto the matching `line_obj` and its words.
6. **Annotate display labels for SRT** (multi mode only): walk lines once, tracking per-`speaker_label` first appearance; populate `line_obj["display_label"]` (full first time, truncated thereafter, `None` for ensemble).
7. **Generate captions**:
   - `generate_srt(line_objects, cfg)` — uses `display_label` for prefixes.
   - `generate_ass(line_objects, cfg)` — keys styles/colors off `dominant_speaker`.

`split_lines_at_speaker_boundaries` is **not** called: this prototype assigns one speaker per whole line, so there are no mid-line boundaries.

---

## 5. New module: `genius.py`

Pure-Python module, only `re` as a dependency.

**Public API**:

```python
def parse_genius_sections(lyrics_text: str) -> list[dict]:
    """Parse Genius headers and return per-line attribution.

    Returns:
        List of GeniusLine dicts (one per emitted lyric line — blank lines and
        fully-parenthesized lines are skipped):
            { "text", "section", "speaker_label", "dominant_speaker", "is_ensemble" }

    Algorithm:
      1. Split on \\n; track current section header.
      2. For each line:
         - Header `[...]`     → parse section + groups; update current attribution.
         - Fully-paren `(...)`→ skip.
         - Blank              → skip.
         - Otherwise          → emit a GeniusLine using current attribution rules:
             * 0 groups            → speaker_label=None, dominant_speaker=None, is_ensemble=True
             * 1 group, "All"      → speaker_label=None, dominant_speaker=None, is_ensemble=True
             * 1 group, single     → speaker_label="Brian", dominant_speaker="Brian"
             * 1 group, pair       → speaker_label="Kevin & AJ", dominant_speaker="Kevin"
             * ≥2 groups           → speaker_label=None, dominant_speaker=None, is_ensemble=True
    """

def genius_singer_mode(genius_lines: list[dict]) -> str:
    """Return "multi" if any genius_line has speaker_label != None, else "solo".

    Note: an all-ensemble file (e.g., every header is `[Chorus: All]`) yields "solo" —
    there is nothing to label, so we skip the speaker-assignment pass.
    """

def split_groups(attribution: str) -> list[list[str]]:
    """Parse an attribution string into groups of names.

    Examples:
        "Brian"             → [["Brian"]]
        "Kevin & AJ"        → [["Kevin", "AJ"]]
        "Nick, All"         → [["Nick"], ["All"]]
        "Brian & AJ, Nick"  → [["Brian", "AJ"], ["Nick"]]
        "All"               → [["All"]]

    Splits on `,` first, then `&` within each group; trims whitespace.
    "All" is a literal token, not split further.
    """

def truncate_speaker_label(speaker_label: str) -> str:
    """Abbreviate a label for SRT subsequent appearances.

    Rules:
      - "All"               → "All" (no change)
      - Single name         → first letter of the first whitespace-delimited token.
                              ("Brian" → "B", "Mary Jane" → "M",
                               "O'Brien" → "O", "Jean-Paul" → "J")
      - Named pair "A & B"  → "<initial(A)> & <initial(B)>"
                              ("Kevin & AJ" → "K & A")
      - 3+ name group       → initial-join with " & "
                              ("A & B & C" → "A & B & C")

    Initial collisions (e.g., two singers named "Brian") are accepted in v1.
    """
```

**Implementation notes**:
- Header regex: `^\[([^\]]+)\]\s*$`; split on first `:` to separate section from attribution.
- Fully-parenthesized line regex: `^\(.*\)$` (matches existing `diarized_captions/genius.py`).
- Centralize attribution-rule application in a small helper to avoid drift across call sites.

### Example parse flow for `back.txt`

```
[Intro: AJ]                       → 1 group, single name
Yeah                              → speaker_label="AJ", dominant_speaker="AJ"

[Verse 1: Brian]                  → 1 group, single name
You are my fire                   → speaker_label="Brian", dominant_speaker="Brian"

[Chorus: AJ, All, Brian]          → 3 groups → ENSEMBLE
Tell me why                       → speaker_label=None, dominant_speaker=None

[Bridge: Kevin & AJ]              → 1 group, named pair
Now I can see                     → speaker_label="Kevin & AJ", dominant_speaker="Kevin"

[Bridge: Kevin, Kevin & AJ, All]  → 3 groups → ENSEMBLE
Show me the meaning               → speaker_label=None, dominant_speaker=None

[Pre-Chorus]                      → no `:`
Anything                          → speaker_label=None, dominant_speaker=None
```

---

## 6. Existing modules (copied & adapted)

### `config.py`

**Source**: `diarized_captions/config.py`.

**Changes**:
- Drop `DiarizeConfig` and the `diarize` field.
- New top-level dataclass `GeniusDiarizeConfig`:
  - `whisper: WhisperConfig` (unchanged)
  - `caption: CaptionConfig` (unchanged — fonts, colors, margins, `speaker_colors` palette)
- CLI args wire `whisper` and `caption` knobs only; no `--num-speakers` etc.

### `caption.py`

**Source**: `diarized_captions/caption.py`.

**Changes**:
- `_generate_ass_events` derives the style name from `line["dominant_speaker"]` (falls back to `line["speaker"]` if absent, for compat with non-Genius callers).
- The style/color palette is built off `dominant_speaker` so that "Kevin" solo and "Kevin & AJ" duet share one color slot.
- `generate_srt` reads `line["display_label"]` when present (set by run.py); otherwise falls back to `line["speaker"]`. Truncation policy lives in run.py, not caption.py.

### `word_extraction.py`

**Source**: `diarized_captions/word_extraction.py`.

**Changes**:
- Adapt `load_genius_lyrics()` to the new `parse_genius_sections` schema. Invariant: returns `(genius_lines, plain_text)` from a single parse so `plain_text` and `[g["text"] for g in genius_lines]` are 1:1.
- Keep `extract_words()`, `segments_to_line_objects()`, `match_words_to_lines()`, `reset_segment_first_flags()`.

### `workers/whisper_worker.py`

**Source**: `diarized_captions/workers/whisper_worker.py`. No changes.

---

## 7. New module: `run.py`

```python
def main():
    args = parse_args()                    # vocal_audio, lyrics_file
    cfg = GeniusDiarizeConfig.from_args(args)

    lyrics_text = args.lyrics_file.read_text()
    genius_lines, plain_text = load_genius_lyrics(lyrics_text)

    mode = genius_singer_mode(genius_lines)

    whisper_worker = WhisperWorker(cfg.whisper)
    whisper_worker.load_model()

    result = whisper_worker.align_and_refine(args.vocal_audio, plain_text)
    words = extract_words(result)
    lyrics_lines = [g["text"] for g in genius_lines]
    line_objects = match_words_to_lines(words, lyrics_lines)
    reset_segment_first_flags(line_objects)
    assert len(line_objects) == len(genius_lines), \
        f"line count mismatch: {len(line_objects)} vs {len(genius_lines)}"

    if mode == "multi":
        assign_speakers_from_genius(line_objects, genius_lines)
        annotate_display_labels(line_objects)

    srt_out = args.vocal_audio.with_suffix(".diarized.srt")
    ass_out = args.vocal_audio.with_suffix(".diarized.ass")
    srt_out.write_text(generate_srt(line_objects, cfg))
    ass_out.write_text(generate_ass(line_objects, cfg))

    whisper_worker.unload_model()
```

### Helper: `assign_speakers_from_genius(line_objects, genius_lines)`

```python
def assign_speakers_from_genius(line_objects, genius_lines):
    """Copy speaker_label and dominant_speaker from each genius_line onto the
    matching line_obj (and onto every word). Modifies in place.
    """
    for line_obj, gl in zip(line_objects, genius_lines):
        line_obj["speaker"] = gl["speaker_label"]
        line_obj["dominant_speaker"] = gl["dominant_speaker"]
        for word in line_obj["words"]:
            word["speaker"] = gl["speaker_label"]
            word["dominant_speaker"] = gl["dominant_speaker"]
```

### Helper: `annotate_display_labels(line_objects)`

```python
def annotate_display_labels(line_objects):
    """Walk lines in time order; first occurrence of each speaker_label gets the
    full label, subsequent occurrences get the truncated label. Sets
    line_obj["display_label"] (str | None).
    """
    seen = set()
    for line_obj in line_objects:
        label = line_obj.get("speaker")
        if label is None:
            line_obj["display_label"] = None
        elif label not in seen:
            seen.add(label)
            line_obj["display_label"] = label
        else:
            line_obj["display_label"] = truncate_speaker_label(label)
```

`split_lines_at_speaker_boundaries` is intentionally omitted (see §4): every word in a line shares the same speaker, so the function would be a no-op. Reintroduce it only if per-word attribution lands later.

---

## 7b. Caption generation: SRT and ASS

### SRT

`generate_srt` reads `line["display_label"]`:
- `None` → emit lyric text only.
- otherwise → emit `f"{display_label}: {lyric_text}"`.

First-appearance/truncation policy lives in `annotate_display_labels` (§7), keeping caption.py agnostic to Genius semantics.

### ASS

Color palette is built off `dominant_speaker`:
1. Walk `line_objects`; collect unique non-None `dominant_speaker` values in first-appearance order.
2. Assign `cfg.caption.speaker_colors[i]` to the i-th unique dominant speaker.
3. Emit one ASS style per unique dominant speaker (`Karaoke_<safe(dominant_speaker)>`) plus an ensemble style for `dominant_speaker is None`.
4. Each line's style is looked up by `line["dominant_speaker"]` (or ensemble style when None).

**Examples**:
```
Line A: speaker="Brian",       dominant_speaker="Brian"   → Karaoke_Brian, color[0]
Line B: speaker="Kevin & AJ",  dominant_speaker="Kevin"   → Karaoke_Kevin, color[1]
Line C: speaker=None,          dominant_speaker=None      → ensemble style (default/white)
Line D: speaker="Brian",       dominant_speaker="Brian"   → Karaoke_Brian, color[0] (reused)
```

---

## 8. File structure

```
genius_diarize/
├── README.md                    # Usage, example, architecture
├── config.py                    # GeniusDiarizeConfig (no DiarizeConfig)
├── caption.py                   # adapted: dominant_speaker for styles, display_label for SRT
├── genius.py                    # NEW: header parsing, mode detection, truncation
├── word_extraction.py           # adapted load_genius_lyrics; rest unchanged
├── run.py                       # NEW: orchestration + helpers
└── workers/
    └── whisper_worker.py        # unchanged
```

### `README.md`

```
# Genius Diarize Prototype

Speaker attribution from Genius.com section headers — no pyannote, no overlap
detection. One label per section, conservatively assigned.

## Usage

python -m genius_diarize.run <vocal_audio> <lyrics_file>

## Example

python -m genius_diarize.run song.m4a song.txt
# Produces: song.diarized.srt, song.diarized.ass

## Format

Lyrics file in Genius format:

[Verse 1: Brian]
You are my fire
The one desire

[Bridge: Kevin & AJ]
Now I can see

[Chorus: Nick, All]   ← multi-group → unlabeled (ensemble)
Tell me why

## How it works

1. Parse Genius headers into per-line attribution.
2. Single-name → label as that name. Named duet → label as "A & B".
   "All", multi-group, or no attribution → unlabeled.
3. Run Whisper to align words to lyric lines.
4. Generate SRT (full label first time, initials thereafter) + ASS
   (colors keyed by the first individual name in the header).
```

---

## 9. Edge cases & mitigations

### A. Line count mismatch
Mitigated by deriving `plain_text` and `lyrics_lines` from the **same** `parse_genius_sections` pass (§4 step 2/4), and asserting `len(line_objects) == len(genius_lines)` after `match_words_to_lines`. If the assert fires, the user gets a clear count mismatch instead of silent mis-attribution.

### B. Named duets
`[Bridge: Kevin & AJ]` → `speaker_label="Kevin & AJ"`, `dominant_speaker="Kevin"`. Labeled, NOT ensemble. SRT shows the pair on first appearance, then "K & A". ASS colors the line with Kevin's slot (shared with any Kevin solo lines).

### C. Multi-group headers
`[Chorus: Nick, All]`, `[Bridge: Brian, AJ]`, `[Verse: Brian & AJ, Nick]` — all → ensemble (None). Per the design decision in §1: line-to-group alignment within a section is unknown, so guessing the dominant voice would be wrong as often as right. Known accuracy tradeoff for multi-group sections; future versions could attempt round-robin if line counts per group are known.

### D. No headers at all
`parse_genius_sections` emits lines with `speaker_label=None`, `is_ensemble=True`. `genius_singer_mode` → "solo". Output is plain karaoke.

### E. Headers with no attribution (`[Verse 1]`)
Same as ensemble: `speaker_label=None`, `is_ensemble=True`. If all headers lack attribution, mode="solo".

### F. Parenthetical asides
v1 strips fully-parenthesized lines (matches existing `diarized_captions/genius.py`). Inline parentheticals (`Sing it (yeah)`) stay attached to the section's main attribution.

### G. Name variations
`"Brian"` and `"Brian Littrell"` are treated as distinct speakers (separate color slots). v1 has no fuzzy matching; users normalize the lyrics file if needed.

### H. Truncation edge cases
- Multi-word names: take initial of first token. `"Mary Jane"` → `"M"`.
- Hyphens / apostrophes: take first letter of token, ignore punctuation. `"O'Brien"` → `"O"`, `"Jean-Paul"` → `"J"`.
- 3+ name groups: initial-join with ` & `. `"A & B & C"` → `"A & B & C"`.
- Initial collisions: accepted in v1 (e.g., two "B"s).

### I. `section` field as debug aid
Preserved on each `genius_line` for log lines like `Verse 3: assigned 'Brian' to 4 lines`. Not consumed by caption code.

---

## 10. Testing plan

### Unit tests

**`test_genius.py`**:
- `test_parse_single_name` — `[Verse: Brian]` → speaker_label="Brian", dominant_speaker="Brian", is_ensemble=False
- `test_parse_named_duet` — `[Bridge: Kevin & AJ]` → speaker_label="Kevin & AJ", dominant_speaker="Kevin", is_ensemble=False
- `test_parse_all_only` — `[Chorus: All]` → speaker_label=None, is_ensemble=True
- `test_parse_multi_group_with_all` — `[Chorus: Nick, All]` → speaker_label=None, is_ensemble=True
- `test_parse_multi_group_named` — `[Bridge: Brian, AJ]` → speaker_label=None, is_ensemble=True
- `test_parse_multi_group_with_pair` — `[Verse: Brian & AJ, Nick]` → speaker_label=None, is_ensemble=True
- `test_parse_no_attribution` — `[Verse 1]` → speaker_label=None, is_ensemble=True
- `test_parse_blank_lines_skipped`
- `test_parse_paren_lines_skipped` — `(ad-lib)` not emitted
- `test_parse_section_field_kept`
- `test_genius_singer_mode_solo` — all None → "solo"
- `test_genius_singer_mode_multi` — at least one labeled → "multi"
- `test_genius_singer_mode_all_ensemble_is_solo` — only `[Chorus: All]` headers → "solo"
- `test_split_groups_*` — variants per docstring
- `test_truncate_single` — "Brian" → "B"
- `test_truncate_pair` — "Kevin & AJ" → "K & A"
- `test_truncate_all` — "All" → "All"
- `test_truncate_multiword` — "Mary Jane" → "M"
- `test_truncate_hyphen` — "Jean-Paul" → "J"
- `test_truncate_apostrophe` — "O'Brien" → "O"
- `test_truncate_triple` — "A & B & C" → "A & B & C"

**`test_assign.py`**:
- `test_assign_single_name` — labels propagate to all words
- `test_assign_duet` — pair label and dominant on words
- `test_assign_ensemble_none` — speaker=None, dominant_speaker=None on all words

**`test_display_labels.py`**:
- `test_display_first_full_then_truncated`
- `test_display_independent_per_speaker` — Brian and Nick tracked separately
- `test_display_none_passes_through`

**`test_caption_srt.py`** (with pre-set `display_label`):
- `test_srt_full_label_first`
- `test_srt_truncated_subsequent`
- `test_srt_no_label_when_none`

**`test_caption_ass.py`**:
- `test_ass_color_keyed_by_dominant_speaker` — Kevin solo and Kevin & AJ share one color slot
- `test_ass_color_first_appearance_order`
- `test_ass_ensemble_uses_default_style`

### Integration tests

**`test_integration_back.txt`**:
- `[Verse 1: Brian]` → labeled "Brian"
- `[Bridge: Kevin & AJ]` → labeled "Kevin & AJ", dominant "Kevin"
- `[Chorus: AJ, All, Brian]`, `[Verse 4: Howie, Nick, All]`, `[Bridge: Kevin, Kevin & AJ, All]` — all multi-group → ensemble (None)
- SRT: full labels on first appearance, truncated thereafter
- ASS: Kevin solo (if any) and Kevin & AJ share the same color slot

**`test_integration_solo.txt`** (no attribution): mode="solo", all unlabeled, plain karaoke.

**`test_integration_mixed.txt`**: mix of labeled and unlabeled headers; mode="multi"; only labeled sections get prefixes/colors.

---

## 11. Implementation order

1. Write `genius.py` (parse / mode / split / truncate); unit tests pass.
2. Copy & adapt `config.py`, `caption.py`, `word_extraction.py`, `whisper_worker.py`.
3. Write `run.py` with `assign_speakers_from_genius` + `annotate_display_labels`.
4. Run integration on `back.txt` + a real audio file; verify SRT/ASS.

---

## Summary

**Genius Diarize** parses Genius headers and assigns one speaker per section, conservatively:
- Single-name and named-duet headers are labeled.
- Multi-group, "All"-only, and unattributed sections are unlabeled (default color).
- Falls back to plain karaoke if no labeled sections exist.

`speaker_label` drives SRT prefixes (full → truncated on repeat). `dominant_speaker` keys ASS color slots so a soloist and their duet share one color. No pyannote, no overlap detection, no per-word attribution.
