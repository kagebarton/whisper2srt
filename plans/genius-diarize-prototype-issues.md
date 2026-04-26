# Genius Diarize Prototype — Potential Issues

Analysis of `plans/genius-diarize-prototype.md` against the existing
`diarized_captions/` codebase and real sample lyrics files.

---

## 1. Schema conflict: plan redefines `GeniusLine` differently from existing `genius.py`

**Plan §3** defines a new `GeniusLine` shape:

```python
{
  "line_text": str,
  "speaker_label": str | None,     # "Brian", "Kevin & AJ", "All", or None
  "dominant_speaker": str | None,   # first individual name, or None
  "is_ensemble": bool
}
```

**Existing `diarized_captions/genius.py`** emits:

```python
{
  "text": str,
  "section": str,
  "singers": list | None,           # ["Brian"] or ["Nick","All"]
  "is_ensemble": bool
}
```

The plan introduces `speaker_label` and `dominant_speaker` (computed fields)
while dropping `section` and `singers` (raw parsed fields). This is a
**hard incompatibility** — every downstream consumer (`caption.py`,
`speaker_labels.py`, `word_extraction.py`) currently reads `singers` and
`text`. If the new `genius_diarize/genius.py` is written to the plan spec,
none of the copied modules will work without modification.

**Impact**: High. Either the plan's schema must be reconciled with the
existing one (add the new fields, keep the old ones), or every copied module
must be rewritten to consume the new schema.

**Recommendation**: Keep the existing schema and add `speaker_label` and
`dominant_speaker` as **derived fields**. Do not drop `singers`, `section`,
or `text`. This avoids rewriting all downstream consumers.

---

## 2. `"All"` speaker_label is set but then treated as unlabeled — contradictory

**Plan §2** says: if first group is `"All"`, set `speaker_label="All"` and
`is_ensemble=True`. But **§5** (`assign_speakers_from_genius`) then assigns
`word["speaker"] = speaker_label` — which would be `"All"`, not `None`.

Later, **§4 step 5** says "set `word["speaker"]` to speaker label (or None
for ensemble/solo)". And **§7b** says "If speaker=None: no prefix" for SRT
and "use ensemble style" for ASS.

But the plan's own `assign_speakers_from_genius` implementation doesn't
produce `None` for "All" — it produces the string `"All"`. The existing
`caption.py` only treats `speaker=None` as ensemble; a string like `"All"`
would be treated as a named speaker with its own style/color.

**Impact**: High. `"All"` sections would get a dedicated ASS style named
`Karaoke_All` with a color from the palette, instead of the intended white
ensemble style. SRT would prefix lines with `"All: "` instead of leaving
them plain.

**Recommendation**: `assign_speakers_from_genius` must explicitly convert
`"All"` to `None` (or the plan must define `speaker_label=None` when
`is_ensemble=True`, making `"All"` an intermediate parse result only).

---

## 3. Named duets are labeled but caption.py can't style them correctly

**Plan §2** says named pairs like `[Bridge: Kevin & AJ]` get
`speaker_label="Kevin & AJ"`. The plan's ASS generation (§7b) then tries to
look up a style based on `dominant_speaker` ("Kevin") — which is a
**different string** from the `speaker` field on the line object
("Kevin & AJ").

The existing `caption.py:_generate_ass_events` derives the style name from
`line_obj["speaker"]` directly:

```python
style = f"Karaoke_{_safe_style_name(speaker)}"
```

So with `speaker="Kevin & AJ"`, it would look for `Karaoke_Kevin___AJ`
(not `Karaoke_Kevin`). The plan says it should use `Karaoke_Kevin` — but
that requires `caption.py` to be rewritten to resolve `dominant_speaker`
from the speaker label, not just use the label as-is.

Similarly, `_speaker_presence()` collects unique `line["speaker"]` values.
If two lines have `speaker="Kevin"` and `speaker="Kevin & AJ"`, they'll
appear as **two** distinct speakers in the palette, consuming two color
slots. The plan intends them to share "Kevin"'s color.

**Impact**: High. ASS output would have wrong/broken style references for
duet lines, and color assignment would be inconsistent (Kevin solo vs
Kevin & AJ treated as different speakers).

**Recommendation**: Either (a) set `line_obj["speaker"]` to
`dominant_speaker` (the individual name) instead of the full pair label,
losing the pair info in captions; or (b) refactor `caption.py` to
accept a separate `style_speaker` field that differs from the display
label; or (c) store `dominant_speaker` on the line object and use that
for style lookup.

---

## 4. `split_lines_at_speaker_boundaries` will split every Genius-attributed line into single-speaker sub-lines — making it a no-op

In the plan, `assign_speakers_from_genius` assigns the **same** speaker
label to every word in a line. Since there are no word-level speaker
changes within a line (unlike the pyannote pipeline where different words
in the same line can map to different clusters), `split_lines_at_speaker_boundaries`
will never find a boundary to split on. It's a pure no-op.

The only case where it *would* split is if a line has `speaker=None` for
some words and a name for others — but the plan's assignment logic doesn't
produce that (it's all-or-nothing per line).

**Impact**: Low. The function is harmless when it has nothing to split.
But including it in the pipeline is misleading — it suggests there's a
case where it matters. It adds dead code and confusion.

**Recommendation**: Remove `split_lines_at_speaker_boundaries` from the
plan's pipeline, or document explicitly that it's retained for forward
compatibility with future per-word attribution.

---

## 5. Line count mismatch risk is under-mitigated

**Plan §9A** acknowledges line count mismatches between Whisper-aligned
lines and Genius-parsed lines, and proposes "pre-process lyrics to remove
blank lines." But this doesn't address the real failure modes:

- **Parenthetical stripping inconsistency**: The existing `genius.py`
  strips fully-parenthesized lines (`_FULL_PAREN_RE`), which removes lines
  like `(ad-lib)` from the genius_lines list. But the plain text passed to
  Whisper still contains them. This creates a count mismatch.

- **Whisper word count errors**: `match_words_to_lines` uses word-count
  pairing. If Whisper hallucinates or drops words, the 1:1 pairing drifts
  silently — line N in `line_objects` no longer corresponds to line N in
  `genius_lines`. The existing `run.py` has an `assert len(line_objects) ==
  len(genius_lines)` guard, but the plan's `run.py` does not.

- **The plan's lyrics_lines extraction is different from the existing
  code**: The plan (§7) builds `lyrics_lines` by splitting on newlines and
  filtering blanks and headers, while `load_genius_lyrics()` builds
  `plain_text` from the parsed `genius_lines` (which already stripped
  parentheticals). If the plan uses its ad-hoc extraction for Whisper but
  `parse_genius_sections` for genius_lines, the line lists may diverge.

**Impact**: High. A count mismatch means `zip(line_objects, genius_lines)`
silently pairs the wrong lines with the wrong speakers. No crash, no
error — just wrong attribution throughout the rest of the file.

**Recommendation**: (a) Use the same code path to produce both the plain
text for Whisper and the genius_lines list (as `load_genius_lyrics` already
does). (b) Add the `assert len(line_objects) == len(genius_lines)` guard
from the existing `run.py`. (c) Ensure parenthetical stripping is
consistent between both paths.

---

## 6. First-group semantics for `speaker_label` conflicts with real-world headers

The plan's "first group determines the dominant voice" rule (§2) produces
incorrect results for several real headers in the sample files:

| Header | Plan's first group | Plan's speaker_label | Problem |
|--------|-------------------|---------------------|---------|
| `[Verse 3: AJ, AJ & Brian]` | `["AJ"]` | `"AJ"` | Ignores that Brian also sings this verse. The "AJ & Brian" group is silently dropped. |
| `[Chorus: AJ, All, Brian]` | `["AJ"]` | `"AJ"` | The header lists three groups (AJ, All, Brian), suggesting these lines rotate between singers. The plan assigns everything to AJ. |
| `[Bridge: Kevin, Kevin & AJ, All]` | `["Kevin"]` | `"Kevin"` | The header lists Kevin solo, then Kevin & AJ duet, then All — this is a sequential progression. The plan assigns everything to Kevin. |
| `[Verse 2: Mystery, Romance, Abby]` | `["Mystery"]` | `"Mystery"` | Three solo singers listed — they likely take turns. Plan assigns everything to Mystery. |
| `[Pre-Chorus: Jinu, Romance, All, Baby]` | `["Jinu"]` | `"Jinu"` | Four groups — plan gives everything to Jinu. |

In the existing `diarized_captions` pipeline, multi-group headers are
handled correctly because `is_ensemble=True` causes pyannote to resolve
per-word. The plan's approach of always taking the first group is a
significant regression for these headers.

**Impact**: Medium-High. For songs with rotating lead vocals within a
section (common in KPop and boy band music), the plan will incorrectly
attribute all lines in that section to the first-listed singer.

**Recommendation**: When a header has multiple groups (comma-separated),
treat the section as ensemble (`speaker_label=None`) rather than
attributing to the first group alone. The first-group rule should only
apply when there's a single group. Alternatively, document this as a
known limitation and recommend users split multi-group headers into
separate sections.

---

## 7. Edge case §9C contradicts the main attribution rule

**§9C** says:

> "Ensemble if *any* group is 'All' OR any group has multiple names
> (detected by `&`)."

This means `[Verse: Brian & AJ]` would be ensemble — but **§2** explicitly
says named pairs get `speaker_label="Brian & AJ"` (NOT ensemble). And the
example at §5 shows `[Bridge: Kevin & AJ]` producing `speaker_label="Kevin
& AJ"`, `is_ensemble=False`.

The contradiction is: §9C says `&` in any group → ensemble, but §2/§5 say
named pairs (groups with `&`) are labeled, not ensemble.

**Impact**: Medium. Implementers following §9C would produce different
output than implementers following §2/§5. Test cases would also conflict.

**Recommendation**: Remove or rewrite §9C. The correct rule (per §2) is:
- Single group with `&` → named pair (labeled, not ensemble)
- Multiple groups → depends on first group (as §2 defines)
- `"All"` as first group → ensemble

---

## 8. SRT truncation logic is not implemented in existing `caption.py`

The plan (§7b) describes SRT generation with full labels on first
appearance and truncated labels on subsequent appearances. The existing
`caption.py:generate_srt` always uses the full `line['speaker']` string —
there is no first-appearance tracking and no truncation.

The plan lists `truncate_speaker_label()` in `genius.py` but doesn't
specify **where** in the pipeline it's called. The `generate_srt` function
in `caption.py` would need to be substantially rewritten to track
first-appearance state across lines.

**Impact**: Medium. The feature is new (not a regression) but the plan
doesn't specify the implementation clearly enough to build it correctly.

**Recommendation**: Add a concrete spec for how `generate_srt` should
track first appearances. Options: (a) pre-process line_objects to add a
`is_first_appearance` flag per line, (b) pass a `seen_speakers` set into
`generate_srt`, or (c) build the truncation logic entirely within
`generate_srt`.

---

## 9. ASS color assignment by `dominant_speaker` is disconnected from data flow

The plan (§7b) says ASS colors are assigned based on "first-appearance
order of dominant speakers." But the `dominant_speaker` field only exists
on `genius_lines`, not on `line_objects`. After `assign_speakers_from_genius`
sets `line_obj["speaker"] = speaker_label`, the `caption.py` code has no
way to recover `dominant_speaker` — it only sees the full label string.

The plan says to "extract dominant speaker (first individual name) from
speaker_label" at render time, but this requires parsing the speaker label
string back into its constituent names — a fragile operation that
duplicates logic already done in `genius.py`.

**Impact**: Medium. Without `dominant_speaker` on the line object,
`caption.py` cannot correctly assign colors for duet lines.

**Recommendation**: Add a `dominant_speaker` field to each line object
during `assign_speakers_from_genius`, or add a `style_key` field that
captures the intended ASS style name.

---

## 10. `genius_singer_mode()` definition is subtly wrong

**Plan §5** defines:

> "multi": at least one line has speakers=[name] (single-name attribution).

But the existing `genius.py:genius_singer_mode()` returns `"multi"` if
**any** line has `singers is not None` — which includes ensemble lines
(e.g. `singers=["All"]` or `singers=["Nick", "All"]`).

The plan's stricter definition (single-name only) would return `"solo"`
for a file where every header has only ensemble attribution (e.g. all
headers are `[Chorus: All]`). This might be intentional, but it differs
from the existing behavior.

**Impact**: Low-Medium. Edge case only — most real files with attribution
have at least one single-name header. But the plan and existing code would
disagree on files like a hypothetical `[Verse: All]`-only song.

**Recommendation**: Clarify whether `"multi"` requires at least one
single-name attribution, or merely any attribution. If the former, the
plan's `genius_singer_mode` implementation needs to check for single-name
specifically, not just `singers is not None`.

---

## 11. Dropped `section` field breaks potential future features

The existing `genius.py` includes a `"section"` field on each line
(e.g. "Verse 1", "Chorus"). The plan's schema drops it. While not used
by current caption generation, `section` is valuable for:

- Debugging (knowing which section a mis-attributed line belongs to)
- Future features (section-based styling, section markers in SRT)
- Logging (the existing `run.py` doesn't log per-section, but could)

**Impact**: Low. No current functionality breaks. But dropping it loses
information that was already parsed for free.

**Recommendation**: Keep the `section` field in the genius_lines schema.

---

## 12. `word_extraction.py` changes are under-specified

**Plan §6** says to "remove `load_genius_lyrics()`" from
`word_extraction.py`, keeping `extract_words()`, `segments_to_line_objects()`,
and `match_words_to_lines()`. But the plan's `run.py` (§7) also uses
`load_genius_lyrics`-style logic (loading raw text, parsing genius sections,
producing plain text). Where does this live now?

The plan's `run.py` shows:

```python
lyrics_text = lyrics_path.read_text()
genius_lines = parse_genius_sections(lyrics_text)
# ...
lyrics_lines = [l.strip() for l in lyrics_text.split("\n")
                if l.strip() and not l.strip().startswith("[")]
```

This ad-hoc extraction is inferior to `load_genius_lyrics()` because:
- It doesn't strip fully-parenthesized lines (genius.py does)
- It constructs `lyrics_lines` independently from `genius_lines`, risking
  divergence (issue #5 above)
- It duplicates logic that already exists

**Impact**: Medium. The plan's `run.py` will produce different `lyrics_lines`
than `genius_lines` in edge cases, causing silent mis-alignment.

**Recommendation**: Keep `load_genius_lyrics()` (or an equivalent) in
`word_extraction.py`. Use the genius_lines output as the source of truth
for both the plain text passed to Whisper and the speaker attribution.

---

## 13. No `DiarizeConfig` removal plan — config.py changes are incomplete

**Plan §6** says "Remove `diarize` config section (no pyannote)" from
`config.py`. But `DiarizedCaptionsConfig.diarize` is referenced by:

- `run.py` line 117-119 (setting num/min/max_speakers)
- `run.py` line 137 (creating `CancelableDiarizeWorker(cfg.diarize)`)
- The `DiarizeConfig` dataclass itself

If the plan removes `DiarizeConfig` and the `diarize` field, the `run.py`
code can't even create a config. The plan doesn't show the updated
`config.py` or specify what replaces `diarize`-related CLI args.

Also, the plan's `run.py` still references `cfg.whisper` but never shows
how `cfg` is constructed. The existing code uses `DiarizedCaptionsConfig()`
which defaults to including `diarize`.

**Impact**: Medium. The plan can't be implemented as-is without resolving
the config structure.

**Recommendation**: Define the new `GeniusDiarizeConfig` dataclass
explicitly (or show the modified `DiarizedCaptionsConfig`), removing
`diarize` but keeping all caption styling fields.

---

## 14. `align_and_refine` API mismatch in plan's `run.py`

**Plan §7** shows:

```python
result = whisper_worker.align_and_refine(vocal_path, lyrics_text_stripped)
```

But the variable `lyrics_text_stripped` is never defined. The existing
code uses `lyrics_text` (the plain text from `load_genius_lyrics`).

Also, the plan then calls:

```python
words = extract_words(result)
line_objects = match_words_to_lines(words, lyrics_lines)
```

But `lyrics_lines` is a list of strings, while `lyrics_text_stripped`
(passed to Whisper) is a single string. The plan doesn't show how
`lyrics_lines` is derived from `lyrics_text_stripped` consistently.

**Impact**: Low-Medium. Variable naming inconsistency and missing
definition. Easy to fix but indicates the plan wasn't fully walked
through as executable code.

**Recommendation**: Walk through the plan's `run.py` as if it were real
code, defining every variable before use. Use `load_genius_lyrics()` or
equivalent to produce both `plain_text` (for Whisper) and `genius_lines`
(for attribution) from the same parse.

---

## 15. Real sample `back.txt` breaks the plan's first-group rule

Walking through `back.txt` with the plan's rules:

| Section header | First group | Plan assigns | What should happen |
|----------------|------------|-------------|-------------------|
| `[Verse 3: AJ, AJ & Brian]` | `["AJ"]` | All lines → "AJ" | Lines alternate: AJ solo then AJ+Brian duet |
| `[Chorus: AJ, All, Brian]` | `["AJ"]` | All lines → "AJ" | Lines rotate: AJ → All → Brian |
| `[Bridge: Kevin, Kevin & AJ, All]` | `["Kevin"]` | All lines → "Kevin" | Lines progress: Kevin → Kevin&AJ → All |
| `[Verse 4: Howie, Nick, All]` | `["Howie"]` | All lines → "Howie" | Lines rotate: Howie → Nick → All |
| `[Break: All, Nick]` | `["All"]` | All lines → None (ensemble) | Lines alternate: All → Nick |
| `[Chorus: Brian, All, Nick, AJ]` | `["Brian"]` | All lines → "Brian" | Lines rotate through 4 singers |

For 6 out of 11 sections in `back.txt`, the first-group rule produces
incorrect attribution. The existing pyannote pipeline handles these
correctly by resolving per-word.

**Impact**: High. The plan's core algorithm is insufficient for the
primary test file.

**Recommendation**: For sections with multiple groups, either (a) treat
as ensemble and leave unlabeled, or (b) attempt round-robin assignment
of groups to lines within the section. Option (a) is simpler and honest
about uncertainty. Option (b) is more useful but requires knowing how
many lines each group covers.

---

## 16. `truncate_speaker_label()` edge cases

The plan defines truncation rules:
- `"Brian"` → `"B"` (first letter)
- `"Kevin & AJ"` → `"K & A"` (initials joined with `&`)
- `"All"` → `"All"` (no truncation)

Unspecified cases:
- Multi-word names: `"Mary Jane"` → `"M"` or `"MJ"`?
- Names with hyphens/apostrophes: `"O'Brien"` → `"O"` or `"OB"`?
- `"Jean-Paul"` → `"J"` or `"JP"`?
- Three-person group: `"A & B & C"` → `"A & B & C"` or `"A & B & C"`? (all initials)
- What if truncation produces the same initial as another speaker? E.g.
  `"Brian"` → `"B"` and `"Brian Littrell"` → `"B"` — duplicate SRT prefixes.

**Impact**: Low-Medium. Edge cases, but the function needs a complete spec
to be implemented correctly.

**Recommendation**: Define truncation for: multi-word names, hyphenated
names, names with apostrophes, 3+ person groups, and initial collisions.

---

## Summary by severity

| # | Issue | Severity |
|---|-------|----------|
| 1 | Schema conflict with existing genius.py | High |
| 2 | `"All"` speaker_label vs None contradiction | High |
| 3 | Named duets break ASS style/color lookup | High |
| 5 | Line count mismatch under-mitigated | High |
| 15 | First-group rule fails on 6/11 sections in back.txt | High |
| 6 | First-group rule wrong for multi-group headers | Medium-High |
| 8 | SRT truncation logic not in existing caption.py | Medium |
| 9 | dominant_speaker not on line_objects | Medium |
| 12 | word_extraction.py changes under-specified | Medium |
| 13 | config.py changes incomplete | Medium |
| 7 | Edge case §9C contradicts main rule §2 | Medium |
| 10 | genius_singer_mode definition mismatch | Low-Medium |
| 14 | Variable naming/API mismatch in plan run.py | Low-Medium |
| 16 | truncate_speaker_label() edge cases | Low-Medium |
| 4 | split_lines_at_boundaries is a no-op | Low |
| 11 | Dropped section field | Low |
