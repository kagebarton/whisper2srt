# Genius-aware Diarization

Use Genius.com section-header annotations to turn anonymous pyannote
clusters (`SPEAKER_00`, `SPEAKER_01`, …) into named singers (`Brian`,
`Nick`, …) and to mark ensemble sections (`All`, `Brian & AJ`)
authoritatively, instead of relying on overlap detection alone.

## 1. Goal & scope

- **Input**: vocal stem audio + a lyrics file. Any `.txt` lyrics file
  is assumed to be Genius-formatted (may or may not have headers).
  Providing a lyrics file always implies **alignment mode** — whisper
  aligns to the text; transcription mode is never combined with Genius.
- **Output**: same `.diarized.srt` + `.diarized.ass` files, but with
  named speakers and reliable ensemble markers.
- **Fallback**: if the lyrics file has no Genius headers (or none carry
  singer attribution), pipeline degrades to today's behavior:
  pyannote clusters → A/B/C, overlap detection → ensemble.

Lyrics files with headers but **no singer attribution** (e.g.
`[Verse 1]`, `[Chorus]`) signal a solo song — `diarize_worker.diarize()`
is not called and the output is plain, unlabeled karaoke. Both workers
still load at startup (no change to the loading sequence); only the
diarization call is skipped.

Out of scope:
- Fetching lyrics from Genius (assume the user provides a `.txt` file
  containing the page body verbatim, headers and all).
- Re-training/fine-tuning pyannote.
- Mapping when Genius attribution disagrees with pyannote *clustering*
  (e.g. if pyannote merged two singers into one cluster) — see §9.

---

## 2. Genius lyric format

Empirically the format is consistent enough to parse with a regex.
Sample (`back.txt`):

```
[Intro: AJ]
Yeah

[Verse 1: Brian]
You are my fire
The one desire
...

[Chorus: Nick, All]
Tell me why
...

[Verse 3: AJ, AJ & Brian]
Am I your fire?
Your one desire
...

[Bridge: Kevin, Kevin & AJ, All]
Now I can see that we've fallen apart
...
```

Header grammar (informal):

```
HEADER     ::= "[" SECTION ":" ATTRIBUTION "]"
SECTION    ::= "Verse" | "Chorus" | "Bridge" | "Break" | "Intro" | "Outro" | "Pre-Chorus" | ...
ATTRIBUTION::= GROUP ("," GROUP)*
GROUP      ::= NAME ("&" NAME)*
NAME       ::= "All" | <free text up to "," / "&" / "]">
```

Attribution semantics (best guess from the Backstreet Boys sample):

- A header lists every group of singers that appears in that section,
  in roughly the order they appear. Each `GROUP` is one configuration
  (solo, duet, or "All").
- Within the body of a section, parenthetical asides — `(Hey, yeah)`,
  `(Don't wanna hear you say)` — are typically the *backup/ad-lib*
  group. The plan does **not** try to attribute parentheticals
  separately in v1 (they ride the line's primary singer).
- `All` means ensemble — multiple simultaneous voices.
- A `GROUP` with two or more `NAME`s (`AJ & Brian`) is also ensemble.

Lines within a section without finer attribution inherit the section's
*first* group. This is a simplification; see §9 for accuracy
limitations.

---

## 3. Architecture

```
lyrics.txt ─► parse_genius_sections ─► [(line_text, [singer,...]), ...]
                                              │
vocal.wav  ─► whisper align ──────────────────┤
                                              ▼
vocal.wav  ─► pyannote diarize ─► turns ─► map_clusters_to_names
                                              │       (majority vote per
                                              │        Genius-labeled line)
                                              ▼
                                       cluster→name dict
                                              │
                                              ▼
                       assign_speakers_to_words (uses Genius attribution
                                              first, mapping table second,
                                              overlap intervals last)
                                              │
                                              ▼
                                       caption.py (emits names)
```

Genius is treated as **ground-truth attribution at the line level**.
Pyannote does timing and provides the cluster→name training signal.
Overlap detection (the work just merged) becomes a fallback for
unlabeled sections only.

---

## 4. Data structures

### Genius-parsed line
```python
{
    "text": str,                # lyric text (no header)
    "section": str,             # "Verse 1", "Chorus", ...
    "singers": list[str] | None,# ["Brian"] for solo, ["Nick","All"]
                                # for ensemble; None if section had no
                                # attribution at all (e.g. "[Verse 1]")
    "is_ensemble": bool,        # True iff len(singers) > 1 or "All" in singers
}
```

### Cluster→name mapping
```python
{
    "SPEAKER_00": "Brian",
    "SPEAKER_01": "Nick",
    ...
}
```
Built once after diarization, applied to every word.

### Word (extension)
```python
{
    "word": str,
    "start": float,
    "end": float,
    "is_segment_first": bool,
    "speaker": str | None,  # named singer ("Brian") or None for ensemble
}
```
The dict shape doesn't change — just the values are now names instead
of letters when Genius attribution is present.

---

## 5. New module: `genius.py`

Lives at `diarized_captions/genius.py`. Pure-python, no external deps
beyond `re`.

### Public API

```python
def parse_genius_sections(lyrics_text: str) -> list[dict]:
    """Parse Genius-formatted lyrics into per-line attribution.

    Returns a list of GeniusLine dicts (see §4) in document order, one
    per non-blank lyric line. Header lines are consumed (not emitted).

    If the file has no recognizable headers, every emitted line has
    singers=None — caller falls back to anonymous diarization.
    """

def genius_singer_mode(genius_lines: list[dict]) -> str:
    """Classify the song's singer attribution level.

    Returns one of:
      "solo"   — no headers at all, OR headers present but none carry
                 singer attribution; diarize_worker.diarize() is skipped
      "multi"  — at least one header carries singer attribution;
                 pyannote + cluster mapping should run

    This is the primary decision gate in run.py.
    """

def split_groups(attribution: str) -> list[list[str]]:
    """'Nick, AJ & Brian' -> [['Nick'], ['AJ', 'Brian']]"""
```

### Parsing rules

1. Walk lines top-to-bottom.
2. Header line matches `^\[([^:\]]+)(?::\s*(.+))?\]\s*$`.
   - Group 1 → section name.
   - Group 2 → attribution string (may be empty).
3. Empty lines are skipped (not emitted).
4. Non-header non-empty lines emit a `GeniusLine` carrying the *first
   group* of the current section's attribution.
5. Stripping: trim whitespace; strip a trailing/leading `(…)` wrapper
   only if the entire line is parenthesized (rare — usually
   parentheticals are mid-line).

### Single-singer detection

A `GeniusLine` is considered *attributed to one named singer* when
`len(singers) == 1` and `singers[0] != "All"`. This is the only case
that contributes to the cluster→name vote (§6).

---

## 6. Cluster → name mapping

New function in `speaker_labels.py`:

```python
def map_clusters_to_names(
    turns: list[dict],
    line_objects: list[dict],
    genius_lines: list[dict],
) -> dict[str, str]:
    """Build {pyannote_label -> singer_name} via majority vote.

    Algorithm:
    1. Pair line_objects with genius_lines by index (they're aligned
       1:1 — match_words_to_lines already preserved order).
    2. For each pair where genius_line is single-named:
       For each word in line_obj["words"]:
         find the pyannote turn covering that word's midpoint;
         credit (pyannote_label, singer_name) += word_duration.
    3. For each pyannote_label, pick the singer_name with max credit.
    4. Resolve collisions (two clusters mapping to the same name): keep
       the one with the higher credit; the other gets the runner-up
       name. If runner-up is also taken, fall back to letter (A/B/C).

    Returns {} if no single-named lines are available.
    """
```

The credit matrix is `dict[(label, name)] -> seconds`. After
construction:

```
votes = {
  ("SPEAKER_00", "Brian"): 47.2,
  ("SPEAKER_00", "Nick"):   3.1,
  ("SPEAKER_01", "Nick"):  31.9,
  ...
}
```

Pick the max per label, then de-duplicate names. The de-dup step
matters: pyannote sometimes produces extra clusters for the same
person (different mic/breathing/falsetto) and we don't want both
clusters to claim "Brian".

### Ambiguity threshold

If the winning name's credit is less than `2×` the runner-up's, log a
warning and **drop** that cluster from the mapping (it gets a letter
fallback). This catches under-determined clusters early instead of
silently mis-naming.

---

## 7. New `assign_speakers_to_words` flow

Rewrite signature and resolution order:

```python
def assign_speakers_to_words(
    line_objects: list[dict],
    turns: list[dict],
    overlap_intervals: list[tuple[float,float]] | None = None,
    genius_lines: list[dict] | None = None,
    cluster_to_name: dict[str,str] | None = None,
) -> list[dict]:
```

Per-word resolution, in order:

1. **Genius ensemble:** if the parent line's `genius_lines[idx]` has
   `is_ensemble=True` → `speaker = None`.
2. **Genius single-name:** if the parent line is single-named and that
   name is in `cluster_to_name.values()` → `speaker = name`. (We trust
   Genius even if pyannote's nearest cluster disagrees — Genius is
   ground truth at line granularity.)
3. **Pyannote + overlap:** existing logic — `_word_speaker(...)` with
   overlap intervals → either `cluster_to_name[label]` or `None`.
4. **Letter fallback:** if a label isn't in `cluster_to_name` (e.g.
   dropped by ambiguity threshold), use the appearance-order letter
   (existing `remap_speakers_by_appearance` output).

Steps 1–2 require pairing `line_objects[i]` with `genius_lines[i]`,
which the alignment-mode caller already enforces (count-based pairing
in `match_words_to_lines`).

---

## 8. File-by-file changes

| File | Change |
|------|--------|
| `diarized_captions/genius.py` | **NEW** — `parse_genius_sections`, `split_groups`. |
| `diarized_captions/word_extraction.py` | `load_lyrics` already returns text; add `load_genius_lyrics(path) -> (lyrics_text, genius_lines)` that calls `parse_genius_sections` and also returns a flattened plain-text version (header-stripped) so whisper alignment still works. |
| `diarized_captions/speaker_labels.py` | Add `map_clusters_to_names`. Extend `assign_speakers_to_words` per §7. Keep existing letter-based path intact for the no-Genius case. |
| `diarized_captions/caption.py` | Generalize from "single character letter" to "speaker label string." Style names become `Karaoke_<safe_label>` where unsafe chars are replaced (e.g. `Karaoke_AJ`, `Karaoke_Brian`). Colors assigned by first-appearance order into `cfg.speaker_colors` palette — same logic as letter assignment but keyed by label string. The `single_speaker` heuristic still applies. |
| `diarized_captions/config.py` | No changes needed — `speaker_colors` palette is reused as-is. |
| `diarized_captions/run.py` | Three-branch dispatch on `genius_singer_mode()` — see §8a. |

### §8a — `run.py` dispatch

Both workers always load at startup. Mode only affects whether
`diarize_worker.diarize()` is called.

```
lyrics_text, lyrics_format = load_lyrics(lyrics_path)
genius_lines = parse_genius_sections(lyrics_text)    # always called
mode = genius_singer_mode(genius_lines)

"solo"   → no headers, OR headers with zero attribution
             whisper align (strip any headers for word count)
             diarize_worker.diarize() NOT called
             all words stay unlabeled → single_speaker caption path
             → plain unlabeled karaoke output
             logs: "Solo mode — skipping diarization."

"multi"  → headers with attribution
             whisper align (strip headers)
             diarize_worker.diarize() called
             map_clusters_to_names → cluster_to_name dict
             assign_speakers_to_words with Genius + pyannote paths
             → named-speaker labeled output
```

---

## 9. Edge cases & decisions

### "solo" mode false positive
A multi-artist song where Genius happened to omit attribution in all
headers (uncommon but possible). The log message makes it visible:
"No singer attribution in headers — solo mode, skipping diarization."
The user can work around it by adding a single dummy attribution to one
header (e.g. `[Intro: Unknown]`), which flips mode to "multi".

### Genius/pyannote count mismatch
Genius lists 5 singers; pyannote detects 3. Two singers will not be
mapped. Their words at line level still get `speaker=name` from the
Genius single-name path (§7 step 2), but their pyannote-cluster
counterparts (if any) will appear under letter fallbacks in unlabeled
sections. Acceptable for v1.

Inverse case (pyannote sees 5, Genius lists 3): extra clusters get
letter fallbacks. The user can constrain pyannote with `--num-speakers
N` derived from the Genius header count — see §10.

### Pyannote merges two singers into one cluster
The two singers' words within Genius single-named lines pull the
cluster in opposite directions. The 2× ambiguity threshold (§6) will
likely drop the cluster, and **both** singers get correct names from
the Genius path on labeled lines. Unlabeled lines belonging to that
cluster fall back to a letter. Logged.

### Parentheticals like `(Don't wanna hear you say)`
v1 treats them as part of the line's primary singer. Real attribution
is usually a backup voice or "All." Refining this is out of scope but
listed under future enhancements (§11).

### Section without attribution `[Verse 2]`
`singers=None`. The line is treated as "not Genius-attributed" — falls
through to the pyannote+overlap path (§7 step 3).

### "All" as the *only* group
`singers=["All"]`, `is_ensemble=True`. Same handling as multi-group:
`speaker = None`.

### Name colors
`caption.py` builds a `{label: color}` dict by walking `present`
(first-appearance order from `_speaker_presence`) and indexing into
`cfg.speaker_colors`. Letter fallbacks and named speakers share the
same palette — whatever labels appear first get the earlier colors.
No config changes needed.

### ASS style names
ASS style names are ASCII-safe. We sanitize: `re.sub(r'\W', '_', name)`
so `O'Brien` becomes `Karaoke_O_Brien`. Collisions are extremely
unlikely in practice; if they happen, log and append a numeric suffix.

### SRT prefix length
Today's SRT prefix is one letter (`A:`); a name prefix can be 5–10
characters (`Brian:`). That changes line layout for some readers but
is the desired behavior here. No mitigation in v1.

---

## 10. Optional: feed singer count back to pyannote

`parse_genius_sections` can return a `unique_named_singers` count
(union of all singletons and group members across the whole song,
excluding "All"). `run.py` can pass this to pyannote as
`min_speakers=count` (not `num_speakers=count` — Genius might miss a
guest vocalist; min is safer). This is a small, optional upgrade that
typically improves pyannote's clustering on short songs.

Default off; opt-in via `--genius-constrain-speakers`.

---

## 11. Future enhancements (not v1)

- Parenthetical-aware attribution (split a line into mainline +
  backup phrases; the backup phrase becomes ensemble).
- Per-word attribution via word-level position within a multi-group
  section header (e.g. `[Verse 3: AJ, AJ & Brian]` — assume the first
  half is AJ solo and the second half is the duet).
- Genius API integration (auto-fetch + cache by song title/artist).
- Fine-tuning pyannote embeddings on songs with confirmed Genius
  attribution as weak labels.
- Confidence scoring per cluster→name mapping, surfaced in logs and
  in an optional `.diarized.json` sidecar.

---

## 12. Test plan

- **Unit**: `parse_genius_sections` against `back.txt` and a header-less
  fallback file. Assert:
  - 5 distinct named singers detected (AJ, Brian, Nick, Kevin, Howie).
  - Verse 1 lines all → `["Brian"]`.
  - Bridge first group → `["Kevin"]`; ensemble flag set on lines after
    a comma group.
  - "Pre-Chorus" / "Outro" sections still parse.

- **Unit**: `map_clusters_to_names` with synthetic `turns` and
  `genius_lines` covering:
  - Clean 1:1 mapping (each cluster wins a unique name).
  - Two clusters competing for the same name → de-dup picks the
    winner; loser gets letter.
  - Ambiguous cluster (votes ~50/50) → dropped, falls back to letter.

- **Integration**: run `python -m diarized_captions.run vocal.m4a
  back.txt` against the Backstreet Boys vocal stem and verify:
  - `.srt` shows `Brian:`, `Nick:`, `AJ:`, `Kevin:`, `Howie:` instead of
    `A:`/`B:`/etc.
  - Chorus and bridge ensemble lines have no prefix.
  - Singer colors are stable across the song.

- **Unit**: `genius_singer_mode` against:
  - `selfish.txt` (headers, no attribution) → `"solo"`
  - plain `.txt` with no headers → `"solo"`
  - `back.txt` (headers + attribution) → `"multi"`

- **Integration**: run `python -m diarized_captions.run vocal.m4a
  selfish.txt` — verify diarize_worker is never instantiated (log
  line absent), output has no speaker prefixes, `.ass` uses the
  single-speaker goldenrod style.

- **Regression**: run with a header-less lyrics file → output is
  byte-identical to today's letter-based output.

---

## 13. Open questions

1. **What if Genius lists `[Verse 3: AJ, AJ & Brian]` and a line
   inside is actually solo Brian (not in any listed group)?** v1
   assigns it to AJ (first group), which is wrong. Needs the
   per-word-position heuristic in §11.

3. **Does count-based `match_words_to_lines` survive Genius headers?**
   Yes, because we strip headers before passing the text to whisper
   alignment. The word counts in the stripped text and the parsed
   Genius lines must agree — the parser produces the same lines whisper
   aligns to. If they ever disagree (e.g. blank-line policy mismatch),
   we get an off-by-one across the song. Worth a guard:
   `assert len(line_objects) == len(genius_lines)` in `run.py` and
   abort with a clear error rather than silently mis-attributing.
