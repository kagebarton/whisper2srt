# LRCLIB-anchored chunked alignment

A new, separate prototype that uses LRCLIB synced lyrics to anchor and
chunk the alignment of Genius lyrics to a vocal stem. Lives at repo root
as `lrclib_align/` — it does **not** modify `genius_align/`.

## Motivation

`genius_align` derives all timing from a single whole-song whisper pass;
the lyrics file contributes only text + speaker attribution. The hard
failure mode is desync (whisper hallucination, skipped sections, remixes)
— the reason `tiling_match.py` and `--match-method auto` escalation exist.

LRCLIB provides per-*line* timestamps for a track. That is coarser than
whisper's per-word timing, so it is **not** a timing-precision upgrade.
Its value is as an independent anchor:

1. **Duration gate** — catch "wrong recording" (extended mix vs radio
   edit, cover, live) before wasting a whisper pass.
2. **Section chunking** — cut the audio on LRCLIB-derived boundaries so a
   whisper desync stays trapped in one chunk; `align()` runs on bounded
   audio + a small reference text and can't smear a missing line across
   the whole song.
3. **Line-time anchoring** — constrain the matcher's search window so it
   can't drift.

## Design principles

### Two tiers — the no-LRCLIB path is a first-class floor

- **LRCLIB found** (duration-matched studio mix): spine + section
  chunking + per-chunk `align()`.
- **LRCLIB not found** (live / remix / obscure): fall through to the
  current whole-song behavior — walk-with-tiling-escalation. A live or
  remixed recording has a different duration and content, so it *won't*
  duration-match LRCLIB and naturally routes here. This is exactly the
  case the tiling matcher was built for, and it must stay intact.

### Source hierarchy — audio is truth, LRCLIB describes the mix, Genius enriches

- **Audio** = ground truth.
- **LRCLIB** = describes *this mix* — which lines exist, in what order,
  roughly when. Its duration matched the video at fetch time.
- **Genius** = enrichment — section headers, speaker attribution, display
  text with parens. Describes the *canonical* song, which may be longer.

Reconciliation is therefore **LRCLIB spine + Genius overlay**, not a
merge of equals.

### Handling Genius/LRCLIB mismatches (e.g. `blood.m4a`)

`blood.m4a` is a shortened mix: LRCLIB has the short version, Genius has
the full song. A cut section shows up as a contiguous run of Genius lines
with no LRCLIB counterpart.

- **Contiguous run** of unmatched Genius lines → deliberately cut section
  → drop from alignment (not chunked, not fed to whisper).
- **Isolated** unmatched Genius line → the two sources just worded it
  differently → keep the Genius line, interpolate its timing.
- **Backstop:** per-chunk `align()` failure ratio. Wrongly *kept* a line
  not in the audio → that chunk fails → escalates to tiling → tiling
  drops it. Wrongly *dropped* one → chunk padding means whisper still saw
  the audio.
- **Log it:** `"N Genius lines (X–Y) had no LRCLIB counterpart — likely
  cut from this mix"` so a missing verse in the output is explained.
- Reverse mismatch (LRCLIB line absent from Genius): rare, low-stakes —
  keep it, attribute as ensemble, inherit section from neighbors.

## Input contract — one user search

The user performs a single interactive search; everything else is
automatic.

1. Prompt for free-text search terms.
2. Query Genius, present the **top 10 results** for the user to pick one.
3. Fetch the chosen result's lyrics + metadata (`title`, `artist`).
4. `ffprobe` the video/vocal file for exact duration.
5. Query LRCLIB `/api/get` with cleaned `title` + `artist` + duration.
   On 404, fall back to `/api/search` and pick the candidate whose
   `duration` is closest to the file; reject if nothing within tolerance.
6. No LRCLIB match (or weak match) → log and fall through to tier 2.

Title normalization: strip parentheticals, `feat.`/`ft.`, `- Live`,
remaster tags on **both** sides before comparing; duration is the
disambiguator when strings still disagree.

No caching of the LRCLIB response for now.

## Reconciliation algorithm — write fresh (recommendation)

Do **not** reuse `_walk_align`. It is heavily tuned for word-token
matching against noisy whisper output (contraction tables, whisper-skip
budget, noise-absorption bias). Genius↔LRCLIB reconciliation is a
different problem: both sides are *clean* lyric text, at *line*
granularity, and the goal is fuzzy line-similarity matching with
gap-runs, not token equality against noise.

Write a small dedicated line-level aligner: normalized line similarity
(token-set ratio or similar) + two-pointer/DP with gap runs. It is the
highest-risk new component — bad reconciliation poisons every chunk
boundary — so it gets its own module and its own golden-output tests,
with `blood.m4a` as the real fixture.

## Phasing

### Phase 1 — fetch + duration gate (low risk, independently valuable)

- Interactive Genius search (prompt → top 10 → select → fetch lyrics +
  metadata).
- `ffprobe` duration probe.
- LRCLIB fetch (`/api/get` → `/api/search` fallback → tolerance reject).
- Duration comparison: close → proceed; large mismatch → warn / abort /
  suggest tier-2. No chunking yet — Phase 1 still runs the existing
  whole-song alignment, it just adds the gate + the fetched synced data.
- Output: a validated `(genius_lyrics, lrclib_synced, durations)` bundle.

### Phase 2 — section chunking

- Line-level reconciliation (the fresh aligner above).
- Derive section boundaries from Genius `[Verse]`/`[Chorus]` headers,
  timed via reconciled LRCLIB stamps; merge tiny sections to ~15–40s.
- Cut the vocal stem per chunk with ~1–2s overlap pad (or stable-ts
  `clip_timestamps`).
- Per-chunk `align()` on bounded audio + that section's text; check
  `last_align_failure_ratio` per chunk; escalate only the failed chunk
  to tiling.
- Stitch: offset timestamps to absolute, concatenate `line_objects`,
  de-dup words landing in overlap pads (keep the copy nearer chunk
  center).
- Speaker assignment + SRT/ASS generation reused from `genius_align`.

### Phase 3 — line-time anchoring

- Thread reconciled per-line timestamps into the matcher as a soft
  prior: clamp candidate search windows to the LRCLIB-anchored time
  region, rejecting matches that imply drift.

## Guarantees / tests

- **No-LRCLIB regression test:** when no LRCLIB match exists, output must
  stay byte-identical to current `genius_align` whole-song behavior.
- **`--match-method tiling` hard override:** bypasses the LRCLIB path
  entirely even if a (bad) LRCLIB entry exists — always able to force
  canonical-lyrics-only behavior.
- **Reconciliation golden tests:** `blood.m4a` and at least one
  clean-match song; extend the existing `tests/test_genius_align_*`
  style.
- **"LRCLIB found" is confidence-gated**, not mere presence — weak match
  logs and falls through rather than trusting a shaky spine.

## Layout

```
lrclib_align/            # new, repo-root prototype — does not touch genius_align/
  __main__.py            # interactive CLI entry
  search.py              # Genius search + top-10 select + lyrics fetch
  lrclib.py              # LRCLIB /api/get + /api/search + duration gate
  reconcile.py           # fresh line-level Genius<->LRCLIB aligner
  chunk.py               # boundary derivation + audio cutting + stitch
  run.py                 # orchestration; reuses genius_align caption/workers
```

Reuse from `genius_align` where unchanged: `workers/whisper_worker.py`,
`caption.py`, `genius.py` parsing, the walk/tiling matchers.
