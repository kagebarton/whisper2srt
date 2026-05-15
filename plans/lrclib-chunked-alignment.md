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
5. **Sweep `/api/get`** with cleaned `title` + `artist`, varying the
   `duration` param across a window around the file's true duration `D`.
   Collect unique returned rows; pick the one whose `duration` is
   closest to `D`. If the sweep returns nothing, fall back to
   `/api/search` as a variant-discovery backstop (it may surface an
   alternate title/artist spelling); otherwise, if the sweep returned
   rows but none are within the acceptable band, that's a weak match.
6. No LRCLIB match (or weak match) → log and fall through to tier 2.

**Why sweep `/api/get` instead of trusting `/api/search`.** LRCLIB's
`/api/get` is a duration-keyed lookup with ~±2s buckets per call: each
distinct duration value returns the closest-indexed recording within
that narrow window, and stepping the param exposes *different* track
ids. `/api/search` caps at 20 results ranked by relevance and silently
ignores any `duration` param — popular studio versions crowd out live /
remix / shortened mixes for the same title, even when those mixes are
the actual file in hand (the `blood.m4a` case). Sweeping `/api/get`
turns the endpoint into a range query and surfaces buried recordings
directly.

**Duration sweep parameters** (constants in `lrclib_align/lrclib.py`,
tunable later):
- Window: ±10s around the file duration (`SWEEP_WINDOW`).
- Step: 2s (`SWEEP_STEP`; ≈ one `/api/get` bucket per probe; 11 calls
  per sweep).
- Acceptable Δ: ≤10s = match; >10s on closest scanned row = weak
  (`ACCEPTABLE_DELTA`). Wide enough to absorb the
  intro/outro/applause variance between a studio recording and a live
  cut of the same arrangement.
- Max related-artist variants probed on escalation: 4
  (`MAX_ARTIST_VARIANTS`).
- HTTP timeout: 15s (`_HTTP_TIMEOUT`). LRCLIB has been observed
  responding at 10–15s under load; a tighter timeout starves all probes.

**Artist-variant escalation.** LRCLIB indexes the same recording under
multiple artist strings (e.g. "Ed Sheeran", "Ed Sheeran & Rudimental",
"Rudimental feat. Ed Sheeran"). Escalation trigger is **no within-
tolerance row** under the primary artist — an empty primary sweep AND
a primary sweep that only found out-of-tolerance rows both escalate.
We then do a title-only `/api/search`, harvest distinct artist names
that share a token-piece with the base (split on `&`, `,`, `/`,
`feat.`/`ft.`, `vs`, `x`), and re-sweep `/api/get` under each variant
up to `MAX_ARTIST_VARIANTS`. **Short-circuit:** as soon as any variant
produces a row within tolerance, remaining variants are skipped. The
title-only search ranks variants by relevance, so popular ones come
first and the short-circuit usually keeps the variant cost low.

**Final backstop.** If neither the primary sweep nor any probed
variant lands a within-tolerance row, run `/api/search` once with both
title and artist filters as a last resort. If even that doesn't yield
a within-tolerance row, return None — caller falls through to tier 2.

**Title normalization** (apply to both Genius title and LRCLIB candidate
title before comparing):
- Lowercase.
- Strip trailing parentheticals: `(feat. ...)`, `(with ...)`,
  `(... Remaster[ed])`, `(... Version)`, `(Live[ at ...])`,
  `(Acoustic)`, `(Remix)`.
- Strip trailing `- Remaster[ed] YYYY?`, `- Live`, `- Single Version`,
  `- Radio Edit` after a dash.
- Strip `feat.`/`ft.`/`featuring ...` from anywhere.
- Collapse whitespace.
- Strings still disagreeing → duration is the disambiguator.

**LRCLIB client:** inline `requests`-based wrapper in
`lrclib_align/lrclib.py` (~30 lines covering `/api/get` + `/api/search`).
`lrclib-python` on PyPI looks like the natural choice but every released
version (0.1–0.4.2) contains nested same-quote f-strings that require
Python ≥3.12; the `pik` env is on 3.11, so it fails to import. Rolling
our own is simpler than upgrading the env or pinning a different
library. No auth needed; no caching of the response for now.

## Reconciliation algorithm — write fresh (recommendation)

Do **not** reuse `_walk_align`. It is heavily tuned for word-token
matching against noisy whisper output (contraction tables, whisper-skip
budget, noise-absorption bias). Genius↔LRCLIB reconciliation is a
different problem: both sides are *clean* lyric text, at *line*
granularity, and the goal is fuzzy line-similarity matching with
gap-runs, not token equality against noise.

Write a small dedicated line-level aligner:

- **Similarity:** `rapidfuzz.fuzz.token_set_ratio` on lowercased,
  punctuation-stripped lines. Threshold: ≥75 = match, 60–75 = weak
  (match only if both neighbors agree), <60 = no match.
- **Algorithm:** Needleman–Wunsch over lines (match = +similarity,
  gap = small negative). Produces an alignment with gap runs on either
  side.
- **Contiguous-run threshold (cut section):** ≥3 consecutive unmatched
  Genius lines bounded by matched lines on both sides → treat as
  deliberately cut from this mix; drop from alignment.
- **Isolated unmatched (≤2):** keep the Genius line; interpolate timing
  from neighbors.

It is the highest-risk new component — bad reconciliation poisons every
chunk boundary — so it gets its own module and its own golden-output
tests, with `blood.m4a` as the real fixture.

## Phasing

### Phase 1 — fetch + duration gate (low risk, independently valuable)

- Interactive Genius search (prompt → top 10 → select → fetch lyrics +
  metadata). Implemented in `lrclib_align/search.py`; if
  `genius_align/genius.py` lacks a `/search` call, add it here, do not
  modify `genius_align`.
- `ffprobe` duration probe.
- LRCLIB fetch: duration-sweep `/api/get` → optional `/api/search`
  backstop → tolerance reject (per the sweep parameters above).
- **Scope of Phase 1 output:** still runs the existing whole-song
  alignment end-to-end (SRT/ASS); the only new behavior is the gate +
  the fetched bundle logged/saved for Phase 2 to consume. No chunking
  yet, no behavioral change when LRCLIB matches.
- Bundle shape: `(genius_lyrics, lrclib_synced, durations)`.

### Phase 2 — section chunking

**Integration shift.** Phase 1 hands off to `genius_align` via
`subprocess.call([sys.executable, "-m", "genius_align", ...])`. Phase 2
needs per-chunk control of `align()`, so this is replaced by
**library-level imports** of `genius_align` modules from inside
`lrclib_align/run.py`. The "no edits to `genius_align`" rule still
holds — Phase 2 only *imports* it. Prerequisite: a reuse audit
confirming the following surface is importable without changes:
- `genius_align.config.GeniusAlignConfig`
- `genius_align.workers.whisper_worker.WhisperWorker` (uses
  `load_model`/`align`/`refine`/`postprocess`/`transcribe`/`regroup`/
  `unload_model`; reload across chunks would be prohibitive, so the
  worker is loaded once and reused).
- `genius_align.word_extraction.{extract_words,load_genius_lyrics,
  match_words_to_lines_walk}`
- `genius_align.tiling_match.match_words_to_lines_tiling`
- `genius_align.genius.genius_singer_mode`
- `genius_align.caption.{generate_srt,generate_ass}`
- `genius_align.run.assign_speakers_from_genius`

**Input.** Phase 2 reads the Phase 1 sidecar
`<vocal_stem>.lrclib.json` (or accepts the equivalent in-memory bundle
from `run.py`). Keys consumed: `lrclib.synced_lyrics` (the LRC string),
`lrclib.duration`, `genius.title`, `genius.artist`, `file_duration`,
`lrclib.source` (logged when surprising chunk-boundary behaviour shows
up — `variant-sweep` / `search-backstop` spines are fuzzier than
`primary`). The `<vocal_stem>.genius.txt` written by Phase 1 has already
been run through `_strip_genius_chrome` (drops everything before the
first `[Section]` header, trims `\d*Embed` trailer) — so
`genius_align.genius.parse_genius_sections` sees clean Genius input with
section headers preserved, which is exactly what it expects.

**New module: `lrc.py`.** Parse `[mm:ss.cc] text` lines out of
`lrclib.synced_lyrics`. Returns `[{start: float, text: str}, ...]`.
Blank-text stamps (LRCLIB sometimes emits trailing empties as section
markers) are kept — they're useful end-of-section anchors. Multi-stamp
lines (`[00:01.00][00:30.00] chorus`) split into one entry per stamp.

**Reconciliation.** Line-level aligner (the fresh aligner above), in
`reconcile.py`. There is no separate "weak match" tier — `find_match`
returns `None` for any below-tolerance result, and the caller falls
through to tier 2 (whole-song tiling). If a third confidence tier is
later wanted, add a `confidence: float` field to `LrclibMatch`; do not
re-introduce the silent weak path.

- Derive section boundaries from Genius `[Verse]`/`[Chorus]` headers,
  timed via reconciled LRCLIB stamps. Sections **are** the chunks —
  no size targeting.
- **Floor-merge only:** if a section is <8s (e.g. a one-line `[Intro]`),
  merge it into the next section. Rationale: per-chunk `align()` has
  fixed overhead and very short audio gives the matcher little to work
  with. No upper bound — a long verse is still a bounded chunk, which
  is the whole point.
- **Overlap pad:** 1.5s on each side (clamped to chunk boundaries at
  song start/end). Implemented by cutting the stem with `ffmpeg` to a
  temp WAV per chunk (simplest reuse path — stable-ts inside
  `WhisperWorker` then runs unmodified against the bounded file).
- Per-chunk `align()` on bounded audio + that section's text; check
  `last_align_failure_ratio` per chunk; escalate only the failed chunk
  to tiling.
- **Stitch:** offset timestamps to absolute (add chunk start), then
  concatenate `line_objects`. Overlap-pad dedup: for words landing in
  the overlap region, keep the copy whose midpoint is nearer its chunk
  center; on exact tie, keep the earlier chunk's.
- Speaker assignment + SRT/ASS generation reused from `genius_align`
  (`assign_speakers_from_genius` + `generate_srt` / `generate_ass`).

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

## Prerequisites (resolve before coding)

- **`genius_align/genius.py` search capability** — confirm it has (or
  add) a function that takes free text and returns the top-10 Genius
  hits with `title`, `primary_artist`, `id`, `url`. Today it appears to
  parse a known URL/file.
- **`ffprobe` on `$PATH` in the `pik` conda env** — verify (the
  `audio-separator` pipeline implies it, but confirm).
- **New deps in `lrclib_align/requirements.txt`:**
  - `requests` — LRCLIB HTTP (inline client; see note above).
  - `lyricsgenius` — Genius search + lyrics fetch.
  - `rapidfuzz` — line-similarity scoring in `reconcile.py` (Phase 2).
- **`GENIUS_API_TOKEN` env var** — required by `lyricsgenius`.
- **Test fixtures committed/available:**
  - `blood.m4a` + its Genius lyrics + expected golden reconciliation
    output (cut-section case).
  - One clean-match song (LRCLIB exact hit, no cuts) for happy-path
    golden test.
- **No-LRCLIB regression baseline** — before any change that could
  perturb output, capture frozen `genius_align` SRT/ASS for a chosen
  song; this becomes the byte-identical comparison fixture.
- **`genius_align` is frozen** — do not modify it. `lrclib_align`
  imports/calls it as-is. Audit `caption.py`,
  `workers/whisper_worker.py`, walk/tiling matchers for hardcoded
  relative paths or `__main__`-only assumptions that block import from
  a sibling package; if any block reuse, work around them inside
  `lrclib_align` (thin wrapper, subprocess invocation, or local
  re-implementation) rather than editing `genius_align`.
- **CLI-only scope** — Phase 1–3 ship as CLI (`python -m lrclib_align`);
  no integration with the `mpv/` mixer web app in scope.

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

Reuse from `genius_align` **as-is, no edits**:
`workers/whisper_worker.py`, `caption.py`, `genius.py` parsing, the
walk/tiling matchers. If something there blocks reuse, wrap or
re-implement inside `lrclib_align` — never patch `genius_align`.
