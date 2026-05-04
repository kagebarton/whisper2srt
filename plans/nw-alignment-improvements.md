# NW Alignment Improvements

Targeted accuracy improvements to the Needleman-Wunsch matcher in
`genius_diarize/word_extraction.py`, prioritized for vocal-stem input
with Genius-formatted lyrics (parens included as first-class tokens).

Revised after review (`nw-alignment-improvements-review.md`).

## Files touched

- `genius_diarize/word_extraction.py` — algorithm changes
- `tests/test_word_extraction.py` — test additions

## Implementation order

Land in this order so each step is independently validatable:

1. **Change 3** (zero-score filter in `match_words_to_lines`) — pure
   filtering on top of unchanged scoring. Validates the fix in
   isolation against current behavior. Doesn't alter NW DP.
2. **Change 1** (anchor bonus + `_PHONETIC_EQUIV`) — only changes
   `_score()`. Filter from step 1 automatically picks up the new +1
   vocal-equivalence scores without re-tuning.
3. **Change 2** (Sakoe-Chiba banding) — restructures the NW DP. The
   filter and scoring changes are already in place, so banding can be
   compared against a known-good baseline.
4. Tests for each change.
5. Run full `tests/test_word_extraction.py` suite. All existing tests
   should pass unchanged.
6. Sanity-check on a real song (vocal stem + Genius lyrics).

---

## Change 3 — `match_words_to_lines()`: filter zero-score pairs

(Listed first because it lands first.)

### Current behavior

```python
for l_idx, w_idx in alignment:
    if l_idx is not None and w_idx is not None:
        lyric_to_whisper[l_idx] = w_idx
```

Records every paired NW result, including pairs where `_score == 0`
(mismatches that NW chose because surrounding context made them the
locally-optimal path).

### Target behavior

```python
for l_idx, w_idx in alignment:
    if l_idx is not None and w_idx is not None:
        # _score() is stateless; recomputing here gives the same
        # value used during NW DP fill. If _score() ever becomes
        # context-aware, switch to returning (l_idx, w_idx, score)
        # triples from _needleman_wunsch().
        if _score(lyric_norms[l_idx], whisper_norms[w_idx]) >= 1:
            lyric_to_whisper[l_idx] = w_idx
```

Lyric tokens that NW paired at score 0 fall through to the existing
"start/end is None → interpolate from neighbors" branch.

### Rationale

Concrete failure case the filter fixes:

- Lyrics: `love me tender`
- Whisper (vocal stem): `love XYZ tender` (XYZ = backing vox or hallucination)
- Without filter: NW picks `love-love(+2), me-XYZ(0), tender-tender(+2) = +4`, pairing "me" to XYZ's timestamp.
- With filter: NW finds the same structural path; the `me-XYZ` pair has score 0 so it's not recorded. "me" gets interpolated timing from neighbors.

### Why not adjust the mismatch DP penalty instead?

I considered making `mismatch = -1` or `-2`. Tracing the DP shows it
doesn't fix the case:

- `mismatch = -1`: pair=-1+2=+1, gap-pair=-3+2=-1 → mismatch wins
- `mismatch = -2`: pair=-2+2=0, gap-pair=-3+2=-1 → mismatch wins
- `mismatch = -4`: pair=-4+2=-2, gap-pair=-3+2=-1 → gap wins

To prevent NW from absorbing the hallucination, mismatch needs to be
< `_GAP_LYRIC + _GAP_WHISPER = -3`. That's harsh enough to break
legitimate single-mismatch cases. The filter cleanly separates
**structural alignment** (NW's job) from **timestamp transfer** (the
filter's job) without penalty-tuning.

### Known edge case (documented, not fixed)

When the filter rejects a pair `(l_idx, w_idx)`, the **whisper** token
at `w_idx` is still consumed by NW's monotone alignment. It's no
longer available to be matched with any other lyric token (which is
fine — NW already decided it has no better match), but it also won't
appear in any line's word list. For hallucinations this is desired
behavior. For the rare case where a real whisper word is paired with
a lyric word at score 0 and a hallucination separately gapped, the
real whisper word is silently dropped. Interpolated timing is usually
close enough that this doesn't show up as a visible bug. Add a comment
at the filter site noting this.

### Filter threshold semantic note

The threshold `>= 1` currently accepts both Levenshtein-1 fuzzy
matches and (after Change 1) `_PHONETIC_EQUIV` vocal-sound
equivalences. Both are higher-trust than mismatch but lower-trust than
exact match. If a future change wants to filter more aggressively
(e.g. accept only score `>= 2`), `_score()` will need to return a
match type alongside the score so fuzzy and vocal-equiv can be
distinguished. Add a comment at the threshold site so this isn't
re-derived later.

---

## Change 1 — `_score()`: anchor bonus + `_PHONETIC_EQUIV`

### Current behavior

```
+2  exact normalized match
+2  contraction equivalence (1:1 or 1:N split)
+1  Levenshtein-1, both tokens len ≥ 3
 0  mismatch
```

### Target behavior

```
+3  exact match, len ≥ 6                      [NEW: anchor bonus]
+2  exact match, len < 6                       (unchanged)
+2  contraction equivalence                    (unchanged)
+1  phonetic equivalence (_PHONETIC_EQUIV)     [NEW]
+1  Levenshtein-1, both tokens len ≥ 3         (unchanged)
 0  mismatch                                   (unchanged)
```

### `_MAX_MATCH` constant

Add a module-level constant:

```python
# Maximum value _score() can return. Used by _needleman_wunsch to
# size the _NEG_INF sentinel and to guard the gap-cost invariant.
# UPDATE this if any branch of _score() returns a higher value.
_MAX_MATCH = 3

# Invariant: a single match must never fully cancel a gap-pair.
# Otherwise NW becomes indifferent between matching a token and
# gapping both sides of it. Must hold whenever _score / gap costs
# are tuned.
assert _MAX_MATCH < -(_GAP_LYRIC + _GAP_WHISPER), \
    "Single match score must be < gap-pair cost"
```

With `_GAP_LYRIC = -2`, `_GAP_WHISPER = -1`, gap-pair cost is 3, so
`_MAX_MATCH = 3` is on the boundary. The strict inequality `< 3`
fails. Two options:

**Option A (chosen):** weaken the invariant to `_MAX_MATCH <= -(_GAP_LYRIC + _GAP_WHISPER)`
and add the test from Issue X.2 to verify that anchor-only paths
don't pull alignment off-diagonal in practice. Banding (Change 2)
provides the real defense — the invariant is a sanity check, not the
mechanism.

**Option B (deferred):** lower `_GAP_WHISPER` to -2 to make gap-pair
cost 4. Risk: changes drop-cost for whisper tokens that don't match
anything, which would force more producer ad-libs / Genius omissions
into mismatches. Defer until Change 2 ships and we see if anchor
drift is actually observed.

### Anchor bonus

```python
if lyric_tok == whisper_tok:
    return 3 if len(lyric_tok) >= 6 else 2
```

Long words ("California", "everywhere", "tomorrow") are stronger
anchors than short function words ("the", "you", "and"). The +3
anchor pinning combined with banding (Change 2) is the chorus-drift
fix.

### `_PHONETIC_EQUIV`

Hand-maintained set of frozenset pairs for short phonetic
equivalences the Levenshtein-1 fuzzy gate (`len ≥ 3`) excludes:

```python
# Phonetic equivalence pairs (vowel-elongation and a few spelling
# variants) that the Levenshtein-1 fuzzy gate (len >= 3) excludes.
# Score returned: +1 (same tier as fuzzy).
#
# NOTE: pairs are NOT transitive. If "mm"~"mmm" and "mmm"~"mmmm" are
# both needed later, add all three pairwise entries explicitly, OR
# refactor to a canonical-form dict:
#     _PHONETIC_CANON = {"mm": "mm", "mmm": "mm", "mmmm": "mm", ...}
#     # match: _PHONETIC_CANON.get(a) == _PHONETIC_CANON.get(b) != None
#
# NOTE: this lookup MUST run after the exact-match branch.
# frozenset({"mm", "mm"}) collapses to frozenset({"mm"}) and won't
# match any pair entry — exact match catches that case at +2 first.
#
# Known gap NOT in this list: "na"/"nah". "nah" is a real English
# word meaning "no", so blanket equivalence with "na" produces false
# positives outside parenthesized vocal-run contexts. Revisit if
# na-na chorus patterns show real misalignment.
_PHONETIC_EQUIV = {
    # vowel-elongation (one side len 2, other len 3)
    frozenset({"mm", "mmm"}),
    frozenset({"uh", "uhh"}),
    frozenset({"oo", "ooh"}),
    frozenset({"hm", "hmm"}),
    frozenset({"wo", "woo"}),
    frozenset({"oh", "ooh"}),
    frozenset({"ah", "aah"}),
    frozenset({"ha", "hah"}),
    frozenset({"ay", "ayy"}),
    # spelling variant (Levenshtein-2; both sides len 4)
    frozenset({"woah", "whoa"}),
}
```

Inserted between the contraction block and the Levenshtein block in
`_score()`:

```python
pair = frozenset({lyric_tok, whisper_tok})
if pair in _PHONETIC_EQUIV:
    return 1
```

### Why a whitelist instead of lowering the fuzzy gate to len ≥ 2?

False-positive rate at len 2 is high: `it/is`, `in/on`, `me/be`,
`my/by`, `am/an`, `at/as` are all Levenshtein-1 distinct words. The
whitelist captures the legitimate short-token cases without admitting
those.

### Why not differentiate vocal-equiv from fuzzy in the score?

Both share +1 (same trust tier). The filter (Change 3) treats them
identically. If future logic needs to distinguish them, refactor
`_score()` to return `(score, match_type)` tuples — see comment at
filter site.

### Contraction ordering note

The contraction branch returns +2 before the anchor-bonus check could
fire. All current `_CONTRACTIONS` keys and split values are < 6 chars,
so no contraction match would benefit from the +3 bonus anyway. If
6+ char contractions are added later, reorder the branches or merge
the checks. Add a one-line comment.

---

## Change 2 — `_needleman_wunsch()`: Sakoe-Chiba band with fallback

### Current behavior

Full O(m·n) DP fill. No constraint on diagonal deviation.

### Target behavior

Constrain the inner DP to a band of width
`band = max(50, max(m, n) // 4)` around the expected diagonal.
Out-of-band cells stay at `_NEG_INF`. **Skip banding entirely for
short sequences** and **fall back to unbanded NW** if the banded
result looks degenerate.

### Banding threshold

```python
_BAND_MIN_LENGTH = 500  # only band sequences longer than this

if max(m, n) <= _BAND_MIN_LENGTH:
    return _needleman_wunsch_unbanded(lyric_norms, whisper_norms)
```

For typical song lengths (100–300 lyric tokens, sometimes up to
~1000), unbanded full DP costs single-digit ms. Banding's main wins
(speed at long-form, drift prevention) only matter above this
threshold. Skipping banding for short sequences avoids the entire
class of "true alignment exits the band" failure modes for the common
case.

If empirical results show chorus drift on songs below the threshold,
lower `_BAND_MIN_LENGTH` or set it to 0.

### `_NEG_INF` derived from `_MAX_MATCH`

```python
# Sentinel for out-of-band cells. Must stay more negative than any
# achievable real path value, even after _MAX_MATCH bonuses are added
# during transitions. The * 2 margin keeps the invariant under
# future scoring tweaks.
_NEG_INF = -(m + n + 1) * _MAX_MATCH * 2
assert _NEG_INF + _MAX_MATCH * max(m, n) < 0, \
    "Sentinel insufficient for current scoring constants"
```

### Banded DP fill with last-K-rows force-fill

Force-filling only the last row creates a cliff: the penultimate row
is still banded, so if the last lyric token needs to match a whisper
token far from the diagonal, the diagonal predecessor cell in row
`m-1` is `_NEG_INF` and the path can only be reached via in-band
left/up transitions, which may not exist.

Use a taper instead: expand the band linearly over the last K rows
to reach full width. K = `min(band, m // 4)`.

```python
m, n = len(lyric_norms), len(whisper_norms)

if max(m, n) <= _BAND_MIN_LENGTH:
    return _needleman_wunsch_unbanded(lyric_norms, whisper_norms)

band = max(50, max(m, n) // 4)
taper_rows = min(band, m // 4)
_NEG_INF = -(m + n + 1) * _MAX_MATCH * 2

dp = [[_NEG_INF] * (n + 1) for _ in range(m + 1)]
for i in range(m + 1):
    dp[i][0] = i * _GAP_LYRIC
for j in range(n + 1):
    dp[0][j] = 0   # free whisper prefix

for i in range(1, m + 1):
    # round() uses banker's rounding (round-half-to-even); the
    # 0.5-token jitter is absorbed by band width >= 50.
    j_exp = round(i * n / m)

    # Taper: widen the band over the last `taper_rows` rows so the
    # free-suffix region is reachable without a cliff.
    rows_from_end = m - i
    if rows_from_end < taper_rows:
        extra = (taper_rows - rows_from_end) * (n - band) // taper_rows
        eff_band = band + max(0, extra)
    else:
        eff_band = band

    j_lo = max(1, j_exp - eff_band)
    j_hi = min(n, j_exp + eff_band)

    for j in range(j_lo, j_hi + 1):
        match_s  = dp[i - 1][j - 1] + _cached_score(lyric_norms[i - 1], whisper_norms[j - 1])
        delete_s = dp[i - 1][j]     + _GAP_LYRIC
        insert_s = dp[i][j - 1]     + _GAP_WHISPER
        dp[i][j] = max(match_s, delete_s, insert_s)
```

### Degenerate-result fallback

Banded NW is only optimal **among paths that stay within the band**.
If the true optimal alignment exits the band (e.g., singer skipped a
long verse, pushing j-deviation beyond `band`), the banded result is
suboptimal and may be visibly wrong.

Detect via final score, not alignment shape:

```python
# Heuristic: a healthy alignment scores at least one match per ~3
# lyric tokens (most lyrics get matched on a clean vocal stem).
# If the score is much worse, the band likely clipped the true path.
expected_floor = (m / 3) * 2 + (m * 2 / 3) * _GAP_LYRIC  # rough
if dp[m][best_j] < expected_floor:
    logger.warning(
        "Banded NW score %d below expected floor %d; re-running unbanded",
        dp[m][best_j], expected_floor,
    )
    return _needleman_wunsch_unbanded(lyric_norms, whisper_norms)
```

The `expected_floor` formula assumes ~1/3 of lyric tokens match (+2
each) and ~2/3 are gapped (-2 each), giving a very loose lower bound.
Tune after observing real-song scores.

### Memory note

The implementation allocates a full `(m+1) × (n+1)` matrix even
though only `O(m * band)` cells are filled. For typical song-length
inputs (m, n < 2000), this is negligible. For long-form inputs
(audiobooks at 10000+ tokens), switch to a row-band representation:
store `2 * (2 * band + 1)` rows for the recurrence plus a full-row
buffer for traceback. Defer until needed.

### Edge cases handled

- **Free prefix** (whisper has lead-in tokens): `dp[0][j] = 0` for
  all j. For small i the band always reaches j=1 because `j_exp -
  band ≤ 0` clamps to 1. Preserved.
- **Free suffix** (whisper trails after last lyric line): the taper
  widens the band over the last `taper_rows` rows so all of row m's
  cells have valid in-band predecessors. `best_j` scan in traceback
  works as before.
- **Singer skips a section** (lyric tokens with no whisper match):
  the "delete" (lyric-gap) path stays available within the band, so
  inline gapping works. If the skip is so large that the alignment
  needs to leave the band, the degenerate-result fallback catches it.

---

## Decisions explicitly NOT in this plan

These were considered and dropped:

### Lower fuzzy gate to `len ≥ 2`

False-positive rate is too high: `it/is`, `in/on`, `me/be`, `my/by`,
`am/an`, `at/as` are all Levenshtein-1 distinct words.
`_PHONETIC_EQUIV` covers the legitimate short-token cases.

### Tighten `_GAP_WHISPER` to -2

Even with Genius coverage, producer ad-libs and breath sounds Genius
didn't document still need to be droppable cheaply. Asymmetric -2/-1
stays. Listed as **Option B** under the `_MAX_MATCH` invariant — defer
unless Change 2 + the gap-cost invariant assertion show drift.

### Line-locality tie-break in NW traceback

Banding (Change 2) addresses the same concern (chorus drift) more
directly. Add traceback complexity only if banded results still show
line-boundary leakage. **Diagnostic:** when assigning whisper indices
to lines, log a warning if a line's matched whisper indices are
non-monotonic relative to the previous line's indices. That's the
signal that line-locality would fix.

### Per-word probability-weighted gap penalty

Already filtered via `_MIN_WORD_PROBABILITY = 0.0001` in
`extract_words`. On vocal stems with VAD, residual mid-band low-
confidence tokens are mostly real audio that Genius covers via
parens.

### Affine gap penalty (open + extend)

With VAD on (default), instrumental sections produce no whisper
output, so long runs of whisper-gaps don't accumulate. Linear penalty
is fine.

### Returning `(score, match_type)` from `_score()`

Would let the filter and downstream code distinguish trust tiers
(exact, contraction, vocal-equiv, fuzzy). Premature: current code has
no use for the distinction. Comment at the filter site flags this as
the refactor path if needed.

---

## Tests to add

In `tests/test_word_extraction.py`:

### `TestScore` additions

- `test_anchor_bonus` — `_score("california", "california") == 3`
  (long exact match)
- `test_anchor_bonus_short_unchanged` — `_score("fire", "fire") == 2`
  (existing-style assertion, stays at +2 below threshold)
- `test_phonetic_equiv_short` — `_score("mm", "mmm") == 1`,
  `_score("uh", "uhh") == 1`, `_score("oh", "ooh") == 1`
- `test_phonetic_equiv_two_edit` — `_score("woah", "whoa") == 1`
- `test_no_false_2char_fuzzy` — `_score("it", "is") == 0`,
  `_score("in", "on") == 0` (verify general fuzzy gate stays at
  len ≥ 3 even after vocal-equiv addition)

### `TestMatchWordsToLines` additions

- `test_zero_score_filter` — lyric "love me tender" vs whisper
  "love XYZ tender": "me" should have None start/end (interpolated),
  not XYZ's timestamps.
- `test_zero_score_filter_orphans_whisper` — verify the documented
  edge case: when filter rejects a pair, the whisper token does not
  end up in any line. (Documents current behavior; not a regression
  guard.)
- `test_banding_prevents_chorus_drift` — synthetic multi-chorus case
  with sequence length above `_BAND_MIN_LENGTH`. Construct whisper
  output where unconstrained NW would mis-align across chorus
  boundaries; verify banded version stays diagonal.
- `test_anchor_bonus_within_band` — long exact-match anchor at the
  diagonal scores +3 and pulls alignment correctly. Same anchor
  placed outside the band (with sequences past `_BAND_MIN_LENGTH`)
  should NOT pull alignment off-diagonal — the lyric anchor token
  is gapped.
- `test_banded_fallback_triggers` — construct a case where the
  optimal alignment requires exiting the band (e.g., singer skips a
  long section); verify the fallback warning fires and the unbanded
  result is returned.
- `test_short_sequences_skip_banding` — sequences below
  `_BAND_MIN_LENGTH` should produce identical results to the legacy
  unbanded implementation. Snapshot a few existing test cases.

---

## Severity disagreements with the review

For the record, where this revision diverges from
`nw-alignment-improvements-review.md`:

- **Issue 1.1 (anchor "tie" with gap-pair):** review marked critical;
  treated here as important. The arithmetic claim is correct
  (+3 == -gap-pair) but doesn't translate into ambiguous alignments
  in practice — mismatches don't get the anchor bonus, so a
  hallucinated long word can't be inflated. Banding is the real
  defense; the `_MAX_MATCH` invariant assertion is a sanity check.
  Did not change scores or gap costs.
- **Issue 3.1 (filter threshold semantics):** review marked critical;
  treated here as a documentation-only concern. The reviewer's own
  recommendation was "no code change now"; severity should reflect
  required action. Comment added at the filter site.
- **Issue 3.2 (orphaned whisper tokens):** review marked important;
  treated here as documentation-only. The recommended action was
  again "no code change"; the edge case is a known limitation, not a
  regression risk. Comment at the filter site, plus a test that
  documents the behavior.

All other review issues incorporated as written.

---

## Outstanding questions for you

1. **`_BAND_MIN_LENGTH` = 500.** Reasonable for your typical song
   length, or do you regularly process inputs (e.g., long-form
   medleys, full-album concatenations) where banding should kick in
   sooner?

2. **Anchor bonus threshold `len ≥ 6`.** A frequency analysis on your
   Genius corpus would tell us whether dropping to `len ≥ 5` (adds
   "dream", "night", "heart", "world" as anchors) hurts more than it
   helps. If you have a few representative lyrics files handy I can
   run a quick count.

3. **Degenerate-fallback floor formula.** `(m/3)*2 + (2m/3)*_GAP_LYRIC`
   is a loose guess. After landing, check actual scores on a handful
   of clean-aligning songs and tighten the floor to ~80% of the
   median.

4. **Defer Option B (`_GAP_WHISPER` = -2)?** Yes by default — only
   revisit if anchor-driven drift shows up in real songs after
   Change 2 ships.

5. **Defer line-locality tie-break?** Yes by default. Add the
   diagnostic log line so we have a trigger signal if it's needed.
