# NW Alignment Improvements — Issue Review

Review of `nw-alignment-improvements.md` against the current codebase
(`genius_diarize/word_extraction.py`, 422 lines;
`tests/test_word_extraction.py`, 201 lines).

Each issue is tagged **[critical]** (will cause wrong output or crashes),
**[important]** (degrades accuracy in real cases), or **[minor]**
(cosmetic, fragility, or documentation).

---

## Change 1 — `_score()`: anchor bonus + vocal-sound equivalences

### Issue 1.1 — Anchor bonus breaks gap-path arithmetic **[critical]**

**Problem:** The plan raises the exact-match score for len >= 6 tokens from
+2 to +3. Currently `_GAP_LYRIC = -2` and `_GAP_WHISPER = -1`, so a
gap-pair (one lyric gap + one whisper gap) costs -3. A single anchor match
now yields +3, which **exactly cancels** one gap-pair. This creates a new
class of tie: NW becomes indifferent between "skip the anchor, pair the
neighbors" and "pair the anchor, gap both neighbors." Ties are broken
arbitrarily by the `max()` evaluation order in the DP fill, so alignment
can flip depending on which branch Python evaluates first — an
implementation-dependent, unstable choice.

With the current +2 scoring, a match is worth *less* than a gap-pair
(2 < 3), so NW always prefers gapping over matching a single isolated
token. At +3 the incentive inverts, and NW may now absorb a hallucinated
long word (e.g., whisper emits "california" from a different section)
even when surrounding context suggests it should be gapped.

The plan's own Change 3 rationale shows the math: a mismatch pair at
score 0 beats a gap-pair at -3 only if the match contributes >= 1. The
same logic applies: a +3 anchor makes a single hallucinated anchor
worth more than the gap alternative, which is the **opposite** of what
banding and zero-score filtering are trying to fix.

**Recommendation:** Two options:

1. **Raise gap penalties proportionally.** Set `_GAP_LYRIC = -3` and
   `_GAP_WHISPER = -1` (gap-pair = -4), restoring the invariant that
   a single match never fully cancels a gap-pair. Risk: this changes
   gap-path preferences globally and may break existing test cases
   where NW currently prefers to gap (e.g., `test_missing_whisper_word`,
   `test_extra_whisper_word`).

2. **Add a tie-break rule instead of changing scores.** Keep the anchor
   at +2 but add a secondary comparison in the DP cell that prefers
   diagonal (match) transitions over gap transitions when scores are
   equal. Implementation: store `(score, match_count)` tuples and
   compare lexicographically. This avoids the arithmetic coupling
   entirely. More complex but doesn't require re-tuning all gap
   behavior.

Either way, the gap-cost/score relationship must be re-examined after
any score change. Add an explicit assertion or comment:

```python
assert _GAP_LYRIC + _GAP_WHISPER < -(_MAX_MATCH_SCORE), \
    "Gap-pair must cost more than any single match to prevent ties"
```

### Issue 1.2 — `_VOCAL_EQUIV` frozenset has no transitivity **[minor]**

**Problem:** The frozenset approach `frozenset({lyric_tok, whisper_tok})
in _VOCAL_EQUIV` works correctly for pairwise equivalence but does not
support transitive chains. If someone later adds `frozenset({"mmm",
"mmmm"})`, then `frozenset({"mm", "mmmm"})` would not match any entry
even though by transitivity mm ~ mmm ~ mmmm. The current 6-entry list
has no such chains, but the API is a trap for future maintainers.

Also: if `lyric_tok == whisper_tok` (e.g., both are "mm"), the
frozenset collapses to a single element and won't match any entry. This
is **not a bug** because the exact-match branch (returning +2) runs
first, but it's worth a comment since a future refactor that reorders
the branches would silently break vocal-sound detection.

**Recommendation:** Add a comment on `_VOCAL_EQUIV`:

```python
# NOTE: frozenset pairs are NOT transitive. If "mm"~"mmm" and
# "mmm"~"mmmm" are both needed, add all three pairwise entries
# explicitly, or switch to a canonical-form dict approach.
# NOTE: this check MUST come after the exact-match branch, since
# frozenset({"mm", "mm"}) == frozenset({"mm"}) won't match any entry.
```

If the list grows past ~15 entries, refactor to a dict mapping each
token to a canonical form:

```python
_VOCAL_CANON = {"mm": "mm", "mmm": "mm", "uh": "uh", "uhh": "uh", ...}
# Then: _VOCAL_CANON.get(a) == _VOCAL_CANON.get(b) and both are not None
```

### Issue 1.3 — Missing vocal-sound pairs that will hit in practice **[important]**

**Problem:** Common vocal sounds absent from the proposed list:

| Pair | Why it matters |
|---|---|
| `oh/ooh` | Whisper elongates back-vowel exclamations; "ooh" is already in the list paired with "oo", but "oh" is a distinct common token |
| `ah/aah` | Same pattern as `uh/uhh` — fronted exclamation |
| `ha/hah` | Laughter tokens appear frequently in ad-libs |
| `ay/ayy` | Slang/vocal emphasis, common in hip-hop |
| `la/laa` | Common in vocal runs (do-re-mi style) |

Also: the very common "na na na" chorus pattern is unaddressed.
`na/nah` is ambiguous because "nah" is also a real English word meaning
"no." The plan correctly excluded it, but didn't flag the `na` / `nah`
chorus case as a known gap.

**Recommendation:**

- Add `oh/ooh`, `ah/aah`, `ha/hah`, `ay/ayy` — these are unambiguous
  vocal sounds with no collision against real English words.
- Do **not** add `na/nah` or `da/dah` without a disambiguation strategy.
  Consider a context-aware rule: only apply the equivalence if **both**
  tokens have length <= 3 (disqualifying real-word "nah" when paired
  with a non-vocal token). Or: only apply when the lyric token is
  inside parenthesized ad-lib text, which Genius already marks.
- Document the `na/nah` gap explicitly in a comment so it's not
  re-discovered from scratch.

### Issue 1.4 — `woah/whoa` is a spelling variant, not a vocal sound **[minor]**

**Problem:** The plan adds `frozenset({"woah", "whoa"})` as a vocal
equivalence pair. This pair is **qualitatively different** from the other
entries: all others are vowel-lengthening variants (same consonants,
different vowel repetition count). `woah/whoa` is a spelling variant
(letter transposition, Levenshtein-2). If more spelling variants are
needed later (e.g., `alright/aight` at Levenshtein-3, not covered),
they don't fit the "vocal sound" category.

**Recommendation:** No code change. Rename `_VOCAL_EQUIV` to
`_PHONETIC_EQUIV` or `_SCORE_EQUIV` to reflect that it's not purely
vocal sounds. This prevents category confusion when adding future
entries that are spelling variants rather than vocal elongations.

---

## Change 2 — `_needleman_wunsch()`: Sakoe-Chiba band

### Issue 2.1 — Banding breaks traceback when optimal path exits the band **[critical]**

**Problem:** The plan claims "by construction, optimal paths never visit
out-of-band cells." This is **not guaranteed**. Banded NW produces the
optimal alignment **only among paths that stay within the band**. If the
true optimal alignment drifts beyond the band, the DP fills out-of-band
cells with `_NEG_INF`, and the traceback may follow a path that reaches
an `_NEG_INF` cell, producing degenerate or meaningless alignments.

Concrete scenario: lyrics have 300 tokens, whisper has 200 tokens
(an entire verse was omitted in the vocal stem). The diagonal
`j_exp = round(i * 200/300)` at `i = 150` predicts `j_exp = 100`.
But the real alignment needs `j = 40` because whisper is far behind
(having skipped the verse tokens early). With `band = 50`,
`j_lo = max(1, 100 - 50) = 50`. The true `j = 40` is out of band.

In the current implementation, the traceback loop (lines 220-239) has a
fallback `break` at line 239 that catches cases where no transition
matches. With banded NW, this fallback triggers when the traceback
reaches an in-band cell whose predecessor is out-of-band
(`_NEG_INF`-valued). The result: the alignment is **silently truncated**
— the traceback stops early, and tokens at the beginning of the
sequences get no alignment.

**Recommendation:** Add a **fallback path**:

```python
alignment = _needleman_wunsch(lyric_norms, whisper_norms)
# Sanity check: if the alignment has a long contiguous unmatched stretch,
# the band may have clipped the true path. Re-run without banding.
if _has_degenerate_alignment(alignment, band):
    logger.warning("Banded NW produced degenerate alignment; re-running without banding")
    alignment = _needleman_wunsch_unbanded(lyric_norms, whisper_norms)
```

Simpler alternative: check if the resulting alignment has a contiguous
stretch of unmatched tokens longer than `band` width. If so, re-run
unbanded. This avoids needing to expose `_NEG_INF` outside the function.

### Issue 2.2 — `_NEG_INF` sentinel is fragile under score changes **[important]**

**Problem:** The plan sets `_NEG_INF = -(m + n + 1) * 10`. With the
anchor bonus (+3), the maximum per-cell contribution is +3 instead of
+2. The invariant required is:

```
_NEG_INF + max_single_match < score_of_any_legitimate_path
```

With `max_single_match = 3` and multiplier 10, `_NEG_INF + 3` is still
massively negative, so it works. But the multiplier 10 is a **magic
number** with no documented relationship to the scoring constants. If a
future change adds +4 or +5 scoring, `_NEG_INF` can become insufficient:
`-(m+n+1)*10 + 5*(m+n)` could be positive for long sequences, making
`_NEG_INF` cells appear as legitimate path values.

**Recommendation:** Compute `_NEG_INF` from the scoring constants:

```python
_MAX_MATCH = 3  # must equal max possible return value of _score()
_NEG_INF = -(m + n + 1) * _MAX_MATCH * 2
```

The `* 2` margin ensures that even after adding the maximum possible
match score, `_NEG_INF` cells remain more negative than any legitimate
path. Add an assertion:

```python
assert _NEG_INF + _MAX_MATCH * max(m, n) < 0, "Sentinel insufficient"
```

### Issue 2.3 — Last-row force-fill creates asymmetric DP behavior **[important]**

**Problem:** The plan force-fills the last row (`i == m`) to full width
to preserve free-suffix behavior. This means the last lyric token can
match any whisper token, no matter how far from the diagonal. But the
**second-to-last** row is still banded. If the true alignment needs the
penultimate lyric token to match a whisper token near the end (within
the free-suffix region), that cell may be out of band on row `m-1`,
forcing the DP to take a suboptimal path to reach the last row.

This creates a **cliff**: alignment quality jumps discontinuously
between the penultimate and last rows. In practice, for songs with
long spoken outros or extended ad-libs after the last lyric line,
the last two lyric tokens may both need to match whisper tokens far
from the diagonal, but only the last one is guaranteed to reach them.

**Recommendation:** Instead of force-filling only the last row, use a
**taper**: expand the band width linearly over the last K rows to reach
full width. For example:

```python
taper_rows = min(band, m // 4)  # taper over last ~25% of rows or band rows
if i > m - taper_rows:
    extra = (i - (m - taper_rows)) * (n - band) // taper_rows
    j_lo = max(1, j_exp - band - extra)
    j_hi = min(n, j_exp + band + extra)
```

This provides a smooth transition to full width and avoids the cliff
effect. Alternatively, if the taper is too complex, force-fill the last
**two** rows (last lyric token and its predecessor). Document why.

### Issue 2.4 — Band center calculation uses banker's rounding **[minor]**

**Problem:** `round(i * n / m)` uses Python 3's banker's rounding
(round-half-to-even), which introduces a systematic 0.5-token jitter
in the band center. For `round(0.5) = 0`, `round(1.5) = 2`, etc.
With band width >= 50, this jitter is negligible. But if the band
formula is ever tuned down for short sequences, it could matter.

**Recommendation:** No code change. Add a comment:

```python
# round() uses banker's rounding (round-half-to-even); the 0.5-token
# jitter is absorbed by the band width and is negligible for band >= 50.
```

### Issue 2.5 — Memory is still O(m*n) despite banding **[minor]**

**Problem:** The implementation sketch allocates a full
`dp = [[_NEG_INF] * (n + 1) for _ in range(m + 1)]` matrix even though
only `O(m * band)` cells are filled. For a 2000-token song with band
250, this allocates 2000 * 2000 = 4M cells but only fills 2000 * 500 =
1M. The memory waste is 4x the useful data.

This isn't a correctness issue, but it undermines one of banding's
benefits. For very long inputs (audiobooks, podcasts at 10000+ tokens),
the full matrix could consume significant memory.

**Recommendation:** If memory is a concern (token lists > 5000), switch
to a row-band representation: store only `2 * (2 * band + 1)` rows (two
for the DP recurrence plus a full-row buffer for traceback). If memory
is fine for song-length inputs (typically < 2000 tokens), leave as-is
and add a comment noting the optimization opportunity.

---

## Change 3 — `match_words_to_lines()`: filter zero-score pairs

### Issue 3.1 — Filter threshold semantics shift with new +1 score types **[critical]**

**Problem:** If the anchor bonus (Change 1) is implemented, the filter
threshold of `>= 1` now has a different semantic meaning. Previously,
score 0 meant "no relationship at all" and +1 meant only "Levenshtein-1
fuzzy match." With Change 1, score +1 also includes **vocal-sound
equivalences**. The filter can't distinguish between "this is a fuzzy
match I trust" and "this is a vocal-sound equivalence I barely trust."
If you later want to filter more aggressively (e.g., only accept scores
>= 2), you'd lose both.

This is critical because it's a **semantic coupling** between two
independent changes: the meaning of `score >= 1` in the filter depends
on which score types exist, and adding a new +1-score type silently
widens what the filter accepts. If the vocal-equiv score were set to
+0 (or a special value), the filter would reject it, but then NW
wouldn't prefer vocal-equiv over mismatch in the DP.

**Recommendation:** No code change now. But consider for the future:
return the **match type** (exact, contraction, vocal-equiv, fuzzy,
mismatch) alongside the score, so the filter can make type-aware
decisions. For the current plan, `>= 1` is the right threshold since
both vocal-equiv and fuzzy are better than interpolation. Add a comment:

```python
# NOTE: score >= 1 now includes both Levenshtein-1 fuzzy matches AND
# vocal-sound equivalences (from _PHONETIC_EQUIV). If you need to filter
# more aggressively, consider returning match_type alongside the score
# so the filter can distinguish fuzzy (trusted) from vocal-equiv (less
# trusted).
```

### Issue 3.2 — Filtered tokens still consume a whisper index **[important]**

**Problem:** When the zero-score filter rejects a pair (l_idx, w_idx),
the lyric token falls through to interpolation, but the **whisper**
token at `w_idx` is still consumed by the alignment. This means a
whisper hallucination that NW paired with a lyric token at score 0
still "uses up" that whisper index. It's no longer available to be
matched with any other lyric token (which is fine — NW already decided
it has no better match). But the whisper token also **doesn't get
assigned to any line's word list**, since the filter removed the lyric
side of the pairing.

This is actually the **desired behavior** for hallucinations (we want
to drop them). But for a different scenario — where a real whisper word
is paired with a lyric word at score 0 because both are short mismatches
— the real whisper word is silently dropped even though it might belong
to the line. Example: lyric "it is" vs whisper "it [it] is" where a
hallucinated "it" appears. NW might pair lyric-it -> whisper-it (+2),
lyric-is -> hallucinated-it (0), with the real whisper-is gapped.
The filter rejects the lyric-is -> hallucinated-it pair, and "is"
gets interpolated. The real whisper "is" is gapped and dropped. This is
suboptimal but not catastrophic — the interpolated timing is usually
close.

**Recommendation:** No code change for now. The filter is a net
improvement over the current behavior (where the 0-score pair would
assign the hallucinated timestamp to the lyric token). Document this
edge case: "filtered pairs leave the whisper token orphaned (gapped in
the alignment), which means it's silently dropped even if it was a real
word. This is acceptable because NW already determined it has no better
lyric match."

### Issue 3.3 — Score recomputation assumes `_score()` is stateless **[minor]**

**Problem:** The plan recommends recomputing `_score()` in the filter
(option 1). This is safe because `_score()` is currently a pure function
of two normalized strings. But if `_score()` is ever made stateful
(e.g., context-aware scoring that depends on neighboring tokens), the
recomputed score may differ from the one NW used during DP fill.

**Recommendation:** Add a comment at the filter site:

```python
# _score() is stateless and deterministic, so recomputing here gives
# the same value used during NW DP fill. If _score() ever becomes
# context-aware, switch to returning (l_idx, w_idx, score) triples
# from _needleman_wunsch().
```

---

## Cross-cutting issues

### Issue X.1 — Implementation order should swap Changes 1 and 3 **[important]**

**Problem:** The plan orders: (1) anchor bonus + vocal-sound, (2)
zero-score filter, (3) banding. But the anchor bonus (Change 1) alters
the score landscape that the filter (Change 3) depends on. Specifically,
the filter threshold `>= 1` interacts with the new vocal-equiv score
(+1): without Change 1, there are no +1 scores from vocal equivalences,
so the filter's behavior is purely about fuzzy matches and mismatches.
With Change 1, the filter also accepts vocal-sound equivalences.

If Change 3 is landed first (before Change 1), it can be validated in
isolation: the filter rejects score-0 pairs, which are only mismatches.
Then when Change 1 adds vocal-equiv at +1, the filter automatically
accepts them — no re-tuning needed.

The plan actually suggests this order ("Independent of Change 2, so
worth landing first to validate the fix on its own") but lists Change 1
as step 1. The text contradicts itself.

**Recommendation:** Land Change 3 (zero-score filter) **before** Change
1 (anchor bonus + vocal-sound). This lets you:

1. Validate the filter in isolation on real songs.
2. Confirm that interpolation handles orphaned lyric tokens well.
3. Then add the scoring changes, which the filter automatically
   accommodates.

Update the implementation order to: 1. Change 3, 2. Change 1, 3. Change
2, 4. Tests, 5. Full suite run, 6. Real-song sanity check.

### Issue X.2 — No test for anchor bonus + banding interaction **[important]**

**Problem:** The anchor bonus gives +3 to long words. Banding constrains
the DP to a diagonal corridor. These two changes interact: the anchor
bonus makes it more attractive for NW to "reach" across the diagonal to
match a long word, but banding prevents that reach. If a long word
appears in the whisper output at a position that's outside the band
(e.g., whisper reordered a verse), the +3 anchor bonus can't help
because the band blocks access.

The plan's test section doesn't include a test for this interaction.

**Recommendation:** Add a test:

```python
def test_anchor_bonus_within_band():
    """Long exact-match tokens at the diagonal get +3;
    same tokens outside the band are unreachable."""
    # Construct lyrics/whisper where a long word ("california")
    # appears at position 60 in lyrics and position 60 in whisper
    # (within band) -> should match at +3.
    # Then construct a case where the same word appears at position 60
    # in lyrics but position 10 in whisper (outside band with band=50)
    # -> should NOT match; lyric token should be gapped.
```

### Issue X.3 — `_CONTRACTIONS` dict + anchor bonus ordering **[minor]**

**Problem:** The contraction check in `_score()` (lines 135-144) returns
+2 for contraction matches before the anchor bonus check could fire.
But contractions can be long: e.g., if someone adds a 6+ char key to
`_CONTRACTIONS`, the contraction match would return +2 before the
anchor bonus check for a long exact match could return +3.

More realistically: the current `_CONTRACTIONS` values are all short
(< 6 chars), so the anchor bonus (len >= 6) will never apply to a
contraction key. But the contraction **expansion** can contain long
words: e.g., `"wanna": "want to"` — "want" is 4 chars, "to" is 2.
Still under the threshold.

**Recommendation:** No code change now. Add a comment noting that the
contraction branch runs before the anchor bonus branch, so contractions
always score +2 even if they happen to be >= 6 chars. If future
contractions are added that are >= 6 chars, consider reordering or
merging the checks.

---

## Answers to the plan's outstanding questions

### Q1: Anchor bonus threshold — len >= 6 for +3?

**Recommendation:** `len >= 6` is reasonable but should be validated
against a corpus. The risk (Issue 1.1) is the score/gap-cost tie, not
the threshold itself. If you address the tie problem, `len >= 5` would
work too and capture more anchors ("dream", "night", "heart", "world"
are all 5 chars). Run a frequency analysis on your Genius lyrics corpus:
count how many unique lyric tokens fall at each length. If the 5-char
bucket is dominated by common function words ("their", "there", "where",
"could", "would"), stick with 6. If it's mostly content words, lower to
5.

### Q2: Vocal-sound list completeness?

See Issue 1.3. Add `oh/ooh`, `ah/aah`, `ha/hah`, `ay/ayy`. Flag
`na/nah` as a known gap requiring disambiguation. Also consider
`la/laa` and `yeah/yea` — though "yeah/yea" is already covered by
Levenshtein-1 fuzzy (both >= 3 chars, edit distance 1).

### Q3: Band width — max(50, max(m, n) // 4)?

**Recommendation:** The formula is reasonable for songs. However, see
Issue 2.1: add a fallback to unbanded NW when the alignment is
degenerate. The 25% ceiling is generous; if empirical results show
chorus drift still occurring, try `max(50, max(m, n) // 6)` (~17%).
If legitimate alignments get cut off, add the taper from Issue 2.3
before widening the band.

### Q4: Force-fill last row?

**Recommendation:** Acceptable with the caveat in Issue 2.3 (cliff
effect on the penultimate row). Implement the taper if you see
penultimate-row misalignment in real songs. Otherwise, force-fill is
the simpler option.

### Q5: Defer line-locality tie-break?

**Recommendation:** Agree — defer. But banding alone may not fully
prevent line-boundary leakage for songs with very similar adjacent
lines (e.g., "I love you" / "I loved you" where only one character
differs). If post-banding results still show leakage, line-locality is
the next lever to pull. Add a log entry when a lyric line's matched
whisper indices are non-monotonic relative to the previous line's
indices — this is the diagnostic signal that line-locality would fix.

---

## Summary table

| ID | Severity | Change | One-line summary |
|---|---|---|---|
| 1.1 | critical | 1 | Anchor +3 creates gap-path tie; must re-examine gap costs |
| 1.2 | minor | 1 | Frozenset equiv has no transitivity; document it |
| 1.3 | important | 1 | Missing `oh/ooh`, `ah/aah`, `ha/hah` vocal pairs |
| 1.4 | minor | 1 | Rename `_VOCAL_EQUIV` to `_PHONETIC_EQUIV` for accuracy |
| 2.1 | critical | 2 | Banded traceback can hit `_NEG_INF` cells; add fallback |
| 2.2 | important | 2 | `_NEG_INF` magic number should derive from `_MAX_MATCH` |
| 2.3 | important | 2 | Last-row force-fill creates cliff on penultimate row |
| 2.4 | minor | 2 | Banker's rounding jitter; document it |
| 2.5 | minor | 2 | Full DP matrix allocated despite banding; note optimization |
| 3.1 | critical | 3 | Filter threshold semantics shift with new +1 score types |
| 3.2 | important | 3 | Filtered pairs orphan the whisper token; document behavior |
| 3.3 | minor | 3 | Recomputation assumes `_score()` is stateless; document it |
| X.1 | important | cross | Implementation order: land Change 3 before Change 1 |
| X.2 | important | cross | No test for anchor bonus + banding interaction |
| X.3 | minor | cross | Contractions bypass anchor bonus; note for future |

**Must-fix before landing:** 1.1, 2.1, 3.1 (critical).
**Should-fix before landing:** 1.3, 2.2, 2.3, 3.2, X.1, X.2 (important).
**Can defer:** 1.2, 1.4, 2.4, 2.5, 3.3, X.3 (minor).
