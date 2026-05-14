# Lyric Alignment Algorithm: Matching Stable-TS Tokens to Lyric Lines

## Problem Statement

Given a set of word-level timestamps from stable-ts (Whisper) and a lyric file, match tokens to lyric lines in a way that:
- Is agnostic to lyric order (handles remixes, live performances)
- Handles repeated sections (chorus appearing multiple times)
- Minimizes unaligned tokens
- Doesn't care how many times a line appears

---

## Algorithm Overview

The problem decomposes into two phases:

**Phase 1** — Find all candidate matches (each lyric line vs. the full token sequence, fuzzily)

**Phase 2** — Pick the best non-overlapping subset of those matches via weighted interval scheduling DP

---

## Phase 1: Candidate Match Generation

For each lyric line, scan the token sequence for approximate matches using a sliding window with word-level edit distance. The window is allowed to be **narrower or wider** than the line — Whisper both drops words (mumbled or fast sections) and inserts filler — so the matcher must tolerate the alignment span being shorter than the reference line, not just longer.

```python
def find_candidates(tokens, lyric_lines, max_edit_ratio=0.25):
    """
    tokens: list of (word, start_time, end_time)
    lyric_lines: list of lists of normalized words
    Returns: list of (token_start_idx, token_end_idx, line_id, score)
    """
    candidates = []
    token_words = [normalize(t[0]) for t in tokens]

    for line_id, line_words in enumerate(lyric_lines):
        n = len(line_words)
        # Floor at 1 so short lines (< 4 words) aren't given zero error
        # tolerance and silently dropped on a single mishear.
        max_allowed = max(1, int(n * max_edit_ratio))
        # Require genuine content overlap — otherwise the narrow-window
        # slack lets a 2-word line "match" a single token by pure deletion.
        min_overlap = min(2, n)
        # Window may be narrower OR wider than the line: whisper drops
        # words (mumbled/fast sections) as well as inserting filler.
        for window_size in range(max(1, n - 2), n + 4):
            for i in range(len(token_words) - window_size + 1):
                window = token_words[i:i + window_size]
                dist = edit_distance(line_words, window)
                if dist <= max_allowed and (n - dist) >= min_overlap:
                    score = (n - dist) / n  # normalized match ratio, 0.0–1.0
                    candidates.append((i, i + window_size, line_id, score))

    return candidates
```

Three robustness properties this buys:
- **Deletion tolerance** — narrower windows (`n-2`) let a line still match when Whisper caught only part of it.
- **Short-line survival** — flooring `max_allowed` at 1 means lines like `"oh oh oh"` or ad-libs get one mishear of slack instead of zero.
- **No degenerate fragments** — the `min_overlap` floor stops the narrow-window slack from tiling near-empty matches (e.g. a 2-word line collapsing onto one token), which the pure-sum DP would otherwise happily stack.

`edit_distance` operates on **word sequences**, not characters — each word is one token. This handles insertions/deletions from Whisper drift.

---

## Phase 1b: Anchor-Run Fallback

The edit-ratio threshold in Phase 1 is deliberately strict, which means a line whose Whisper transcription is badly garbled produces **zero candidates** and is silently dropped. Sometimes that's correct (the line genuinely wasn't sung), but often the transcription still contains a solid recognizable phrase — e.g. lyric `"Oh, so take my hand, it's open"` heard as `"so take my hand, it's a bone"` shares the contiguous run `"so take my hand its"`.

After Phase 1, re-scan **only the zero-candidate lines** with a relaxed rule: accept a window if it shares a *contiguous anchor run* of `≥ max(3, n // 3)` consecutive words with the line, regardless of total edit distance. The anchor run is the confidence signal — a long verbatim phrase is proof the line was sung there even if the surrounding words diverge.

```python
def find_anchor_candidates(token_words, lyric_lines, line_ids):
    candidates = []
    for line_id in line_ids:                       # only zero-candidate lines
        line_words = lyric_lines[line_id]
        n = len(line_words)
        min_run = max(3, n // 3)
        if n < min_run:
            continue
        for window_size in range(max(1, n - 2), n + 4):
            for i in range(len(token_words) - window_size + 1):
                window = token_words[i:i + window_size]
                run = longest_contiguous_run(line_words, window)
                if run >= min_run:
                    score = run / n               # only the anchored span counts
                    candidates.append((i, i + window_size, line_id, score))
    return candidates
```

This is **non-polluting** by construction:
- It only runs for lines that already failed Phase 1 — it can't change the result for lines that matched cleanly.
- Score is `anchor_run / n` (anchored coverage only), always conservative, so a recovered match never outweighs a real Phase 1 match in the Phase 2 DP.
- Restricting the scan to a handful of zero-candidate lines keeps the cost bounded.

`longest_contiguous_run` is the word-level longest common substring — an O(n·m) DP over the (short) line and window.

---

## Phase 2: Weighted Interval Scheduling DP

The candidates form a set of intervals `(start, end, score)` over the token index space. The goal is to pick the highest-scoring non-overlapping subset. Runs in **O(m log m)**.

```python
def best_tiling(candidates, n_tokens):
    candidates.sort(key=lambda c: c[1])

    def last_non_overlapping(i):
        lo, hi = 0, i - 1
        target = candidates[i][0]
        while lo <= hi:
            mid = (lo + hi) // 2
            if candidates[mid][1] <= target:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi

    m = len(candidates)
    dp = [0] * (m + 1)
    choice = [None] * (m + 1)

    for i in range(1, m + 1):
        c = candidates[i - 1]
        j = last_non_overlapping(i - 1)
        include = c[3] + (dp[j + 1] if j >= 0 else 0)
        exclude = dp[i - 1]
        if include > exclude:
            dp[i] = include
            choice[i] = (i - 1, j)
        else:
            dp[i] = exclude
            choice[i] = None

    selected = []
    i = m
    while i > 0:
        if choice[i] is not None:
            ci, j = choice[i]
            selected.append(candidates[ci])
            i = j + 1
        else:
            i -= 1

    return selected[::-1]
```

---

## Normalization

Normalize aggressively before matching — Whisper and Genius diverge in predictable ways:

```python
import re, unicodedata

def normalize(word):
    word = unicodedata.normalize('NFD', word.lower())
    word = re.sub(r"[^\w\s']", '', word)
    word = re.sub(r"\s+", ' ', word).strip()
    # Optionally: map contractions ("don't" → "dont", "i'm" → "im")
    return word
```

Consider **phonetic normalization** (Soundex/Metaphone) as a fallback for words where spelling diverges but pronunciation matches — especially useful for names.

---

## Resilience to Live/Remix Lyric Scrambling

Order-independence falls out naturally from the design — Phase 1 and Phase 2 are fully decoupled from lyric order. Phase 1 generates candidates by scanning every lyric line against every token position independently. Phase 2 scores purely on token coverage with no concept of "line 3 should follow line 2."

### Edge cases

| Case | Behavior |
|---|---|
| **Chorus repeated 3x** | Generates candidates at all 3 positions; DP selects all if non-overlapping and high-scoring |
| **Dropped verse** | No high-scoring candidates generated; line is silently dropped from output |
| **Ad-libs / improvised lines** | Appear as token gaps between selected intervals; DP has no incentive to force a bad match |
| **Heavy paraphrasing / garbled transcription** | Edit distance exceeds the Phase 1 threshold → zero candidates. The Phase 1b anchor-run fallback recovers the line if it still contains a contiguous verbatim phrase; otherwise it's dropped. |

---

## Confidence Scoring

With a 9-word line and `max_edit_ratio=0.25`, `max_allowed = max(1, int(9 * 0.25)) = 2`. A single missing token (edit distance 1) passes easily, scoring `(9-1)/9 = 0.89` vs a perfect match at `1.0`. No separate confidence filter is needed — confidence is implicit in the score gradient, and the DP naturally prefers higher-scoring matches. The `max(1, …)` floor matters only for lines shorter than 4 words, where `int(n * 0.25)` would otherwise round to 0.

### Score normalization for short lines

Raw scores (covered token count) aren't comparable across line lengths. An 8/9 match (raw score 8) would be treated as twice as valuable as a 4/4 perfect match (raw score 4), so short lines (e.g. "oh oh oh", "yeah yeah") get squeezed out by longer mediocre matches.

This is why Phase 1 emits the **normalized match ratio** `(n - dist) / n` rather than a raw count. A 4/4 perfect match (1.0) correctly outweighs an 8/9 near-miss (0.89) on a per-line basis, while the DP still favors tiling more lines overall (more intervals → more summed score).

---

## Output

- Tokens inside selected intervals → assigned the timestamp range from stable-ts
- Tokens in gaps → unaligned; can be discarded, interpolated, or flagged for review
- The same lyric line can be selected at multiple positions → chorus repeats handled naturally
