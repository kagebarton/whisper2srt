"""Order-independent lyric matcher: candidate generation + interval-scheduling DP.

An alternative to the walk matcher (``word_extraction.match_words_to_lines_walk``).
Where the walk matcher trusts stable-ts to emit whisper words in reference
order, this matcher is order-agnostic: it finds every fuzzy occurrence of each
lyric line in the whisper token stream and picks the best non-overlapping
tiling. That makes it resilient to remixes, live re-orderings, and repeated
choruses — at the cost of silently dropping lyric lines it can't find.

Two phases (see plans/lyric-alignment-algorithm.md):
  1. find_candidates — sliding-window fuzzy match of each line vs. the token
     stream. Windows range narrower AND wider than the line so the matcher
     tolerates whisper dropping words as well as inserting filler.
     find_anchor_candidates is a relaxed fallback re-scan for lines that got
     zero candidates: it accepts a window on a contiguous anchor run alone,
     regardless of total edit distance.
  2. best_tiling — weighted interval scheduling DP over the candidate
     intervals, O(m log m).

Before matching, lyric lines are split into *match units*: most lines yield
one unit, but a line with parenthetical phrases ("main (backing vocal)")
splits into the main phrase + each parenthetical as separate units — a
parenthetical is typically a simultaneous backing vocal that whisper
transcribes as its own run, never contiguous with the main phrase.

Output line_objects carry a ``line_id`` back-reference into ``lines`` (the
same lyric line may appear zero times or many times — dropped, repeated, or
split across parenthetical units), so downstream speaker assignment maps by
index rather than positional zip.
"""

import logging
import re
from bisect import bisect_right

from genius_align.word_extraction import _normalize_token, _walk_align

logger = logging.getLogger(__name__)

# Parenthetical phrases — typically simultaneous backing vocals — are split
# into their own match units (see _split_paren_units).
_PAREN_RE = re.compile(r"\(([^()]*)\)")


def _split_paren_units(text: str) -> list:
    """Split a lyric line into match units on parenthetical phrases.

    "All the voices in my mind (Tell me when it kicks in)" yields
    ["All the voices in my mind", "Tell me when it kicks in"]. A
    parenthetical is usually a simultaneous backing vocal that whisper
    transcribes as its own run, never contiguous with the main phrase, so
    the combined line can never match a single contiguous window.
    Splitting lets the main phrase and each parenthetical match
    independently. Lines with no parens return unchanged as one unit.
    """
    parens = _PAREN_RE.findall(text)
    if not parens:
        return [text]
    units = []
    # Collapse the whitespace sub() leaves where parens were removed.
    main = " ".join(_PAREN_RE.sub(" ", text).split())
    if main:
        units.append(main)
    for p in parens:
        p = p.strip()
        if p:
            units.append(p)
    return units or [text]


def _tokenize_unit(text: str) -> list:
    """Tokenize a match unit into [(normalized, raw), ...], dropping tokens
    that normalize to empty (punctuation-only)."""
    text = text.replace("—", " ").replace("--", " ")
    toks = []
    for tok in text.split():
        norm = _normalize_token(tok)
        if norm:
            toks.append((norm, tok))
    return toks


def _edit_distance(a: list, b: list) -> int:
    """Word-level Levenshtein distance between two token sequences."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _longest_contiguous_run(a: list, b: list) -> int:
    """Length of the longest run of consecutive equal elements common to
    sequences a and b — word-level longest common substring.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    best = 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def find_candidates(
    token_norms: list,
    lyric_lines: list,
    max_edit_ratio: float = 0.25,
    window_slack_lo: int = 2,
    window_slack_hi: int = 3,
) -> list:
    """Scan the token sequence for fuzzy occurrences of each lyric line.

    token_norms: list of normalized whisper word strings
    lyric_lines: list of lists of normalized lyric words
    Returns: list of (start_idx, end_idx, line_id, score) where score is the
    normalized match ratio in (0.0, 1.0].

    Windows range from ``n - window_slack_lo`` to ``n + window_slack_hi`` wide:
    narrower windows catch lines whisper only partially heard, wider windows
    absorb inserted filler. ``max_allowed`` is floored at 1 so short lines
    (< 4 words) aren't given zero error tolerance.

    A candidate must also have at least ``min(2, n)`` tokens of genuine
    content overlap (``n - dist``). Without this floor, the narrow-window
    slack lets a 2-word line "match" a single token by pure deletion
    (score 0.5), and the DP happily tiles those degenerate fragments.
    """
    candidates = []
    T = len(token_norms)
    for line_id, line_words in enumerate(lyric_lines):
        n = len(line_words)
        if n == 0:
            continue
        max_allowed = max(1, int(n * max_edit_ratio))
        min_overlap = min(2, n)
        lo = max(1, n - window_slack_lo)
        hi = n + window_slack_hi
        for window_size in range(lo, hi + 1):
            if window_size > T:
                break
            for i in range(T - window_size + 1):
                window = token_norms[i : i + window_size]
                dist = _edit_distance(line_words, window)
                if dist <= max_allowed and (n - dist) >= min_overlap:
                    score = (n - dist) / n
                    candidates.append((i, i + window_size, line_id, score))
    return candidates


def find_anchor_candidates(
    token_norms: list,
    lyric_lines: list,
    line_ids: list,
    min_run_floor: int = 3,
    window_slack_lo: int = 2,
    window_slack_hi: int = 3,
) -> list:
    """Relaxed fallback re-scan for specific lyric lines.

    Intended for lines that produced zero candidates in the main
    find_candidates pass — typically because whisper's transcription of
    that line is too garbled for the edit-ratio threshold. A window is
    accepted here on a *contiguous anchor run* alone: it must share a run
    of >= min_run consecutive words with the line, regardless of total
    edit distance. The anchor run is the confidence signal — a solid
    recognizable phrase is proof the line was sung here even if the
    surrounding words diverge.

    Score is ``anchor_run / n`` — only the confidently-matched span counts
    — so these weaker matches never outweigh a real find_candidates match
    in the tiling DP. Restricting the scan to ``line_ids`` keeps it bounded
    and means it never pollutes lines that already matched cleanly.
    """
    candidates = []
    T = len(token_norms)
    for line_id in line_ids:
        line_words = lyric_lines[line_id]
        n = len(line_words)
        min_run = max(min_run_floor, n // 3)
        if n < min_run:
            continue  # too short to anchor confidently
        lo = max(1, n - window_slack_lo)
        hi = n + window_slack_hi
        for window_size in range(lo, hi + 1):
            if window_size > T:
                break
            for i in range(T - window_size + 1):
                window = token_norms[i : i + window_size]
                run = _longest_contiguous_run(line_words, window)
                if run >= min_run:
                    candidates.append((i, i + window_size, line_id, run / n))
    return candidates


def best_tiling(candidates: list) -> list:
    """Pick the highest-scoring non-overlapping subset of candidate intervals.

    Weighted interval scheduling DP, O(m log m). candidates are
    (start_idx, end_idx, line_id, score) tuples over token-index space.
    Returns the selected subset in start order.
    """
    if not candidates:
        return []

    cands = sorted(candidates, key=lambda c: c[1])
    ends = [c[1] for c in cands]
    m = len(cands)

    dp = [0.0] * (m + 1)
    choice = [None] * (m + 1)

    for i in range(1, m + 1):
        c = cands[i - 1]
        # Index of the last candidate whose end <= this candidate's start.
        j = bisect_right(ends, c[0], 0, i - 1) - 1
        include = c[3] + (dp[j + 1] if j >= 0 else 0.0)
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
            selected.append(cands[ci])
            i = j + 1
        else:
            i -= 1
    return selected[::-1]


def _build_line_object(
    text: str, line_id: int, line_toks: list, win_words: list, lookahead: int
) -> dict:
    """Assign per-word timing within a selected window.

    A candidate match only ties a whole lyric line to a token span — for
    karaoke each lyric word still needs its own start/end. Reuse the walk
    aligner *within the window* to pair words, then linearly interpolate any
    unmatched runs across the surrounding anchors (same gap-fill logic as the
    walk matcher).
    """
    lyric_norms = [t[0] for t in line_toks]
    win_norms = [_normalize_token(w["word"]) for w in win_words]
    # Neutralize _walk_align's whole-song biases here: a selected tiling
    # window is already a tight fuzzy match, so there's no long stretch of
    # noise to absorb. confirm_matches=1 and whisper_skip_budget=1 reproduce
    # the original symmetric "advance both on no-resync" behavior.
    mapping = _walk_align(
        lyric_norms,
        win_norms,
        lyric_lookahead=lookahead,
        whisper_lookahead=lookahead,
        confirm_matches=1,
        whisper_skip_budget=1,
    )

    n = len(line_toks)
    tw: list = [None] * n
    for k, (_, raw) in enumerate(line_toks):
        wi = mapping[k]
        if wi is not None:
            src = win_words[wi]
            tw[k] = {
                "word": raw,
                "start": src["start"],
                "end": src["end"],
                "speaker": None,
                "dominant_speaker": None,
            }

    win_start = win_words[0]["start"]
    win_end = win_words[-1]["end"]
    k = 0
    while k < n:
        if tw[k] is not None:
            k += 1
            continue
        run_end = k
        while run_end < n and tw[run_end] is None:
            run_end += 1
        prev_end = tw[k - 1]["end"] if k > 0 else win_start
        next_start = tw[run_end]["start"] if run_end < n else win_end
        if next_start < prev_end:
            next_start = prev_end
        run_len = run_end - k
        slot = (next_start - prev_end) / run_len if run_len > 0 else 0.0
        for off in range(run_len):
            s = prev_end + off * slot
            e = prev_end + (off + 1) * slot
            _, raw = line_toks[k + off]
            tw[k + off] = {
                "word": raw,
                "start": s,
                "end": e,
                "speaker": None,
                "dominant_speaker": None,
            }
        k = run_end

    return {
        "text": text,
        "line_id": line_id,
        "words": tw,
        "start": tw[0]["start"],
        "end": tw[-1]["end"],
    }


def match_words_to_lines_tiling(
    words: list,
    lines: list,
    align_lines: list = None,
    max_edit_ratio: float = 0.25,
    lookahead: int = 3,
    anchor_fallback: bool = True,
) -> list:
    """Order-independent matcher — see module docstring.

    Returns line_objects sorted by start time. Unlike the walk matcher this
    does NOT emit one object per lyric line: dropped lines are absent,
    repeated lines (choruses) appear multiple times, and lines with
    parenthetical phrases are split into separate objects (main phrase +
    each parenthetical). Each object carries a ``line_id`` back-reference
    into ``lines`` — multiple objects may share one ``line_id``.

    When ``anchor_fallback`` is set, match units that produced zero
    candidates in the main pass get a relaxed contiguous-anchor re-scan
    (see find_anchor_candidates) before the tiling DP runs.
    """
    if align_lines is None:
        align_lines = lines

    # Build match units. Most lines yield one unit; a line with
    # parenthetical phrases splits into the main phrase + each paren as
    # separate units (see _split_paren_units). Each unit keeps a line_id
    # back-reference into ``lines`` for downstream speaker assignment.
    unit_line_ids: list = []   # unit index -> original line index
    unit_texts: list = []      # unit index -> display text
    unit_tokens: list = []     # unit index -> [(norm, raw), ...]
    for line_idx, aline in enumerate(align_lines):
        for unit_text in _split_paren_units(aline):
            unit_line_ids.append(line_idx)
            unit_texts.append(unit_text)
            unit_tokens.append(_tokenize_unit(unit_text))

    if not words:
        return []

    token_norms = [_normalize_token(w["word"]) for w in words]
    unit_lines_norm = [[t[0] for t in toks] for toks in unit_tokens]

    candidates = find_candidates(token_norms, unit_lines_norm, max_edit_ratio)

    if anchor_fallback:
        matched_units = {c[2] for c in candidates}
        zero_candidate_units = [
            uid
            for uid, toks in enumerate(unit_lines_norm)
            if toks and uid not in matched_units
        ]
        if zero_candidate_units:
            anchor_cands = find_anchor_candidates(
                token_norms, unit_lines_norm, zero_candidate_units
            )
            recovered = len({c[2] for c in anchor_cands})
            logger.info(
                "Anchor fallback: re-scanned %d zero-candidate units → "
                "%d candidates recovering %d units",
                len(zero_candidate_units),
                len(anchor_cands),
                recovered,
            )
            candidates.extend(anchor_cands)

    selected = best_tiling(candidates)

    logger.info(
        "Tiling match: %d candidates → %d intervals selected, covering "
        "%d/%d match units across %d/%d lyric lines",
        len(candidates),
        len(selected),
        len({c[2] for c in selected}),
        len(unit_texts),
        len({unit_line_ids[c[2]] for c in selected}),
        len(lines),
    )

    line_objects = []
    for start_idx, end_idx, unit_idx, _score in selected:
        win_words = words[start_idx:end_idx]
        obj = _build_line_object(
            unit_texts[unit_idx],
            unit_line_ids[unit_idx],
            unit_tokens[unit_idx],
            win_words,
            lookahead,
        )
        line_objects.append(obj)

    line_objects.sort(key=lambda o: o["start"])
    return line_objects
