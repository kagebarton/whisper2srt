"""Word extraction and line-building helpers.

Adapted from diarized_captions/word_extraction.py. Key change:
load_genius_lyrics() returns (genius_lines, plain_text) from a single
parse so both outputs are 1:1 in line count and order.
"""

import logging
import re
import unicodedata
from pathlib import Path

import srt

from genius_diarize.genius import parse_genius_sections

logger = logging.getLogger(__name__)


def load_lyrics(lyrics_path: Path) -> tuple:
    """Return (lyrics_text, lyrics_format) where format is 'txt' or 'srt'.

    If lyrics_path is .srt: parse with srt library, concatenate text,
    discard original timestamps. If .txt: read raw text.
    """
    suffix = Path(lyrics_path).suffix.lower()
    if suffix == ".srt":
        raw = lyrics_path.read_text(encoding="utf-8")
        subs = list(srt.parse(raw))
        lyrics_text = "\n".join(sub.content for sub in subs)
        return lyrics_text, "srt"
    else:
        lyrics_text = lyrics_path.read_text(encoding="utf-8")
        return lyrics_text, "txt"


def load_genius_lyrics(lyrics_text: str) -> tuple:
    """Return (genius_lines, plain_text) from a single parse.

    Calls parse_genius_sections on the raw lyrics text and also returns
    a flattened plain-text version (headers stripped) so whisper alignment
    still works. The flattened text preserves the same non-blank lyric
    lines in the same order as the genius_lines list — both come from the
    same parse, guaranteeing 1:1 line correspondence.

    Returns:
        (genius_lines, plain_text) where plain_text is a newline-joined
        string of the ``text`` field of each genius_line, and
        genius_lines is the output of parse_genius_sections.
    """
    genius_lines = parse_genius_sections(lyrics_text)
    plain_lines = [gl["align_text"] for gl in genius_lines]
    plain_text = "\n".join(plain_lines)
    return genius_lines, plain_text


# Words with probability below this threshold are treated as Whisper
# hallucinations — they appear when the model is forced to emit tokens
# for a silent/unintelligible region and clusters them at a single
# zero-duration timestamp. Letting NW match them collapses whole lines
# to that timestamp. Real low-confidence words sit around 0.05+; the
# phantom cluster is in the 1e-5 to 1e-3 range.
_MIN_WORD_PROBABILITY = 0.0001


def extract_words(result) -> list:
    """Flatten WhisperResult into [{word, start, end, is_segment_first, speaker}, ...].

    Drops words with probability below _MIN_WORD_PROBABILITY (whisper
    hallucinations from silent regions). Each word dict is initialized
    with ``speaker: None`` and ``dominant_speaker: None`` so the type
    annotation is honest before speaker assignment runs.
    """
    all_words = []
    dropped = 0
    for segment in result.segments:
        kept_in_segment = 0
        for word in segment.words:
            prob = getattr(word, "probability", None)
            if prob is not None and prob < _MIN_WORD_PROBABILITY:
                dropped += 1
                continue
            all_words.append(
                {
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "is_segment_first": kept_in_segment == 0,
                    "speaker": None,
                    "dominant_speaker": None,
                }
            )
            kept_in_segment += 1
    if dropped:
        logger.info(
            "Dropped %d low-probability whisper words (< %.2f) — likely silent-region hallucinations",
            dropped, _MIN_WORD_PROBABILITY,
        )
    return all_words


_STRIP_NONWORD_RE = re.compile(r"[^\w]", re.UNICODE)

# Contraction expansions for normalization (both directions checked at match time)
_CONTRACTIONS = {
    "im": "i am", "ive": "i have", "id": "i would", "ill": "i will",
    "youre": "you are", "youve": "you have", "youd": "you would",
    "youll": "you will", "hes": "he is", "shes": "she is",
    "its": "it is", "weve": "we have", "were": "we are",
    "theyre": "they are", "theyve": "they have", "dont": "do not",
    "doesnt": "does not", "didnt": "did not", "cant": "cannot",
    "wont": "will not", "isnt": "is not", "arent": "are not",
    "wasnt": "was not", "werent": "were not", "wanna": "want to",
    "gonna": "going to", "gotta": "got to", "aint": "are not",
}


def _normalize_token(token: str) -> str:
    """Lowercase, NFKC-normalize, strip non-word chars."""
    token = unicodedata.normalize("NFKC", token).lower()
    return _STRIP_NONWORD_RE.sub("", token)


def _score(lyric_tok: str, whisper_tok: str) -> int:
    """Alignment score for a lyric/whisper token pair.

    +3 exact match, len >= 6 (anchor bonus — long words pin alignment)
    +2 exact match, len < 6
    +2 contraction equivalence (1:1 or 1:N split)
    +1 phonetic equivalence (_PHONETIC_EQUIV — vowel elongation, spelling variants)
    +1 one-edit-distance match (tokens >=3 chars) — handles minor typos
    0 mismatch
    """
    # Exact match: anchor bonus for long words
    if lyric_tok == whisper_tok:
        return 3 if len(lyric_tok) >= 6 else 2
    # Contraction equivalence.
    # All current _CONTRACTIONS keys and split values are <= 6 chars, so no
    # contraction match would benefit from the +3 anchor bonus. If 6+ char
    # contractions are added later, reorder the branches or merge the checks.
    l_exp = _CONTRACTIONS.get(lyric_tok)
    w_exp = _CONTRACTIONS.get(whisper_tok)
    if (l_exp == whisper_tok) or (lyric_tok == w_exp) or (l_exp and l_exp == w_exp):
        return 2
    # Handle 1:N splits: whisper may split a contraction the lyrics keep whole, or vice versa.
    # e.g. lyric "dont" vs whisper "do"/"not": check if whisper_tok is any word in l_exp.
    if l_exp and whisper_tok in l_exp.split():
        return 2
    if w_exp and lyric_tok in w_exp.split():
        return 2
    # Phonetic equivalence (vowel elongation, spelling variants)
    pair = frozenset({lyric_tok, whisper_tok})
    if pair in _PHONETIC_EQUIV:
        return 1
    # Fuzzy: Levenshtein-1 for tokens long enough that a 1-char edit is meaningful
    if len(lyric_tok) >= 3 and len(whisper_tok) >= 3:
        if _levenshtein(lyric_tok, whisper_tok) <= 1:
            return 1
    return 0


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance (no substitution weighting needed here)."""
    if abs(len(a) - len(b)) > 1:
        return 2  # early exit — can't be ≤1
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


_GAP_LYRIC = -2 # penalty for lyric word with no whisper match
_GAP_WHISPER = -1 # penalty for whisper token with no lyric match

# Maximum value _score() can return. Used by _needleman_wunsch to size the
# _NEG_INF sentinel and to guard the gap-cost invariant.
# UPDATE this if any branch of _score() returns a higher value.
_MAX_MATCH = 3

# Invariant: a single match must never exceed the cost of gapping both sides
# of a token. Otherwise NW becomes indifferent between matching a token and
# gapping both sides of it. Anchor bonus +3 ties gap-pair cost -3; this is
# acceptable because mismatches don't get the anchor bonus, and banding
# (Change 2) provides the real defense against drift. Must hold whenever
# _score / gap costs are tuned.
assert _MAX_MATCH <= -(_GAP_LYRIC + _GAP_WHISPER), \
    "Single match score must not exceed gap-pair cost"

# Phonetic equivalence pairs (vowel-elongation and a few spelling variants)
# that the Levenshtein-1 fuzzy gate (len >= 3) excludes.
# Score returned: +1 (same tier as fuzzy).
#
# NOTE: pairs are NOT transitive. If "mm"~"mmm" and "mmm"~"mmmm" are both
# needed later, add all three pairwise entries explicitly, OR refactor to a
# canonical-form dict:
#   _PHONETIC_CANON = {"mm": "mm", "mmm": "mm", "mmmm": "mm", ...}
#   match: _PHONETIC_CANON.get(a) == _PHONETIC_CANON.get(b) != None
#
# NOTE: this lookup MUST run after the exact-match branch.
# frozenset({"mm", "mm"}) collapses to frozenset({"mm"}) and won't match any
# pair entry — exact match catches that case at +2 first.
#
# Known gap NOT in this list: "na"/"nah". "nah" is a real English word meaning
# "no", so blanket equivalence with "na" produces false positives outside
# parenthesized vocal-run contexts. Revisit if na-na chorus patterns show real
# misalignment.
_PHONETIC_EQUIV = {
    # vowel-elongation (one side len 2, other len 3)
    frozenset({"mm", "mmm"}),
    frozenset({"uh", "uhh"}),
    frozenset({"oo", "ooh"}),
    frozenset({"hm", "hmm"}),
    frozenset({"wo", "woo"}),
    frozenset({"oh", "ooh"}),
    frozenset({"ah", "aah"}),
    frozenset({"ha", "hah"}),  # borderline: both are dictionary words
    frozenset({"ay", "ayy"}),
    frozenset({"la", "laa"}),  # common in sung choruses
    frozenset({"da", "daa"}),  # common in sung choruses
    # spelling variant (Levenshtein-2; both sides len 4)
    frozenset({"woah", "whoa"}),
}


_BAND_MIN_LENGTH = 500  # only band sequences longer than this


def _needleman_wunsch_unbanded(lyric_norms: list, whisper_norms: list) -> list:
    """Unbanded (full O(m*n)) Semi-Global NW alignment.

    Returns list of (lyric_idx | None, whisper_idx | None) pairs.
    None on either side means a gap on that sequence.
    Handles asymmetric penalties and free whisper prefix/suffix.
    """
    m, n = len(lyric_norms), len(whisper_norms)

    # Scoped score cache — same pair appears in both DP fill and traceback.
    score_cache: dict = {}

    def _cached_score(a: str, b: str) -> int:
        key = (a, b)
        if key not in score_cache:
            score_cache[key] = _score(a, b)
        return score_cache[key]

    # Fill DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * _GAP_LYRIC
    for j in range(n + 1):
        dp[0][j] = 0  # Free Whisper prefix

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + _cached_score(lyric_norms[i - 1], whisper_norms[j - 1])
            delete = dp[i - 1][j] + _GAP_LYRIC  # lyric token with no whisper match
            insert = dp[i][j - 1] + _GAP_WHISPER  # whisper token with no lyric match
            dp[i][j] = max(match, delete, insert)

    # Traceback
    alignment = []

    # Find best Whisper suffix point (free suffix)
    best_j = n
    max_score = dp[m][n]
    for j_opt in range(n):
        if dp[m][j_opt] > max_score:
            max_score = dp[m][j_opt]
            best_j = j_opt

    i, j = m, best_j

    # Mark skipped suffix whisper tokens
    for j_skip in range(n, best_j, -1):
        alignment.append((None, j_skip - 1))

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = _cached_score(lyric_norms[i - 1], whisper_norms[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + s:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + _GAP_LYRIC:
            alignment.append((i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + _GAP_WHISPER:
            alignment.append((None, j - 1))
            j -= 1
        elif j > 0 and i == 0:
            # Reached top row, free prefix traceback
            alignment.append((None, j - 1))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment


def _needleman_wunsch_banded(lyric_norms: list, whisper_norms: list) -> list:
    """Banded (Sakoe-Chiba) Semi-Global NW alignment with taper and fallback.

    Constrain the inner DP to a band around the expected diagonal. If the
    banded result looks degenerate (true optimal alignment likely exited the
    band), fall back to unbanded NW.
    """
    m, n = len(lyric_norms), len(whisper_norms)

    # Scoped score cache — same pair appears in both DP fill and traceback.
    score_cache: dict = {}

    def _cached_score(a: str, b: str) -> int:
        key = (a, b)
        if key not in score_cache:
            score_cache[key] = _score(a, b)
        return score_cache[key]

    band = max(50, max(m, n) // 4)
    taper_rows = min(band, m // 4)

    # Sentinel for out-of-band cells. Must stay more negative than any
    # achievable real path value, even after _MAX_MATCH bonuses are added
    # during transitions. The * 2 margin keeps the invariant under future
    # scoring tweaks.
    neg_inf = -(m + n + 1) * _MAX_MATCH * 2
    assert neg_inf + _MAX_MATCH * max(m, n) < 0, \
        "Sentinel insufficient for current scoring constants"

    # Fill DP table (full allocation; defer row-band repr for long-form)
    dp = [[neg_inf] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * _GAP_LYRIC
    for j in range(n + 1):
        dp[0][j] = 0  # Free Whisper prefix

    for i in range(1, m + 1):
        # round() uses banker's rounding (round-half-to-even); the
        # 0.5-token jitter is absorbed by band width >= 50.
        j_exp = round(i * n / m)

        # Taper: widen the band over the last `taper_rows` rows so the
        # free-suffix region is reachable without a cliff.
        rows_from_end = m - i
        if taper_rows > 0 and rows_from_end < taper_rows:
            extra = (taper_rows - rows_from_end) * (n - band) // taper_rows
            eff_band = band + max(0, extra)
        else:
            eff_band = band

        j_lo = max(1, j_exp - eff_band)
        j_hi = min(n, j_exp + eff_band)

        for j in range(j_lo, j_hi + 1):
            match_s = dp[i - 1][j - 1] + _cached_score(lyric_norms[i - 1], whisper_norms[j - 1])
            delete_s = dp[i - 1][j] + _GAP_LYRIC
            insert_s = dp[i][j - 1] + _GAP_WHISPER
            dp[i][j] = max(match_s, delete_s, insert_s)

    # Find best Whisper suffix point (free suffix)
    best_j = n
    max_score = dp[m][n]
    for j_opt in range(n):
        if dp[m][j_opt] > max_score:
            max_score = dp[m][j_opt]
            best_j = j_opt

    # Degenerate-result fallback: a healthy alignment should score at least
    # ~1 match per 3 lyric tokens. If the banded result is much worse, the
    # band likely clipped the true path.
    expected_floor = (m / 3) * 2 + (m * 2 / 3) * _GAP_LYRIC
    if dp[m][best_j] < expected_floor:
        logger.warning(
            "Banded NW score %d below expected floor %d; re-running unbanded",
            dp[m][best_j], int(expected_floor),
        )
        return _needleman_wunsch_unbanded(lyric_norms, whisper_norms)

    # Traceback (same logic as unbanded, but some dp cells are neg_inf)
    alignment = []

    i, j = m, best_j

    # Mark skipped suffix whisper tokens
    for j_skip in range(n, best_j, -1):
        alignment.append((None, j_skip - 1))

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = _cached_score(lyric_norms[i - 1], whisper_norms[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + s:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + _GAP_LYRIC:
            alignment.append((i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + _GAP_WHISPER:
            alignment.append((None, j - 1))
            j -= 1
        elif j > 0 and i == 0:
            # Reached top row, free prefix traceback
            alignment.append((None, j - 1))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment


def _needleman_wunsch(lyric_norms: list, whisper_norms: list) -> list:
    """Semi-Global sequence alignment dispatcher.

    Returns list of (lyric_idx | None, whisper_idx | None) pairs.
    None on either side means a gap on that sequence.

    For short sequences (max(m,n) <= _BAND_MIN_LENGTH), uses full O(m*n)
    unbanded NW. For longer sequences, uses Sakoe-Chiba banded NW with
    taper and degenerate-result fallback to unbanded.
    """
    if max(len(lyric_norms), len(whisper_norms)) <= _BAND_MIN_LENGTH:
        return _needleman_wunsch_unbanded(lyric_norms, whisper_norms)
    return _needleman_wunsch_banded(lyric_norms, whisper_norms)


def match_words_to_lines(words: list, lines: list, align_lines: list = None) -> list:
    """Assign whisper words to lyric lines via Needleman-Wunsch alignment.

    Replaces the old count-based approach. Each lyric token is globally
    aligned to a whisper token using a scoring function (+2 exact, +1
    fuzzy/contraction, 0 mismatch, -1 gap). This tolerates contraction
    splitting, punctuation differences, and limited whisper hallucinations
    without cascading the error across all subsequent lines.

    Whisper words that match no lyric token (hallucinations, backing vox)
    are silently dropped. Lyric lines where no whisper words aligned get
    an empty words list and inherit the previous line's end time — the
    ASS generator skips them; SRT still emits the text.

    Args:
        words: flat whisper word list from extract_words().
        lines: display text per lyric line (may include inline parens).
        align_lines: paren-stripped text per lyric line for alignment.
            If None, falls back to lines.
    """
    if align_lines is None:
        align_lines = lines

    # Build flat lyric token list tagged with line index
    lyric_tokens = []  # [(norm_tok, line_idx)]
    for line_idx, aline in enumerate(align_lines):
        # Handle em-dashes and en-dashes by splitting them into separate words before normalization
        aline_split = aline.replace("—", " ").replace("--", " ")
        for tok in aline_split.split():
            norm = _normalize_token(tok)
            if norm:
                lyric_tokens.append((norm, line_idx))

    if not lyric_tokens or not words:
        return [{"text": d, "words": [], "start": 0.0, "end": 0.0} for d in lines]

    whisper_norms = [_normalize_token(w["word"]) for w in words]
    lyric_norms = [lt[0] for lt in lyric_tokens]

    alignment = _needleman_wunsch(lyric_norms, whisper_norms)

    # Map each lyric token index → whisper word index (or None = no match).
    # Filter: only record pairs where _score() >= 1. NW may pair tokens at
    # score 0 (mismatch) when surrounding context makes that the locally-optimal
    # path; those pairs are structurally correct but timestamp-unreliable.
    # Rejected lyric tokens fall through to the "start/end is None → interpolate
    # from neighbors" branch below.
    #
    # Threshold semantics: >= 1 accepts exact match (+2, +3 after Change 1),
    # contraction (+2), Levenshtein-1 fuzzy (+1), and _PHONETIC_EQUIV (+1 after
    # Change 1). Score 0 (mismatch) is rejected. If a future change wants to
    # filter more aggressively (e.g. accept only >= 2), _score() will need to
    # return (score, match_type) tuples so fuzzy and vocal-equiv can be
    # distinguished.
    #
    # _score() is stateless; recomputing here gives the same value used during
    # NW DP fill. If _score() ever becomes context-aware, switch to returning
    # (l_idx, w_idx, score) triples from _needleman_wunsch().
    #
    # Edge case: when the filter rejects a pair (l_idx, w_idx), the whisper
    # token at w_idx is still consumed by NW's monotone alignment and is no
    # longer available for any other lyric token. It also won't appear in any
    # line's word list. For hallucinations this is desired; for the rare case
    # where a real whisper word is paired at score 0, the token is silently
    # dropped and interpolated timing is used instead.
    lyric_to_whisper = [None] * len(lyric_tokens)
    for l_idx, w_idx in alignment:
        if l_idx is not None and w_idx is not None:
            if _score(lyric_norms[l_idx], whisper_norms[w_idx]) >= 1:
                lyric_to_whisper[l_idx] = w_idx

    # Group whisper word indices by lyric line (monotone → already ordered).
    # Diagnostic: warn if a line's matched whisper indices are non-monotonic
    # relative to the previous line's last index — that's the signal that a
    # line-locality tie-break in NW traceback would help.
    n_lines = len(lines)
    line_whisper_indices: list[list[int]] = [[] for _ in range(n_lines)]
    prev_line_last_widx = -1
    for ltok_idx, (_, line_idx) in enumerate(lyric_tokens):
        w_idx = lyric_to_whisper[ltok_idx]
        if w_idx is not None:
            line_whisper_indices[line_idx].append(w_idx)
            if w_idx < prev_line_last_widx and line_idx > 0:
                logger.warning(
                    "Non-monotonic whisper index: line %d got w_idx %d "
                    "but previous line ended at w_idx %d — possible line-boundary leakage",
                    line_idx, w_idx, prev_line_last_widx,
                )
            prev_line_last_widx = max(prev_line_last_widx, w_idx)

    # Build one line_obj per lyric line
    line_objects = []
    for line_idx in range(n_lines):
        display_text = lines[line_idx]
        w_indices = line_whisper_indices[line_idx]

        if not w_indices:
            line_objects.append({
                "text": display_text,
                "words": [],
                "start": None,
                "end": None,
            })
        else:
            # Deduplicate while preserving order (monotone alignment)
            seen: set = set()
            unique: list = []
            for idx in w_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique.append(idx)
            line_words = [words[i] for i in unique]
            line_objects.append({
                "text": display_text,
                "words": line_words,
                "start": line_words[0]["start"],
                "end": line_words[-1]["end"],
            })

    # Interpolate missing lines
    for i, obj in enumerate(line_objects):
        if obj["start"] is None:
            logger.warning("No aligned whisper words for line: %r. Interpolating timestamps.", obj["text"])
            
            prev_end = 0.0
            for j in range(i - 1, -1, -1):
                if line_objects[j]["end"] is not None:
                    prev_end = line_objects[j]["end"]
                    break
                    
            next_start = prev_end
            for j in range(i + 1, len(line_objects)):
                if line_objects[j]["start"] is not None:
                    next_start = line_objects[j]["start"]
                    break
                    
            obj["start"] = prev_end
            obj["end"] = next_start

    return line_objects


def _match_simple(lyric_tok: str, whisper_tok: str) -> bool:
    """Cheap equivalence check: exact, contraction, or 1:N contraction split.

    Strict-by-design: no Levenshtein, no phonetic table. The walk matcher
    relies on stable-ts emitting words in reference order, so we just need
    to recognize the cases where a single reference token corresponds to
    a different surface form in the whisper output.
    """
    if lyric_tok == whisper_tok:
        return True
    l_exp = _CONTRACTIONS.get(lyric_tok)
    w_exp = _CONTRACTIONS.get(whisper_tok)
    if l_exp == whisper_tok or lyric_tok == w_exp:
        return True
    if l_exp and l_exp == w_exp:
        return True
    if l_exp and whisper_tok in l_exp.split():
        return True
    if w_exp and lyric_tok in w_exp.split():
        return True
    return False


def _walk_align(lyric_norms: list, whisper_norms: list, lookahead: int = 3) -> list:
    """Two-pointer lockstep alignment with bounded lookahead.

    Returns mapping[lyric_idx] = whisper_idx | None. On mismatch, scans an
    (lookahead+1) x (lookahead+1) window for the nearest re-sync point and
    advances both pointers to it (skipped lyric tokens stay None; skipped
    whisper words are dropped). If no re-sync is found within the window,
    advances both pointers by one.
    """
    m, n = len(lyric_norms), len(whisper_norms)
    mapping: list = [None] * m
    i = j = 0
    while i < m and j < n:
        if _match_simple(lyric_norms[i], whisper_norms[j]):
            mapping[i] = j
            i += 1
            j += 1
            continue
        best = None
        best_cost = lookahead * 2 + 1
        for di in range(min(lookahead + 1, m - i)):
            for dj in range(min(lookahead + 1, n - j)):
                if di == 0 and dj == 0:
                    continue
                cost = di + dj
                if cost >= best_cost:
                    continue
                if _match_simple(lyric_norms[i + di], whisper_norms[j + dj]):
                    best = (di, dj)
                    best_cost = cost
        if best is None:
            i += 1
            j += 1
        else:
            di, dj = best
            i += di
            j += dj
            mapping[i] = j
            i += 1
            j += 1
    return mapping


def match_words_to_lines_walk(
    words: list, lines: list, align_lines: list = None, lookahead: int = 3
) -> list:
    """Two-pointer walk matcher with gap interpolation.

    Trusts stable-ts to emit whisper words in reference order. For each
    reference token we either pair it with a whisper word (and use that
    word's timing) or, if it has no match, synthesize a placeholder word
    with timestamps interpolated linearly between the surrounding matched
    anchors. Every reference token therefore ends up in the karaoke
    output, and the final line word lists are gap-free.

    Args:
        words: flat whisper word list from extract_words().
        lines: display text per lyric line (may include inline parens).
        align_lines: paren-stripped text per lyric line for alignment.
            If None, falls back to lines.
        lookahead: search radius (in tokens) used to recover from local
            desyncs (contraction splits, dropped words, hallucinations).
    """
    if align_lines is None:
        align_lines = lines

    # Build lyric tokens: (norm, line_idx, raw_display_text)
    lyric_tokens: list = []
    for line_idx, aline in enumerate(align_lines):
        aline_split = aline.replace("—", " ").replace("--", " ")
        for tok in aline_split.split():
            norm = _normalize_token(tok)
            if norm:
                lyric_tokens.append((norm, line_idx, tok))

    n_lines = len(lines)
    if not lyric_tokens or not words:
        return [{"text": d, "words": [], "start": 0.0, "end": 0.0} for d in lines]

    whisper_norms = [_normalize_token(w["word"]) for w in words]
    lyric_norms = [lt[0] for lt in lyric_tokens]

    mapping = _walk_align(lyric_norms, whisper_norms, lookahead)

    # Build a per-token word dict — matched tokens take whisper timing,
    # unmatched runs get linearly-interpolated timings between anchors.
    n_tokens = len(lyric_tokens)
    token_words: list = [None] * n_tokens
    for k, (_, _, raw) in enumerate(lyric_tokens):
        w_idx = mapping[k]
        if w_idx is not None:
            wsrc = words[w_idx]
            token_words[k] = {
                "word": raw,
                "start": wsrc["start"],
                "end": wsrc["end"],
                "is_segment_first": False,
                "speaker": None,
                "dominant_speaker": None,
            }

    matched_count = sum(1 for tw in token_words if tw is not None)
    logger.info(
        "Walk align: matched %d/%d reference tokens (%.1f%%); %d whisper words consumed",
        matched_count, n_tokens,
        100.0 * matched_count / n_tokens if n_tokens else 0.0,
        sum(1 for v in mapping if v is not None),
    )

    k = 0
    while k < n_tokens:
        if token_words[k] is not None:
            k += 1
            continue
        run_end = k
        while run_end < n_tokens and token_words[run_end] is None:
            run_end += 1
        prev_end = token_words[k - 1]["end"] if k > 0 else 0.0
        next_start = token_words[run_end]["start"] if run_end < n_tokens else prev_end
        # Guard against pathologically out-of-order whisper output
        if next_start < prev_end:
            next_start = prev_end
        run_len = run_end - k
        slot = (next_start - prev_end) / run_len if run_len > 0 else 0.0
        for offset in range(run_len):
            s = prev_end + offset * slot
            e = prev_end + (offset + 1) * slot
            _, _, raw = lyric_tokens[k + offset]
            token_words[k + offset] = {
                "word": raw,
                "start": s,
                "end": e,
                "is_segment_first": False,
                "speaker": None,
                "dominant_speaker": None,
            }
        k = run_end

    # Group by line_idx
    line_word_lists: list = [[] for _ in range(n_lines)]
    for k, (_, line_idx, _) in enumerate(lyric_tokens):
        line_word_lists[line_idx].append(token_words[k])

    line_objects = []
    for line_idx in range(n_lines):
        text = lines[line_idx]
        line_words = line_word_lists[line_idx]
        if line_words:
            line_objects.append({
                "text": text,
                "words": line_words,
                "start": line_words[0]["start"],
                "end": line_words[-1]["end"],
            })
        else:
            line_objects.append({
                "text": text,
                "words": [],
                "start": None,
                "end": None,
            })

    # Lines with no normalizable tokens (e.g. paren-only display lines)
    # have no anchors of their own — interpolate start/end from neighbors.
    # No warning: paren-only lines are expected, not an alignment failure.
    for i, obj in enumerate(line_objects):
        if obj["start"] is None:
            prev_end = 0.0
            for j in range(i - 1, -1, -1):
                if line_objects[j]["end"] is not None:
                    prev_end = line_objects[j]["end"]
                    break
            next_start = prev_end
            for j in range(i + 1, len(line_objects)):
                if line_objects[j]["start"] is not None:
                    next_start = line_objects[j]["start"]
                    break
            obj["start"] = prev_end
            obj["end"] = next_start

    return line_objects


def match_words_to_lines_by_count(words: list, lines: list, align_lines: list = None) -> list:
    """Legacy count-based pairing — kept as a CLI fallback for comparison.

    Slices the flat whisper word list by lyric-line word count, in order.
    Brittle: any mismatch between lyric word count and aligned word count
    cascades to all subsequent lines.

    Args:
        words: flat whisper word list from extract_words().
        lines: display text per lyric line (may include inline parens).
        align_lines: stripped text per lyric line used for word counting.
            If None, falls back to lines.
    """
    if align_lines is None:
        align_lines = lines

    line_objects = []
    word_index = 0

    for display_line, align_line in zip(lines, align_lines):
        line_word_count = len(align_line.split())
        line_words = words[word_index : word_index + line_word_count]
        word_index += line_word_count

        if not line_words:
            line_objects.append({"text": display_line, "words": [], "start": 0.0, "end": 0.0})
            continue

        line_objects.append(
            {
                "text": display_line,
                "words": line_words,
                "start": line_words[0]["start"],
                "end": line_words[-1]["end"],
            }
        )

    return line_objects


def segments_to_line_objects(result) -> list:
    """Build line objects directly from stable-ts segments (transcription mode).

    Each segment becomes one subtitle line; its words are used for karaoke
    timing. Segments with no words are skipped. Each word dict is
    initialized with ``speaker: None`` and ``dominant_speaker: None``.
    """
    line_objects = []
    for segment in result.segments:
        if not segment.words:
            continue
        words = [
            {
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "is_segment_first": i == 0,
                "speaker": None,
                "dominant_speaker": None,
            }
            for i, w in enumerate(segment.words)
        ]
        line_objects.append(
            {
                "text": segment.text.strip(),
                "words": words,
                "start": words[0]["start"],
                "end": words[-1]["end"],
            }
        )
    return line_objects
