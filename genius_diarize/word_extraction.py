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

    +2 exact normalized match
    +1 one-edit-distance match (tokens ≥3 chars) — handles minor typos
         and contracted vs. expanded forms
     0 mismatch
    """
    if lyric_tok == whisper_tok:
        return 2
    # Check contraction equivalence
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


_GAP = -1  # gap penalty for NW


def _needleman_wunsch(lyric_norms: list, whisper_norms: list) -> list:
    """Global sequence alignment (Needleman-Wunsch).

    Returns list of (lyric_idx | None, whisper_idx | None) pairs.
    None on either side means a gap on that sequence.
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
        dp[i][0] = i * _GAP
    for j in range(n + 1):
        dp[0][j] = j * _GAP

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + _cached_score(lyric_norms[i - 1], whisper_norms[j - 1])
            delete = dp[i - 1][j] + _GAP  # lyric token with no whisper match
            insert = dp[i][j - 1] + _GAP  # whisper token with no lyric match
            dp[i][j] = max(match, delete, insert)

    # Traceback
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = _cached_score(lyric_norms[i - 1], whisper_norms[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + s:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + _GAP:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1

    alignment.reverse()
    return alignment


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
        for tok in aline.split():
            norm = _normalize_token(tok)
            if norm:
                lyric_tokens.append((norm, line_idx))

    if not lyric_tokens or not words:
        return [{"text": d, "words": [], "start": 0.0, "end": 0.0} for d in lines]

    whisper_norms = [_normalize_token(w["word"]) for w in words]
    lyric_norms = [lt[0] for lt in lyric_tokens]

    alignment = _needleman_wunsch(lyric_norms, whisper_norms)

    # Map each lyric token index → whisper word index (or None = no match)
    lyric_to_whisper = [None] * len(lyric_tokens)
    for l_idx, w_idx in alignment:
        if l_idx is not None and w_idx is not None:
            lyric_to_whisper[l_idx] = w_idx

    # Group whisper word indices by lyric line (monotone → already ordered)
    n_lines = len(lines)
    line_whisper_indices: list[list[int]] = [[] for _ in range(n_lines)]
    for ltok_idx, (_, line_idx) in enumerate(lyric_tokens):
        w_idx = lyric_to_whisper[ltok_idx]
        if w_idx is not None:
            line_whisper_indices[line_idx].append(w_idx)

    # Build one line_obj per lyric line
    line_objects = []
    prev_end = 0.0
    for line_idx in range(n_lines):
        display_text = lines[line_idx]
        w_indices = line_whisper_indices[line_idx]

        if not w_indices:
            logger.warning("No aligned whisper words for line: %r", display_text)
            line_obj = {"text": display_text, "words": [], "start": prev_end, "end": prev_end}
        else:
            # Deduplicate while preserving order (monotone alignment)
            seen: set = set()
            unique: list = []
            for idx in w_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique.append(idx)
            line_words = [words[i] for i in unique]
            line_obj = {
                "text": display_text,
                "words": line_words,
                "start": line_words[0]["start"],
                "end": line_words[-1]["end"],
            }
            prev_end = line_obj["end"]

        line_objects.append(line_obj)

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
