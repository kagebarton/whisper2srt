"""Word extraction and walk-matcher for genius_align.

Pruned fork of ``genius_diarize.word_extraction``: only the two-pointer
walk matcher is kept. NW alignment and count-based slicing have been
removed in favor of the walk matcher with gap interpolation.
"""


import logging
import re
import unicodedata
from pathlib import Path

import srt

from genius_align.genius import parse_genius_sections

logger = logging.getLogger(__name__)


def load_lyrics(lyrics_path: Path) -> tuple:
    """Return (lyrics_text, lyrics_format) where format is 'txt' or 'srt'."""
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

    The plain-text version preserves the same non-blank lyric lines in
    the same order as ``genius_lines`` so whisper alignment and
    speaker-attribution stay 1:1.
    """
    genius_lines = parse_genius_sections(lyrics_text)
    plain_lines = [gl["align_text"] for gl in genius_lines]
    plain_text = "\n".join(plain_lines)
    return genius_lines, plain_text


# Whisper hallucinations from silent regions cluster at near-zero probability.
_MIN_WORD_PROBABILITY = 0.0001


def extract_words(result, min_prob: float = 0.0001) -> list:
    """Flatten WhisperResult into [{word, start, end, is_segment_first, speaker}, ...].

    Drops words below ``min_prob`` (silent-region phantoms).
    """
    all_words = []
    dropped = 0
    for segment in result.segments:
        kept_in_segment = 0
        for word in segment.words:
            prob = getattr(word, "probability", None)
            if prob is not None and prob < min_prob:
                dropped += 1
                continue
            all_words.append(
                {
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "speaker": None,
                    "dominant_speaker": None,
                }
            )
            kept_in_segment += 1
    if dropped:
        logger.info(
            "Dropped %d low-probability whisper words (< %.4f)",
            dropped, min_prob,
        )
    return all_words


_STRIP_NONWORD_RE = re.compile(r"[^\w]", re.UNICODE)

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


def _walk_align(
    lyric_norms: list,
    whisper_norms: list,
    lyric_lookahead: int = 3,
    whisper_lookahead: int = 10,
    confirm_matches: int = 2,
    whisper_skip_budget: int = 10,
) -> list:
    """Two-pointer alignment biased toward preserving lyric anchors.

    Returns mapping[lyric_idx] = whisper_idx | None.

    On mismatch, scans an asymmetric (lyric_lookahead+1) x (whisper_lookahead+1)
    window for a re-sync point. A candidate is only accepted if the next
    confirm_matches-1 token pairs also match — this filters out spurious
    single-word anchors on common short tokens ("I", "the", "you") during
    a desync.

    When no confirmed re-sync exists in the window, advances only the
    whisper pointer (dropping noisy/extra whisper output) up to
    whisper_skip_budget steps before reluctantly skipping the current
    lyric token. This bias matches the common failure mode: the lyrics
    file is the source of truth, whisper output is noise to absorb.
    Skipped lyric tokens stay None and get interpolated downstream.
    """
    m, n = len(lyric_norms), len(whisper_norms)
    mapping: list = [None] * m
    i = j = 0
    whisper_skips = 0
    while i < m and j < n:
        if _match_simple(lyric_norms[i], whisper_norms[j]):
            mapping[i] = j
            i += 1
            j += 1
            whisper_skips = 0
            continue
        best = None
        best_cost = lyric_lookahead + whisper_lookahead + 1
        max_di = min(lyric_lookahead + 1, m - i)
        max_dj = min(whisper_lookahead + 1, n - j)
        for di in range(max_di):
            for dj in range(max_dj):
                if di == 0 and dj == 0:
                    continue
                cost = di + dj
                if cost >= best_cost:
                    continue
                if not _match_simple(lyric_norms[i + di], whisper_norms[j + dj]):
                    continue
                # Confirm with the next (confirm_matches - 1) pairs. If we
                # run off either sequence before confirming, accept anyway
                # — end-of-sequence anchors can't be confirmed but are
                # usually still correct.
                confirmed = True
                for k in range(1, confirm_matches):
                    li, wi = i + di + k, j + dj + k
                    if li >= m or wi >= n:
                        break
                    if not _match_simple(lyric_norms[li], whisper_norms[wi]):
                        confirmed = False
                        break
                if confirmed:
                    best = (di, dj)
                    best_cost = cost
        if best is None:
            # Advance whisper pointer only: preserve the current lyric
            # token as an unmatched candidate. Cap consecutive whisper-only
            # advances so a truly missing lyric token (skipped chorus,
            # ad-libbed line) doesn't stall the walker forever.
            j += 1
            whisper_skips += 1
            if whisper_skips >= whisper_skip_budget:
                i += 1
                whisper_skips = 0
        else:
            di, dj = best
            i += di
            j += dj
            mapping[i] = j
            i += 1
            j += 1
            whisper_skips = 0
    return mapping


def match_words_to_lines_walk(
    words: list,
    lines: list,
    align_lines: list = None,
    lyric_lookahead: int = 3,
    whisper_lookahead: int = 10,
    confirm_matches: int = 2,
    whisper_skip_budget: int = 10,
    max_interp_run: int = 5,
    min_interp_slot: float = 0.1,
    max_collapsed_run: int = 8,
    collapse_window: float = 0.3,
) -> list:
    """Two-pointer walk matcher with gap interpolation.

    Trusts stable-ts to emit whisper words in reference order. For each
    reference token we either pair it with a whisper word (and use that
    word's timing) or synthesize a placeholder word with timestamps
    interpolated linearly between the surrounding matched anchors. Short
    unmatched runs are interpolated so the line stays gap-free; a run is
    dropped entirely — rather than interpolated — when either it is
    longer than ``max_interp_run`` or its interpolation slot would fall
    below ``min_interp_slot``. Both signal lyrics absent from the audio
    (skipped section, spurious anchor): faking timing would just animate
    phantom words, and a near-zero slot crams them illegibly.

    Lines whose tokens are all dropped via the interp cap return with
    ``words=[]`` and ``start=None`` — the SRT/ASS generators skip these.
    Paren-only display lines (no normalizable tokens at all) still inherit
    timing from neighbors and render statically.

    Args:
        words: flat whisper word list from the worker.
        lines: display text per lyric line (may include inline parens).
        align_lines: paren-stripped text per lyric line for alignment.
            If None, falls back to lines.
        lyric_lookahead: max lyric tokens to skip when re-syncing.
        whisper_lookahead: max whisper words to skip when re-syncing —
            asymmetrically larger so the walker can absorb stretches of
            extra/noisy whisper output without losing lyric anchors.
        confirm_matches: required consecutive matches at a re-sync point
            before accepting it (filters spurious common-word anchors).
        whisper_skip_budget: max consecutive whisper-only advances before
            the walker reluctantly skips the current lyric token.
        max_interp_run: largest unmatched-token run that still gets
            linearly interpolated. Longer runs are dropped — assumed to
            be lyrics absent from the audio (skipped verse, restructured
            chorus). Set to a very large number to disable the cap.
        min_interp_slot: minimum per-token interpolation slot, in seconds.
            If the bracketing anchors are too close to give each token at
            least this much time, the run is dropped instead of crammed.
            Catches the case where spurious common-word matches fragment a
            skipped section into short, length-cap-passing sub-runs.
            Set to 0.0 to disable.
        max_collapsed_run: longest run of *matched* tokens allowed to share
            essentially one timestamp. When align() can't locate a lyric
            section it force-places every word in it at a single instant;
            the walk matcher pairs those 1:1, so the interp caps never see
            them. A run of more than this many matched tokens spanning less
            than ``collapse_window`` seconds is demoted to unmatched so the
            interp/drop logic applies. The length threshold is the
            disambiguator — a handful of words sharing a timestamp is
            normal fast singing; dozens is align() giving up. Set to a very
            large number to disable.
        collapse_window: max span, in seconds, for a matched-token run to
            count as "collapsed onto one instant" for the
            ``max_collapsed_run`` check.
    """
    if align_lines is None:
        align_lines = lines

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

    mapping = _walk_align(
        lyric_norms,
        whisper_norms,
        lyric_lookahead=lyric_lookahead,
        whisper_lookahead=whisper_lookahead,
        confirm_matches=confirm_matches,
        whisper_skip_budget=whisper_skip_budget,
    )

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
                "speaker": None,
                "dominant_speaker": None,
            }

    # Demote collapsed matched runs: when align() can't locate a lyric
    # section it force-places every word in it at one timestamp. The walk
    # matcher pairs those 1:1 so the interp caps never see them — find
    # long runs of matched tokens crammed into < collapse_window seconds
    # and demote them to None so the interp/drop logic below applies. The
    # *length* is the disambiguator: a few words at one timestamp is
    # normal fast singing, dozens is align() giving up.
    collapsed_tokens = 0
    if max_collapsed_run < n_tokens:
        k = 0
        while k < n_tokens:
            if token_words[k] is None:
                k += 1
                continue
            run_end = k + 1
            while (
                run_end < n_tokens
                and token_words[run_end] is not None
                and token_words[run_end]["start"] - token_words[k]["start"]
                < collapse_window
            ):
                run_end += 1
            run_len = run_end - k
            if run_len > max_collapsed_run:
                for idx in range(k, run_end):
                    token_words[idx] = None
                collapsed_tokens += run_len
                k = run_end
            else:
                k += 1

    matched_count = sum(1 for tw in token_words if tw is not None)

    # Interpolate short unmatched runs; drop long runs entirely (token_words
    # stays None — those tokens won't appear in the karaoke output).
    dropped_tokens = 0
    k = 0
    while k < n_tokens:
        if token_words[k] is not None:
            k += 1
            continue
        run_end = k
        while run_end < n_tokens and token_words[run_end] is None:
            run_end += 1
        run_len = run_end - k
        if run_len > max_interp_run:
            dropped_tokens += run_len
            k = run_end
            continue
        prev_end = token_words[k - 1]["end"] if k > 0 else 0.0
        next_start = token_words[run_end]["start"] if run_end < n_tokens else prev_end
        if next_start < prev_end:
            next_start = prev_end
        slot = (next_start - prev_end) / run_len if run_len > 0 else 0.0
        if run_len > 0 and slot < min_interp_slot:
            # Degenerate interpolation: the bracketing anchors are too
            # close to fit these tokens at a readable pace. Almost always
            # a skipped section whose run was fragmented by spurious
            # common-word anchors — drop it rather than cram near-zero-
            # duration phantom words into a sliver of time.
            dropped_tokens += run_len
            k = run_end
            continue
        for offset in range(run_len):
            s = prev_end + offset * slot
            e = prev_end + (offset + 1) * slot
            _, _, raw = lyric_tokens[k + offset]
            token_words[k + offset] = {
                "word": raw,
                "start": s,
                "end": e,
                "speaker": None,
                "dominant_speaker": None,
            }
        k = run_end

    logger.info(
        "Walk align: matched %d/%d reference tokens (%.1f%%); "
        "%d whisper words consumed; %d tokens demoted (collapsed run); "
        "%d tokens dropped (uninterpolatable)",
        matched_count, n_tokens,
        100.0 * matched_count / n_tokens if n_tokens else 0.0,
        sum(1 for v in mapping if v is not None),
        collapsed_tokens,
        dropped_tokens,
    )

    # Track which lines had any normalizable tokens at all, so we can
    # distinguish "paren-only display line" (inherit neighbor timing,
    # render statically) from "every token was dropped by the interp cap"
    # (suppress the line — audio doesn't contain it).
    lines_with_tokens: set = set()
    for _, line_idx, _ in lyric_tokens:
        lines_with_tokens.add(line_idx)

    line_word_lists: list = [[] for _ in range(n_lines)]
    for k, (_, line_idx, _) in enumerate(lyric_tokens):
        if token_words[k] is not None:
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

    # Lines with no normalizable tokens (paren-only display lines) inherit
    # timing from neighbors. Lines whose tokens were all dropped via the
    # interp cap stay with start=None so the SRT/ASS generators skip them.
    for i, obj in enumerate(line_objects):
        if obj["start"] is not None or i in lines_with_tokens:
            continue
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
