"""Word extraction and line-building helpers.

Adapted from diarized_captions/word_extraction.py. Key change:
load_genius_lyrics() returns (genius_lines, plain_text) from a single
parse so both outputs are 1:1 in line count and order.
"""

import logging
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


def extract_words(result) -> list:
    """Flatten WhisperResult into [{word, start, end, is_segment_first, speaker}, ...].

    Each word dict is initialized with ``speaker: None`` and
    ``dominant_speaker: None`` so the type annotation is honest before
    speaker assignment runs.
    """
    all_words = []
    for segment in result.segments:
        for i, word in enumerate(segment.words):
            all_words.append(
                {
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "is_segment_first": i == 0,
                    "speaker": None,
                    "dominant_speaker": None,
                }
            )
    return all_words


def match_words_to_lines(words: list, lines: list, align_lines: list = None) -> list:
    """Assign aligned words to lyrics lines by count.

    Count-based pairing: assumes the lyrics file has the same word count
    and order as what stable-ts aligned.

    Args:
        words: flat whisper word list from extract_words().
        lines: display text per lyric line (may include inline parens).
        align_lines: stripped text per lyric line used for word counting.
            If None, falls back to lines. Pass genius_line["align_text"]
            values here so inline parentheticals don't inflate the count.
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
            continue

        line_obj = {
            "text": display_line,
            "words": line_words,
            "start": line_words[0]["start"],
            "end": line_words[-1]["end"],
        }
        line_objects.append(line_obj)

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
