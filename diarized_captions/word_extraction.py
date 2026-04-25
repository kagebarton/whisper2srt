"""Word extraction and line-building helpers.

Ported from pipeline/stages/lyric_align.py (methods on LyricAlignStage
converted to standalone functions).
"""

import logging
from pathlib import Path

import srt

from diarized_captions.genius import parse_genius_sections

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


def load_genius_lyrics(lyrics_path: Path) -> tuple:
    """Return (lyrics_text, genius_lines) for a Genius-formatted lyrics file.

    Calls ``parse_genius_sections`` on the raw file text and also returns
    a flattened plain-text version (headers stripped) so whisper alignment
    still works.  The flattened text preserves the same non-blank lyric
    lines in the same order as the genius_lines list.

    Returns:
        (plain_text, genius_lines) where plain_text is a newline-joined
        string of lyric lines (no headers, no blank lines) and
        genius_lines is the output of ``parse_genius_sections``.
    """
    raw_text = lyrics_path.read_text(encoding="utf-8")
    genius_lines = parse_genius_sections(raw_text)
    # Flatten: headers are already consumed by the parser; reconstruct
    # a plain-text version from the parsed line dicts so whisper
    # alignment sees the exact same lines.
    plain_lines = [gl["text"] for gl in genius_lines]
    plain_text = "\n".join(plain_lines)
    return plain_text, genius_lines


def extract_words(result) -> list:
    """Flatten WhisperResult into [{word, start, end, is_segment_first, speaker}, ...].

    Each word dict is initialized with ``speaker: None`` so the type
    annotation is honest before speaker assignment runs.
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
                }
            )
    return all_words


def match_words_to_lines(words: list, lines: list) -> list:
    """Assign aligned words to lyrics lines by count.

    Count-based pairing: assumes the lyrics file has the same word
    count and order as what stable-ts aligned.
    """
    line_objects = []
    word_index = 0

    for line in lines:
        line_word_count = len(line.split())
        line_words = words[word_index : word_index + line_word_count]
        word_index += line_word_count

        if not line_words:
            continue

        line_obj = {
            "text": line,
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
    initialized with ``speaker: None``.
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
