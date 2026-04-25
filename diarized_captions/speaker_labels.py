"""Speaker assignment: appearance-order remap, word→speaker assignment, line splitting.

Functions:
- remap_speakers_by_appearance: raw pyannote labels → A/B/C by first appearance
- assign_speakers_to_words: attach a speaker letter to every word
- reset_segment_first_flags: fix is_segment_first after word regrouping
- split_lines_at_speaker_boundaries: break lines at speaker hand-offs
"""

import logging

logger = logging.getLogger(__name__)


def remap_speakers_by_appearance(turns, cfg):
    """Remap raw pyannote speaker labels to A/B/C by appearance order.

    Walks turns sorted by start time; the first-seen pyannote label is
    mapped to cfg.speaker_letters[0] (= "A"), second to "B", etc.

    Returns:
        (turns, label_map) — turns are mutated in-place (each gets a
        "letter" key), label_map maps raw label → letter.

    Raises:
        ValueError: if more speakers are detected than there are
        configured letters.
    """
    label_map = {}
    for t in sorted(turns, key=lambda t: t["start"]):
        raw = t["speaker"]
        if raw not in label_map:
            idx = len(label_map)
            if idx >= len(cfg.speaker_letters):
                raise ValueError(
                    f"More speakers than configured letters "
                    f"({len(cfg.speaker_letters)}). "
                    f"Bump speaker_letters / speaker_colors in config, "
                    f"or set max_speakers."
                )
            label_map[raw] = cfg.speaker_letters[idx]
    for t in turns:
        t["letter"] = label_map[t["speaker"]]
    return turns, label_map


def assign_speakers_to_words(line_objects, turns):
    """Attach a speaker letter to every word in every line object.

    For each word, find the diarization turn with the largest temporal
    overlap. If the word falls entirely outside any turn (gap), pick the
    closest turn by midpoint distance — keeps every word labeled.

    Raises:
        RuntimeError: if turns is empty (cannot label any words).
    """
    if not turns:
        raise RuntimeError("Diarization produced zero turns — cannot label words")
    for line in line_objects:
        for w in line["words"]:
            w["speaker"] = _word_speaker(w, turns)
    return line_objects


def _word_speaker(word, turns):
    """Find the best speaker letter for a single word given diarization turns.

    Strategy: largest overlap first; gap fallback to nearest turn by midpoint.
    """
    best_turn, best_overlap = None, 0.0
    for t in turns:
        overlap = max(0.0, min(word["end"], t["end"]) - max(word["start"], t["start"]))
        if overlap > best_overlap:
            best_turn, best_overlap = t, overlap
    if best_turn is not None:
        return best_turn["letter"]
    # gap fallback: nearest turn by midpoint (turns guaranteed non-empty)
    mid = 0.5 * (word["start"] + word["end"])
    nearest = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
    return nearest["letter"]


def reset_segment_first_flags(line_objects):
    """Reset is_segment_first so only the first word of each line is True.

    Called from run.py after match_words_to_lines (alignment path only).
    The whisper alignment marks is_segment_first=True on the first word
    of each *whisper* segment, but those boundaries don't match the
    lyric-line boundaries we just regrouped to. Stray True flags in
    the middle of a line would cause first_word_nudge_cs to fire at
    the wrong position.

    Not needed in the transcription path — segments_to_line_objects
    builds each line from one whisper segment, so the flags already
    line up.
    """
    for line in line_objects:
        for i, w in enumerate(line["words"]):
            w["is_segment_first"] = i == 0


def split_lines_at_speaker_boundaries(line_objects):
    """Split lines wherever the speaker changes between consecutive words.

    Option (b): walk each line's words; whenever the speaker letter
    changes between consecutive words, close the current sub-line and
    start a new one. The first word of every emitted sub-line gets
    is_segment_first=True so the existing first_word_nudge_cs logic
    still applies inside the karaoke generator.
    """
    out = []
    for line in line_objects:
        words = line["words"]
        if not words:
            continue
        cur_words = [dict(words[0], is_segment_first=True)]
        cur_speaker = words[0]["speaker"]
        for w in words[1:]:
            if w["speaker"] != cur_speaker:
                out.append(_make_sub(cur_words, cur_speaker))
                cur_words = [dict(w, is_segment_first=True)]
                cur_speaker = w["speaker"]
            else:
                cur_words.append(w)
        out.append(_make_sub(cur_words, cur_speaker))
    return out


def _make_sub(words, speaker):
    """Build a line-object dict from a contiguous run of same-speaker words."""
    return {
        "text": " ".join(w["word"] for w in words),
        "words": words,
        "speaker": speaker,
        "start": words[0]["start"],
        "end": words[-1]["end"],
    }
