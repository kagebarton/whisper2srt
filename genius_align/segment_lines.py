"""Strict 1:1 segment→line builder for the ``original_split=True`` path.

When ``align()`` runs with ``original_split=True``, each ``\\n``-separated
input line becomes one fixed-boundary output segment, so the segment list
is 1:1 with ``genius_lines``. Two things break that mapping:

  1. Worker post-processing: ``adjust_gaps()`` and ``merge_by_gap()`` merge
     adjacent segments based on timing — collapsing zero-duration segments
     (the "failed to align" ones) into their neighbors and breaking the
     mapping silently.
  2. Stable-ts itself can mark a segment as failed (``end - start <= 0``).

This module provides:
  * ``align_for_strict_split()`` — runs align + refine but skips the
    line-destroying post-processing in ``align_and_refine()``.
  * ``segments_to_line_objects_strict()`` — strict 1:1 builder that
    asserts the count and interpolates timing for zero-duration segments
    from their neighbors so karaoke timing degrades gracefully instead
    of producing 0-cs subtitle lines.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def align_for_strict_split(worker, vocal_path: Path, plain_text: str):
    """Run align + refine without the merge/adjust post-processing.

    The standard ``worker.align_and_refine()`` runs ``adjust_gaps()`` and
    ``merge_by_gap()`` after refine — both merge adjacent segments and
    break the 1:1 mapping that ``original_split=True`` relies on.
    """
    result = worker.align(vocal_path, plain_text)
    refined = worker.refine(vocal_path, result)
    return refined


def segments_to_line_objects_strict(
    result,
    genius_lines: list,
    min_prob: float = 0.0001,
) -> list:
    """Build line_objects 1:1 from segments, asserting count match.

    Segments with ``end - start <= 0`` (stable-ts "failed to align" lines)
    get timing interpolated from the nearest non-failed neighbors so
    karaoke output stays usable. Their words list will be empty.
    """
    segments = list(result.segments)
    if len(segments) != len(genius_lines):
        raise ValueError(
            f"segment/genius_line count mismatch: "
            f"{len(segments)} segments vs {len(genius_lines)} genius_lines. "
            f"original_split=True requires 1:1; check that worker "
            f"post-processing (merge_by_gap/adjust_gaps) is disabled."
        )

    line_objects = []
    dropped = 0
    failed_indices = []

    for i, (segment, gl) in enumerate(zip(segments, genius_lines)):
        words = []
        for w in segment.words:
            prob = getattr(w, "probability", None)
            if prob is not None and prob < min_prob:
                dropped += 1
                continue
            words.append({
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "speaker": None,
                "dominant_speaker": None,
            })
        seg_failed = segment.end - segment.start <= 0
        if seg_failed:
            failed_indices.append(i)

        line_objects.append({
            "text": gl["text"],
            "words": words,
            "start": segment.start,
            "end": segment.end,
            "_failed": seg_failed,
        })

    # Interpolate timing for failed segments from nearest non-failed neighbors.
    for i in failed_indices:
        prev_end = None
        for j in range(i - 1, -1, -1):
            if not line_objects[j]["_failed"]:
                prev_end = line_objects[j]["end"]
                break
        next_start = None
        for j in range(i + 1, len(line_objects)):
            if not line_objects[j]["_failed"]:
                next_start = line_objects[j]["start"]
                break
        if prev_end is not None and next_start is not None:
            line_objects[i]["start"] = prev_end
            line_objects[i]["end"] = next_start
        elif prev_end is not None:
            line_objects[i]["start"] = prev_end
            line_objects[i]["end"] = prev_end + 1.0
        elif next_start is not None:
            line_objects[i]["start"] = max(0.0, next_start - 1.0)
            line_objects[i]["end"] = next_start

    for lo in line_objects:
        lo.pop("_failed", None)

    if dropped:
        logger.info(
            "Dropped %d low-probability whisper words (< %.4f)",
            dropped, min_prob,
        )
    if failed_indices:
        logger.warning(
            "Interpolated timing for %d failed segments (indices: %s)",
            len(failed_indices),
            failed_indices if len(failed_indices) <= 10 else f"{failed_indices[:10]}...",
        )
    return line_objects
