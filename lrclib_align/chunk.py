"""Chunk planning + per-chunk audio cutting for LRCLIB-anchored alignment.

Sections (as defined by Genius headers) are the chunks. Cut-section
lines (status="cut" from reconcile) are dropped here — they never make
it into any chunk's text and won't appear in the final output.

This module is intentionally narrow: chunk planning, audio cutting, and
small helpers for absolute-time stitching. The per-chunk `align()` loop
lives in `lrclib_align.run` because it needs the `WhisperWorker` plus
the matcher selection machinery.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lrclib_align.reconcile import ReconciledLine

logger = logging.getLogger(__name__)

# Tunables.
MIN_SECTION_SEC = 8.0
OVERLAP_PAD_SEC = 1.5


@dataclass
class Chunk:
    section_name: str                 # debug label
    genius_idxs: list[int] = field(default_factory=list)
    chunk_start: float = 0.0          # padded; absolute file time
    chunk_end: float = 0.0            # padded; absolute file time
    section_start: float = 0.0        # unpadded; for logging
    section_end: float = 0.0


def plan_chunks(
    genius_lines: list[dict],
    reconciled: list[ReconciledLine],
    file_duration: float,
) -> list[Chunk]:
    """Group surviving Genius lines into section chunks, floor-merge,
    and pad. Cut-section lines are dropped here.
    """
    assert len(genius_lines) == len(reconciled)

    # Pass 1: group surviving (non-cut) lines by section, in source order.
    sections: list[tuple[str, list[int]]] = []  # (section_name, [genius_idx])
    for gi, (gl, rl) in enumerate(zip(genius_lines, reconciled)):
        if rl.status == "cut":
            continue
        section_name = gl.get("section") or ""
        if sections and sections[-1][0] == section_name:
            sections[-1][1].append(gi)
        else:
            sections.append((section_name, [gi]))

    if not sections:
        return []

    # Pass 2: assign per-section start times from first line's lrclib_time.
    # Interpolated times count — they're our best guess.
    section_starts: list[Optional[float]] = []
    for _, idxs in sections:
        first_idx = idxs[0]
        section_starts.append(reconciled[first_idx].lrclib_time)

    # Backfill any None starts by averaging neighbors. In practice this
    # only happens if a section's leading line is unmatched and no
    # neighbors helped during interpolation (e.g. song starts with cut).
    for i, s in enumerate(section_starts):
        if s is not None:
            continue
        prev_t = next(
            (section_starts[k] for k in range(i - 1, -1, -1)
             if section_starts[k] is not None),
            0.0,
        )
        next_t = next(
            (section_starts[k] for k in range(i + 1, len(section_starts))
             if section_starts[k] is not None),
            file_duration,
        )
        section_starts[i] = (prev_t + next_t) / 2.0

    # Pass 3: section ends = next section's start, last = file_duration.
    section_ends: list[float] = []
    for i in range(len(sections)):
        if i + 1 < len(sections):
            section_ends.append(section_starts[i + 1])  # type: ignore[arg-type]
        else:
            section_ends.append(file_duration)

    # Pass 4: floor-merge — fold any section shorter than MIN_SECTION_SEC
    # into the next one. Iterate until stable.
    merged_sections: list[tuple[str, list[int], float, float]] = [
        (name, list(idxs), section_starts[i] or 0.0, section_ends[i])
        for i, (name, idxs) in enumerate(sections)
    ]
    changed = True
    while changed and len(merged_sections) > 1:
        changed = False
        for i in range(len(merged_sections) - 1):
            _, _, s, e = merged_sections[i]
            if e - s < MIN_SECTION_SEC:
                name_a, idxs_a, sa, _ = merged_sections[i]
                name_b, idxs_b, _sb, eb = merged_sections[i + 1]
                merged_sections[i + 1] = (
                    f"{name_a}+{name_b}", idxs_a + idxs_b, sa, eb,
                )
                merged_sections.pop(i)
                changed = True
                break
    # If the very last section is still <MIN, merge it backward.
    if len(merged_sections) > 1:
        name, idxs, s, e = merged_sections[-1]
        if e - s < MIN_SECTION_SEC:
            pname, pidxs, ps, _pe = merged_sections[-2]
            merged_sections[-2] = (f"{pname}+{name}", pidxs + idxs, ps, e)
            merged_sections.pop()

    # Pass 5: build padded Chunks. Anchor first chunk's start at 0 and
    # last chunk's end at file_duration so flanking audio is included.
    chunks: list[Chunk] = []
    for i, (name, idxs, s, e) in enumerate(merged_sections):
        c = Chunk(
            section_name=name,
            genius_idxs=idxs,
            section_start=s,
            section_end=e,
            chunk_start=max(0.0, s - OVERLAP_PAD_SEC),
            chunk_end=min(file_duration, e + OVERLAP_PAD_SEC),
        )
        if i == 0:
            c.chunk_start = 0.0
        if i == len(merged_sections) - 1:
            c.chunk_end = file_duration
        chunks.append(c)

    for c in chunks:
        logger.info(
            "Chunk %-30s lines=%d  audio=[%.2f, %.2f]  section=[%.2f, %.2f]",
            c.section_name[:30], len(c.genius_idxs),
            c.chunk_start, c.chunk_end, c.section_start, c.section_end,
        )

    return chunks


def cut_audio(
    vocal_path: Path,
    start: float,
    end: float,
    out_path: Path,
) -> None:
    """Extract [start, end] from `vocal_path` to a 16kHz mono WAV at `out_path`.

    Uses `-ss` before `-i` for fast seek (sample-accurate enough given
    our 1.5s overlap pad). Raises CalledProcessError if ffmpeg fails.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(vocal_path),
        "-t", f"{duration:.3f}",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def offset_line_objects(line_objects: list[dict], offset: float) -> None:
    """Add `offset` to every time field in `line_objects` (in place).

    The walk matcher leaves ``start``/``end`` as ``None`` for lines it
    couldn't place; the SRT/ASS generators skip those, so we just pass
    them through unmodified here.
    """
    for lo in line_objects:
        if lo.get("start") is not None:
            lo["start"] = lo["start"] + offset
        if lo.get("end") is not None:
            lo["end"] = lo["end"] + offset
        for w in lo.get("words", []):
            if w.get("start") is not None:
                w["start"] = w["start"] + offset
            if w.get("end") is not None:
                w["end"] = w["end"] + offset
