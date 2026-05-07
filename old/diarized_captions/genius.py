"""Genius.com lyrics parser — extract per-line singer attribution.

Parses Genius-formatted lyrics (with section headers like
``[Verse 1: Brian]``) into per-line attribution dicts.  If the file
has no recognizable headers, every emitted line has ``singers=None``
— caller falls back to anonymous diarization.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Header regex: [Section] or [Section: Attribution]
_HEADER_RE = re.compile(r"^\[([^:\]]+)(?::\s*(.+))?\]\s*$")

# Regex to detect if the entire line is parenthesized
_FULL_PAREN_RE = re.compile(r"^\((.+)\)$")


def parse_genius_sections(lyrics_text: str) -> list[dict]:
    """Parse Genius-formatted lyrics into per-line attribution.

    Returns a list of GeniusLine dicts (see §4) in document order, one
    per non-blank lyric line. Header lines are consumed (not emitted).

    If the file has no recognizable headers, every emitted line has
    singers=None — caller falls back to anonymous diarization.

    Each dict has the shape::

        {
            "text": str,             # lyric text (no header)
            "section": str,          # "Verse 1", "Chorus", ...
            "singers": list | None,  # ["Brian"] for solo, ["Nick","All"]
                                     # for ensemble; None if section had
                                     # no attribution at all (e.g. "[Verse 1]")
            "is_ensemble": bool,     # True iff len(singers) > 1 or "All" in singers
        }
    """
    lines = lyrics_text.split("\n")
    result = []

    current_section = ""
    current_groups: list[list[str]] | None = None  # list of groups from header
    current_first_singers: list[str] | None = None  # first group's name list

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        header_match = _HEADER_RE.match(stripped)
        if header_match:
            current_section = header_match.group(1).strip()
            attribution = header_match.group(2)
            if attribution is not None:
                current_groups = split_groups(attribution.strip())
                current_first_singers = current_groups[0] if current_groups else None
            else:
                current_groups = None
                current_first_singers = None
            continue

        # Non-header, non-empty line — emit a GeniusLine
        text = stripped
        # Strip a leading/trailing parenthetical wrapper only if the
        # entire line is parenthesized (rare — usually parentheticals
        # are mid-line).
        full_paren = _FULL_PAREN_RE.match(text)
        if full_paren:
            text = full_paren.group(1)

        singers = current_first_singers
        is_ensemble = False
        if singers is not None:
            is_ensemble = len(singers) > 1 or "All" in singers

        result.append(
            {
                "text": text,
                "section": current_section,
                "singers": list(singers) if singers is not None else None,
                "is_ensemble": is_ensemble,
            }
        )

    return result


def split_groups(attribution: str) -> list[list[str]]:
    """Split a Genius attribution string into groups of names.

    ``'Nick, AJ & Brian'`` → ``[['Nick'], ['AJ', 'Brian']]``

    Groups are separated by ``,``; names within a group are separated
    by ``&``.  Each name is stripped of whitespace.
    """
    groups = []
    for group_str in attribution.split(","):
        group_str = group_str.strip()
        if not group_str:
            continue
        names = [n.strip() for n in group_str.split("&") if n.strip()]
        if names:
            groups.append(names)
    return groups


def genius_singer_mode(genius_lines: list[dict]) -> str:
    """Classify the song's singer attribution level.

    Returns one of:

    ``"solo"`` — no headers at all, OR headers present but none carry
        singer attribution (e.g. ``[Verse 1]``, ``[Chorus]`` — Genius
        omits attribution for single-artist songs; a plain .txt with
        no headers at all is also solo).
        ``diarize_worker.diarize()`` should be skipped.

    ``"multi"`` — at least one header carries singer attribution.
        Pyannote + cluster→name mapping should run.

    This is the primary decision gate in ``run.py``.
    """
    has_attribution = False
    for gl in genius_lines:
        if gl["singers"] is not None:
            has_attribution = True
            break

    if has_attribution:
        return "multi"
    return "solo"


def unique_named_singers(genius_lines: list[dict]) -> list[str]:
    """Return the union of all named singers (excluding ``All``).

    Useful for feeding the count back to pyannote as ``min_speakers``.
    """
    seen: set[str] = set()
    order: list[str] = []
    for gl in genius_lines:
        if gl["singers"] is None:
            continue
        for name in gl["singers"]:
            if name == "All":
                continue
            if name not in seen:
                seen.add(name)
                order.append(name)
    return order
