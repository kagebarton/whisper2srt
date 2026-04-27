"""Genius.com lyrics parser — extract per-line speaker attribution.

Parses Genius-formatted lyrics (with section headers like
``[Verse 1: Brian]``) into per-line attribution dicts using conservative
rules:

- Single group, single name  → labeled
- Single group, named duet   → labeled with pair
- Single group "All"         → unlabeled (ensemble)
- Multiple groups            → unlabeled (ensemble)
- No attribution (no ``:``)  → unlabeled

No pyannote, no overlap detection, no cluster mapping — just headers.
"""

import re

# Header regex: [Section] or [Section: Attribution]
_HEADER_RE = re.compile(r"^\[([^\]]+)\]\s*$")

# Regex to detect if the entire line is parenthesized
_FULL_PAREN_RE = re.compile(r"^\((.+)\)$")

# Strip inline backing-vocal parentheticals for whisper alignment
_INLINE_PAREN_RE = re.compile(r"\s*\([^)]*\)")


# ---------------------------------------------------------------------------
# Attribution-rule resolver
# ---------------------------------------------------------------------------

def _resolve_attribution(groups):
    """Apply attribution rules to a parsed list of groups.

    Args:
        groups: list of groups, each a list of names, as returned by
            split_groups(). ``None`` means no attribution (header has no ``:``).

    Returns:
        (speaker_label, dominant_speaker, is_ensemble) — see GeniusLine
        schema in §3.

    Rules:
        - 0 groups / no attribution    → ensemble
        - 1 group, "All"               → ensemble
        - 1 group, single name         → labeled
        - 1 group, named pair          → labeled with pair
        - 2+ groups, any is "All"      → ensemble
        - 2+ named groups              → ensemble
        - 1 group                      → label from it
    """
    if groups is None:
        return None, None, True

    group_count = len(groups)
    if group_count == 0:
        return None, None, True

    if group_count >= 2:
        return None, None, True

    # Any group is "All" → ensemble
    if any(g == ["All"] for g in groups):
        return None, None, True

    # Exactly 1 fully-named group — label from it
    return _resolve_first_group(groups[0])


def _resolve_first_group(first_group):
    """Return (speaker_label, dominant_speaker, is_ensemble) for a named group."""
    if len(first_group) == 1:
        name = first_group[0]
        return name, name, False
    # Named pair / duet
    pair_label = " & ".join(first_group)
    dominant = first_group[0]
    return pair_label, dominant, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_genius_sections(lyrics_text: str) -> list[dict]:
    """Parse Genius-formatted lyrics into per-line attribution.

    Returns a list of GeniusLine dicts in document order, one per
    non-blank, non-fully-parenthesized lyric line. Header lines are
    consumed (not emitted).

    Each dict::

        {
            "text": str,              # lyric text (display; includes parens)
            "align_text": str,        # text with inline parens stripped (for whisper)
            "section": str,           # "Verse 1", "Chorus", ...
            "speaker_label": str|None,  # "Brian", "Kevin & AJ", or None
            "dominant_speaker": str|None,  # first individual name, or None
            "is_ensemble": bool,      # True iff unlabeled by rule
        }
    """
    lines = lyrics_text.split("\n")
    result = []

    current_section = ""
    current_groups = None  # None = no attribution yet

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        header_match = _HEADER_RE.match(stripped)
        if header_match:
            bracket_content = header_match.group(1)
            # Split on first ':' to separate section from attribution
            if ":" in bracket_content:
                section_part, attr_part = bracket_content.split(":", 1)
                current_section = section_part.strip()
                current_groups = split_groups(attr_part.strip())
            else:
                current_section = bracket_content.strip()
                current_groups = None
            continue

        # Skip fully-parenthesized lines (e.g., ``(ad-lib)``)
        if _FULL_PAREN_RE.match(stripped):
            continue

        # Alignment text: strip inline backing-vocal parentheticals so
        # whisper isn't asked to match "(Bye, bye)" or "(Yeah)" tokens
        # that may be absent or overlapping in the vocal stem.
        align_text = _INLINE_PAREN_RE.sub("", stripped).strip()
        if not align_text:
            continue  # entire content was parenthetical

        # Non-header, non-blank, non-paren — emit a GeniusLine
        speaker_label, dominant_speaker, is_ensemble = _resolve_attribution(
            current_groups
        )

        result.append(
            {
                "text": stripped,
                "align_text": align_text,
                "section": current_section,
                "speaker_label": speaker_label,
                "dominant_speaker": dominant_speaker,
                "is_ensemble": is_ensemble,
            }
        )

    return result


def genius_singer_mode(genius_lines: list[dict]) -> str:
    """Detect whether to run speaker assignment.

    Returns:
        ``"solo"`` — no line has a non-None speaker_label (all ensemble
        or no attribution). Skip speaker assignment; output plain karaoke.
        ``"multi"`` — at least one line has speaker_label != None.

    Note: an all-ensemble file (e.g., every header is ``[Chorus: All]``)
    yields ``"solo"`` — there is nothing to label.
    """
    for gl in genius_lines:
        if gl["speaker_label"] is not None:
            return "multi"
    return "solo"


def split_groups(attribution: str) -> list[list[str]]:
    """Parse an attribution string into groups of names.

    Splits on ``,`` first (groups), then ``&`` within each group.
    Each name is stripped of whitespace.

    Examples::

        "Brian"             → [["Brian"]]
        "Kevin & AJ"        → [["Kevin", "AJ"]]
        "Nick, All"         → [["Nick"], ["All"]]
        "Brian & AJ, Nick"  → [["Brian", "AJ"], ["Nick"]]
        "All"               → [["All"]]

    Returns:
        List of groups; each group is a list of name strings.
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


def truncate_speaker_label(speaker_label: str) -> str:
    """Abbreviate a speaker label for SRT subsequent appearances.

    Rules:
        - ``"All"`` → ``"All"`` (no change)
        - Single name → first letter of first whitespace-delimited token.
          (``"Brian"`` → ``"B"``, ``"Mary Jane"`` → ``"M"``,
          ``"O'Brien"`` → ``"O"``, ``"Jean-Paul"`` → ``"J"``)
        - Named pair ``"A & B"`` → ``"<initial(A)> & <initial(B)>"``
          (``"Kevin & AJ"`` → ``"K & A"``)
        - 3+ name group → initial-join with ``" & "``.

    Initial collisions (e.g., two singers named "Brian") are accepted in v1.

    Returns:
        Abbreviated label string.
    """
    if speaker_label == "All":
        return "All"

    # Check if it's a pair/group (contains " & ")
    if " & " in speaker_label:
        parts = speaker_label.split(" & ")
        initials = [_initial(p.strip()) for p in parts]
        return " & ".join(initials)

    # Single name
    return _initial(speaker_label)


def _initial(name: str) -> str:
    """Return the first letter of the first whitespace-delimited token."""
    token = name.split()[0] if name.split() else ""
    return token[0] if token else ""
