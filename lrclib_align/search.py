"""Genius search + interactive picker + lyrics fetch.

Phase 1 input contract:
    1. Prompt for free-text search terms (or accept a pre-supplied query).
    2. Query Genius, present top 10 results, user picks one.
    3. Fetch the chosen result's lyrics + metadata.

Genius lyrics are returned **with section headers preserved** — those
headers are what `genius_align.genius.parse_genius_sections` uses for
speaker attribution. Bracket content is NOT stripped here.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass

import lyricsgenius

logger = logging.getLogger(__name__)

# Prototype default — same personal token already committed at
# old/snippets/genius.py:61. Override with GENIUS_API_TOKEN env var.
_DEFAULT_GENIUS_TOKEN = "OQNe-SALiHKew5tn4fwBEl5mcyiIBTiYS62tjWxhtiFQ2z7nvQcQJdEW05CZcdjB"


# Cyrillic→Latin homoglyph correction. Genius pages occasionally contain
# Cyrillic lookalikes (а/е/о/р/с/х) that whisper won't match against the
# real audio. Copied from old/snippets/genius.py.
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a", "А": "A", "е": "e", "Е": "E", "о": "o", "О": "O",
        "р": "p", "Р": "P", "с": "c", "С": "C", "х": "x", "Х": "X",
        "у": "y", "У": "Y", "К": "K", "М": "M", "Н": "H", "Т": "T",
        "В": "B", "З": "3",
    }
)


def _fix_cyrillic_homoglyphs(text: str) -> str:
    return text.translate(_CYRILLIC_TO_LATIN)


@dataclass
class GeniusHit:
    """One row in the search results."""

    song_id: int
    title: str
    artist: str
    url: str

    @property
    def display(self) -> str:
        return f"{self.artist} - {self.title}"


@dataclass
class GeniusSelection:
    """The picked song + its fetched lyrics."""

    hit: GeniusHit
    lyrics: str  # with section headers preserved


def _get_token() -> str:
    return os.environ.get("GENIUS_API_TOKEN") or _DEFAULT_GENIUS_TOKEN


def _make_client() -> lyricsgenius.Genius:
    # Keep section headers — genius_align parses them for speaker attribution.
    g = lyricsgenius.Genius(
        _get_token(),
        timeout=10,
        remove_section_headers=False,
        skip_non_songs=True,
    )
    g.verbose = False
    return g


def search_genius(query: str, limit: int = 10) -> list[GeniusHit]:
    g = _make_client()
    raw = g.search_songs(query, per_page=limit)
    hits_raw = raw.get("hits", []) if isinstance(raw, dict) else []
    hits: list[GeniusHit] = []
    for h in hits_raw:
        s = h.get("result") if isinstance(h, dict) else None
        if not s:
            continue
        artist = (s.get("primary_artist") or {}).get("name") or "Unknown"
        hits.append(
            GeniusHit(
                song_id=int(s["id"]),
                title=s.get("title") or "Unknown",
                artist=artist,
                url=s.get("url") or "",
            )
        )
    return hits


def fetch_lyrics(hit: GeniusHit) -> str:
    """Download lyrics for a chosen hit; section headers preserved."""
    g = _make_client()
    lyrics = g.lyrics(song_id=hit.song_id, remove_section_headers=False)
    if not lyrics:
        return ""
    lyrics = _fix_cyrillic_homoglyphs(lyrics)
    # lyricsgenius often prepends a non-lyric preamble like
    # "<Title> Lyrics" before the first `[Section]` header, and appends
    # trailing "<n>Embed" / "You might also like" cruft. Trim both.
    lyrics = _strip_genius_chrome(lyrics)
    return lyrics.strip() + "\n"


def _strip_genius_chrome(lyrics: str) -> str:
    """Remove lyricsgenius preamble/trailer noise."""
    # Drop everything before the first [Section] header if present.
    first_hdr = re.search(r"^\[", lyrics, flags=re.MULTILINE)
    if first_hdr:
        lyrics = lyrics[first_hdr.start():]
    # Drop trailing 'NNNEmbed' marker some pages include.
    lyrics = re.sub(r"\d*Embed\s*$", "", lyrics)
    return lyrics


def _prompt(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(130)


def interactive_search(initial_query: str | None = None) -> GeniusSelection:
    """Prompt → search → top-10 picker → fetch lyrics.

    Returns the chosen `GeniusSelection`. Re-prompts on empty results or
    bad input; exits with code 130 on Ctrl-C / EOF.
    """
    query = initial_query
    while True:
        if not query:
            query = _prompt("Search Genius: ")
            if not query:
                continue

        print(f'Searching Genius for: "{query}"')
        try:
            hits = search_genius(query, limit=10)
        except Exception as exc:
            logger.error("Genius search failed: %s", exc)
            query = None
            continue

        if not hits:
            print("No results. Try a different query.")
            query = None
            continue

        print()
        print("Results:")
        print("-" * 60)
        for i, h in enumerate(hits, start=1):
            print(f"  {i:>2}. {h.display}")
        print("-" * 60)

        choice = _prompt("Pick (1-{n}), 's' to search again, 'q' to quit: ".format(n=len(hits)))
        if choice.lower() == "q":
            sys.exit(0)
        if choice.lower() == "s":
            query = None
            continue
        try:
            idx = int(choice) - 1
        except ValueError:
            print("Not a number.")
            continue
        if not (0 <= idx < len(hits)):
            print(f"Out of range (1-{len(hits)}).")
            continue

        picked = hits[idx]
        print(f"Fetching lyrics for: {picked.display}")
        lyrics = fetch_lyrics(picked)
        if not lyrics:
            print("Genius returned no lyrics for that hit. Try another.")
            continue
        return GeniusSelection(hit=picked, lyrics=lyrics)
