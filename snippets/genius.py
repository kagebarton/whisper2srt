#!/usr/bin/env python3
"""Test script for lyricsgenius search functionality."""

import sys
import os
import re
import lyricsgenius

# ==============================================================================
#  CYRILLIC HOMOGLYPH CORRECTION
# ==============================================================================
# Maps Cyrillic characters that look like Latin letters to their Latin equivalents.
# This fixes the common Genius API issue where lyrics contain Cyrillic lookalikes.
CYRILLIC_TO_LATIN = str.maketrans({
    'а': 'a',   # U+0430 → U+0061
    'А': 'A',   # U+0410 → U+0041
    'е': 'e',   # U+0435 → U+0065
    'Е': 'E',   # U+0415 → U+0045
    'о': 'o',   # U+043E → U+006F
    'О': 'O',   # U+041E → U+004F
    'р': 'p',   # U+0440 → U+0070  (lowercase Cyrillic эр looks like Latin p)
    'Р': 'P',   # U+0420 → U+0050
    'с': 'c',   # U+0441 → U+0063
    'С': 'C',   # U+0421 → U+0043
    'х': 'x',   # U+0445 → U+0078
    'Х': 'X',   # U+0425 → U+0058
    'у': 'y',   # U+0443 → U+0079  (sometimes used as y lookalike)
    'У': 'Y',   # U+0423 → U+0059
    'К': 'K',   # U+041A → U+004B
    'М': 'M',   # U+041C → U+004D
    'Н': 'H',   # U+041D → U+0048
    'Т': 'T',   # U+0422 → U+0054
    'В': 'B',   # U+0412 → U+0042
    'З': '3',   # U+0417 → digit (occasionally used as leetspeak)
})


def fix_cyrillic_homoglyphs(text: str) -> str:
    """Replace Cyrillic characters that look like Latin letters with actual Latin ones."""
    return text.translate(CYRILLIC_TO_LATIN)


def clean_bracket_content(text: str) -> str:
    """
    Remove anything enclosed in square brackets (including the brackets themselves).
    Handles multi-line bracket blocks and inline brackets like [Chorus].
    Also cleans up any resulting blank lines.
    """
    # Remove [...] blocks (non-greedy, across lines via DOTALL)
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    # Collapse multiple blank lines into single newlines, strip trailing whitespace
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    return text

# ==============================================================================
#  CONFIG
# ==============================================================================

# Genius API token for lyricsgenius
# Get your token at: https://genius.com/api_clients
GENIUS_API_TOKEN = "OQNe-SALiHKew5tn4fwBEl5mcyiIBTiYS62tjWxhtiFQ2z7nvQcQJdEW05CZcdjB"


def search_songs(genius, query: str, limit: int = 10):
    """Search for songs on Genius and return top results."""
    results = genius.search_songs(query, per_page=limit)

    if not results:
        return []

    # results is a dict with 'hits' key
    hits = results.get('hits', []) if isinstance(results, dict) else results

    if not hits:
        return []

    songs = []
    for hit in hits:
        song = hit.get('result', {}) if isinstance(hit, dict) else hit
        if not song:
            continue
        artist = song.get('primary_artist', {}).get('name', 'Unknown')
        title = song.get('title', 'Unknown')
        songs.append((song, f"{artist} - {title}"))

    return songs


def download_lyrics(genius, song) -> str:
    """Download lyrics from Genius."""
    # Use search_song to get full lyrics object
    artist_name = song.get('primary_artist', {}).get('name', '') if isinstance(song, dict) else ''
    title = song.get('title', '') if isinstance(song, dict) else ''

    # Search for the specific song to get lyrics
    result = genius.search_song(title, artist_name)

    if not result or not result.lyrics:
        return ""

    return clean_bracket_content(fix_cyrillic_homoglyphs(result.lyrics))


def main():
    if len(sys.argv) < 2:
        print("Usage: python genius.py <search_term>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    if not GENIUS_API_TOKEN:
        print("Error: GENIUS_API_TOKEN not set")
        sys.exit(1)

    genius = lyricsgenius.Genius(GENIUS_API_TOKEN, timeout=10)

    # Configure options
    #genius.verbose = False
    genius.remove_section_headers = True
    genius.skip_non_songs = True
    #genius.excluded_terms = ["(Remix)", "(Live)"]

    print(f"Searching for: \"{query}\"")
    print()

    songs = search_songs(genius, query)

    if not songs:
        print("No results found.")
        return

    print("Top search results:")
    print("-" * 50)
    for i, (_, display) in enumerate(songs, start=1):
        print(f"  {i}. {display}")
    print("-" * 50)
    print()

    while True:
        choice = input("Enter choice (1-10) or 'q' to quit: ").strip().lower()

        if choice == 'q':
            print("Done.")
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(songs):
                selected_song = songs[idx][0]
                display_name = songs[idx][1].replace(' - ', '_').replace(' ', '_')

                print(f"\nDownloading lyrics for: {songs[idx][1]}...")

                lyrics = download_lyrics(genius, selected_song)

                if lyrics:
                    output_file = f"{display_name}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(lyrics)
                    print(f"Lyrics saved to {output_file}")
                else:
                    print("No lyrics found.")
            else:
                print(f"Please enter a number between 1 and {len(songs)}, or 'q'.")
        except ValueError:
            print("Invalid input. Please enter a number (1-10) or 'q'.")


if __name__ == "__main__":
    main()
