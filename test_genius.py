#!/usr/bin/env python3
"""Test script for lyricsgenius search functionality."""

import sys
import re
import config
import lyricsgenius


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

    return result.lyrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_genius.py <search_term>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    if not config.GENIUS_API_TOKEN:
        print("Error: GENIUS_API_TOKEN not set in config.py")
        sys.exit(1)

    genius = lyricsgenius.Genius(config.GENIUS_API_TOKEN, timeout=10)

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
