import os
import re
import glob
import argparse
import config_genius as config
import stable_whisper
import lyricsgenius

GAP_THRESHOLD = 0.4  # seconds; tune this if needed


def load_model():
    return stable_whisper.load_faster_whisper(
        config.WHISPER_MODEL,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE
    )


def init_genius_client():
    """Initialize LyricsGenius client with API token from config."""
    if not config.GENIUS_API_TOKEN:
        print("Warning: GENIUS_API_TOKEN not set in config_genius.py. Lyrics search will be unavailable.")
        return None
    genius = lyricsgenius.Genius(config.GENIUS_API_TOKEN, timeout=10)
    genius.verbose = False
    genius.remove_section_headers = True
    genius.skip_non_songs = True
    #genius.excluded_terms = ["(Remix)", "(Live)"]
    return genius


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
    artist_name = song.get('primary_artist', {}).get('name', '') if isinstance(song, dict) else ''
    title = song.get('title', '') if isinstance(song, dict) else ''

    result = genius.search_song(title, artist_name)

    if not result or not result.lyrics:
        return ""

    return result.lyrics


def transcribe_audio(model, audio_file_path: str):
    """
    Transcribes audio using faster-whisper via stable-ts.
    Used when user chooses transcription mode.
    """
    print("Running in transcription mode.")
    return model.transcribe(
        audio_file_path,
        word_timestamps=True,
        language="en",
        beam_size=10,
        best_of=10,
        temperature=0.0,
        condition_on_previous_text=True,
        initial_prompt="Lyrics:"
    )


def align_audio(model, audio_file_path: str, lyrics_file_path: str):
    """
    Aligns audio to verified lyrics using stable-ts.
    """
    print(f"Running in alignment mode using {lyrics_file_path}")
    with open(lyrics_file_path, "r", encoding="utf-8") as f:
        lyrics = f.read()
    return model.align(audio_file_path, lyrics, language="en")


def seconds_to_srt(seconds: float) -> str:
    """Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt_from_transcription(result, output_path: str):
    """
    Saves SRT from a transcription result.
    Splits lines on capital letters using word-level timestamps.
    """
    blocks = []

    for segment in result.segments:
        words = segment.words

        current_line = []
        for word in words:
            text = word.word.strip()
            is_capital = bool(re.match(r'[A-Z]', text))
            is_new_line = False

            if current_line and is_capital:
                if text == "I":
                    prev_word = current_line[-1]
                    gap = word.start - prev_word.end
                    ends_with_punctuation = bool(re.search(r'[.,!?]$', prev_word.word.strip()))
                    is_new_line = gap > GAP_THRESHOLD or ends_with_punctuation
                else:
                    is_new_line = True

            if is_new_line:
                blocks.append({
                    "start": seconds_to_srt(current_line[0].start),
                    "end": seconds_to_srt(current_line[-1].end),
                    "text": " ".join(w.word.strip() for w in current_line)
                })
                current_line = []

            current_line.append(word)

        if current_line:
            blocks.append({
                "start": seconds_to_srt(current_line[0].start),
                "end": seconds_to_srt(current_line[-1].end),
                "text": " ".join(w.word.strip() for w in current_line)
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks, start=1):
            f.write(f"{i}\n{block['start']} --> {block['end']}\n{block['text']}\n\n")


def save_srt_from_alignment(result, lyrics_file_path: str, output_path: str):
    """
    Saves SRT from an alignment result, using the lyrics file line breaks
    to determine subtitle blocks.
    """
    all_words = []
    for segment in result.segments:
        for word in segment.words:
            all_words.append(word)

    with open(lyrics_file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    blocks = []
    word_index = 0

    for line in lines:
        line_word_count = len(line.split())
        line_words = all_words[word_index: word_index + line_word_count]
        word_index += line_word_count

        if not line_words:
            continue

        blocks.append({
            "start": seconds_to_srt(line_words[0].start),
            "end": seconds_to_srt(line_words[-1].end),
            "text": line
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks, start=1):
            f.write(f"{i}\n{block['start']} --> {block['end']}\n{block['text']}\n\n")


def get_user_selection(songs: list):
    """
    Get user's song selection.
    Returns: selected song object, or None if user chose transcription mode.
    """
    while True:
        choice = input("Enter choice (1-10) or 't' for transcription: ").strip().lower()

        if choice == 't':
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(songs):
                return songs[idx][0]
            else:
                print(f"Please enter a number between 1 and {len(songs)}, or 't'.")
        except ValueError:
            print("Invalid input. Please enter a number (1-10) or 't'.")


def process_file(model, file_path: str, genius=None):
    prefix = os.path.splitext(file_path)[0]
    lyrics_path = prefix + ".txt"
    output_path = prefix + ".srt"

    print(f"\nProcessing: {file_path}")

    # Always search for lyrics first
    if genius:
        query = os.path.basename(prefix)
        print(f"Searching for: \"{query}\"")
        songs = search_songs(genius, query)

        if songs:
            print("\nTop search results:")
            print("-" * 50)
            for i, (_, display) in enumerate(songs, start=1):
                print(f"  {i}. {display}")
            print("-" * 50)
            print("  [t] Use transcription mode")
            print()

            selected_song = get_user_selection(songs)

            if selected_song:
                print(f"\nDownloading lyrics...")
                lyrics = download_lyrics(genius, selected_song)

                if lyrics:
                    with open(lyrics_path, "w", encoding="utf-8") as f:
                        f.write(lyrics)
                    print(f"Lyrics saved to {lyrics_path}")

                    result = align_audio(model, file_path, lyrics_path)
                    save_srt_from_alignment(result, lyrics_path, output_path)
                    print(f"SRT saved to {output_path}")
                    return
                else:
                    print("No lyrics found. Falling back to transcription mode.")
            else:
                print("Using transcription mode.")
        else:
            print("No search results found. Using transcription mode.")

    # Fall back to transcription mode
    result = transcribe_audio(model, file_path)
    save_srt_from_transcription(result, output_path)
    print(f"SRT saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe or align audio to SRT subtitles.")
    parser.add_argument("files", nargs="+", help="Audio file path(s); supports wildcards e.g. *.mp3")
    args = parser.parse_args()

    # Expand wildcards manually
    file_paths = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            file_paths.extend(sorted(matches))
        else:
            print(f"Warning: no files matched '{pattern}'")

    if not file_paths:
        print("No files to process.")
    else:
        model = load_model()
        genius = init_genius_client()
        for file_path in file_paths:
            process_file(model, file_path, genius)
