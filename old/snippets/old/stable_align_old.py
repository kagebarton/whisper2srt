#!/usr/bin/env python3
"""
================================================================================
whisper2srt_karaoke.py — Generate karaoke .ass files with progressive highlighting
================================================================================

USAGE:
  python whisper2srt_karaoke.py <audio_file> <lyrics_file>

  Example:
    python whisper2srt_karaoke.py "song.mp3" "song_lyrics.txt"

OUTPUT:
  An .ass file with word-level karaoke highlighting:
    song.ass

The .ass file uses the \\k tag for progressive word highlighting that can be
used in video players like MPC-HC, VLC, or Aegisub.

REQUIREMENTS:
  pip install stable-ts faster-whisper
  ffmpeg must be on PATH
================================================================================
"""

import os
import argparse
import stable_whisper


# ==============================================================================
#  CONFIG
# ==============================================================================

# Path to pre-downloaded model folder
# Set to None to use model name (will download if not cached)
# Example: r"C:\models\faster-whisper-large-v3" or "/home/user/models/faster-whisper-large-v3"
MODEL_PATH = "/home/ken/whisper2srt/whisper_models"  # Set absolute path to your model folder

# Device for inference: "cpu", "cuda", or "auto"
DEVICE = "auto"

# Compute type for faster-whisper: "float16", "int8", "int8_float16", or "float32"
COMPUTE_TYPE = "int8"

# ASS styling
FONT_NAME = "Arial"
FONT_SIZE = 48
PRIMARY_COLOR = "&H0000FFFF"  # Cyan (highlighted)
SECONDARY_COLOR = "&H00FFFFFF"  # Black (not yet sung)
OUTLINE_COLOR = "&H00000000"  # Black outline
BACK_COLOR = "&H00000000"  # Transparent background
OUTLINE_WIDTH = 2
SHADOW_OFFSET = 1
MARGIN_LEFT = 50
MARGIN_RIGHT = 50
MARGIN_VERTICAL = 150


# ==============================================================================


def load_model():
    """Load the faster-whisper model via stable-ts."""
    # Use MODEL_PATH if set, otherwise fall back to default model name
    model_source = MODEL_PATH if MODEL_PATH else "faster-whisper-large-v3"

    print(f"Loading model: {model_source}")
    return stable_whisper.load_faster_whisper(
        model_source,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )


def align_audio(model, audio_file_path: str, lyrics_file_path: str):
    """
    Aligns audio to confirmed lyrics using stable-ts.
    Returns alignment result with word-level timestamps.
    """
    print(f"Aligning audio to lyrics: {lyrics_file_path}")
    with open(lyrics_file_path, "r", encoding="utf-8") as f:
        lyrics = f.read()
    
    result = model.align(
        audio_file_path,
        lyrics,
        language="en",
        verbose=False
    )
    print("Alignment complete.")
    return result


def seconds_to_ass_time(seconds: float) -> str:
    """
    Convert seconds to ASS timestamp format: H:MM:SS.cc (centiseconds)
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def extract_words_from_alignment(result):
    """
    Extract all words with timestamps from alignment result.
    Returns list of dicts: [{word, start, end}, ...]
    """
    all_words = []
    for segment in result.segments:
        for word in segment.words:
            all_words.append({
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end
            })
    return all_words


def load_lyrics_lines(lyrics_file_path: str):
    """
    Load lyrics file and return list of lines (non-empty only).
    """
    with open(lyrics_file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def match_words_to_lines(words, lyrics_lines):
    """
    Match aligned words to lyrics lines.
    
    Returns list of line objects:
    [{
        "text": "original line text",
        "words": [{word, start, end}, ...],
        "start": first word start time,
        "end": last word end time
    }, ...]
    """
    line_objects = []
    word_index = 0
    
    for line in lyrics_lines:
        line_word_count = len(line.split())
        line_words = words[word_index:word_index + line_word_count]
        word_index += line_word_count
        
        if not line_words:
            continue
        
        line_obj = {
            "text": line,
            "words": line_words,
            "start": line_words[0]["start"],
            "end": line_words[-1]["end"]
        }
        line_objects.append(line_obj)
    
    return line_objects


def create_karaoke_effect(words):
    """
    Create ASS karaoke effect string for a line.
    
    Uses \\k tag for progressive highlighting.
    \\k<duration> highlights characters over <duration> centiseconds.
    
    For word-level highlighting, we use:
    {\\k0}word{\\k<duration>} for each word
    
    The \\k tag works on character level, so we calculate duration
    per word and apply it after the word.
    """
    if not words:
        return ""

    parts = []

    for i, word_data in enumerate(words):
        word = word_data["word"]
        duration_cs = int((word_data["end"] - word_data["start"]) * 100)
        duration_cs = max(duration_cs, 10)  # Minimum 10 centiseconds

        # Add word with karaoke timing
        # \\k after the word means "highlight this word over this duration"
        if i == 0:
            # First word starts immediately
            parts.append(f"{word}{{\\k{duration_cs}}}")
        else:
            # Subsequent words - add space before the word
            parts.append(f" {word}{{\\k{duration_cs}}}")

    return "".join(parts)


def create_karaoke_effect_word_by_word(words):
    """
    Create word-level karaoke highlighting.

    Each word lights up progressively as it's sung.
    Uses character-level \\k timing within each word.
    """
    if not words:
        return ""

    parts = []

    for i, word_data in enumerate(words):
        word = word_data["word"]
        duration_cs = int((word_data["end"] - word_data["start"]) * 100)
        duration_cs = max(duration_cs, 10)

        # Distribute duration across characters for progressive highlight
        char_count = len(word)
        if char_count > 0:
            per_char_cs = max(duration_cs // char_count, 5)
            # Apply \\k to each character for smooth progressive effect
            highlighted_word = ""
            for char in word:
                highlighted_word += f"{char}{{\\k{per_char_cs}}}"
            # Add space before word (except first)
            if i == 0:
                parts.append(highlighted_word)
            else:
                parts.append(" " + highlighted_word)
        else:
            parts.append(word)

    return "".join(parts)


def create_simple_word_highlight(words):
    """
    Create simpler word-level highlighting.

    Each word appears with its own timing using \\k.
    This is more compatible with various players.
    """
    if not words:
        return ""

    parts = []

    for i, word_data in enumerate(words):
        word = word_data["word"]
        duration_cs = int((word_data["end"] - word_data["start"]) * 100)
        duration_cs = max(duration_cs, 10)

        if i == 0:
            parts.append(f"{{\\k{duration_cs}}}{word}")
        else:
            # Add space before each word (except first)
            parts.append(f"{{\\k{duration_cs}}} {word}")

    return "".join(parts)


def generate_ass_header():
    """Generate the ASS file header with styles."""
    return f"""[Script Info]
Title: Karaoke Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOR},{SECONDARY_COLOR},{OUTLINE_COLOR},{BACK_COLOR},0,0,0,0,100,100,0,0,3,{OUTLINE_WIDTH},{SHADOW_OFFSET},2,{MARGIN_LEFT},{MARGIN_RIGHT},{MARGIN_VERTICAL},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass_events(line_objects):
    """Generate ASS event lines for each lyrics line."""
    events = []

    for line_obj in line_objects:
        start_time = seconds_to_ass_time(line_obj["start"])
        end_time = seconds_to_ass_time(line_obj["end"])

        # Create karaoke text with word-level highlighting
        karaoke_text = create_simple_word_highlight(line_obj["words"])

        events.append(
            f"Dialogue: 0,{start_time},{end_time},Karaoke,,0,0,0,,{karaoke_text}"
        )

    return "\n".join(events)


def generate_enhanced_karaoke_ass(line_objects):
    """
    Generate enhanced .ass with better karaoke effects.

    Uses \\K (capital K) for fill-style karaoke where characters
    fill in progressively.
    """
    header = generate_ass_header()
    events = []

    for line_obj in line_objects:
        start_time = seconds_to_ass_time(line_obj["start"])
        end_time = seconds_to_ass_time(line_obj["end"])

        # Build karaoke text with \\K for progressive fill
        parts = []
        cumulative_cs = 0

        for i, word_data in enumerate(line_obj["words"]):
            word = word_data["word"]
            duration_cs = int((word_data["end"] - word_data["start"]) * 100)
            duration_cs = max(duration_cs, 10)

            if i == 0:
                parts.append(f"{{\\K{duration_cs}}}{word}")
            else:
                # Add space before word
                parts.append(f"{{\\K{duration_cs}}} {word}")

        karaoke_text = "".join(parts)

        events.append(
            f"Dialogue: 0,{start_time},{end_time},Karaoke,,0,0,0,,{karaoke_text}"
        )

    return header + "\n".join(events)


def save_ass(result, lyrics_file_path: str, output_path: str):
    """
    Generate and save .ass file with karaoke highlighting.
    """
    # Extract words and match to lyrics lines
    words = extract_words_from_alignment(result)
    lyrics_lines = load_lyrics_lines(lyrics_file_path)
    line_objects = match_words_to_lines(words, lyrics_lines)
    
    # Generate enhanced ASS content
    ass_content = generate_enhanced_karaoke_ass(line_objects)
    
    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    
    print(f"Karaoke .ass saved to {output_path}")
    print(f"  Total lines: {len(line_objects)}")
    print(f"  Duration: {seconds_to_ass_time(line_objects[-1]['end']) if line_objects else 'N/A'}")


def process_files(model, audio_file_path: str, lyrics_file_path: str):
    """Process audio and lyrics to generate karaoke .ass file."""
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found: {audio_file_path}")
        return
    
    if not os.path.exists(lyrics_file_path):
        print(f"Error: Lyrics file not found: {lyrics_file_path}")
        return
    
    # Output path: same base name as audio, .ass extension
    base_name = os.path.splitext(audio_file_path)[0]
    output_path = base_name + ".ass"
    
    print(f"\nAudio: {audio_file_path}")
    print(f"Lyrics: {lyrics_file_path}")
    
    # Align audio to lyrics
    result = align_audio(model, audio_file_path, lyrics_file_path)
    
    # Generate and save .ass
    save_ass(result, lyrics_file_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate karaoke .ass files with progressive word highlighting."
    )
    parser.add_argument("audio_file", help="Audio file path (e.g., song.mp3)")
    parser.add_argument("lyrics_file", help="Confirmed lyrics text file (e.g., song.txt)")
    args = parser.parse_args()
    
    # Load model
    model = load_model()
    
    # Process files
    process_files(model, args.audio_file, args.lyrics_file)


if __name__ == "__main__":
    main()
