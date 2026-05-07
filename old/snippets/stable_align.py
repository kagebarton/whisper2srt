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
MODEL_PATH = "/home/ken/whisper2srt/whisper-model/large-v3"  # Set absolute path to your model folder

# Device for inference: "cpu", "cuda", or "auto"
DEVICE = "auto"

# Compute type for faster-whisper: "float16", "int8", "int8_float16", or "float32"
COMPUTE_TYPE = "int8"

# ASS styling
FONT_NAME = "Arial"
FONT_SIZE = 60
PRIMARY_COLOR = "&H00D7FF&"  # Soft Yellow
SECONDARY_COLOR = "&H00FFFFFF"  # White (not yet sung)
OUTLINE_COLOR = "&H00000000"  # Black outline
BACK_COLOR = "&H80000000&"  # Translucent shadow
OUTLINE_WIDTH = 3
SHADOW_OFFSET = 2
MARGIN_LEFT = 50
MARGIN_RIGHT = 50
MARGIN_VERTICAL = 150

# Karaoke overlap timing
#
# These two values create the classic karaoke "next line preview" effect:
#
#   LINE_LEAD_IN_CS   — how many centiseconds before the first sung word the
#                       subtitle card appears on screen. During this window the
#                       line sits in SECONDARY_COLOR (unsung/white) while the
#                       previous line is still finishing. This gives the singer
#                       time to read ahead before they need to start singing.
#
#   LINE_LEAD_OUT_CS  — how many centiseconds after the last sung word the card
#                       stays on screen. Combined with the next line's lead-in
#                       this creates the brief overlap where both lines are
#                       visible simultaneously.
#
#   FIRST_WORD_NUDGE_CS — how many centiseconds to push back the first word of a
#                         segment when it starts almost immediately after the
#                         previous segment's end (within ~50ms of the expected
#                         lead-in gap). Helps prevent word clipping by adding a
#                         small buffer before the first word.
#
# The lead-in is encoded as a silent {\k<lead_in_cs>} tag at the very start of
# each Dialogue event. This advances the karaoke cursor through the lead-in
# period without sweeping any colour, so the fill only begins at the exact
# moment the first word is sung – not when the card first appears.

LINE_LEAD_IN_CS       = 80   # 0.5 seconds
LINE_LEAD_OUT_CS      = 20   # 0.5 seconds
FIRST_WORD_NUDGE_CS   = 0    # Disabled by default; 15 = ~150ms, tune by ear


# ==============================================================================


def load_model():
    """Load the faster-whisper model via stable-ts."""
    # Use MODEL_PATH if set, otherwise fall back to default model name
    model_source = MODEL_PATH if MODEL_PATH else "faster-whisper-large-v3"

    print(f"Loading model: {model_source}")
    return stable_whisper.load_faster_whisper(
        model_source,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
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
        verbose=False,
        vad=True,           # Silero VAD — much better at first-word boundaries
        vad_threshold=0.25,
        suppress_silence=True,
        suppress_word_ts=True,
        only_voice_freq=True,  # you're already on vocal stems, costs nothing
    )

    model.refine(
        audio_file_path,
        result,
         # --- What to refine ---
        steps="s",              # 's' = refine starts, 'e' = refine ends, 'se' = both
        word_level=False,         # False to only refine segment start/end, not individual words

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
    all_words = []
    for segment in result.segments:
        for i, word in enumerate(segment.words):
            all_words.append({
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end,
                "is_segment_first": i == 0   # ← add this
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
Style: Karaoke,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOR},{SECONDARY_COLOR},{OUTLINE_COLOR},{BACK_COLOR},0,0,0,0,100,100,0,0,1,{OUTLINE_WIDTH},{SHADOW_OFFSET},2,{MARGIN_LEFT},{MARGIN_RIGHT},{MARGIN_VERTICAL},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass_events(line_objects):
    """Unused — kept for reference. See generate_enhanced_karaoke_ass."""
    pass


def generate_enhanced_karaoke_ass(line_objects):
    """
    Generate the .ass file content with correct karaoke overlap timing.

    Each Dialogue event is structured in three layers:

      1. EVENT WINDOW (Start / End timestamps)
         Start = first_word_start - LINE_LEAD_IN_CS
         End   = last_word_end   + LINE_LEAD_OUT_CS

         The lead-in makes the card appear early so the singer can read ahead.
         The lead-out keeps it visible briefly after the last word.
         When adjacent lines' windows overlap, ASS stacks both on screen
         simultaneously — the new line sits in SECONDARY_COLOR while the old
         line finishes its sweep.

      2. SILENT LEAD-IN CURSOR TAG  {\\k<lead_in_cs>}
         The karaoke cursor runs from the Dialogue Start time, not from when
         words are sung. Without this tag the fill sweep would begin the moment
         the card appears (lead_in_cs too early). This silent \\k tag burns
         through the lead-in period without changing any colour, so the sweep
         only starts at the correct moment.

         The cursor is initialised to line_start (= first_word_start - lead_in).
         For the first word the gap is therefore exactly LINE_LEAD_IN_CS, which
         becomes the silent tag. For subsequent words any gap between the
         previous word's end and the next word's start is also encoded as a
         silent \\k tag — this keeps the cursor accurate across natural pauses.

      3. FILL SWEEP TAGS  {\\kf<duration_cs>}word
         \\kf produces a left-to-right colour fill over <duration_cs>
         centiseconds, sweeping from SECONDARY_COLOR (unsung) to PRIMARY_COLOR
         (sung) as each word is reached.

         A leading {\\k0} resets the karaoke state at the start of each line
         so it always begins fully in SECONDARY_COLOR regardless of any prior
         karaoke state from the previous event.
    """
    header = generate_ass_header()
    events = []

    for line_obj in line_objects:
        words = line_obj["words"]
        if not words:
            continue

        # Pad the event window around the sung word boundaries
        event_start = max(0.0, words[0]["start"] - LINE_LEAD_IN_CS / 100.0)
        event_end   = words[-1]["end"] + LINE_LEAD_OUT_CS / 100.0

        # Karaoke cursor starts at the event start time.
        # Tracking prev_end from here means the gap to the first word
        # automatically becomes the silent lead-in tag.
        prev_end = event_start
        #parts    = ["{\\k0}"]   # reset karaoke state to fully unsung
        parts    = []

        for i, word_data in enumerate(words):
            word        = word_data["word"]

            word_start = word_data["start"]
            # Apply first_word_nudge_cs: push back the first word of a segment if it
            # starts almost immediately after the previous segment (within ~50ms of
            # the expected lead-in gap), preventing word clipping.
            if word_data.get("is_segment_first") and abs(word_start - prev_end - LINE_LEAD_IN_CS / 100.0) < 0.05:
                word_start += FIRST_WORD_NUDGE_CS / 100.0

            word_end    = word_data["end"]
            word_dur_cs = max(10, round((word_end - word_start) * 100))

            # Silent cursor advance through any gap before this word.
            # For the first word this gap = LINE_LEAD_IN_CS (the lead-in).
            # For subsequent words it covers natural pauses between words.
            gap_cs = max(0, round((word_start - prev_end) * 100))
            if gap_cs > 0:
                parts.append(f"{{\\k{gap_cs}}}")

            # \\kf = left-to-right fill sweep over word_dur_cs centiseconds
            parts.append(f"{{\\kf{word_dur_cs}}}{word}")
            prev_end = word_end

            if i < len(words) - 1:
                parts.append(" ")

        karaoke_text = "".join(parts)
        events.append(
            f"Dialogue: 0,{seconds_to_ass_time(event_start)},"
            f"{seconds_to_ass_time(event_end)},Karaoke,,0,0,0,,{karaoke_text}"
        )

    return header + "\n".join(events) + "\n"


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
