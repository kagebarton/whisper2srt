import os
import re
import glob
import argparse
import config
import stable_whisper

GAP_THRESHOLD = 0.4  # seconds; tune this if needed


def load_model():
    return stable_whisper.load_faster_whisper(
        config.WHISPER_MODEL,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE
    )


def transcribe_audio(model, audio_file_path: str):
    """Transcribes audio using faster-whisper via stable-ts."""
    print("Transcribing audio...")
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


def seconds_to_srt(seconds: float) -> str:
    """Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(result, output_path: str):
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


def process_file(model, file_path: str):
    prefix = os.path.splitext(file_path)[0]
    output_path = prefix + ".srt"

    print(f"\nProcessing: {file_path}")

    result = transcribe_audio(model, file_path)
    save_srt(result, output_path)

    print(f"SRT saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT subtitles.")
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
        for file_path in file_paths:
            process_file(model, file_path)
