import os
import glob
import argparse
import config
import stable_whisper


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
        condition_on_previous_text=False,
        initial_prompt="Lyrics",
    )


def seconds_to_srt(seconds: float) -> str:
    """Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(result, output_path: str):
    """Saves transcription result to SRT format, one segment per subtitle."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result.segments, start=1):
            start = seconds_to_srt(segment.start)
            end = seconds_to_srt(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def save_raw_text(result, output_path: str):
    """Saves raw Whisper transcription to plain text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in result.segments:
            f.write(segment.text.strip() + "\n")


def process_file(model, file_path: str):
    prefix = os.path.splitext(file_path)[0]
    srt_path = prefix + ".srt"
    txt_path = prefix + ".txt"

    print(f"\nProcessing: {file_path}")

    result = transcribe_audio(model, file_path)
    save_srt(result, srt_path)
    save_raw_text(result, txt_path)

    print(f"SRT saved to {srt_path}")
    print(f"Raw text saved to {txt_path}")


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
