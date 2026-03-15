import re
import config
from faster_whisper import WhisperModel


GAP_THRESHOLD = 0.4  # seconds; tune this if needed


def transcribe_audio(audio_file_path: str) -> list:
    """
    Returns a list of transcribed segments from an audio file.
    Each segment contains:
        text -> the transcribed text for this segment;
        start -> segment start time in seconds;
        end -> segment end time in seconds;
        words -> list of word-level timestamps, each with word, start, and end
    """
    model = WhisperModel(
        config.WHISPER_MODELS_DIR,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE
    )

    segments, info = model.transcribe(
        audio_file_path,
        word_timestamps=True,
        language="en",
        beam_size=10,
        best_of=10,
        temperature=0.0,
        condition_on_previous_text=True,
        initial_prompt="Lyrics:"
    )

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # faster-whisper returns a generator, so we resolve it into a list here
    return list(segments)


def seconds_to_srt(seconds: float) -> str:
    """Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(segments: list, output_path: str):
    """
    Saves transcript segments to a paginated SRT subtitle file.
    Uses word-level timestamps to split lines at capital letters,
    so each subtitle block gets accurate start/end times.
    "I" is treated specially since it is always capitalized in English —
    it only triggers a new line if preceded by a pause or punctuation.
    """
    blocks = []

    for segment in segments:
        words = segment.words  # faster-whisper uses attributes, not dict keys

        current_line = []
        for word in words:
            text = word.word.strip()
            is_capital = bool(re.match(r'[A-Z]', text))
            is_new_line = False

            if current_line and is_capital:
                if text == "I":
                    # "I" is always capitalized, so require extra evidence
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

        # Don't forget the last line in the segment
        if current_line:
            blocks.append({
                "start": seconds_to_srt(current_line[0].start),
                "end": seconds_to_srt(current_line[-1].end),
                "text": " ".join(w.word.strip() for w in current_line)
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks, start=1):
            f.write(f"{i}\n{block['start']} --> {block['end']}\n{block['text']}\n\n")


if __name__ == "__main__":
    file_path = "/home/ken/Downloads/KPop Demon Hunters  - Free.webm"
    output_path = "/home/ken/Downloads/KPop Demon Hunters  - Free.srt"

    segments = transcribe_audio(file_path)
    save_srt(segments, output_path)

    print(f"SRT saved to {output_path}")
