"""Genius Diarize — native-segmentation entry point.

Variant of run.py that delegates lyric-line segmentation to stable-ts via
``original_split=True`` instead of running a post-whisper matcher.

Each '\\n' in the alignment text becomes a stable-ts segment boundary, so
``result.segments`` returns already grouped 1:1 with the genius lines.
This skips the entire word→line matching step (no NW, no walk, no count).

Usage:
    python -m genius_diarize.run_native <vocal_audio> <lyrics_file>
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# --- Bootstrap: sys.path before any package imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genius_diarize.config import GeniusDiarizeConfig
from genius_diarize.caption import generate_srt, generate_ass
from genius_diarize.genius import genius_singer_mode
from genius_diarize.run import assign_speakers_from_genius
from genius_diarize.word_extraction import load_genius_lyrics
from genius_diarize.workers.whisper_worker import WhisperWorker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genius_diarize.run_native")


_MIN_WORD_PROBABILITY = 0.0001


def segments_to_line_objects(result) -> list:
    """Build one line_object per stable-ts segment.

    With ``original_split=True``, each segment corresponds to a single
    lyric line, so segments map 1:1 onto genius_lines. Low-probability
    words (whisper hallucinations from silent regions) are dropped, but
    the segment is preserved — its line_object simply ends up with a
    shorter ``words`` list. Empty segments retain segment-level start/end
    so downstream attribution still aligns by index.
    """
    line_objects = []
    dropped = 0
    for segment in result.segments:
        words = []
        for w in segment.words:
            prob = getattr(w, "probability", None)
            if prob is not None and prob < _MIN_WORD_PROBABILITY:
                dropped += 1
                continue
            words.append({
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "is_segment_first": len(words) == 0,
                "speaker": None,
                "dominant_speaker": None,
            })
        if words:
            start = words[0]["start"]
            end = words[-1]["end"]
        else:
            start = segment.start
            end = segment.end
        line_objects.append({
            "text": segment.text.strip(),
            "words": words,
            "start": start,
            "end": end,
        })
    if dropped:
        logger.info(
            "Dropped %d low-probability whisper words (< %.4f)",
            dropped, _MIN_WORD_PROBABILITY,
        )
    return line_objects


def main():
    parser = argparse.ArgumentParser(
        description="Generate diarized .srt + .ass captions using stable-ts native segmentation."
    )
    parser.add_argument(
        "vocal_audio",
        type=Path,
        help="Path to vocal stem audio file (.wav, .m4a, etc.)",
    )
    parser.add_argument(
        "lyrics_file",
        type=Path,
        help="Genius-formatted lyrics file (.txt) with section headers",
    )
    parser.add_argument(
        "--save-whisper",
        action="store_true",
        help="Save raw Whisper alignment result as <stem>.whisper.json next to the audio file.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Silero VAD during whisper alignment.",
    )
    args = parser.parse_args()

    vocal_path = args.vocal_audio
    lyrics_path = args.lyrics_file

    if not vocal_path.exists():
        logger.error(f"Vocal file not found: {vocal_path}")
        sys.exit(1)
    if not lyrics_path.exists():
        logger.error(f"Lyrics file not found: {lyrics_path}")
        sys.exit(1)

    cfg = GeniusDiarizeConfig()
    cfg.whisper.original_split = True
    if args.no_vad:
        cfg.whisper.vad = False
        logger.info("VAD disabled — whisper will attempt to align every region.")

    # --- Parse lyrics (single source of truth) ---
    lyrics_text = lyrics_path.read_text(encoding="utf-8")
    genius_lines, plain_text = load_genius_lyrics(lyrics_text)
    mode = genius_singer_mode(genius_lines)
    logger.info(f"Genius singer mode: {mode}")

    # --- Load whisper model ---
    logger.info("Loading whisper model...")
    whisper_worker = WhisperWorker(cfg.whisper)
    t0 = time.time()
    whisper_worker.load_model()
    logger.info(f"Whisper model loaded in {time.time() - t0:.1f}s")

    try:
        # --- Align (segments come back grouped per lyric line) ---
        logger.info(f"Aligning lyrics: {lyrics_path.name} → {vocal_path.name}")
        result = whisper_worker.align_and_refine(vocal_path, plain_text)

        if args.save_whisper:
            whisper_json = vocal_path.with_name(vocal_path.stem + ".whisper.json")
            result.save_as_json(str(whisper_json))
            logger.info(f"Whisper result saved: {whisper_json}")

        line_objects = segments_to_line_objects(result)

        # 1:1 invariant: stable-ts must emit one segment per lyric line.
        # If this trips, original_split didn't behave as expected — bail
        # rather than silently mis-attribute speakers.
        assert len(line_objects) == len(genius_lines), (
            f"Segment count mismatch: stable-ts produced {len(line_objects)} "
            f"segments but lyrics have {len(genius_lines)} lines. "
            f"original_split=True did not preserve line breaks 1:1."
        )

        if mode == "multi":
            assign_speakers_from_genius(line_objects, genius_lines)
            logger.info("Speaker assignment complete (Genius headers only)")
        else:
            logger.info(
                "Solo mode — no labeled sections. "
                "Output will be plain karaoke (no speaker labels)."
            )

        # --- Write outputs ---
        srt_out = vocal_path.with_suffix(".diarized.srt")
        ass_out = vocal_path.with_suffix(".diarized.ass")

        srt_content = generate_srt(line_objects, cfg)
        srt_out.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT written: {srt_out}")

        ass_content = generate_ass(line_objects, cfg)
        ass_out.write_text(ass_content, encoding="utf-8")
        logger.info(f"ASS written: {ass_out}")

    finally:
        whisper_worker.unload_model()
        logger.info("Models unloaded.")


if __name__ == "__main__":
    main()
