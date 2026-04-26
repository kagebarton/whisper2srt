"""Genius Diarize — CLI entry point.

Usage:
    python -m genius_diarize.run <vocal_audio> <lyrics_file>

Takes a vocal stem audio file and a Genius-formatted lyrics file, then
produces .srt and .ass caption files where lines are labeled and colored
by speaker attribution parsed from Genius section headers.

Output files are written next to the vocal stem:
    <vocal_stem>.diarized.srt
    <vocal_stem>.diarized.ass
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
from genius_diarize.genius import genius_singer_mode, truncate_speaker_label
from genius_diarize.word_extraction import (
    extract_words,
    load_genius_lyrics,
    match_words_to_lines,
)
from genius_diarize.workers.whisper_worker import WhisperWorker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genius_diarize.run")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assign_speakers_from_genius(line_objects, genius_lines):
    """Copy speaker_label and dominant_speaker from each genius_line onto
    the matching line_obj (and onto every word). Modifies in place.
    """
    for line_obj, gl in zip(line_objects, genius_lines):
        line_obj["speaker"] = gl["speaker_label"]
        line_obj["dominant_speaker"] = gl["dominant_speaker"]
        for word in line_obj["words"]:
            word["speaker"] = gl["speaker_label"]
            word["dominant_speaker"] = gl["dominant_speaker"]


def annotate_display_labels(line_objects):
    """Walk lines in time order; first occurrence of each speaker_label gets
    the full label, subsequent occurrences get the truncated label. Sets
    line_obj["display_label"] (str | None).
    """
    seen = set()
    for line_obj in line_objects:
        label = line_obj.get("speaker")
        if label is None:
            line_obj["display_label"] = None
        elif label not in seen:
            seen.add(label)
            line_obj["display_label"] = label
        else:
            line_obj["display_label"] = truncate_speaker_label(label)


def reset_segment_first_flags(line_objects):
    """Reset is_segment_first so only the first word of each line is True.

    Called after match_words_to_lines. The whisper alignment marks
    is_segment_first=True on the first word of each *whisper* segment,
    but those boundaries don't match the lyric-line boundaries we just
    regrouped to.
    """
    for line in line_objects:
        for i, w in enumerate(line["words"]):
            w["is_segment_first"] = i == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate diarized .srt + .ass captions from Genius headers."
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
        # --- Align ---
        logger.info(f"Aligning lyrics: {lyrics_path.name} → {vocal_path.name}")
        result = whisper_worker.align_and_refine(vocal_path, plain_text)
        words = extract_words(result)
        lyrics_lines = [g["text"] for g in genius_lines]
        line_objects = match_words_to_lines(words, lyrics_lines)
        reset_segment_first_flags(line_objects)

        # Guard: line_objects and genius_lines must agree in count
        assert len(line_objects) == len(genius_lines), (
            f"Line count mismatch: whisper aligned {len(line_objects)} lines "
            f"but Genius parser produced {len(genius_lines)} lines. "
            f"Aborting to avoid silent mis-attribution."
        )

        if mode == "multi":
            # --- Assign speakers from Genius headers ---
            assign_speakers_from_genius(line_objects, genius_lines)
            annotate_display_labels(line_objects)
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
