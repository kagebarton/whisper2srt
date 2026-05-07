"""Diarized captions CLI entry point.

Usage:
    python -m diarized_captions.run <vocal_audio> [lyrics_file]

Takes a vocal stem audio file and (optionally) a lyrics file, then produces
.srt and .ass caption files where every line is labeled and colored by speaker.

Output files are written next to the vocal stem:
    <vocal_stem>.diarized.srt
    <vocal_stem>.diarized.ass
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# --- Bootstrap: sys.path + HF_HOME before any package imports ---

# Ensure the repo root is on sys.path so `diarized_captions` is importable
# even if invoked directly (python diarized_captions/run.py) instead of
# via `python -m diarized_captions.run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point HF cache at the local pre-downloaded model folder BEFORE importing
# anything that pulls in huggingface_hub or pyannote.
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

from diarized_captions.config import DiarizedCaptionsConfig
from diarized_captions.caption import generate_srt, generate_ass
from diarized_captions.genius import genius_singer_mode, unique_named_singers
from diarized_captions.speaker_labels import (
    assign_speakers_to_words,
    map_clusters_to_names,
    remap_speakers_by_appearance,
    reset_segment_first_flags,
    split_lines_at_speaker_boundaries,
)
from diarized_captions.word_extraction import (
    extract_words,
    load_genius_lyrics,
    load_lyrics,
    match_words_to_lines,
    segments_to_line_objects,
)
from diarized_captions.workers.whisper_worker import WhisperWorker
from diarized_captions.workers.diarize_worker import CancelableDiarizeWorker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diarized_captions.run")


def main():
    parser = argparse.ArgumentParser(
        description="Generate diarized .srt + .ass captions from a vocal stem."
    )
    parser.add_argument(
        "vocal_audio",
        type=Path,
        help="Path to vocal stem audio file (.wav, .m4a, etc.)",
    )
    parser.add_argument(
        "lyrics_file",
        type=Path,
        nargs="?",
        default=None,
        help="Optional lyrics file (.txt or .srt) for alignment mode",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=0,
        metavar="N",
        help="Exact number of speakers (0 = auto-detect)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=0,
        metavar="N",
        help="Minimum speakers (0 = no constraint)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        metavar="N",
        help="Maximum speakers (0 = no constraint)",
    )
    parser.add_argument(
        "--genius-constrain-speakers",
        action="store_true",
        default=False,
        help="Use singer count from Genius headers as min_speakers for pyannote",
    )
    args = parser.parse_args()

    vocal_path = args.vocal_audio
    lyrics_path = args.lyrics_file

    if not vocal_path.exists():
        logger.error(f"Vocal file not found: {vocal_path}")
        sys.exit(1)
    if lyrics_path is not None and not lyrics_path.exists():
        logger.error(f"Lyrics file not found: {lyrics_path}")
        sys.exit(1)

    cfg = DiarizedCaptionsConfig()
    cfg.diarize.num_speakers = args.num_speakers
    cfg.diarize.min_speakers = args.min_speakers
    cfg.diarize.max_speakers = args.max_speakers

    # --- Detect Genius singer mode early (before loading models) ---
    genius_lines = None
    mode = "solo"  # default for no-lyrics transcription path
    if lyrics_path is not None:
        lyrics_text, genius_lines = load_genius_lyrics(lyrics_path)
        mode = genius_singer_mode(genius_lines)
        logger.info(f"Genius singer mode: {mode}")

    # --- Load models (both always load) ---
    logger.info("Loading whisper model...")
    whisper_worker = WhisperWorker(cfg.whisper)
    t0 = time.time()
    whisper_worker.load_model()
    logger.info(f"Whisper model loaded in {time.time() - t0:.1f}s")

    logger.info("Loading pyannote diarization pipeline...")
    diarize_worker = CancelableDiarizeWorker(cfg.diarize)
    t0 = time.time()
    diarize_worker.load_model()
    logger.info(f"Pyannote pipeline loaded in {time.time() - t0:.1f}s")

    try:
        # --- 1. Transcribe or align via whisper ---
        if lyrics_path is None:
            logger.info(f"Transcribing: {vocal_path.name}")
            result = whisper_worker.transcribe_and_refine(vocal_path)
            line_objects = segments_to_line_objects(result)
        else:
            logger.info(f"Aligning lyrics: {lyrics_path.name} → {vocal_path.name}")
            result = whisper_worker.align_and_refine(vocal_path, lyrics_text)
            words = extract_words(result)
            lyric_lines = [l.strip() for l in lyrics_text.split("\n") if l.strip()]
            line_objects = match_words_to_lines(words, lyric_lines)
            # Fix is_segment_first flags after word regrouping
            reset_segment_first_flags(line_objects)

        if mode == "solo":
            # --- Solo mode: skip diarization entirely ---
            # All words stay speaker=None → single_speaker caption path
            # produces plain unlabeled karaoke output.
            logger.info(
                "Solo mode — skipping diarization. "
                "No singer attribution in headers."
            )

        elif mode == "multi":
            # --- Multi mode: run pyannote + Genius cluster mapping ---
            # Optionally constrain speaker count from Genius headers
            if args.genius_constrain_speakers and genius_lines is not None:
                singer_names = unique_named_singers(genius_lines)
                genius_count = len(singer_names)
                if genius_count > 0:
                    # Use as min_speakers (not num_speakers) — Genius
                    # might miss a guest vocalist; min is safer.
                    if cfg.diarize.min_speakers == 0 or genius_count > cfg.diarize.min_speakers:
                        cfg.diarize.min_speakers = genius_count
                    logger.info(
                        f"Constraining pyannote: min_speakers={cfg.diarize.min_speakers} "
                        f"(from Genius headers: {singer_names})"
                    )

            # Guard: line_objects and genius_lines must agree in count
            assert len(line_objects) == len(genius_lines), (
                f"Line count mismatch: whisper aligned {len(line_objects)} lines "
                f"but Genius parser produced {len(genius_lines)} lines. "
                f"Aborting to avoid silent mis-attribution."
            )

            logger.info(f"Diarizing: {vocal_path.name}")
            turns, overlap_intervals = diarize_worker.diarize(vocal_path)

            # Remap pyannote labels → A/B/C by appearance order
            turns, label_map = remap_speakers_by_appearance(turns, cfg)
            logger.info(f"Speakers detected: {list(label_map.values())}")

            # Build cluster → name mapping via Genius majority vote
            cluster_to_name = map_clusters_to_names(turns, line_objects, genius_lines)
            if cluster_to_name:
                logger.info(f"Cluster → name mapping: {cluster_to_name}")
            else:
                logger.warning(
                    "No cluster→name mappings established — "
                    "falling back to letter-based speakers"
                )

            # Attach speaker labels (names or letters) to every word
            line_objects = assign_speakers_to_words(
                line_objects,
                turns,
                overlap_intervals,
                genius_lines=genius_lines,
                cluster_to_name=cluster_to_name,
            )

            # Split lines wherever speaker changes mid-line
            line_objects = split_lines_at_speaker_boundaries(line_objects)

        # --- Write outputs alongside stem ---
        srt_out = vocal_path.with_suffix(".diarized.srt")
        ass_out = vocal_path.with_suffix(".diarized.ass")

        srt_content = generate_srt(line_objects, cfg)
        srt_out.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT written: {srt_out}")

        ass_content = generate_ass(line_objects, cfg)
        ass_out.write_text(ass_content, encoding="utf-8")
        logger.info(f"ASS written: {ass_out}")

    finally:
        # --- Unload models ---
        whisper_worker.unload_model()
        diarize_worker.unload_model()
        logger.info("Models unloaded.")


if __name__ == "__main__":
    main()
