#!/usr/bin/env python3
"""Interactive test runner for mid-diarization cancellation with caption output.

Uses the two-layer hook architecture: Layer 1 (pyannote hook callback for
stage-boundary progress) + Layer 2 (PyTorch forward_pre_hooks on both
segmentation and embedding models for per-batch cancel).

Usage:
python run_test.py <vocal_audio> [srt/ass/both] [--cancel-after SECONDS] [--auto] [--cancel-stage STAGE]

Modes:
Direct (default):
- Loads the pyannote pipeline
- Runs diarization with optional cancellation
- Writes caption files (SRT, ASS, or both) next to the input

Auto (--auto):
- Cancels after --cancel-after seconds (or when --cancel-stage starts)
- Then re-runs to prove model is still loaded
- Writes caption files after the successful re-run

--cancel-stage watches the pyannote hook callback and fires the cancel
event when the requested stage starts. This makes embedding-phase cancel
testing deterministic regardless of audio length.

Examples:
# Generate SRT from diarization
python run_test.py vocals.wav srt

# Generate ASS from diarization
python run_test.py vocals.wav ass

# Generate both SRT and ASS
python run_test.py vocals.wav both

# Cancel after 5 seconds, then re-run and write both
python run_test.py vocals.wav both --auto --cancel-after 5

# Cancel when embedding stage starts (deterministic embedding-phase cancel)
python run_test.py vocals.wav both --auto --cancel-stage embedding
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Point HF cache at the local pre-downloaded model folder BEFORE importing
# anything that pulls in huggingface_hub or pyannote.
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

# Add cancel_tests to sys.path so diarize package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diarize.caption import generate_ass, generate_srt
from diarize.config import DiarizeConfig
from diarize.workers.cancelable_diarize_worker import (
    CancelableDiarizeWorker,
    DiarizationCancelledError,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_test")


def write_captions(turns, vocal_path, caption_format, config):
    """Write caption files based on diarization turns.

    Args:
        turns: List of diarization turn dicts.
        vocal_path: Path to the vocal audio file (used for output naming).
        caption_format: "srt", "ass", or "both".
        config: DiarizeConfig for styling.

    Returns:
        List of output file paths.
    """
    output_paths = []
    vocal_stem = vocal_path.stem

    if caption_format in ("srt", "both"):
        srt_path = vocal_path.parent / f"{vocal_stem}.diarization.srt"
        srt_content = generate_srt(turns, config)
        srt_path.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT written: {srt_path}")
        output_paths.append(srt_path)

    if caption_format in ("ass", "both"):
        ass_path = vocal_path.parent / f"{vocal_stem}.diarization.ass"
        ass_content = generate_ass(turns, config)
        ass_path.write_text(ass_content, encoding="utf-8")
        logger.info(f"ASS written: {ass_path}")
        output_paths.append(ass_path)

    return output_paths


def direct_mode(
    vocal_path: str,
    caption_format: str,
    cancel_after: float,
    auto: bool,
    config: DiarizeConfig,
    cancel_stage: str = "",
) -> None:
    """Direct test: load model, run diarization, write captions.

    If --auto is set, cancels after cancel_after seconds (or when
    --cancel-stage starts), then re-runs.
    """
    worker = CancelableDiarizeWorker(config)
    vocal = Path(vocal_path)

    if not vocal.exists():
        logger.error(f"Vocal file not found: {vocal}")
        return

    logger.info("=== CANCEL DIARIZE TEST (two-layer hooks) ===")
    logger.info("Loading pyannote pipeline (this takes a few seconds)...")

    start = time.time()
    worker.load_model()
    model_load_time = time.time() - start
    logger.info(f"Pipeline loaded in {model_load_time:.1f}s")

    if auto:
        # --- Cancel then re-run ---
        logger.info(f"\n{'=' * 60}")
        logger.info("TEST 1: Cancel diarization mid-stream")
        if cancel_stage:
            logger.info(f"Cancel trigger: when stage '{cancel_stage}' starts")
        else:
            logger.info(f"Cancel trigger: after {cancel_after}s")
        logger.info(f"{'=' * 60}")

        cancel_event = threading.Event()
        sep_result = [None, None]  # [result, exception]
        stage_seen = threading.Event()

        def run_diarize():
            try:
                # If cancel_stage is set, wrap the diarize call with a
                # stage-monitoring approach: we use a separate mechanism
                # to detect when the target stage starts.
                # The worker's pyannote hook logs progress; we'll use
                # a timer-based approach for simplicity, but the
                # --cancel-stage flag controls which stage we target.
                turns = worker.diarize(vocal, cancel_event=cancel_event)
                sep_result[0] = turns
            except Exception as e:
                sep_result[1] = e

        sep_thread = threading.Thread(target=run_diarize, daemon=True)
        sep_thread.start()

        if cancel_stage:
            # Wait for the target stage by polling worker's _current_stage
            # The worker updates _current_stage from the pyannote hook callback
            logger.info(f"Waiting for stage '{cancel_stage}' to start...")
            stage_timeout = 300  # 5 min max wait
            stage_start = time.time()
            while not stage_seen.is_set() and time.time() - stage_start < stage_timeout:
                current = getattr(worker, "_current_stage", None)
                if current and cancel_stage in (current or "").lower():
                    logger.info(f"Stage '{current}' detected — setting cancel event")
                    cancel_event.set()
                    stage_seen.set()
                    break
                time.sleep(0.1)
            if not stage_seen.is_set():
                logger.warning(
                    f"Stage '{cancel_stage}' not seen within {stage_timeout}s "
                    f"— falling back to timer-based cancel after {cancel_after}s"
                )
                cancel_event.set()
        else:
            # Timer-based cancel
            logger.info(f"Waiting {cancel_after}s before cancelling...")
            time.sleep(cancel_after)

            logger.info(">>> SETTING CANCEL EVENT <<<")
            cancel_event.set()

        # Wait for the thread to finish
        sep_thread.join(timeout=120)

        if sep_result[1] is not None:
            if isinstance(sep_result[1], DiarizationCancelledError):
                logger.info(
                    "Got DiarizationCancelledError -- "
                    "diarization cancelled (hook caught the event)"
                )
            else:
                logger.error(f"Got unexpected error: {sep_result[1]}")
        elif sep_result[0] is not None:
            logger.info(
                "Diarization completed before cancel took effect "
                "(increase --cancel-after)"
            )

        # Check model is still loaded
        logger.info(f"Model still loaded after cancel: {worker.model_loaded}")

        if not worker.model_loaded:
            logger.error("Model was LOST -- cannot re-run!")
            return

        # --- Re-run diarization ---
        logger.info(f"\n{'=' * 60}")
        logger.info("TEST 2: Re-run diarization (model should still be loaded)")
        logger.info(f"{'=' * 60}")

    # --- Run (or re-run) diarization ---
    start = time.time()
    try:
        turns = worker.diarize(vocal)
        elapsed = time.time() - start
        logger.info(f"Diarization completed in {elapsed:.1f}s")

        if turns:
            speakers = sorted(set(t["speaker"] for t in turns))
            logger.info(f" Speakers detected: {speakers}")
            logger.info(f" Turns: {len(turns)}")
            logger.info(f" Duration: {turns[-1]['end']:.1f}s")
        else:
            logger.warning("No turns returned from diarization")

    except DiarizationCancelledError:
        logger.error("Unexpected cancellation on final run!")
        return
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return

    # --- Write caption files ---
    if turns and caption_format != "none":
        logger.info(f"\nWriting {caption_format.upper()} captions...")
        output_paths = write_captions(turns, vocal, caption_format, config)
        for p in output_paths:
            logger.info(f" Output: {p}")

    # Cleanup
    worker.unload_model()
    logger.info("\n=== TEST COMPLETE ===")


def main():
    parser = argparse.ArgumentParser(
        description="Test mid-diarization cancellation with caption output"
    )
    parser.add_argument(
        "vocal_audio", help="Path to vocal stem audio file (WAV, M4A, etc.)"
    )
    parser.add_argument(
        "caption_format",
        nargs="?",
        default="both",
        choices=["srt", "ass", "both"],
        help="Caption output format: srt, ass, or both (default: both)",
    )
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=5.0,
        help="Seconds to wait before auto-cancelling (default: 5)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-cancel mode: cancel then re-run to prove model survives",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Path to HuggingFace token file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, or auto",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=0,
        help="Number of speakers (0 = auto-detect)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=0,
        help="Minimum speakers (0 = no constraint)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        help="Maximum speakers (0 = no constraint)",
    )
    parser.add_argument(
        "--cancel-stage",
        type=str,
        default="",
        choices=["segmentation", "embedding", ""],
        help="Cancel when this pyannote stage starts (deterministic stage cancel)",
    )

    args = parser.parse_args()

    if not Path(args.vocal_audio).exists():
        logger.error(f"File not found: {args.vocal_audio}")
        sys.exit(1)

    config = DiarizeConfig()
    if args.hf_token:
        config.hf_token_path = args.hf_token
    if args.device is not None:
        config.device = args.device
    config.num_speakers = args.num_speakers
    config.min_speakers = args.min_speakers
    config.max_speakers = args.max_speakers

    direct_mode(
        args.vocal_audio,
        args.caption_format,
        args.cancel_after,
        args.auto,
        config,
        cancel_stage=args.cancel_stage,
    )


if __name__ == "__main__":
    main()
