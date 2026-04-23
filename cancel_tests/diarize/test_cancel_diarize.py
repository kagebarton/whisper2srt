#!/usr/bin/env python3
"""Standalone test for cancelable pyannote diarization (two-layer hook architecture).

This script proves that pyannote's SpeakerDiarization pipeline can be
cancelled mid-computation without losing the loaded model, using two
complementary hook layers:

  Layer 1 — pyannote hook= callback: fires at stage boundaries (covers clustering).
  Layer 2 — PyTorch forward_pre_hook on segmentation + embedding models:
            fires per batch (~0.1-1s latency within either stage).

With the old monkey-patch design, cancel only worked during the segmentation
phase. Now it works during embedding too — try --cancel-after 8 on a 45s file
to cancel during the embedding phase.

USAGE:
python test_cancel_diarize.py <vocal_audio> [--cancel-after SECONDS]

Example:
python test_cancel_diarize.py vocals.wav --cancel-after 5
python test_cancel_diarize.py vocals.m4a --cancel-after 2
python test_cancel_diarize.py vocals.wav --hf-token /path/to/token.txt
python test_cancel_diarize.py vocals.wav --num-speakers 2

REQUIREMENTS:
pip install pyannote.audio
A HuggingFace token with access to pyannote/speaker-diarization-3.1
(Set HF_TOKEN env var or use --hf-token flag)
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
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "pyann-models"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

# Add project root to path so cancel_diarize is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cancel_diarize.config import DiarizeConfig
from cancel_diarize.workers.cancelable_diarize_worker import (
    CancelableDiarizeWorker,
    DiarizationCancelledError,
)


def setup_logging() -> logging.Logger:
    """Configure logging for the test."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    for name in ("pyannote", "cancel_diarize", "lightning", "torch"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False

    return logging.getLogger("test_cancel_diarize")


def print_turns(turns, log):
    """Print a summary of diarization turns."""
    speakers = sorted(set(t["speaker"] for t in turns))
    log.info(f"  Speakers: {speakers}")
    log.info(f"  Turns: {len(turns)}")
    if turns:
        log.info(f"  Duration: {turns[-1]['end']:.1f}s")
        log.info("  First 10 turns:")
        for turn in turns[:10]:
            log.info(
                f"    [{turn['start']:.2f}s - {turn['end']:.2f}s] {turn['speaker']}"
            )
        if len(turns) > 10:
            log.info(f"  ... and {len(turns) - 10} more")


def run_test(vocal_path: str, cancel_after: float, config: DiarizeConfig):
    """Run the cancellation proof-of-concept test.

    Steps:
    1. Load the pyannote pipeline (expensive, one-time)
    2. Run diarization with a cancel event that fires after cancel_after seconds
    3. Verify the model is still loaded after cancellation
    4. Re-run diarization to completion to prove the model survived
    5. Print diarization results
    """
    log = logging.getLogger("test_cancel_diarize")
    vocal = Path(vocal_path)

    if not vocal.exists():
        log.error(f"Vocal file not found: {vocal}")
        sys.exit(1)

    # --- STEP 1: Load model ---
    log.info("=" * 70)
    log.info("STEP 1: Loading pyannote diarization pipeline")
    log.info("=" * 70)
    worker = CancelableDiarizeWorker(config)
    start = time.time()
    worker.load_model()
    model_load_time = time.time() - start
    log.info(f"Pipeline loaded in {model_load_time:.1f}s")

    # --- STEP 2: Run diarization with cancellation ---
    log.info("")
    log.info("=" * 70)
    log.info(f"STEP 2: Running diarization (will cancel after {cancel_after}s)")
    log.info("=" * 70)

    cancel_event = threading.Event()

    # Schedule the cancel event to be set after cancel_after seconds
    cancel_timer = threading.Timer(cancel_after, cancel_event.set)
    cancel_timer.daemon = True

    start = time.time()
    cancel_timer.start()

    try:
        turns = worker.diarize(vocal, cancel_event=cancel_event)
        elapsed = time.time() - start

        if turns is not None:
            log.warning(
                f"Diarization completed in {elapsed:.1f}s -- "
                f"cancel timer did not fire fast enough. "
                f"Try a larger --cancel-after value."
            )
            log.info("Skipping cancel test -- diarization was too fast.")
            log.info("")
            log.info("=" * 70)
            log.info("STEP 3: Model still loaded (trivially true)")
            log.info("=" * 70)
            log.info(f"Model is still loaded: {worker.model_loaded}")
            print_turns(turns, log)
            return

    except DiarizationCancelledError as e:
        elapsed = time.time() - start
        log.info(f"Diarization cancelled after {elapsed:.1f}s (as expected)")
        log.info(f"  {e}")

    # --- STEP 3: Verify model survived ---
    log.info("")
    log.info("=" * 70)
    log.info("STEP 3: Verify model is still loaded after cancellation")
    log.info("=" * 70)
    log.info(f"Model is still loaded: {worker.model_loaded}")

    if not worker.model_loaded:
        log.error("Model was lost after cancellation -- this should not happen!")
        sys.exit(1)

    log.info("Model survived cancellation -- no reload needed")

    # --- STEP 4: Re-run diarization to completion ---
    log.info("")
    log.info("=" * 70)
    log.info("STEP 4: Re-running diarization (model should still be loaded)")
    log.info("=" * 70)

    start = time.time()
    try:
        turns = worker.diarize(vocal, cancel_event=None)
    except DiarizationCancelledError:
        log.error("Unexpected cancellation on re-run!")
        sys.exit(1)

    diarize_time = time.time() - start

    if not turns:
        log.error("Re-run diarization returned no turns!")
        sys.exit(1)

    log.info(f"Second diarization completed in {diarize_time:.1f}s")
    print_turns(turns, log)

    # --- Summary ---
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"  Model load: {model_load_time:.1f}s")
    log.info(f"  Cancelled + re-run: {diarize_time:.1f}s (no model reload)")
    log.info("")
    log.info("MODEL WAS STILL LOADED AFTER CANCELLATION -- NO RELOAD NEEDED")
    log.info("")
    log.info("Two-layer hook architecture allows cancellation in both")
    log.info("segmentation and embedding phases (PyTorch forward_pre_hooks),")
    log.info("with stage-boundary coverage via the pyannote hook callback.")
    log.info("Model weights survive as GPU/CPU attributes on the pipeline.")


def main():
    parser = argparse.ArgumentParser(
        description="Test cancelable pyannote diarization without losing the model."
    )
    parser.add_argument(
        "vocal_audio", help="Path to the vocal stem audio file (WAV, M4A, etc.)"
    )
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=5.0,
        help="Seconds to wait before cancelling diarization (default: 5.0)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Path to HuggingFace token file (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, or auto (overrides config default)",
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
        help="Minimum number of speakers (0 = no constraint)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        help="Maximum number of speakers (0 = no constraint)",
    )

    args = parser.parse_args()

    # Build config from defaults + CLI overrides
    config = DiarizeConfig()
    if args.hf_token:
        config.hf_token_path = args.hf_token
    if args.device is not None:
        config.device = args.device
    config.num_speakers = args.num_speakers
    config.min_speakers = args.min_speakers
    config.max_speakers = args.max_speakers

    setup_logging()
    run_test(args.vocal_audio, args.cancel_after, config)


if __name__ == "__main__":
    main()
