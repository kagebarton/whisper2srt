#!/usr/bin/env python3
"""Standalone test for cancelable stable-ts alignment.

This script proves that stable-ts model.align() and model.refine() can be
cancelled mid-computation without losing the loaded model, using a monkey-
patch on model.encode() that checks a threading.Event before each encoder
forward pass.

USAGE:
    python test_cancel_whisper.py <vocal_wav> <lyrics_txt> [--cancel-after SECONDS]

Example:
    # Cancel alignment after 5 seconds, then re-run to prove model survives
    python test_cancel_whisper.py vocals.wav lyrics.txt --cancel-after 5

    # Cancel sooner (if you have a fast GPU)
    python test_cancel_whisper.py vocals.wav lyrics.txt --cancel-after 2

    # Use a custom model path
    python test_cancel_whisper.py vocals.wav lyrics.txt --model-path /path/to/model

REQUIREMENTS:
    pip install stable-ts faster-whisper
    A faster-whisper model must be available at the configured path.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

# Add project root to path so cancel_whisper is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cancel_whisper.config import WhisperModelConfig
from cancel_whisper.workers.cancelable_whisper_worker import (
    AlignmentCancelledError,
    CancelableWhisperWorker,
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

    # Also set stable_whisper / faster_whisper loggers
    for name in ("stable_whisper", "faster_whisper", "cancel_whisper"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False

    return logging.getLogger("test_cancel_whisper")


def read_lyrics(lyrics_path: str) -> str:
    """Read lyrics from a text file."""
    with open(lyrics_path, "r", encoding="utf-8") as f:
        return f.read()


def run_test(
    vocal_path: str, lyrics_path: str, cancel_after: float, config: WhisperModelConfig
):
    """Run the cancellation proof-of-concept test.

    Steps:
    1. Load the whisper model (expensive, one-time)
    2. Run alignment with a cancel event that fires after cancel_after seconds
    3. Verify the model is still loaded after cancellation
    4. Re-run alignment to completion to prove the model survived
    """
    log = logging.getLogger("test_cancel_whisper")
    vocal = Path(vocal_path)
    lyrics_text = read_lyrics(lyrics_path)

    if not vocal.exists():
        log.error(f"Vocal file not found: {vocal}")
        sys.exit(1)

    # --- STEP 1: Load model ---
    log.info("=" * 70)
    log.info("STEP 1: Loading whisper model")
    log.info("=" * 70)
    worker = CancelableWhisperWorker(config)
    start = time.time()
    worker.load_model()
    model_load_time = time.time() - start
    log.info(f"Model loaded in {model_load_time:.1f}s")

    # --- STEP 2: Run alignment with cancellation ---
    log.info("")
    log.info("=" * 70)
    log.info(f"STEP 2: Running alignment (will cancel after {cancel_after}s)")
    log.info("=" * 70)

    cancel_event = threading.Event()

    # Schedule the cancel event to be set after cancel_after seconds
    cancel_timer = threading.Timer(cancel_after, cancel_event.set)
    cancel_timer.daemon = True

    start = time.time()
    cancel_timer.start()

    try:
        result = worker.align(vocal, lyrics_text, cancel_event=cancel_event)
        elapsed = time.time() - start

        if result is not None:
            log.warning(
                f"Alignment completed in {elapsed:.1f}s — "
                f"cancel timer didn't fire fast enough (try a larger --cancel-after). "
                f"Alignment finished before the {cancel_after}s cancel window."
            )
            log.info("Skipping cancel test — alignment was too fast to cancel.")
            log.info("")
            log.info("=" * 70)
            log.info(
                "STEP 3: Model still loaded (trivially true — alignment completed)"
            )
            log.info("=" * 70)
            log.info(f"Model is still loaded: {worker.model_loaded}")
            return

    except AlignmentCancelledError as e:
        elapsed = time.time() - start
        log.info(f"✓ Alignment cancelled after {elapsed:.1f}s (as expected)")
        log.info(f"  {e}")

    # --- STEP 3: Verify model survived ---
    log.info("")
    log.info("=" * 70)
    log.info("STEP 3: Verify model is still loaded after cancellation")
    log.info("=" * 70)
    log.info(f"Model is still loaded: {worker.model_loaded}")

    if not worker.model_loaded:
        log.error("✗ Model was lost after cancellation — this should not happen!")
        sys.exit(1)

    log.info("✓ Model survived cancellation — no reload needed")

    # --- STEP 4: Re-run alignment to completion ---
    log.info("")
    log.info("=" * 70)
    log.info("STEP 4: Re-running alignment (model should still be loaded)")
    log.info("=" * 70)

    # Use a fresh cancel event that never fires (no cancellation)
    start = time.time()
    try:
        result = worker.align(vocal, lyrics_text, cancel_event=None)
    except AlignmentCancelledError:
        log.error("✗ Unexpected cancellation on re-run!")
        sys.exit(1)

    align_time = time.time() - start

    if result is None:
        log.error("✗ Re-run alignment returned None!")
        sys.exit(1)

    log.info(f"✓ Second alignment completed in {align_time:.1f}s")
    log.info(f"  Segments: {len(result.segments)}")
    log.info(f"  Words: {len(result.all_words())}")

    # --- STEP 5: Refine with cancellation ---
    log.info("")
    log.info("=" * 70)
    log.info(f"STEP 5: Running refinement (will cancel after {cancel_after}s)")
    log.info("=" * 70)

    cancel_event2 = threading.Event()
    cancel_timer2 = threading.Timer(cancel_after, cancel_event2.set)
    cancel_timer2.daemon = True

    start = time.time()
    cancel_timer2.start()

    try:
        refined = worker.refine(vocal, result, cancel_event=cancel_event2)
        elapsed = time.time() - start

        if refined is not None:
            log.warning(
                f"Refinement completed in {elapsed:.1f}s — "
                f"cancel timer didn't fire fast enough. "
                f"Refine finished before the {cancel_after}s cancel window."
            )
        else:
            log.info("Refinement returned None (unexpected)")

    except AlignmentCancelledError as e:
        elapsed = time.time() - start
        log.info(f"✓ Refinement cancelled after {elapsed:.1f}s (as expected)")
        log.info(f"  {e}")

    # --- STEP 6: Re-run refinement to completion ---
    log.info("")
    log.info("=" * 70)
    log.info("STEP 6: Re-running refinement (model should still be loaded)")
    log.info("=" * 70)

    start = time.time()
    try:
        # Use a fresh result from a completed alignment for refinement
        fresh_result = worker.align(vocal, lyrics_text, cancel_event=None)
        refined = worker.refine(vocal, fresh_result, cancel_event=None)
    except AlignmentCancelledError:
        log.error("✗ Unexpected cancellation on re-run!")
        sys.exit(1)

    refine_time = time.time() - start

    if refined is None:
        log.error("✗ Re-run refinement returned None!")
        sys.exit(1)

    log.info(f"✓ Second alignment+refine completed in {refine_time:.1f}s")

    # --- Summary ---
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"  Model load: {model_load_time:.1f}s")
    log.info(f"  Cancelled align + re-run: {align_time:.1f}s (no model reload)")
    log.info(f"  Cancelled refine + re-run: {refine_time:.1f}s (no model reload)")
    log.info("")
    log.info("✓✓✓ MODEL WAS STILL LOADED AFTER CANCELLATION — NO RELOAD NEEDED ✓✓✓")
    log.info("")
    log.info("The monkey-patch on model.encode() allows cancellation between")
    log.info("encoder passes in both align() and refine(), while the CTranslate2")
    log.info("model weights survive as instance attributes on the WhisperModel.")


def main():
    parser = argparse.ArgumentParser(
        description="Test cancelable stable-ts alignment without losing the model."
    )
    parser.add_argument(
        "vocal_wav", help="Path to the vocal stem WAV file (from stem separation)"
    )
    parser.add_argument("lyrics_txt", help="Path to the confirmed lyrics text file")
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=5.0,
        help="Seconds to wait before cancelling alignment (default: 5.0)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to faster-whisper model folder (overrides config default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu', 'cuda', or 'auto' (overrides config default)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="Compute type: 'float16', 'int8', etc. (overrides config default)",
    )

    args = parser.parse_args()

    # Build config from defaults + CLI overrides
    config = WhisperModelConfig()
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.device is not None:
        config.device = args.device
    if args.compute_type is not None:
        config.compute_type = args.compute_type

    setup_logging()

    run_test(args.vocal_wav, args.lyrics_txt, args.cancel_after, config)


if __name__ == "__main__":
    main()
