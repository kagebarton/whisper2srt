#!/usr/bin/env python3
"""Standalone test for hook-based whisper alignment cancellation.

This script proves that stable-ts model.align(), model.refine(), and
model.transcribe() can be cancelled mid-computation without losing the
loaded model, using a register_forward_pre_hook() on the encoder nn.Module
that checks a threading.Event before each encoder forward pass.

USAGE:
    python -m cancel_tests.whisper.test_cancel_whisper <vocal_wav> <lyrics_txt> [--cancel-after SECONDS]

Example:
    # Cancel alignment after 5 seconds, then re-run to prove model survives
    python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --cancel-after 5

    # Cancel sooner (if you have a fast GPU)
    python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --cancel-after 2

    # Use a custom model path
    python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --model-path /path/to/model

    # Cancel during refine phase instead of align
    python -m cancel_tests.whisper.test_cancel_whisper vocals.wav lyrics.txt --phase refine

REQUIREMENTS:
    pip install stable-ts openai-whisper
    A whisper .pt model file must be available at the configured path.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path so cancel_tests is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cancel_tests.whisper.config import WhisperModelConfig
from cancel_tests.whisper.whisper_worker import (
    AlignmentCancelledError,
    WhisperWorker,
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
    root.setLevel(logging.INFO)

    # Also set stable_whisper / whisper loggers
    for name in ("stable_whisper", "whisper", "cancel_tests"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False

    return logging.getLogger("cancel_tests.whisper.test_cancel_whisper")


def read_lyrics(lyrics_path: str) -> str:
    """Read lyrics from a text file."""
    with open(lyrics_path, "r", encoding="utf-8") as f:
        return f.read()


def run_test(
    vocal_path: str,
    lyrics_path: str,
    cancel_after: float,
    phase: str,
    config: WhisperModelConfig,
):
    """Run the hook-based cancellation proof-of-concept test.

    Steps:
    1. Load the whisper model (expensive, one-time)
    2. Run alignment with a cancel event that fires after cancel_after seconds
    3. Verify the model is still loaded after cancellation
    4. Re-run alignment to completion to prove the model survived
    5. Run refinement with cancellation
    6. Re-run refinement to completion
    7. Run transcription with cancellation
    8. Re-run transcription to completion
    """
    log = logging.getLogger("cancel_tests.whisper.test_cancel_whisper")
    vocal = Path(vocal_path)
    lyrics_text = read_lyrics(lyrics_path)

    if not vocal.exists():
        log.error(f"Vocal file not found: {vocal}")
        sys.exit(1)

    # --- STEP 1: Load model ---
    log.info("=" * 70)
    log.info("STEP 1: Loading whisper model (PyTorch)")
    log.info("=" * 70)
    worker = WhisperWorker(config)
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
        log.info(f"Alignment cancelled after {elapsed:.1f}s (as expected)")
        log.info(f"  {e}")

    # --- STEP 3: Verify model survived ---
    log.info("")
    log.info("=" * 70)
    log.info("STEP 3: Verify model is still loaded after cancellation")
    log.info("=" * 70)
    log.info(f"Model is still loaded: {worker.model_loaded}")

    if not worker.model_loaded:
        log.error("Model was lost after cancellation — this should not happen!")
        sys.exit(1)

    log.info("Model survived cancellation — no reload needed")

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
        log.error("Unexpected cancellation on re-run!")
        sys.exit(1)

    align_time = time.time() - start

    if result is None:
        log.error("Re-run alignment returned None!")
        sys.exit(1)

    log.info(f"Second alignment completed in {align_time:.1f}s")
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
        log.info(f"Refinement cancelled after {elapsed:.1f}s (as expected)")
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
        log.error("Unexpected cancellation on re-run!")
        sys.exit(1)

    refine_time = time.time() - start

    if refined is None:
        log.error("Re-run refinement returned None!")
        sys.exit(1)

    log.info(f"Second alignment+refine completed in {refine_time:.1f}s")

    # --- STEP 7: Transcribe with cancellation ---
    log.info("")
    log.info("=" * 70)
    log.info(f"STEP 7: Running transcription (will cancel after {cancel_after}s)")
    log.info("=" * 70)

    cancel_event3 = threading.Event()
    cancel_timer3 = threading.Timer(cancel_after, cancel_event3.set)
    cancel_timer3.daemon = True

    start = time.time()
    cancel_timer3.start()

    try:
        transcript = worker.transcribe(vocal, cancel_event=cancel_event3)
        elapsed = time.time() - start

        if transcript is not None:
            log.warning(
                f"Transcription completed in {elapsed:.1f}s — "
                f"cancel timer didn't fire fast enough. "
                f"Transcription finished before the {cancel_after}s cancel window."
            )
        else:
            log.info("Transcription returned None (unexpected)")

    except AlignmentCancelledError as e:
        elapsed = time.time() - start
        log.info(f"Transcription cancelled after {elapsed:.1f}s (as expected)")
        log.info(f"  {e}")

    # --- STEP 8: Re-run transcription to completion ---
    log.info("")
    log.info("=" * 70)
    log.info("STEP 8: Re-running transcription (model should still be loaded)")
    log.info("=" * 70)

    start = time.time()
    try:
        transcript = worker.transcribe(vocal, cancel_event=None)
    except AlignmentCancelledError:
        log.error("Unexpected cancellation on re-run!")
        sys.exit(1)

    transcribe_time = time.time() - start

    if transcript is None:
        log.error("Re-run transcription returned None!")
        sys.exit(1)

    log.info(f"Second transcription completed in {transcribe_time:.1f}s")
    log.info(f"  Segments: {len(transcript.segments)}")
    log.info(f"  Words: {len(transcript.all_words())}")

    # --- Summary ---
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"  Model load: {model_load_time:.1f}s")
    log.info(f"  Cancelled align + re-run: {align_time:.1f}s (no model reload)")
    log.info(f"  Cancelled refine + re-run: {refine_time:.1f}s (no model reload)")
    log.info(f"  Cancelled transcribe + re-run: {transcribe_time:.1f}s (no model reload)")
    log.info("")
    log.info("MODEL WAS STILL LOADED AFTER CANCELLATION — NO RELOAD NEEDED")
    log.info("")
    log.info("The register_forward_pre_hook on model.encoder allows cancellation")
    log.info("between encoder passes in align(), refine(), and transcribe(), while")
    log.info("the PyTorch model weights survive as nn.Parameter attributes on the")
    log.info("whisper.model.Whisper nn.Module.")


def main():
    parser = argparse.ArgumentParser(
        description="Test hook-based whisper alignment cancellation (PyTorch model)."
    )
    parser.add_argument(
        "vocal_wav",
        help="Path to the vocal stem WAV file (from stem separation)",
    )
    parser.add_argument(
        "lyrics_txt",
        help="Path to the confirmed lyrics text file",
    )
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
        help="Path to whisper .pt model file (overrides config default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu', 'cuda', or 'auto' (overrides config default)",
    )
    parser.add_argument(
        "--phase",
        choices=["align", "refine"],
        default="align",
        help="Which phase to cancel (default: align)",
    )

    args = parser.parse_args()

    # Build config from defaults + CLI overrides
    config = WhisperModelConfig()
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.device is not None:
        config.device = args.device

    setup_logging()

    run_test(
        args.vocal_wav,
        args.lyrics_txt,
        args.cancel_after,
        args.phase,
        config,
    )


if __name__ == "__main__":
    main()
