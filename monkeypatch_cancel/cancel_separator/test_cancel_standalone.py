#!/usr/bin/env python3
"""Minimal standalone test: proves mid-separation cancellation preserves the loaded model.

This script is self-contained — it only requires audio-separator to be installed.
It does NOT import from the cancel-test package or from pikaraoke.

Usage:
    python test_cancel_standalone.py <audio_file> [--cancel-after SECONDS]

What it does:
1. Loads the audio-separator model (takes ~10-30s depending on hardware)
2. Runs separation on the audio file
3. After --cancel-after seconds, injects a cancel check into the model
4. The cancel check aborts separation between chunks (not mid-chunk)
5. Verifies the model is still usable by running a second separation

The injection mechanism: monkey-patches model_instance.model_run.forward()
to check a threading.Event before each forward pass. Since the Roformer
demix() loop calls self.model_run(part.unsqueeze(0))[0] once per chunk,
our check runs between chunks — the model weights are untouched.
"""

import argparse
import logging
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

# --- Configuration (same as production pikaraoke) ---
MODEL_NAME = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent.parent.parent / "models")
SEPARATION_FORMAT = "wav"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("standalone_test")


class CancelledBetweenChunks(Exception):
    """Raised by the patched forward() when cancellation is detected."""


def test_cancellation(audio_path: str, cancel_after: float, model_dir: str) -> None:
    """Main test: load model, cancel mid-separation, prove model survives."""

    from audio_separator.separator import Separator

    # --- 1. Load model ---
    logger.info("=" * 60)
    logger.info("STEP 1: Loading audio-separator model")
    logger.info("=" * 60)
    load_start = time.time()

    separator = Separator(
        model_file_dir=model_dir,
        output_format=SEPARATION_FORMAT,
    )
    separator.load_model(model_filename=MODEL_NAME)

    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.1f}s")
    logger.info(f"Model instance: {type(separator.model_instance).__name__}")
    logger.info(f"Model run obj:   {type(separator.model_instance.model_run).__name__}")

    # --- 2. First separation (cancelled) ---
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 2: Running separation (will cancel after {cancel_after}s)")
    logger.info("=" * 60)

    tmp_dir1 = Path(tempfile.mkdtemp(prefix="cancel_test_1_"))
    separator.output_dir = str(tmp_dir1)
    separator.model_instance.output_dir = str(tmp_dir1)

    cancel_event = threading.Event()
    chunk_count = [0]

    # --- Monkey-patch forward() ---
    model_run = separator.model_instance.model_run
    original_forward = model_run.forward

    def cancelable_forward(*args, **kwargs):
        if cancel_event.is_set():
            logger.info(
                f"Cancel detected before chunk #{chunk_count[0] + 1} — aborting!"
            )
            raise CancelledBetweenChunks()
        result = original_forward(*args, **kwargs)
        chunk_count[0] += 1
        return result

    model_run.forward = cancelable_forward
    logger.info("Patched model_run.forward() with cancel check")

    # Start separation in a background thread
    sep_exc = [None]

    def run_sep():
        try:
            separator.separate(str(audio_path))
        except CancelledBetweenChunks as e:
            sep_exc[0] = e
        except Exception as e:
            sep_exc[0] = e

    sep_thread = threading.Thread(target=run_sep, daemon=True)
    sep_thread.start()

    # Wait then cancel
    logger.info(f"Waiting {cancel_after}s before cancelling...")
    time.sleep(cancel_after)

    logger.info(">>> SETTING CANCEL EVENT <<<")
    cancel_event.set()

    sep_thread.join(timeout=300)  # max 5 min for the current chunk to finish

    # Restore original forward()
    model_run.forward = original_forward
    logger.info(
        f"Restored original forward() — processed {chunk_count[0]} chunks before cancel"
    )

    if isinstance(sep_exc[0], CancelledBetweenChunks):
        logger.info("✓ Separation cancelled between chunks (as expected)")
    elif sep_exc[0] is not None:
        logger.error(f"✗ Unexpected error: {sep_exc[0]}")
        shutil.rmtree(tmp_dir1, ignore_errors=True)
        return
    else:
        logger.info("Separation completed before cancel (increase --cancel-after)")
        shutil.rmtree(tmp_dir1, ignore_errors=True)
        return

    # Clean up GPU state
    try:
        separator.model_instance.clear_gpu_cache()
        separator.model_instance.clear_file_specific_paths()
    except Exception as e:
        logger.warning(f"GPU cleanup warning: {e}")

    logger.info(f"Model is still loaded: {separator.model_instance is not None}")
    logger.info(f"Model run is still on device: {next(model_run.parameters()).device}")

    # --- 3. Second separation (proves model survives) ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Re-running separation (model should still be loaded)")
    logger.info("=" * 60)

    tmp_dir2 = Path(tempfile.mkdtemp(prefix="cancel_test_2_"))
    separator.output_dir = str(tmp_dir2)
    separator.model_instance.output_dir = str(tmp_dir2)

    sep2_start = time.time()
    try:
        output_paths = separator.separate(str(audio_path))
        sep2_time = time.time() - sep2_start
        logger.info(f"✓ Second separation completed in {sep2_time:.1f}s")
        logger.info(f"  Output files: {[Path(p).name for p in output_paths]}")
        logger.info("")
        logger.info("✓✓✓ MODEL WAS STILL LOADED — NO RELOAD NEEDED ✓✓✓")
    except Exception as e:
        sep2_time = time.time() - sep2_start
        logger.error(f"✗ Second separation failed after {sep2_time:.1f}s: {e}")

    # Clean up
    del separator
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    shutil.rmtree(tmp_dir1, ignore_errors=True)
    shutil.rmtree(tmp_dir2, ignore_errors=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone test: mid-separation cancellation preserves loaded model"
    )
    parser.add_argument("audio_file", help="Path to audio/video file")
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=5.0,
        help="Seconds before cancelling (default: 5)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing audio-separator model files (default: ./audio-separator/models)",
    )
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        logger.error(f"File not found: {args.audio_file}")
        sys.exit(1)

    test_cancellation(args.audio_file, args.cancel_after, args.model_dir)


if __name__ == "__main__":
    main()
