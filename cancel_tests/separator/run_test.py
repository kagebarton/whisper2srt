#!/usr/bin/env python3
"""Cancel-then-rerun proof for the hook-based StemWorker prototype.

Usage:
    python -m cancel_tests.separator.run_test <input.wav> [--cancel-after SECONDS]

Demonstrates:
1. Start worker, load model in subprocess
2. Phase A: cancel separation mid-stream (after --cancel-after seconds)
   → expect WorkerCancelledError
3. Phase B: re-run separation without restarting the subprocess
   → proves model survived the cancellation
"""

import argparse
import logging
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path so cancel_tests is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cancel_tests.separator.stem_worker import StemWorker, WorkerCancelledError

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cancel_tests.separator.run_test")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hook-based StemWorker: cancel-then-rerun proof"
    )
    parser.add_argument(
        "input_wav",
        help="Path to input WAV file",
    )
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=3.0,
        help="Seconds to wait before cancelling (default: 3.0)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Directory containing audio-separator model files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Model filename (basename only) inside --model-dir",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="",
        help="Temporary directory for processing outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input_wav)
    if not input_path.exists():
        logger.error(f"File not found: {args.input_wav}")
        sys.exit(1)

    temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="hook_cancel_test_")
    logger.info(f"Temp dir: {temp_dir}")

    # --- Start worker ---
    worker_kwargs: dict = {
        "temp_dir": temp_dir,
        "log_level": logging.INFO,
    }
    if args.model_dir:
        worker_kwargs["model_dir"] = args.model_dir
    if args.model_name:
        worker_kwargs["model_name"] = args.model_name

    worker = StemWorker(**worker_kwargs)
    logger.info("Starting worker (model loading in subprocess)...")
    worker.start()

    # Wait for model to load
    time.sleep(5)
    if not worker.is_alive():
        logger.error("Worker failed to start — aborting")
        worker.kill()
        sys.exit(1)
    logger.info("Worker is alive — model loaded, ready for jobs")

    output_dir_a = Path(temp_dir) / "phase_a"
    output_dir_a.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Phase A: cancel test
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("PHASE A: Cancel separation mid-stream")
    logger.info(f"{'=' * 60}")

    cancel_event = threading.Event()

    # Schedule cancellation
    timer = threading.Timer(args.cancel_after, cancel_event.set)
    timer.daemon = True
    timer.start()

    start_time = time.time()
    try:
        vocal, instrumental = worker.separate(input_path, output_dir_a, cancel_event)
        elapsed = time.time() - start_time
        logger.warning(
            f"Separation completed in {elapsed:.1f}s before cancel took effect "
            f"— increase --cancel-after"
        )
        logger.info(f"  vocal: {vocal}")
        logger.info(f"  instrumental: {instrumental}")
    except WorkerCancelledError:
        elapsed = time.time() - start_time
        logger.info(
            f"WorkerCancelledError after {elapsed:.1f}s — "
            f"separation cancelled between chunks (model still loaded)"
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Unexpected error after {elapsed:.1f}s: {e}")
        worker.kill()
        sys.exit(1)
    finally:
        timer.cancel()

    # Verify worker is still alive
    if not worker.is_alive():
        logger.error(
            "Worker DIED after cancellation — model would need to be reloaded!"
        )
        worker.kill()
        sys.exit(1)
    logger.info("Worker is still alive — model survived cancellation")

    # Clean up Phase A output
    if output_dir_a.is_dir():
        for f in output_dir_a.iterdir():
            try:
                f.unlink()
            except OSError:
                pass

    # ================================================================
    # Phase B: prove model survived — re-run without cancellation
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("PHASE B: Re-run separation (should be fast — model already loaded)")
    logger.info(f"{'=' * 60}")

    output_dir_b = Path(temp_dir) / "phase_b"
    output_dir_b.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        vocal, instrumental = worker.separate(
            input_path, output_dir_b, cancel_event=None
        )
        elapsed = time.time() - start_time
        logger.info(f"Second separation completed in {elapsed:.1f}s")
        logger.info(f"  vocal: {vocal}")
        logger.info(f"  instrumental: {instrumental}")
        logger.info("MODEL WAS STILL LOADED — no subprocess restart needed!")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Second separation failed after {elapsed:.1f}s: {e}")
        worker.kill()
        sys.exit(1)

    # --- Cleanup ---
    worker.stop()
    if not args.temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("\n=== HOOK-BASED CANCEL TEST COMPLETE ===")


if __name__ == "__main__":
    main()
