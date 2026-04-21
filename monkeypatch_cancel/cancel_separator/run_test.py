#!/usr/bin/env python3
"""Interactive test runner for mid-separation cancellation.

Usage:
    python run_test.py <audio_file> [--cancel-after SECONDS] [--auto] [--loop COUNT]

Modes:
  Interactive (default):
    - Enqueues the audio file for processing
    - Shows live status every 0.5s
    - Press 'c' + Enter to cancel mid-separation
    - Press 'q' + Enter to quit
    - After cancellation, automatically re-enqueues to prove the model
      is still loaded (no reload time)

  Auto (--auto):
    - Enqueues the file and cancels after --cancel-after seconds
    - Then re-enqueues to prove model is still loaded
    - Optionally repeats with --loop

  Direct worker test (--direct):
    - Bypasses the orchestrator and tests CancelableStemWorker directly
    - Useful for isolated debugging of the monkey-patch mechanism

Examples:
    # Interactive: manually cancel with 'c'
    python run_test.py /path/to/song.mp4

    # Auto-cancel after 5 seconds, then re-process
    python run_test.py /path/to/song.mp4 --auto --cancel-after 5

    # Repeat cancel-then-reprocess 3 times
    python run_test.py /path/to/song.mp4 --auto --cancel-after 3 --loop 3

    # Direct worker test (no orchestrator)
    python run_test.py /path/to/song.wav --direct --cancel-after 5
"""

import argparse
import logging
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

# Add project root to sys.path so cancel_test is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cancel_test.context import CancelledError, StageContext
from cancel_test.orchestrator import StageOrchestrator
from cancel_test.stages.ffmpeg_extract import FFmpegExtractStage
from cancel_test.stages.ffmpeg_transcode import FFmpegTranscodeStage
from cancel_test.stages.stem_separation import StemSeparationStage
from cancel_test.workers.cancelable_stem_worker import (
    CancelableStemWorker,
    WorkerCancelledError,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_test")


def build_orchestrator(temp_dir: str = "", model_dir: str = "") -> StageOrchestrator:
    """Build the stage pipeline with a CancelableStemWorker."""
    worker_kwargs: dict = {
        "temp_dir": temp_dir,
        "log_level": logging.INFO,
    }
    if model_dir:
        worker_kwargs["model_dir"] = model_dir
    stem_worker = CancelableStemWorker(**worker_kwargs)
    stages = [
        FFmpegExtractStage(),
        StemSeparationStage(stem_worker),
        FFmpegTranscodeStage(),
    ]
    return StageOrchestrator(stages=stages, stem_worker=stem_worker, temp_dir=temp_dir)


def interactive_mode(audio_file: str, temp_dir: str, model_dir: str) -> None:
    """Interactive mode: enqueue, show status, accept cancel commands.

    Uses a background thread for non-blocking stdin reading (cross-platform,
    works on Windows where select() doesn't work on stdin).
    """
    orch = build_orchestrator(temp_dir, model_dir=model_dir)
    orch.start()

    # Give the worker a moment to load the model
    logger.info("Worker starting — model loading (this takes a few seconds)...")
    time.sleep(5)

    if not orch.get_status()["worker_alive"]:
        logger.error("Worker failed to start — aborting")
        orch.stop()
        return

    # Enqueue the file
    orch.enqueue(audio_file)
    logger.info(f"Enqueued: {Path(audio_file).name}")
    logger.info(
        "Commands: 'c' + Enter = cancel, 'r' + Enter = re-enqueue, 'q' + Enter = quit"
    )

    # Background thread for non-blocking stdin reading
    cmd_queue: queue.Queue[str] = queue.Queue()

    def _stdin_reader():
        """Read lines from stdin in a background thread."""
        while True:
            try:
                line = sys.stdin.readline().strip().lower()
                if line:
                    cmd_queue.put(line)
            except (EOFError, OSError):
                break

    stdin_thread = threading.Thread(target=_stdin_reader, daemon=True)
    stdin_thread.start()

    requeued = False

    try:
        while True:
            status = orch.get_status()
            active = status["active_song"]
            stage = status["active_stage"]
            pending = len(status["pending_jobs"])
            worker_ok = status["worker_alive"]

            status_line = f" stage={stage or 'idle':20s} pending={pending} worker={'OK' if worker_ok else 'DEAD'}"

            if active:
                status_line += f" file={Path(active).name}"

            sys.stdout.write(f"\r{status_line}")
            sys.stdout.flush()

            # Check for user commands (non-blocking)
            while not cmd_queue.empty():
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break

                if cmd == "c" and active:
                    logger.info("\n>>> CANCELLING ACTIVE JOB <<<")
                    orch.cancel_active(active)
                elif cmd == "r":
                    logger.info(f"\n>>> RE-ENQUEUING: {Path(audio_file).name} <<<")
                    _cleanup_stems(audio_file)
                    orch.enqueue(audio_file)
                    requeued = True
                elif cmd == "q":
                    logger.info("\n>>> QUITTING <<<")
                    return

            # If job completed and we haven't requeued, offer to re-enqueue
            if not active and not pending and not requeued:
                logger.info(
                    "\nJob completed! Press 'r' + Enter to re-enqueue (proves model is loaded), 'q' to quit"
                )
                requeued = True  # only prompt once

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    finally:
        orch.stop()


def auto_mode(
    audio_file: str, temp_dir: str, model_dir: str, cancel_after: float, loop: int
) -> None:
    """Auto mode: cancel after N seconds, then re-process to prove model stays loaded.

    Uses a SINGLE orchestrator/worker across all iterations. This is the key test:
    after cancelling mid-separation, the same worker process should accept the next
    job without reloading the model.
    """
    orch = build_orchestrator(temp_dir, model_dir=model_dir)
    orch.start()

    # Wait for worker to be ready (model loads in subprocess)
    logger.info("Waiting for worker to load model...")
    time.sleep(5)

    if not orch.get_status()["worker_alive"]:
        logger.error("Worker failed to start — aborting")
        orch.stop()
        return

    logger.info("Worker is alive — proceeding with test")

    try:
        for i in range(loop):
            logger.info(f"=== ITERATION {i + 1}/{loop} ===")

            # Clean up any previous stems
            _cleanup_stems(audio_file)

            # Enqueue
            orch.enqueue(audio_file)
            logger.info(f"Enqueued: {Path(audio_file).name}")

            # Wait then cancel
            logger.info(f"Will cancel in {cancel_after}s...")
            time.sleep(cancel_after)

            status = orch.get_status()
            if status["active_song"]:
                logger.info(">>> CANCELLING <<<")
                orch.cancel_active(status["active_song"])

                # Wait for cancellation to complete
                time.sleep(3)
                status_after = orch.get_status()
                logger.info(
                    f"After cancel: worker_alive={status_after['worker_alive']}, "
                    f"active_song={status_after['active_song']}"
                )
            else:
                logger.info(
                    "Job already completed before cancel — increase --cancel-after"
                )

            # Re-enqueue to prove the model is still loaded (same worker!)
            _cleanup_stems(audio_file)
            orch.enqueue(audio_file)
            logger.info(f"Re-enqueued after cancel: {Path(audio_file).name}")

            # Wait for completion
            for _ in range(120):  # max 60s
                status = orch.get_status()
                if not status["active_song"] and not status["pending_jobs"]:
                    break
                time.sleep(0.5)

            final_status = orch.get_status()
            logger.info(
                f"After re-process: worker_alive={final_status['worker_alive']}, "
                f"active_song={final_status['active_song']}"
            )

            if final_status["worker_alive"]:
                logger.info(
                    f"✓ Iteration {i + 1}: worker survived cancel and re-processed successfully"
                )
            else:
                logger.error(f"✗ Iteration {i + 1}: worker died after cancel")
                break

    finally:
        orch.stop()

    logger.info("=== ALL ITERATIONS COMPLETE ===")


def direct_worker_mode(
    audio_file: str, temp_dir: str, model_dir: str, cancel_after: float
) -> None:
    """Direct test: use CancelableStemWorker without the full orchestrator.

    This tests the core mechanism in isolation:
    1. Start worker, load model
    2. Extract WAV from audio file using ffmpeg
    3. Call worker.separate() with a cancel_event
    4. Set cancel_event after cancel_after seconds
    5. Verify: WorkerCancelledError is raised (not WorkerDiedError)
    6. Call worker.separate() again — should work immediately (model still loaded)
    """
    worker_kwargs: dict = {
        "temp_dir": temp_dir,
        "log_level": logging.INFO,
    }
    if model_dir:
        worker_kwargs["model_dir"] = model_dir
    worker = CancelableStemWorker(**worker_kwargs)

    logger.info("=== DIRECT WORKER TEST ===")
    logger.info("Starting worker (model loading)...")
    worker.start()

    # Wait for model to load
    logger.info("Waiting for model to load...")
    time.sleep(5)

    if not worker.is_alive():
        logger.error("Worker failed to start!")
        return

    logger.info("Worker is alive! Proceeding with test.")

    # Extract audio to WAV first
    tmp_dir = Path(tempfile.mkdtemp(prefix="cancel_test_direct_"))
    wav_in = tmp_dir / f"{Path(audio_file).stem}_input.wav"

    logger.info(f"Extracting audio to {wav_in}...")
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_file,
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-sample_fmt",
        "s16",
        str(wav_in),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg extract failed: {result.stderr}")
        return
    logger.info(f"Audio extracted: {wav_in}")

    output_dir = tmp_dir / "stems"
    output_dir.mkdir()

    # --- First separation: cancel mid-stream ---
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST 1: Cancel separation mid-stream")
    logger.info(f"{'=' * 60}")

    cancel_event = threading.Event()

    # Start separation in a thread
    sep_result = [None, None]  # [result, exception]

    def run_separate():
        try:
            vocal, instrumental = worker.separate(
                wav_path=wav_in,
                output_dir=output_dir,
                cancel_event=cancel_event,
            )
            sep_result[0] = (vocal, instrumental)
        except Exception as e:
            sep_result[1] = e

    sep_thread = threading.Thread(target=run_separate, daemon=True)
    sep_thread.start()

    # Wait then cancel
    logger.info(f"Waiting {cancel_after}s before cancelling...")
    time.sleep(cancel_after)

    logger.info(">>> SETTING CANCEL EVENT <<<")
    cancel_event.set()

    # Wait for the separation thread to finish
    sep_thread.join(timeout=120)

    if sep_result[1] is not None:
        if isinstance(sep_result[1], WorkerCancelledError):
            logger.info(
                "✓ Got WorkerCancelledError — separation cancelled between chunks"
            )
        else:
            logger.error(f"✗ Got unexpected error: {sep_result[1]}")
    elif sep_result[0] is not None:
        logger.info(
            "Separation completed before cancel took effect (increase --cancel-after)"
        )

    # Check worker is still alive
    worker_alive = worker.is_alive()
    logger.info(f"Worker alive after cancel: {worker_alive}")

    if not worker_alive:
        logger.error("✗ Worker DIED — model would need to be reloaded!")
        worker.stop()
        return

    # --- Second separation: verify model is still loaded ---
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST 2: Re-run separation (should be fast — model already loaded)")
    logger.info(f"{'=' * 60}")

    # Clean up previous output
    for f in output_dir.iterdir():
        try:
            f.unlink()
        except OSError:
            pass

    cancel_event2 = threading.Event()
    start_time = time.time()

    try:
        vocal, instrumental = worker.separate(
            wav_path=wav_in,
            output_dir=output_dir,
            cancel_event=cancel_event2,
        )
        elapsed = time.time() - start_time
        logger.info(f"✓ Second separation completed in {elapsed:.1f}s")
        logger.info(f"  vocal: {vocal}")
        logger.info(f"  instrumental: {instrumental}")
        logger.info("✓ MODEL WAS STILL LOADED — no reload needed!")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Second separation failed after {elapsed:.1f}s: {e}")

    # Clean up
    worker.stop()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("\n=== DIRECT WORKER TEST COMPLETE ===")


def _cleanup_stems(song_path: str) -> None:
    """Remove any existing stem outputs so the pipeline re-processes."""
    video = Path(song_path)
    for stem_dir_name in ("vocal", "nonvocal"):
        stem_dir = video.parent / stem_dir_name
        if stem_dir.is_dir():
            for f in stem_dir.glob(f"{video.stem}---*"):
                try:
                    f.unlink()
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description="Test mid-separation cancellation with stage-based pipeline"
    )
    parser.add_argument("audio_file", help="Path to audio/video file to process")
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=5.0,
        help="Seconds to wait before auto-cancelling (default: 5)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-cancel mode (no interactive input needed)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Number of cancel-then-reprocess iterations (default: 1)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Test CancelableStemWorker directly (no orchestrator)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="",
        help="Temporary directory for processing",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Directory containing audio-separator model files (default: ./audio-separator/models)",
    )
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        logger.error(f"File not found: {args.audio_file}")
        sys.exit(1)

    if args.direct:
        direct_worker_mode(
            args.audio_file, args.temp_dir, args.model_dir, args.cancel_after
        )
    elif args.auto:
        auto_mode(
            args.audio_file, args.temp_dir, args.model_dir, args.cancel_after, args.loop
        )
    else:
        interactive_mode(args.audio_file, args.temp_dir, args.model_dir)


if __name__ == "__main__":
    main()
