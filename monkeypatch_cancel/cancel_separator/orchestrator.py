"""Stage orchestrator: coordinates pipeline stages with cancellation.

From the A1 architecture: the orchestrator owns the worker lifecycle
and runs stages sequentially, checking for cancellation between stages.
If a stage is cancelled, the orchestrator skips remaining stages and
emits a cancellation event.

Key design points:
- Workers (StemWorker) are owned by the orchestrator, NOT by stages.
  Stages hold a reference but never create/destroy workers.
- Cancel semantics differ by stage:
  - FFmpeg stages: immediate (kill Popen)
  - Stem stage: delayed (set flag, let current chunk finish)
- After cancellation, the orchestrator verifies the worker is alive.
  If cancel killed it (shouldn't happen with delayed cancel), restart it.
"""

import logging
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Sequence

from cancel_test.context import CancelledError, StageContext
from cancel_test.stages.base import PipelineStage
from cancel_test.stages.stem_separation import StemSeparationStage
from cancel_test.workers.cancelable_stem_worker import CancelableStemWorker

logger = logging.getLogger(__name__)


class StageOrchestrator:
    """Orchestrates pipeline stages with surgical cancellation.

    Runs a real-thread orchestrator loop that:
    1. Pulls songs from a pending queue
    2. Creates a per-job StageContext
    3. Runs stages sequentially, passing context forward
    4. Checks cancel flag between stages
    5. On cancel: calls stage.cancel(), cleans up, and moves to next job

    Worker lifecycle is managed here, not in stages. The StemWorker
    is started once and stays alive across jobs.
    """

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        stem_worker: CancelableStemWorker,
        temp_dir: str = "",
    ) -> None:
        self._stages = list(stages)
        self._stem_worker = stem_worker
        self._temp_dir = temp_dir
        self._pending_queue: queue.Queue[str | None] = queue.Queue()
        self.pending_jobs: list[str] = []
        self._state_lock = threading.Lock()
        self._active_song: str | None = None
        self._active_stage: PipelineStage | None = None
        self._active_ctx: StageContext | None = None
        self._stop_event = threading.Event()
        self._orchestrator_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the stem worker and orchestrator thread."""
        self._stem_worker.start()
        self._orchestrator_thread = threading.Thread(
            target=self._orchestrator_loop, daemon=True
        )
        self._orchestrator_thread.start()
        logger.info("Stage orchestrator started")

    def stop(self) -> None:
        """Shut down orchestrator thread and stem worker gracefully."""
        self._stop_event.set()
        try:
            self._pending_queue.put(None)  # shutdown sentinel
        except Exception:
            pass
        if (
            self._orchestrator_thread is not None
            and self._orchestrator_thread.is_alive()
        ):
            self._orchestrator_thread.join(timeout=10)
        self._stem_worker.stop()
        logger.info("Stage orchestrator stopped")

    def enqueue(self, song_path: str) -> None:
        """Add a song to the processing queue."""
        with self._state_lock:
            self.pending_jobs.append(song_path)
        self._pending_queue.put(song_path)
        logger.info(f"Queued: {Path(song_path).name}")

    def cancel_active(self, song_path: str) -> None:
        """Cancel the currently active job.

        For the stem separation stage, this sets the cancel flag but
        does NOT kill the worker — the model stays loaded. The worker's
        patched forward() will detect the flag between chunks and abort.
        """
        with self._state_lock:
            if self._active_song != song_path:
                logger.warning(
                    f"Cancel requested for non-active job: {Path(song_path).name}"
                )
                return
            stage = self._active_stage
            ctx = self._active_ctx

        if stage is not None and ctx is not None:
            logger.info(
                f"Cancelling active stage '{stage.name}': {Path(song_path).name}"
            )
            stage.cancel(ctx)
        else:
            logger.warning(f"No active stage to cancel for: {Path(song_path).name}")

    # -- Orchestrator loop --------------------------------------------------

    def _orchestrator_loop(self) -> None:
        while not self._stop_event.is_set():
            song_path = self._pending_queue.get()  # blocks until item or sentinel
            if song_path is None:
                return  # shutdown sentinel

            with self._state_lock:
                self._active_song = song_path

            try:
                self._run_pipeline(song_path)
                logger.info(f"Pipeline complete: {Path(song_path).name}")
            except CancelledError:
                logger.info(f"Pipeline cancelled: {Path(song_path).name}")
            except Exception as e:
                logger.error(f"Pipeline failed for {Path(song_path).name}: {e}")
            finally:
                with self._state_lock:
                    if song_path in self.pending_jobs:
                        self.pending_jobs.remove(song_path)
                    self._active_song = None
                    self._active_stage = None
                    self._active_ctx = None
                # Eager restart: if the worker somehow died, bring it back
                if not self._stem_worker.is_alive():
                    try:
                        logger.warning("Stem worker died — restarting")
                        self._stem_worker.start()
                    except Exception as e:
                        logger.error(f"Failed to restart stem worker: {e}")

    def _run_pipeline(self, song_path: str) -> None:
        """Run all stages sequentially, checking for cancellation between stages."""
        video = Path(song_path)
        if not video.exists():
            raise FileNotFoundError(f"Song file not found: {song_path}")

        tmp_dir_parent = self._temp_dir if self._temp_dir else None
        tmp_dir = Path(tempfile.mkdtemp(prefix="cancel_test_", dir=tmp_dir_parent))

        ctx = StageContext(
            song_path=video,
            tmp_dir=tmp_dir,
        )

        with self._state_lock:
            self._active_ctx = ctx

        try:
            for stage in self._stages:
                with self._state_lock:
                    self._active_stage = stage

                logger.info(f"[pipeline] Running stage: {stage.name}")
                stage.run(ctx)
                logger.info(f"[pipeline] Stage complete: {stage.name}")

                # Check cancellation between stages
                ctx.check_cancelled()
        finally:
            logger.debug(f"[pipeline] Cleanup tmp: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- Status API ---------------------------------------------------------

    def get_status(self) -> dict:
        """Return current orchestrator status (for display/logging)."""
        with self._state_lock:
            return {
                "active_song": self._active_song,
                "active_stage": self._active_stage.name if self._active_stage else None,
                "pending_jobs": list(self.pending_jobs),
                "worker_alive": self._stem_worker.is_alive(),
            }
