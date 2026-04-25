"""Pipeline orchestrator with cancellation support.

Owns both worker lifecycles (StemWorker, WhisperWorker), creates and
cleans up the temp directory, and runs stages sequentially.  Supports
both synchronous one-shot execution (``run_one``) and asynchronous
execution with cancellation (``run_one_async`` + ``join``).

Each ``run_one_async()`` call creates a **fresh** CancelToken with a
**fresh** threading.Event — never reused across jobs.  This isolates
jobs from each other, preventing stale cancel-forwarder daemon threads
from injecting spurious signals into subsequent jobs.
"""

import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Sequence

from pipeline.config import PipelineConfig
from pipeline.context import CancelToken, PipelineCancelled, StageContext
from pipeline.stages.base import PipelineStage
from pipeline.workers.stem_worker import StemWorker
from pipeline.workers.whisper_worker import WhisperWorker

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        stages: Sequence[PipelineStage],
        stem_worker: StemWorker,
        whisper_worker: WhisperWorker,
        config: PipelineConfig,
    ) -> None:
        self._stages = list(stages)
        self._stem_worker = stem_worker
        self._whisper_worker = whisper_worker
        self._config = config
        self._workers_started = False

        # Async state — set by run_one_async, read by join/cancel_active
        self._cancel_token: Optional[CancelToken] = None
        self._pipeline_thread: Optional[threading.Thread] = None
        self._result: dict[str, object] = {}  # {"ctx": ..., "exception": ...}

    # ------------------------------------------------------------------
    # Worker lifecycle (idempotent)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn workers (idempotent).  Driver calls this once before any run."""
        if self._workers_started:
            return
        logger.info("Starting stem worker...")
        self._stem_worker.start()
        logger.info("Loading whisper model...")
        self._whisper_worker.load_model()
        self._workers_started = True

    def stop(self) -> None:
        """Tear down workers (idempotent).  Driver calls this once at end."""
        if not self._workers_started:
            return
        logger.info("Stopping stem worker...")
        try:
            self._stem_worker.stop()
        except Exception as e:
            logger.warning(f"stem_worker.stop() failed: {e}")
        logger.info("Unloading whisper model...")
        try:
            self._whisper_worker.unload_model()
        except Exception as e:
            logger.warning(f"whisper_worker.unload_model() failed: {e}")
        self._workers_started = False

    # ------------------------------------------------------------------
    # Read-only access for the test driver
    # ------------------------------------------------------------------

    @property
    def stem_worker(self) -> StemWorker:
        return self._stem_worker

    @property
    def whisper_worker(self) -> WhisperWorker:
        return self._whisper_worker

    # ------------------------------------------------------------------
    # Synchronous execution
    # ------------------------------------------------------------------

    def run(self, song_path: Path, lyrics_path: Optional[Path] = None) -> StageContext:
        """Back-compat: start workers, run one song, stop workers."""
        self.start()
        try:
            return self.run_one(song_path, lyrics_path)
        finally:
            self.stop()

    def run_one(self, song_path: Path, lyrics_path: Optional[Path] = None) -> StageContext:
        """Run one song synchronously.  Workers must already be started."""
        token = self.run_one_async(song_path, lyrics_path)
        return self.join()

    # ------------------------------------------------------------------
    # Asynchronous execution + cancellation
    # ------------------------------------------------------------------

    def run_one_async(self, song_path: Path, lyrics_path: Optional[Path] = None) -> CancelToken:
        """Start the pipeline on a background thread; return its CancelToken.

        Each call creates a FRESH CancelToken and a FRESH threading.Event —
        never reused across jobs.  The driver uses the returned token to
        observe phase and call cancel_active().  Workers must already be
        started.
        """
        self._validate_inputs(song_path, lyrics_path)

        # Fresh token per job (closes review issue #3)
        self._cancel_token = CancelToken(event=threading.Event())
        self._result = {}

        def _thread_target():
            try:
                ctx = self._run_pipeline(song_path, lyrics_path, self._cancel_token)
                self._result["ctx"] = ctx
            except Exception as exc:
                self._result["exception"] = exc

        self._pipeline_thread = threading.Thread(
            target=_thread_target,
            name="pipeline-worker",
            daemon=True,
        )
        self._pipeline_thread.start()
        return self._cancel_token

    def join(self, timeout: Optional[float] = None) -> StageContext:
        """Wait for the pipeline thread to finish.

        Re-raises whatever exception the thread caught (PipelineCancelled,
        FileNotFoundError, RuntimeError, etc.) — the caller is responsible
        for distinguishing cancellation from other failures.
        """
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=timeout)
            if self._pipeline_thread.is_alive():
                raise TimeoutError("Pipeline thread did not finish within timeout")

        exc = self._result.get("exception")
        if exc is not None:
            raise exc

        ctx = self._result.get("ctx")
        if ctx is not None:
            return ctx

        raise RuntimeError("Pipeline thread finished without result or exception")

    def cancel_active(self) -> None:
        """Request cancellation of the currently-running pipeline.

        Delegates to self._cancel_token.cancel().
        """
        if self._cancel_token is not None:
            self._cancel_token.cancel()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_inputs(self, song_path: Path, lyrics_path: Optional[Path]) -> None:
        if not song_path.exists():
            raise FileNotFoundError(f"Song file not found: {song_path}")
        if lyrics_path is not None:
            if not lyrics_path.exists():
                raise FileNotFoundError(f"Lyrics file not found: {lyrics_path}")
            if lyrics_path.suffix.lower() not in (".txt", ".srt"):
                raise ValueError(f"Lyrics must be .txt or .srt, got: {lyrics_path.suffix}")

    def _run_pipeline(
        self,
        song_path: Path,
        lyrics_path: Optional[Path],
        cancel_token: CancelToken,
    ) -> StageContext:
        tmp_parent = self._config.intermediate_dir or None
        tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_", dir=tmp_parent))
        logger.info(f"Temp dir: {tmp_dir}")

        ctx = StageContext(
            song_path=song_path,
            tmp_dir=tmp_dir,
            config=self._config,
            cancel=cancel_token,
        )
        ctx.artifacts["lyrics_path"] = lyrics_path

        try:
            for stage in self._stages:
                cancel_token.check_cancelled()  # cancel between stages
                logger.info(f"[pipeline] ▶ {stage.name}")
                stage.run(ctx)
                logger.info(f"[pipeline] ✓ {stage.name}")
            return ctx
        finally:
            logger.debug(f"[pipeline] Cleanup tmp: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
