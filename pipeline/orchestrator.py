"""Synchronous pipeline orchestrator for single-song processing.

Owns both worker lifecycles (StemWorker, WhisperWorker), creates and
cleans up the temp directory, and runs stages sequentially. No job queue,
no orchestrator thread, no cancel flag — the pipeline runs synchronously
in the main thread.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from pipeline.config import PipelineConfig
from pipeline.context import StageContext
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

    def run(self, song_path: Path, lyrics_path: Optional[Path] = None) -> StageContext:
        self._validate_inputs(song_path, lyrics_path)
        self._start_workers()
        try:
            return self._run_pipeline(song_path, lyrics_path)
        finally:
            self._stop_workers()

    # --- Internals ---

    def _validate_inputs(self, song_path: Path, lyrics_path: Optional[Path]) -> None:
        if not song_path.exists():
            raise FileNotFoundError(f"Song file not found: {song_path}")
        if lyrics_path is not None:
            if not lyrics_path.exists():
                raise FileNotFoundError(f"Lyrics file not found: {lyrics_path}")
            if lyrics_path.suffix.lower() not in (".txt", ".srt"):
                raise ValueError(f"Lyrics must be .txt or .srt, got: {lyrics_path.suffix}")

    def _start_workers(self) -> None:
        logger.info("Starting stem worker...")
        self._stem_worker.start()
        logger.info("Loading whisper model...")
        self._whisper_worker.load_model()

    def _stop_workers(self) -> None:
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

    def _run_pipeline(self, song_path: Path, lyrics_path: Optional[Path]) -> StageContext:
        tmp_parent = self._config.intermediate_dir or None
        tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_", dir=tmp_parent))
        logger.info(f"Temp dir: {tmp_dir}")

        ctx = StageContext(
            song_path=song_path,
            tmp_dir=tmp_dir,
            config=self._config,
        )
        ctx.artifacts["lyrics_path"] = lyrics_path

        try:
            for stage in self._stages:
                logger.info(f"[pipeline] ▶ {stage.name}")
                stage.run(ctx)
                logger.info(f"[pipeline] ✓ {stage.name}")
            return ctx
        finally:
            logger.debug(f"[pipeline] Cleanup tmp: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
