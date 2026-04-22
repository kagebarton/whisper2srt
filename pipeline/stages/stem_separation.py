"""Stem separation stage: delegates to a persistent StemWorker.

Ported from cancel_separator/stages/stem_separation.py.
Removes cancel logic, keeps worker delegation.
"""

import logging
from pathlib import Path

from pipeline.context import StageContext
from pipeline.stages.base import BaseStage
from pipeline.workers.stem_worker import (
    StemWorker,
    WorkerCancelledError,
    WorkerDiedError,
)

logger = logging.getLogger(__name__)


class StemSeparationStage(BaseStage):
    """Thin adapter: delegates to a persistent StemWorker.

    The worker is injected (not owned) — its lifecycle is managed by
    the orchestrator, not by this stage. This ensures the model stays
    loaded across jobs.
    """

    name = "stem_separation"

    def __init__(self, stem_worker: StemWorker) -> None:
        self._worker = stem_worker

    def run(self, ctx: StageContext) -> None:
        extracted_wav = ctx.artifacts.get("extracted_wav")
        if extracted_wav is None:
            raise RuntimeError(f"[{self.name}] No extracted_wav in artifacts")

        if not self._worker.is_alive():
            logger.info(f"[{self.name}] Worker not alive — starting")
            self._worker.start()

        logger.info(f"[{self.name}] Starting separation: {Path(extracted_wav).name}")
        try:
            vocal_wav, instrumental_wav = self._worker.separate(
                wav_path=extracted_wav,
                output_dir=ctx.tmp_dir,
                cancel_event=None,
            )
        except WorkerCancelledError:
            raise RuntimeError("Separation cancelled between chunks (model still loaded)")
        except WorkerDiedError:
            raise RuntimeError("Stem worker died unexpectedly during separation")

        ctx.artifacts["vocal_wav"] = vocal_wav
        ctx.artifacts["instrumental_wav"] = instrumental_wav
        logger.info(
            f"[{self.name}] Separation complete: vocal={vocal_wav.name}, "
            f"instrumental={instrumental_wav.name}"
        )
