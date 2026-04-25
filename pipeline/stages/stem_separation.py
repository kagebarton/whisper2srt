"""Stem separation stage: delegates to a persistent StemWorker.

Wraps worker.separate() in a cancel activity scope so the orchestrator
can cancel mid-separation via SetEvent.  Translates WorkerCancelledError
to PipelineCancelled.  The stale auto-start path has been removed —
with orchestrator-managed lifecycle, a dead worker mid-pipeline is a
real failure.
"""

import logging
from pathlib import Path

from pipeline.context import Phase, PipelineCancelled, SetEvent, StageContext
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

        logger.info(f"[{self.name}] Starting separation: {Path(extracted_wav).name}")

        if ctx.cancel is not None:
            cancel_event = ctx.cancel.event
            try:
                with ctx.cancel.activity(Phase.STEM_SEPARATION, SetEvent(cancel_event)):
                    vocal_wav, instrumental_wav = self._worker.separate(
                        wav_path=extracted_wav,
                        output_dir=ctx.tmp_dir,
                        cancel_event=cancel_event,
                    )
            except WorkerCancelledError:
                # Worker raised before activity() noticed cancelled flag; translate.
                raise PipelineCancelled(Phase.STEM_SEPARATION)
            except WorkerDiedError:
                raise RuntimeError("Stem worker died unexpectedly during separation")
        else:
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
