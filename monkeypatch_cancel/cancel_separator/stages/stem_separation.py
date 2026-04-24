"""Stem separation stage: delegates to a persistent CancelableStemWorker.

This is the key stage for mid-separation cancellation. The worker
subprocess loads the model once and stays alive across jobs. The stage
injects a per-chunk cancellation check into the model's demix loop,
allowing the separation to abort between chunks without killing the
process (and thus without losing the loaded model).

Cancel semantics: **delayed**. The stage's cancel() sets the
ctx.cancelled flag, but does NOT kill the worker. The injected
check inside demix() raises CancelledError between chunks. The worker
catches it, cleans up the partial GPU state, and returns a cancel
result to the orchestrator. The model stays loaded and ready for
the next job.
"""

import logging
from pathlib import Path

from cancel_separator.context import CancelledError, StageContext
from cancel_separator.stages.base import BaseStage
from cancel_separator.workers.cancelable_stem_worker import (
    CancelableStemWorker,
    WorkerCancelledError,
    WorkerDiedError,
)

logger = logging.getLogger(__name__)


class StemSeparationStage(BaseStage):
    """Thin adapter: delegates to a persistent CancelableStemWorker.

    The worker is injected (not owned) — its lifecycle is managed by
    the orchestrator, not by this stage. This ensures the model stays
    loaded across jobs.
    """

    name = "stem_separation"

    def __init__(self, stem_worker: CancelableStemWorker) -> None:
        self._worker = stem_worker

    def run(self, ctx: StageContext) -> None:
        wav_in = ctx.artifacts.get("wav_in")
        if wav_in is None:
            raise RuntimeError(f"[{self.name}] No wav_in in artifacts")

        if not self._worker.is_alive():
            logger.info(f"[{self.name}] Worker not alive — starting")
            self._worker.start()

        logger.info(f"[{self.name}] Starting separation: {wav_in.name}")
        try:
            vocal_wav, instrumental_wav = self._worker.separate(
                wav_path=wav_in,
                output_dir=ctx.tmp_dir,
                cancel_event=ctx.cancelled,
            )
        except WorkerCancelledError:
            # Worker detected cancellation between chunks — model is still loaded
            logger.info(
                f"[{self.name}] Separation cancelled mid-stream (model still loaded)"
            )
            raise CancelledError()
        except WorkerDiedError:
            ctx.check_cancelled()  # re-raises CancelledError if cancelled
            raise RuntimeError("Stem worker died unexpectedly during separation")

        ctx.artifacts["vocal_wav"] = vocal_wav
        ctx.artifacts["instrumental_wav"] = instrumental_wav
        logger.info(
            f"[{self.name}] Separation complete: vocal={vocal_wav.name}, instrumental={instrumental_wav.name}"
        )

    def cancel(self, ctx: StageContext) -> None:
        """Delayed cancel: set the flag, let the worker finish the current chunk.

        The worker's demix() loop checks ctx.cancelled between chunks.
        When it sees the flag, it raises CancelledError inside the worker
        process, which is caught and returned as a cancel result. The
        model stays loaded.
        """
        logger.info(
            f"[{self.name}] Cancel requested — delayed (letting current chunk finish)"
        )
        ctx.cancelled.set()

    # Do NOT call self._worker.kill() — that would force a model reload.
