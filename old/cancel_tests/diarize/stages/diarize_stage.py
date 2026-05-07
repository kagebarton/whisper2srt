"""Diarization stage: apply speaker diarization to a vocal stem.

This stage delegates to CancelableDiarizeWorker for the actual inference.
The worker uses a two-layer hook architecture for cancellation:
  Layer 1: pyannote hook= callback (stage-boundary progress + cancel)
  Layer 2: PyTorch forward_pre_hooks (per-batch cancel in segmentation + embedding)
The cancel() method on context is a threading.Event — the worker's hooks
check it. The model stays loaded across cancellations.
"""

import logging

from diarize.context import StageContext
from diarize.stages.base import BaseStage
from diarize.workers.cancelable_diarize_worker import (
    CancelableDiarizeWorker,
    DiarizationCancelledError,
)

logger = logging.getLogger(__name__)


class DiarizeStage(BaseStage):
    """Apply speaker diarization to a vocal stem audio file."""

    name = "diarize"

    def __init__(self, worker: CancelableDiarizeWorker) -> None:
        self._worker = worker

    def run(self, ctx: StageContext) -> None:
        vocal_path = ctx.vocal_path
        if vocal_path is None:
            raise RuntimeError(f"[{self.name}] No vocal_path in context")

        logger.info(f"[{self.name}] Diarizing: {vocal_path.name}")

        try:
            turns = self._worker.diarize(
                vocal_path=vocal_path,
                cancel_event=ctx.cancelled,
            )
            ctx.artifacts["diarization_turns"] = turns
            logger.info(f"[{self.name}] Diarization complete: {len(turns)} turns")
        except DiarizationCancelledError:
            logger.info(f"[{self.name}] Diarization cancelled")
            raise
