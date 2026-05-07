"""Pipeline stage protocol and base class.

From the cancel_separator architecture: stages are stateless, instantiated
once, and reused for every job. Each stage receives a StageContext and
advances the pipeline. Cancellation is cooperative: stages check
ctx.cancelled between work units.

Two cancel semantics:
1. Immediate cancel — for stages that control a subprocess.
   The stage's cancel() method kills the Popen.
2. Delayed cancel — for stages that delegate to a persistent worker.
   The stage's cancel() is a no-op; the worker finishes its current
   model pass, and the orchestrator checks ctx.cancelled after return.
   The model stays loaded.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from diarize.context import StageContext

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStage(Protocol):
    """Protocol for a single pipeline stage."""

    name: str

    def run(self, ctx: StageContext) -> None:
        """Execute this stage, mutating ctx.artifacts as needed."""
        ...

    def cancel(self, ctx: StageContext) -> None:
        """Request cancellation of in-progress work.

        Default is delayed cancel (flag-only). Stages that own a
        subprocess override this to kill the Popen.
        """
        ...


class BaseStage:
    """Base class providing default (delayed) cancel semantics."""

    name: str = "base"

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError

    def cancel(self, ctx: StageContext) -> None:
        """Delayed cancel: set the flag, let the current work finish."""
        logger.info(
            f"[{self.name}] Cancel requested — delayed (letting current work finish)"
        )
        ctx.cancelled.set()
