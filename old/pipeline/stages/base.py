from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from pipeline.context import StageContext

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStage(Protocol):
    name: str

    def run(self, ctx: StageContext) -> None:
        """Execute this stage, mutating ctx.artifacts as needed."""
        ...


class BaseStage:
    """Base class providing the name attribute; subclasses override run()."""

    name: str = "base"

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError
