"""Stage context: per-job state bag passed between pipeline stages.

Mirrors the A1 architecture from the pipeline-robustness-fixes plan:
- song_path, tmp_dir, artifacts are per-job
- cancelled is the cancel flag shared between orchestrator and stages
- Workers (StemWorker, etc.) are NOT in the context — they are
  manager-scoped, injected into stages, and persist across jobs.
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageContext:
    """Per-job context passed forward through the pipeline stages."""

    song_path: Path
    tmp_dir: Path
    cancelled: threading.Event = field(default_factory=threading.Event)
    artifacts: dict[str, Path] = field(default_factory=dict)

    # -- Convenience -------------------------------------------------------

    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()

    def check_cancelled(self) -> None:
        """Raise CancelledError if the job has been cancelled."""
        if self.cancelled.is_set():
            raise CancelledError()


class CancelledError(Exception):
    """Raised when a running job has been cancelled."""
