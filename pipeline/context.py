"""Pipeline context: per-job state and cancellation infrastructure.

Defines Phase, CancelToken (with activity() contextmanager), Cancellable
protocol + concrete implementations (KillProcess, SetEvent),
PipelineCancelled exception, and StageContext.
"""

import enum
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class Phase(enum.Enum):
    """Cancellation grain — one per stage or per model-call within a stage."""
    EXTRACT = "extract"                  # ffmpeg_extract stage
    LOUDNORM = "loudnorm"                # loudnorm_analyze stage
    STEM_SEPARATION = "stem_separation"  # stem_separation stage (per-chunk)
    TRANSCODE = "transcode"              # ffmpeg_transcode stage (both stems)
    ALIGN = "align"                      # lyric_align: model.align()
    TRANSCRIBE = "transcribe"            # lyric_align: model.transcribe()
    REFINE = "refine"                    # lyric_align: model.refine()


# ---------------------------------------------------------------------------
# Cancellable protocol + concrete implementations
# ---------------------------------------------------------------------------

class Cancellable(Protocol):
    """Abstraction for anything the orchestrator can cancel mid-flight."""
    def cancel(self) -> None: ...


class KillProcess:
    """Cancel an ffmpeg subprocess by SIGKILL."""
    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc

    def cancel(self) -> None:
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass


class SetEvent:
    """Cancel a model worker call by setting its threading.Event."""
    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def cancel(self) -> None:
        self._event.set()


# ---------------------------------------------------------------------------
# PipelineCancelled exception
# ---------------------------------------------------------------------------

class PipelineCancelled(Exception):
    """Raised when the active job has been cancelled."""
    def __init__(self, phase: Optional[Phase] = None) -> None:
        self.phase = phase
        if phase is not None:
            super().__init__(f"Pipeline cancelled at phase {phase.value}")
        else:
            super().__init__("Pipeline cancelled")


# ---------------------------------------------------------------------------
# CancelToken
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class CancelToken:
    """Holds the cancellation state for one pipeline job.

    Owned by the orchestrator, referenced from StageContext.  Each
    ``run_one_async()`` call creates a fresh CancelToken with a fresh
    ``threading.Event`` — never reused across jobs.

    Key invariant: phase + active are always set/cleared together under
    the lock.  ``cancelled`` is set *before* any ``target.cancel()`` call
    so that the activity's exit check reliably sees it.
    """

    event: threading.Event               # per-job model-worker event
    cancelled: bool = False              # sticky flag, set by cancel()
    phase: Optional[Phase] = None       # current phase (None = between activities)
    active: Optional[Cancellable] = None  # registered cancel target
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Stage-facing -----------------------------------------------------------

    @contextmanager
    def activity(self, phase: Phase, target: Cancellable):
        """Open an activity for *phase*, with *target* as the cancel mechanism.

        Atomically sets phase + active under the lock.  On exit (normal,
        exception, or cancel), atomically clears active.  If a cancel
        arrived before the activity opened, raises PipelineCancelled
        immediately so the work inside never starts.

        On exit, only synthesises PipelineCancelled if the work itself
        did NOT raise — otherwise we'd mask the original exception.
        """
        with self.lock:
            if self.cancelled:
                raise PipelineCancelled(phase)
            self.phase = phase
            self.active = target
        try:
            yield
        finally:
            with self.lock:
                self.active = None
            # Only raise if the work itself completed without an exception.
            # If yield raised (PipelineCancelled, RuntimeError, anything),
            # let it propagate untouched.
            if self.cancelled and sys.exc_info()[0] is None:
                raise PipelineCancelled(phase)

    def is_cancelled(self) -> bool:
        """Return True if cancelled (lock-acquired, free-threading-safe)."""
        with self.lock:
            return self.cancelled

    def check_cancelled(self) -> None:
        """Raise PipelineCancelled if cancelled is set.  Used between stages."""
        with self.lock:
            if self.cancelled:
                raise PipelineCancelled(self.phase)

    # Driver-facing ----------------------------------------------------------

    def get_phase(self) -> Optional[Phase]:
        """Return the current phase (lock-acquired)."""
        with self.lock:
            return self.phase

    def cancel(self) -> None:
        """Mark cancelled, signal whichever Cancellable is currently active.

        Sets ``cancelled=True`` FIRST (under the lock), THEN calls
        ``target.cancel()``.  This ordering means by the time the
        cancelled work observes the cancel (Popen wakes up from SIGKILL,
        worker sees the event), ``cancelled`` is already True — so the
        activity()'s exit check reliably sees it.  Do not reorder.
        """
        with self.lock:
            self.cancelled = True
            target = self.active
            if target is not None:
                target.cancel()


# ---------------------------------------------------------------------------
# StageContext
# ---------------------------------------------------------------------------

@dataclass
class StageContext:
    """Per-job context passed forward through the pipeline stages."""

    song_path: Path                          # input audio/video file
    tmp_dir: Path                            # per-job temp directory
    config: PipelineConfig                   # shared config reference
    artifacts: dict[str, Any] = field(default_factory=dict)
    cancel: Optional[CancelToken] = None     # None when run without cancel
