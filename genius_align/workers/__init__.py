"""Local WhisperWorker — forked from pipeline.workers to wire the full
tuning surface defined in ``genius_align.config.WhisperModelConfig``.
"""

from genius_align.workers.whisper_worker import (
    AlignmentCancelledError,
    WhisperWorker,
)

__all__ = ["WhisperWorker", "AlignmentCancelledError"]
