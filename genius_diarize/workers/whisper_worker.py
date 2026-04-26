"""Re-export pipeline's WhisperWorker so genius_diarize uses a single implementation."""

from pipeline.workers.whisper_worker import AlignmentCancelledError, WhisperWorker

__all__ = ["WhisperWorker", "AlignmentCancelledError"]
