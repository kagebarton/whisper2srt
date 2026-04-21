"""FFmpeg extract stage: extract audio from video to WAV.

Spawns an ffmpeg subprocess. Cancel is **immediate**: cancel() kills
the Popen, and the orchestrator raises CancelledError when wait() returns.
"""

import logging
import subprocess
from pathlib import Path

from cancel_test.context import CancelledError, StageContext
from cancel_test.stages.base import BaseStage

logger = logging.getLogger(__name__)


class FFmpegExtractStage(BaseStage):
    """Extract audio from a video file to a temporary WAV using ffmpeg."""

    name = "ffmpeg_extract"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def run(self, ctx: StageContext) -> None:
        wav_out = ctx.tmp_dir / f"{ctx.song_path.stem}_input.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(ctx.song_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-sample_fmt",
            "s16",
            str(wav_out),
        ]
        logger.info(f"[{self.name}] Extracting audio: {ctx.song_path.name}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            rc = self._proc.wait()
        finally:
            self._proc = None

        if rc != 0:
            if ctx.is_cancelled():
                raise CancelledError()
            raise RuntimeError(f"ffmpeg extract failed (exit code {rc})")

        ctx.artifacts["wav_in"] = wav_out
        logger.info(f"[{self.name}] Extract complete: {wav_out.name}")

    def cancel(self, ctx: StageContext) -> None:
        """Immediate cancel: kill the ffmpeg process."""
        logger.info(f"[{self.name}] Cancel requested — killing ffmpeg process")
        ctx.cancelled.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
