"""FFmpeg extract stage: extract audio from video to WAV.

Ported from cancel_separator/stages/ffmpeg_extract.py.
Removes cancel logic, uses subprocess.run(), reads ffmpeg_threads from config.
"""

import logging
import subprocess
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.context import StageContext
from pipeline.stages.base import BaseStage

logger = logging.getLogger(__name__)


class FFmpegExtractStage(BaseStage):
    """Extract audio from a video file to a temporary WAV using ffmpeg."""

    name = "ffmpeg_extract"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> None:
        wav_out = ctx.tmp_dir / f"{ctx.song_path.stem}_input.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            self._config.ffmpeg_threads,
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
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg extract failed (exit code {result.returncode})")

        ctx.artifacts["extracted_wav"] = wav_out
        logger.info(f"[{self.name}] Extract complete: {wav_out.name}")
