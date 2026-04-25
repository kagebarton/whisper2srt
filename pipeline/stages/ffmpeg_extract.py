"""FFmpeg extract stage: extract audio from video to WAV.

Ported from cancel_separator/stages/ffmpeg_extract.py.
Uses run_ffmpeg() from _ffmpeg_helpers for cancellation support.
"""

import logging
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.context import Phase, StageContext
from pipeline.stages._ffmpeg_helpers import run_ffmpeg
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
            "-threads", self._config.ffmpeg_threads,
            "-i", str(ctx.song_path),
            "-vn",
            "-ac", "2",
            "-ar", "44100",
            "-sample_fmt", "s16",
            str(wav_out),
        ]
        logger.info(f"[{self.name}] Extracting audio: {ctx.song_path.name}")
        run_ffmpeg(cmd, ctx, Phase.EXTRACT)

        ctx.artifacts["extracted_wav"] = wav_out
        logger.info(f"[{self.name}] Extract complete: {wav_out.name}")
