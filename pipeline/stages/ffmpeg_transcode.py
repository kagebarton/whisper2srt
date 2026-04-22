"""FFmpeg transcode stage: transcode WAV stems to M4A (AAC).

Ported from cancel_separator/stages/ffmpeg_transcode.py.
Removes cancel logic, reads AAC params from config, drops ctx from _transcode.
"""

import logging
import subprocess
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.context import StageContext
from pipeline.stages.base import BaseStage

logger = logging.getLogger(__name__)


class FFmpegTranscodeStage(BaseStage):
    """Transcode vocal and instrumental WAV stems to M4A."""

    name = "ffmpeg_transcode"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> None:
        vocal_wav = ctx.artifacts.get("vocal_wav")
        instrumental_wav = ctx.artifacts.get("instrumental_wav")

        if not vocal_wav or not instrumental_wav:
            raise RuntimeError(f"[{self.name}] Missing stem WAVs in artifacts")

        video = ctx.song_path
        vocal_dir = video.parent / "vocal"
        nonvocal_dir = video.parent / "nonvocal"
        vocal_dir.mkdir(exist_ok=True)
        nonvocal_dir.mkdir(exist_ok=True)

        vocal_out = vocal_dir / f"{video.stem}---vocal.m4a"
        nonvocal_out = nonvocal_dir / f"{video.stem}---nonvocal.m4a"

        logger.info(f"[{self.name}] Transcoding vocal: {Path(vocal_wav).name}")
        self._transcode(vocal_wav, vocal_out)

        logger.info(f"[{self.name}] Transcoding instrumental: {Path(instrumental_wav).name}")
        self._transcode(instrumental_wav, nonvocal_out)

        ctx.artifacts["vocal_m4a"] = vocal_out
        ctx.artifacts["nonvocal_m4a"] = nonvocal_out
        logger.info(f"[{self.name}] Transcode complete")

    def _transcode(self, wav_path: Path, output_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            self._config.ffmpeg_threads,
            "-i",
            str(wav_path),
            "-c:a",
            "aac",
            "-q:a",
            self._config.aac_quality,
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg transcode failed (exit code {result.returncode})")
