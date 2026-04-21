"""FFmpeg transcode stage: transcode WAV stems to M4A (AAC).

Spawns ffmpeg processes. Cancel is **immediate** for the same reason
as FFmpegExtractStage.
"""

import logging
import subprocess
from pathlib import Path

from cancel_test.context import CancelledError, StageContext
from cancel_test.stages.base import BaseStage

logger = logging.getLogger(__name__)

# M4A encoding quality: "2" ≈ 128 kbps VBR AAC
AAC_QUALITY = "2"
FFMPEG_THREADS = "4"


class FFmpegTranscodeStage(BaseStage):
    """Transcode vocal and instrumental WAV stems to M4A."""

    name = "ffmpeg_transcode"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

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

        logger.info(f"[{self.name}] Transcoding vocal: {vocal_wav.name}")
        self._transcode(vocal_wav, vocal_out, ctx)
        ctx.check_cancelled()

        logger.info(f"[{self.name}] Transcoding instrumental: {instrumental_wav.name}")
        self._transcode(instrumental_wav, nonvocal_out, ctx)
        ctx.check_cancelled()

        ctx.artifacts["vocal_m4a"] = vocal_out
        ctx.artifacts["nonvocal_m4a"] = nonvocal_out
        logger.info(f"[{self.name}] Transcode complete")

    def _transcode(self, wav_path: Path, output_path: Path, ctx: StageContext) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            FFMPEG_THREADS,
            "-i",
            str(wav_path),
            "-c:a",
            "aac",
            "-q:a",
            AAC_QUALITY,
            str(output_path),
        ]
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
            raise RuntimeError(f"ffmpeg transcode failed (exit code {rc})")

    def cancel(self, ctx: StageContext) -> None:
        """Immediate cancel: kill the active ffmpeg process."""
        logger.info(f"[{self.name}] Cancel requested — killing ffmpeg process")
        ctx.cancelled.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
