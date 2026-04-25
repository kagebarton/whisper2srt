"""FFmpeg transcode stage: transcode WAV stems to M4A (AAC).

Writes transcoded files to ctx.tmp_dir first, then moves them to the
final output directory only after both transcodes succeed — preventing
orphan files on cancellation.

Uses run_ffmpeg() from _ffmpeg_helpers for cancellation support.
"""

import logging
import shutil
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.context import Phase, StageContext
from pipeline.stages._ffmpeg_helpers import run_ffmpeg
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

        song = ctx.song_path

        # Final destinations (created lazily; only after both transcodes succeed)
        final_vocal = song.parent / "vocal" / f"{song.stem}---vocal.m4a"
        final_nonvocal = song.parent / "nonvocal" / f"{song.stem}---nonvocal.m4a"

        # Write transcoded output INSIDE tmp_dir so a cancel during either
        # transcode leaves the partial file in tmp_dir (wiped by the
        # orchestrator's shutil.rmtree).
        tmp_vocal = ctx.tmp_dir / final_vocal.name
        tmp_nonvocal = ctx.tmp_dir / final_nonvocal.name

        logger.info(f"[{self.name}] Transcoding vocal: {Path(vocal_wav).name}")
        self._transcode(vocal_wav, tmp_vocal, ctx)

        # Cancel between the two transcodes is caught here (outside any
        # activity scope after the first run_ffmpeg exits).
        if ctx.cancel is not None:
            ctx.cancel.check_cancelled()

        logger.info(f"[{self.name}] Transcoding instrumental: {Path(instrumental_wav).name}")
        self._transcode(instrumental_wav, tmp_nonvocal, ctx)

        # Both transcodes succeeded — promote tmp files to final locations.
        final_vocal.parent.mkdir(exist_ok=True)
        final_nonvocal.parent.mkdir(exist_ok=True)
        shutil.move(str(tmp_vocal), str(final_vocal))
        shutil.move(str(tmp_nonvocal), str(final_nonvocal))

        ctx.artifacts["vocal_m4a"] = final_vocal
        ctx.artifacts["nonvocal_m4a"] = final_nonvocal
        logger.info(f"[{self.name}] Transcode complete")

    def _transcode(self, wav_path: Path, output_path: Path, ctx: StageContext) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-threads", self._config.ffmpeg_threads,
            "-i", str(wav_path),
            "-c:a", "aac",
            "-q:a", self._config.aac_quality,
            str(output_path),
        ]
        run_ffmpeg(cmd, ctx, Phase.TRANSCODE)
