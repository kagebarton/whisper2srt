"""Loudnorm analyze stage: FFmpeg loudnorm 1st pass (analysis only).

Runs ffmpeg loudnorm on the extracted WAV to capture loudness measurements
for later use. No output file is produced — only measurement values are
stored in the context artifacts.
"""

import json
import logging
import subprocess
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.context import StageContext
from pipeline.stages.base import BaseStage

logger = logging.getLogger(__name__)


class LoudnormAnalyzeStage(BaseStage):
    """Run FFmpeg loudnorm 1st pass and capture measurement JSON."""

    name = "loudnorm_analyze"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, ctx: StageContext) -> None:
        extracted_wav = ctx.artifacts.get("extracted_wav")
        if extracted_wav is None:
            raise RuntimeError(f"[{self.name}] No extracted_wav in artifacts")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-threads",
            self._config.ffmpeg_threads,
            "-i",
            str(extracted_wav),
            "-af",
            f"loudnorm=I={self._config.loudnorm_target_i}"
            f":TP={self._config.loudnorm_target_tp}"
            f":LRA={self._config.loudnorm_target_lra}"
            f":print_format=json",
            "-f",
            "null",
            "-",
        ]
        logger.info(f"[{self.name}] Running loudnorm analysis on: {Path(extracted_wav).name}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else ""
            raise RuntimeError(f"ffmpeg loudnorm failed: {stderr_tail}")

        # Parse the JSON block from stderr by walking backward from the end.
        # FFmpeg may emit other {...} fragments in log lines earlier in stderr,
        # so we find the last balanced JSON block.
        json_str = self._extract_json_from_stderr(result.stderr)
        if json_str is None:
            raise RuntimeError("loudnorm did not produce JSON output")

        try:
            measurements = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"loudnorm JSON parse error: {e}") from e

        # Store measurement fields in artifacts
        try:
            ctx.artifacts["loudnorm_input_i"] = float(measurements["input_i"])
            ctx.artifacts["loudnorm_input_tp"] = float(measurements["input_tp"])
            ctx.artifacts["loudnorm_input_lra"] = float(measurements["input_lra"])
            ctx.artifacts["loudnorm_input_thresh"] = float(measurements["input_thresh"])
            ctx.artifacts["loudnorm_target_offset"] = float(measurements["target_offset"])
            ctx.artifacts["loudnorm_type"] = str(measurements["normalization_type"])
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"loudnorm JSON missing or invalid field: {e}") from e

        logger.info(
            f"[{self.name}] I={measurements['input_i']} LUFS "
            f"TP={measurements['input_tp']} dBTP "
            f"LRA={measurements['input_lra']} LU "
            f"offset={measurements['target_offset']} dB "
            f"type={measurements['normalization_type']}"
        )

    @staticmethod
    def _extract_json_from_stderr(stderr: str) -> str | None:
        """Walk backward through stderr lines to find the JSON block.

        Loudnorm writes the stats JSON to stderr. We iterate from the last
        line toward the first, finding the closing '}' then collecting
        backward until the opening '{'.
        """
        lines = stderr.strip().split("\n")
        end_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().endswith("}"):
                end_idx = i
                break

        if end_idx is None:
            return None

        # Walk backward from the closing '}' to find the opening '{'
        start_idx = None
        for i in range(end_idx, -1, -1):
            if lines[i].strip().startswith("{"):
                start_idx = i
                break

        if start_idx is None:
            return None

        return "\n".join(lines[start_idx:end_idx + 1])
