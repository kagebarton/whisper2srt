from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig


@dataclass
class StageContext:
    """Per-job context passed forward through the pipeline stages."""

    song_path: Path  # input audio/video file
    tmp_dir: Path  # per-job temp directory
    config: PipelineConfig  # shared config reference
    artifacts: dict[str, Any] = field(default_factory=dict)
