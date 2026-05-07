from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class WhisperModelConfig:
    model_path: str = str(_REPO_ROOT / "models" / "large-v3-turbo.pt")
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    vad: bool = True
    vad_threshold: float = 0.25
    suppress_silence: bool = True
    suppress_word_ts: bool = True
    only_voice_freq: bool = True
    refine_steps: str = "s"
    refine_word_level: bool = False
