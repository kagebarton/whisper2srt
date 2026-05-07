"""Configuration dataclasses for the diarized captions prototype.

Three dataclasses mirroring the source prototypes so the copied workers
don't need rewiring beyond their import lines.
"""

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class WhisperModelConfig:
    """Whisper model configuration — identical to pipeline.config.WhisperModelConfig."""

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
    regroup: str = ""


@dataclass
class DiarizeConfig:
    """Diarization configuration — subset of cancel_tests/diarize/config.py.

    ASS styling fields and speaker_label_format are intentionally omitted;
    those move into DiarizedCaptionsConfig. The worker never reads them
    (verified against cancelable_diarize_worker.py).
    """

    pipeline_name: str = "pyannote/speaker-diarization-community-1"
    hf_token_path: str = ""
    device: str = "auto"
    num_speakers: int = 0  # 0 = auto-detect
    min_speakers: int = 0
    max_speakers: int = 0


@dataclass
class DiarizedCaptionsConfig:
    """Top-level configuration. Owns whisper + diarize sub-configs and caption styling.

    All ASS colors normalized to 8-hex ``&HAABBGGRR&`` form with trailing ``&``.
    The pipeline uses the 6-hex ``&HBBGGRR&`` form for primary_color — the
    8-hex form is visually identical (alpha=00 is the default) but textually
    different.
    """

    whisper: WhisperModelConfig = field(default_factory=WhisperModelConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)

    # ASS karaoke styling (cloned from pipeline.config.PipelineConfig)
    font_name: str = "Arial"
    font_size: int = 60
    secondary_color: str = "&H00FFFFFF&"  # not-yet-sung (white)
    outline_color: str = "&H00000000&"  # black outline
    back_color: str = "&H80000000&"  # 50% translucent shadow
    outline_width: int = 3
    shadow_offset: int = 2
    margin_left: int = 50
    margin_right: int = 50
    margin_vertical: int = 150

    # Karaoke timing (centiseconds)
    line_lead_in_cs: int = 80
    line_lead_out_cs: int = 20
    first_word_nudge_cs: int = 0

    # Per-speaker colors. Index 0 → speaker A, index 1 → B, etc.
    # Format: &HAABBGGRR& (8 hex digits, alpha + reversed RGB, trailing &).
    speaker_colors: list = field(
        default_factory=lambda: [
            "&H0000D7FF&",  # A — goldenrod (alpha=00 form of pipeline's &H00D7FF&)
            "&H00FFFF00&",  # B — cyan
            "&H00B469FF&",  # C — pink
            "&H0000FF00&",  # D — green
            "&H000080FF&",  # E — orange
            "&H00FA82FA&",  # F — lavender
        ]
    )

    # Speaker letters. Mapped 1:1 to speaker_colors by index.
    speaker_letters: str = "ABCDEFGHIJ"
