"""Configuration for the Genius Diarize prototype.

Drops DiarizeConfig (no pyannote). Caption styling fields are kept flat
on the top-level GeniusDiarizeConfig for compatibility with caption.py
(avoiding cfg.caption.xxx indirection).
"""

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import WhisperModelConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GeniusDiarizeConfig:
    """Top-level configuration. Owns whisper sub-config + flat caption styling.

    No DiarizeConfig — this prototype has no pyannote dependency.

    All ASS colors normalized to 8-hex ``&HAABBGGRR&`` form with
    trailing ``&``.
    """

    whisper: WhisperModelConfig = field(default_factory=WhisperModelConfig)

    # ASS karaoke styling (flat — no CaptionConfig indirection)
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

    # Ensemble (unlabeled) lines use this color for the karaoke sweep.
    ensemble_color: str = "&H0000D7FF&"  # goldenrod

    # Per-speaker colors. Index 0 → first dominant_speaker, etc.
    # Format: &HAABBGGRR& (8 hex digits, alpha + reversed RGB, trailing &).
    # Goldenrod is reserved for ensemble; these should be distinct from it.
    speaker_colors: list = field(
        default_factory=lambda: [
            "&H00FFFF00&",  # 0 — cyan
            "&H00B469FF&",  # 1 — pink
            "&H0000FF00&",  # 2 — green
            "&H000080FF&",  # 3 — orange
            "&H00FA82FA&",  # 4 — lavender
            "&H000000FF&",  # 5 — red
        ]
    )
