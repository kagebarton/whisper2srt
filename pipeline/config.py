from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = str(_REPO_ROOT / "models")


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
    regroup: str = ""


@dataclass
class PipelineConfig:
    # --- Model paths ---
    whisper_model_path: str = str(_REPO_ROOT / "models" / "large-v3-turbo.pt")
    separator_model_dir: str = _MODELS_DIR
    separator_model_name: str = "vocals_mel_band_roformer.ckpt"

    # --- Device/compute ---
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"

    # --- Intermediate files directory ---
    intermediate_dir: str = "/mnt/ramdisk"  # Empty = system temp dir

    # --- Loudnorm targets ---
    loudnorm_target_i: float = -24.0  # Target integrated loudness (LUFS)
    loudnorm_target_tp: float = -2.0  # Target true peak (dBTP)
    loudnorm_target_lra: float = 7.0  # Target loudness range (LU)

    # --- Whisper alignment options ---
    whisper_language: str = "en"
    whisper_vad: bool = True
    whisper_vad_threshold: float = 0.25
    whisper_suppress_silence: bool = True
    whisper_suppress_word_ts: bool = True
    whisper_only_voice_freq: bool = True
    whisper_refine_steps: str = "s"  # 's' = refine starts, 'e' = refine ends, 'se' = both
    whisper_refine_word_level: bool = False
    whisper_regroup: str = ""

    # --- ASS styling ---
    font_name: str = "Arial"
    font_size: int = 60
    primary_color: str = "&H00D7FF&"  # Soft Yellow
    secondary_color: str = "&H00FFFFFF"  # White (not yet sung)
    outline_color: str = "&H00000000"  # Black outline
    back_color: str = "&H80000000&"  # Translucent shadow
    outline_width: int = 3
    shadow_offset: int = 2
    margin_left: int = 50
    margin_right: int = 50
    margin_vertical: int = 150

    # --- Karaoke timing (centiseconds) ---
    line_lead_in_cs: int = 80
    line_lead_out_cs: int = 20
    first_word_nudge_cs: int = 0

    # --- FFmpeg transcoding ---
    aac_quality: str = "2"  # ≈ 128 kbps VBR AAC
    ffmpeg_threads: str = "4"
