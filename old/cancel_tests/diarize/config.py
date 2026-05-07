from dataclasses import dataclass


@dataclass
class DiarizeConfig:
    hf_token_path: str = ""
    device: str = "auto"
    num_speakers: int = 0
    min_speakers: int = 0
    max_speakers: int = 0
    speaker_label_format: str = "{speaker}"
    font_name: str = "Arial"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"
    secondary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    outline_width: int = 2
    shadow_offset: int = 1
    margin_left: int = 50
    margin_right: int = 50
    margin_vertical: int = 100
    line_lead_in_cs: int = 0
    line_lead_out_cs: int = 0
