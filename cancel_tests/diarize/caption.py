"""Caption generation from diarization results.

Generates SRT and ASS subtitle files from pyannote diarization output
(turns with speaker labels and timestamps). This is the diarization
equivalent of pipeline/stages/lyric_align.py — but instead of whisper-
aligned lyrics, it labels speaker turns.

ASS output uses speaker-labeled Dialogue events. Each speaker turn
becomes one subtitle line showing "[Speaker_00] Hello world".

SRT output follows the same segmentation as ASS (one entry per turn).
"""

import datetime
import logging
from pathlib import Path

import srt

from cancel_diarize.config import DiarizeConfig

logger = logging.getLogger(__name__)


def generate_srt(turns: list[dict], config: DiarizeConfig) -> str:
    """Build .srt content from diarization turns.

    Each turn becomes one SRT entry with speaker label + turn text.
    For diarization-only output (no transcription), the content is
    just the speaker label and time range.

    Args:
        turns: List of {"speaker": str, "start": float, "end": float} dicts.
        config: DiarizeConfig for speaker label formatting.

    Returns:
        SRT-formatted string.
    """
    subtitles = []
    for i, turn in enumerate(turns, start=1):
        label = config.speaker_label_format.format(
            speaker=turn["speaker"],
            index=int(turn["speaker"].split("_")[-1]) if "_" in turn["speaker"] else 0,
        )
        duration = turn["end"] - turn["start"]
        content = f"[{label}] ({duration:.1f}s)"

        subtitles.append(
            srt.Subtitle(
                index=i,
                start=datetime.timedelta(seconds=turn["start"]),
                end=datetime.timedelta(seconds=turn["end"]),
                content=content,
            )
        )

    return srt.compose(subtitles)


def generate_ass(turns: list[dict], config: DiarizeConfig) -> str:
    """Build .ass content from diarization turns.

    Each turn becomes one ASS Dialogue event with the speaker label
    as the subtitle text. Lead-in/lead-out timing gives the classic
    karaoke-style overlap between adjacent turns.

    Args:
        turns: List of {"speaker": str, "start": float, "end": float} dicts.
        config: DiarizeConfig for styling and timing.

    Returns:
        ASS-formatted string.
    """
    header = _generate_ass_header(config)
    events = _generate_ass_events(turns, config)
    return header + "\n".join(events) + "\n"


def write_srt(turns: list[dict], output_path: Path, config: DiarizeConfig) -> None:
    """Generate and write SRT file from diarization turns."""
    content = generate_srt(turns, config)
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"SRT written: {output_path}")


def write_ass(turns: list[dict], output_path: Path, config: DiarizeConfig) -> None:
    """Generate and write ASS file from diarization turns."""
    content = generate_ass(turns, config)
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"ASS written: {output_path}")


def _generate_ass_header(config: DiarizeConfig) -> str:
    """Generate the ASS file header with styles."""
    return (
        f"[Script Info]\n"
        f"Title: Speaker Diarization Subtitles\n"
        f"ScriptType: v4.00+\n"
        f"PlayResX: 1920\n"
        f"PlayResY: 1080\n"
        f"Timer: 100.0000\n"
        f"\n"
        f"[V4+ Styles]\n"
        f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        f"OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        f"ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Diarize,{config.font_name},{config.font_size},"
        f"{config.primary_color},{config.secondary_color},"
        f"{config.outline_color},{config.back_color},"
        f"0,0,0,0,100,100,0,0,1,"
        f"{config.outline_width},{config.shadow_offset},2,"
        f"{config.margin_left},{config.margin_right},{config.margin_vertical},1\n"
        f"\n"
        f"[Events]\n"
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _generate_ass_events(turns: list[dict], config: DiarizeConfig) -> list[str]:
    """Generate ASS Dialogue events from diarization turns.

    Each turn gets a single Dialogue line with:
    - Event window padded by lead-in/lead-out from config
    - Speaker label as text content
    """
    events = []
    lead_in_s = config.line_lead_in_cs / 100.0
    lead_out_s = config.line_lead_out_cs / 100.0

    for turn in turns:
        event_start = max(0.0, turn["start"] - lead_in_s)
        event_end = turn["end"] + lead_out_s

        label = config.speaker_label_format.format(
            speaker=turn["speaker"],
            index=int(turn["speaker"].split("_")[-1]) if "_" in turn["speaker"] else 0,
        )
        text = f"{{\\k0}}{label}"

        events.append(
            f"Dialogue: 0,{_seconds_to_ass_time(event_start)},"
            f"{_seconds_to_ass_time(event_end)},Diarize,,0,0,0,,{text}"
        )

    return events


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
