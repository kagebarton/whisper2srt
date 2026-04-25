"""Caption generation: SRT + per-speaker karaoke ASS.

Cloned and adapted from pipeline/stages/lyric_align.py. Two public
functions, no class.
"""

import datetime
import logging

import srt

logger = logging.getLogger(__name__)


def generate_srt(line_objects, cfg):
    """Build .srt content from line objects.

    Each line is prefixed with the speaker letter (e.g. "A: Hello world").
    In single-speaker mode (only one speaker letter appears in all lines),
    the prefix is omitted — output is plain lyric text, matching the
    non-diarized pipeline output.
    """
    present = sorted({l["speaker"] for l in line_objects}) if line_objects else []
    single_speaker = len(present) <= 1

    subs = []
    for i, line in enumerate(line_objects, start=1):
        if single_speaker:
            content = line["text"]
        else:
            content = f"{line['speaker']}: {line['text']}"
        subs.append(
            srt.Subtitle(
                index=i,
                start=datetime.timedelta(seconds=line["start"]),
                end=datetime.timedelta(seconds=line["end"]),
                content=content,
            )
        )
    return srt.compose(subs)


def generate_ass(line_objects, cfg):
    """Build .ass content from line objects using config styling/timing.

    Header gets one Style per speaker letter that actually appears in the
    line objects — keeps the file lean if only A/B are present.

    Single-speaker fallback: if only one speaker letter appears, emit a
    single ``Style: Karaoke`` (no letter suffix) with goldenrod primary
    color and skip the speaker prefix in every Dialogue line. Output is
    visually identical to a standard non-diarized karaoke file.
    """
    present = sorted({l["speaker"] for l in line_objects}) if line_objects else []
    single_speaker = len(present) <= 1

    header = _generate_ass_header(cfg, present, single_speaker)
    events = _generate_ass_events(line_objects, cfg, single_speaker)
    return header + "\n".join(events) + "\n"


# ---------------------------------------------------------------------------
# ASS header
# ---------------------------------------------------------------------------


def _generate_ass_header(cfg, present, single_speaker):
    """Generate the [Script Info] + [V4+ Styles] + [Events] Format header."""
    styles_block = _generate_styles(cfg, present, single_speaker)
    return (
        f"[Script Info]\n"
        f"Title: Karaoke Subtitles\n"
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
        f"{styles_block}"
        f"\n"
        f"[Events]\n"
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _generate_styles(cfg, present, single_speaker):
    """Generate one Style row per speaker, or a single Karaoke style for single-speaker."""
    rows = []
    if single_speaker:
        # Use goldenrod (pipeline default) — first entry in speaker_colors
        primary = cfg.speaker_colors[0] if cfg.speaker_colors else "&H0000D7FF&"
        rows.append(
            f"Style: Karaoke,{cfg.font_name},{cfg.font_size},"
            f"{primary},{cfg.secondary_color},"
            f"{cfg.outline_color},{cfg.back_color},"
            f"0,0,0,0,100,100,0,0,1,"
            f"{cfg.outline_width},{cfg.shadow_offset},2,"
            f"{cfg.margin_left},{cfg.margin_right},{cfg.margin_vertical},1"
        )
    else:
        for letter in present:
            idx = cfg.speaker_letters.index(letter)
            primary = cfg.speaker_colors[idx]
            rows.append(
                f"Style: Karaoke_{letter},{cfg.font_name},{cfg.font_size},"
                f"{primary},{cfg.secondary_color},"
                f"{cfg.outline_color},{cfg.back_color},"
                f"0,0,0,0,100,100,0,0,1,"
                f"{cfg.outline_width},{cfg.shadow_offset},2,"
                f"{cfg.margin_left},{cfg.margin_right},{cfg.margin_vertical},1"
            )
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# ASS events
# ---------------------------------------------------------------------------


def _generate_ass_events(line_objects, cfg, single_speaker):
    """Generate one Dialogue event per line object."""
    events = []
    for line_obj in line_objects:
        words = line_obj["words"]
        if not words:
            continue

        # Determine style name
        if single_speaker:
            style = "Karaoke"
        else:
            style = f"Karaoke_{line_obj['speaker']}"

        # Pad the event window around the sung word boundaries
        event_start = max(0.0, words[0]["start"] - cfg.line_lead_in_cs / 100.0)
        event_end = words[-1]["end"] + cfg.line_lead_out_cs / 100.0

        # Karaoke cursor starts at the event start time.
        # Tracking prev_end from here means the gap to the first word
        # automatically becomes the silent lead-in tag.
        prev_end = event_start
        parts = []

        # Speaker prefix (plain text, no \kf) — omitted in single-speaker mode
        if not single_speaker:
            parts.append(f"{line_obj['speaker']}: ")

        for i, word_data in enumerate(words):
            word = word_data["word"]
            word_start = word_data["start"]

            # Apply first_word_nudge_cs: push the first word of a segment
            # forward if it starts almost immediately after the previous
            # segment (within ~50ms of the expected lead-in gap), preventing
            # word clipping.
            if (
                word_data.get("is_segment_first")
                and abs(word_start - prev_end - cfg.line_lead_in_cs / 100.0) < 0.05
            ):
                word_start += cfg.first_word_nudge_cs / 100.0

            word_end = word_data["end"]
            word_dur_cs = max(10, round((word_end - word_start) * 100))

            # Silent cursor advance through any gap before this word.
            # For the first word this gap = line_lead_in_cs (the lead-in).
            # For subsequent words it covers natural pauses between words.
            gap_cs = max(0, round((word_start - prev_end) * 100))
            if gap_cs > 0:
                parts.append(f"{{\\k{gap_cs}}}")

            # \kf = left-to-right fill sweep over word_dur_cs centiseconds
            parts.append(f"{{\\kf{word_dur_cs}}}{word}")
            prev_end = word_end

            if i < len(words) - 1:
                parts.append(" ")

        karaoke_text = "".join(parts)
        events.append(
            f"Dialogue: 0,{_seconds_to_ass_time(event_start)},"
            f"{_seconds_to_ass_time(event_end)},{style},,0,0,0,,{karaoke_text}"
        )

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
