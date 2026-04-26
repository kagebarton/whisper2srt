"""Caption generation: SRT + per-speaker karaoke ASS.

Adapted from diarized_captions/caption.py with these changes:

- SRT reads ``line["display_label"]`` when present (set by run.py's
  annotate_display_labels); falls back to ``line["speaker"]`` for compat
  with non-Genius callers.
- ASS style/color palette is built off ``line["dominant_speaker"]`` so
  that a Kevin solo and a Kevin & AJ duet share one color slot. Falls
  back to ``line["speaker"]`` if dominant_speaker is absent.
- No ``speaker_letters`` dependency — speaker labels are arbitrary strings.
"""

import datetime
import re

import srt

# Pattern for sanitising speaker labels into safe ASS style-name tokens
_UNSAFE_CHAR_RE = re.compile(r"\W")

# Ensemble style constants
_ENSEMBLE_STYLE = "Karaoke_ensemble"
_ENSEMBLE_COLOR_FALLBACK = "&H0000D7FF&"  # goldenrod — fallback if not in cfg


def generate_srt(line_objects, cfg):
    """Build .srt content from line objects.

    Uses ``line["display_label"]`` when present (full name on first
    appearance, truncated on subsequent — set by annotate_display_labels
    in run.py). Falls back to ``line["speaker"]`` for non-Genius callers.

    Lines with display_label=None (ensemble) have no prefix.
    In single-speaker mode (only one speaker label, no ensemble lines),
    the prefix is omitted entirely — plain lyric text.
    """
    present, has_ensemble = _speaker_presence(line_objects)
    single_speaker = len(present) <= 1 and not has_ensemble

    subs = []
    for i, line in enumerate(line_objects, start=1):
        if single_speaker or line.get("speaker") is None:
            content = line["text"]
        else:
            # Use display_label if available (Genius pipeline), else speaker
            label = line.get("display_label", line.get("speaker"))
            if label is None:
                content = line["text"]
            else:
                content = f"{label}: {line['text']}"
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

    Style/color palette is built off ``dominant_speaker`` so that a solo
    line and a duet line for the same singer share one color slot.

    Lines with dominant_speaker=None (or speaker=None) get the white
    ensemble style with no speaker prefix.

    Single-speaker fallback: if only one dominant speaker appears and
    there are no ensemble lines, emit a single ``Style: Karaoke``
    (goldenrod) and skip the speaker prefix.
    """
    present, has_ensemble = _dominant_speaker_presence(line_objects)
    single_speaker = len(present) <= 1 and not has_ensemble

    header = _generate_ass_header(cfg, present, single_speaker, has_ensemble)
    events = _generate_ass_events(line_objects, cfg, single_speaker)
    return header + "\n".join(events) + "\n"


# ---------------------------------------------------------------------------
# ASS header
# ---------------------------------------------------------------------------


def _generate_ass_header(cfg, present, single_speaker, has_ensemble=False):
    """Generate the [Script Info] + [V4+ Styles] + [Events] Format header."""
    styles_block = _generate_styles(cfg, present, single_speaker, has_ensemble)
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


def _safe_style_name(label: str) -> str:
    """Sanitise a speaker label into an ASCII-safe ASS style name component."""
    return _UNSAFE_CHAR_RE.sub("_", label)


def _generate_styles(cfg, present, single_speaker, has_ensemble=False):
    """Generate one Style row per dominant speaker, or a single Karaoke style."""
    rows = []
    if single_speaker:
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
        # Assign colours by first-appearance order of dominant_speaker
        for color_idx, label in enumerate(present):
            primary = (
                cfg.speaker_colors[color_idx]
                if color_idx < len(cfg.speaker_colors)
                else cfg.speaker_colors[-1]
            )
            safe = _safe_style_name(label)
            rows.append(
                f"Style: Karaoke_{safe},{cfg.font_name},{cfg.font_size},"
                f"{primary},{cfg.secondary_color},"
                f"{cfg.outline_color},{cfg.back_color},"
                f"0,0,0,0,100,100,0,0,1,"
                f"{cfg.outline_width},{cfg.shadow_offset},2,"
                f"{cfg.margin_left},{cfg.margin_right},{cfg.margin_vertical},1"
            )
        if has_ensemble:
            ensemble_color = getattr(cfg, "ensemble_color", _ENSEMBLE_COLOR_FALLBACK)
            rows.append(
                f"Style: {_ENSEMBLE_STYLE},{cfg.font_name},{cfg.font_size},"
                f"{ensemble_color},{cfg.secondary_color},"
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

        # Determine style from dominant_speaker (falls back to speaker
        # for compat with non-Genius callers).
        speaker = line_obj.get("speaker")
        dominant = line_obj.get("dominant_speaker", speaker)
        if single_speaker:
            style = "Karaoke"
        elif dominant is None:
            style = _ENSEMBLE_STYLE
        else:
            style = f"Karaoke_{_safe_style_name(dominant)}"

        # Pad the event window around the sung word boundaries
        event_start = max(0.0, words[0]["start"] - cfg.line_lead_in_cs / 100.0)
        event_end = words[-1]["end"] + cfg.line_lead_out_cs / 100.0

        # Karaoke cursor starts at the event start time.
        prev_end = event_start
        parts = []

        # Speaker prefix (plain text, no \kf) — omitted in single-speaker
        # or ensemble. Uses display_label when present for SRT consistency;
        # falls back to speaker.
        if not single_speaker and speaker is not None:
            prefix = line_obj.get("display_label", speaker)
            if prefix is not None:
                parts.append(f"{prefix}: ")

        for i, word_data in enumerate(words):
            word = word_data["word"]
            word_start = word_data["start"]

            # Apply first_word_nudge_cs: push the first word of a segment
            # forward if it starts almost immediately after the previous
            # segment (within ~50ms of the expected lead-in gap).
            if (
                word_data.get("is_segment_first")
                and abs(word_start - prev_end - cfg.line_lead_in_cs / 100.0) < 0.05
            ):
                word_start += cfg.first_word_nudge_cs / 100.0

            word_end = word_data["end"]
            word_dur_cs = max(10, round((word_end - word_start) * 100))

            # Silent cursor advance through any gap before this word.
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


def _dominant_speaker_presence(line_objects):
    """Return (present_labels, has_ensemble) keyed by dominant_speaker.

    present_labels: list of non-None dominant_speaker values in
    **first-appearance order**.

    has_ensemble: True if any line has speaker=None.
    """
    seen = set()
    present = []
    for line in line_objects:
        ds = line.get("dominant_speaker", line.get("speaker"))
        if ds is not None and ds not in seen:
            seen.add(ds)
            present.append(ds)
    has_ensemble = any(l.get("speaker") is None for l in line_objects)
    return present, has_ensemble


def _speaker_presence(line_objects):
    """Return (present_labels, has_ensemble) keyed by speaker.

    present_labels: list of non-None speaker values in first-appearance
    order.

    has_ensemble: True if any line has speaker=None.
    """
    seen = set()
    present = []
    for line in line_objects:
        s = line.get("speaker")
        if s is not None and s not in seen:
            seen.add(s)
            present.append(s)
    has_ensemble = any(l.get("speaker") is None for l in line_objects)
    return present, has_ensemble


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
