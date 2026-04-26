"""Unit tests for genius_diarize/caption.py: SRT with display_label,
ASS with dominant_speaker color keying."""

import pytest

from genius_diarize.caption import (
    _dominant_speaker_presence,
    _speaker_presence,
    generate_ass,
    generate_srt,
)
from genius_diarize.config import GeniusDiarizeConfig


def _make_line(text, speaker, dominant_speaker=None, display_label=None,
               start=0.0, end=1.0):
    """Build a minimal line object for caption testing."""
    ds = dominant_speaker if dominant_speaker is not None else speaker
    words = [
        {
            "word": w,
            "start": start + i * 0.3,
            "end": start + i * 0.3 + 0.2,
            "speaker": speaker,
            "dominant_speaker": ds,
            "is_segment_first": i == 0,
        }
        for i, w in enumerate(text.split())
    ]
    obj = {
        "text": text,
        "words": words,
        "speaker": speaker,
        "dominant_speaker": ds,
        "start": start,
        "end": end,
    }
    if display_label is not None:
        obj["display_label"] = display_label
    return obj


def _cfg():
    return GeniusDiarizeConfig()


# ---------------------------------------------------------------------------
# SRT tests
# ---------------------------------------------------------------------------


class TestSRT:
    def test_full_label_first(self):
        """First appearance uses full speaker label."""
        lines = [
            _make_line("Hello world", "Brian", display_label="Brian"),
            _make_line("Ensemble", None, display_label=None),
        ]
        srt_text = generate_srt(lines, _cfg())
        assert "Brian: Hello world" in srt_text

    def test_truncated_subsequent(self):
        """Subsequent appearance uses truncated label."""
        lines = [
            _make_line("First", "Brian", display_label="Brian"),
            _make_line("Second", "Brian", display_label="B"),
            _make_line("Ensemble", None, display_label=None),
        ]
        srt_text = generate_srt(lines, _cfg())
        assert "Brian: First" in srt_text
        assert "B: Second" in srt_text

    def test_no_label_when_none(self):
        """Ensemble lines (speaker=None) have no prefix."""
        lines = [_make_line("All together", None)]
        srt_text = generate_srt(lines, _cfg())
        # Should NOT have "None:" or any prefix
        assert "All together" in srt_text
        assert ": " not in srt_text.replace("Hello world", "")

    def test_single_speaker_no_prefix(self):
        """Single speaker mode omits prefix entirely."""
        lines = [_make_line("Hello world", "Brian")]
        srt_text = generate_srt(lines, _cfg())
        # Single speaker: no prefix
        assert "Brian:" not in srt_text
        assert "Hello world" in srt_text


# ---------------------------------------------------------------------------
# ASS tests
# ---------------------------------------------------------------------------


class TestASS:
    def test_color_keyed_by_dominant_speaker(self):
        """Kevin solo and Kevin & AJ share one color slot (Karaoke_Kevin)."""
        lines = [
            _make_line("Solo line", "Kevin", dominant_speaker="Kevin"),
            _make_line("Duet line", "Kevin & AJ", dominant_speaker="Kevin"),
            _make_line("Ensemble", None, dominant_speaker=None),
        ]
        ass_text = generate_ass(lines, _cfg())
        # Should have exactly one Karaoke_Kevin style
        assert "Style: Karaoke_Kevin," in ass_text
        # Should NOT have a separate style for the duet label
        assert "Karaoke_Kevin___AJ" not in ass_text

    def test_color_first_appearance_order(self):
        """Colors assigned in first-appearance order of dominant_speaker."""
        lines = [
            _make_line("Brian line", "Brian", dominant_speaker="Brian"),
            _make_line("Nick line", "Nick", dominant_speaker="Nick"),
        ]
        present, _ = _dominant_speaker_presence(lines)
        assert present == ["Brian", "Nick"]

    def test_ensemble_uses_default_style(self):
        """Lines with speaker=None use ensemble style."""
        lines = [
            _make_line("Ensemble", None, dominant_speaker=None),
        ]
        ass_text = generate_ass(lines, _cfg())
        assert "Karaoke_ensemble" in ass_text

    def test_single_speaker_uses_karaoke_style(self):
        """Single dominant speaker → single Karaoke style (goldenrod)."""
        lines = [_make_line("Solo", "Brian", dominant_speaker="Brian")]
        ass_text = generate_ass(lines, _cfg())
        assert "Style: Karaoke," in ass_text
        # No per-speaker style
        assert "Karaoke_Brian" not in ass_text

    def test_duet_event_uses_dominant_style(self):
        """A duet line's Dialogue event references the dominant speaker style."""
        lines = [
            _make_line("Duet line", "Kevin & AJ", dominant_speaker="Kevin"),
        ]
        # Need an ensemble line to avoid single-speaker mode
        lines.append(_make_line("Ensemble", None, dominant_speaker=None))
        ass_text = generate_ass(lines, _cfg())
        # The duet line should use Karaoke_Kevin, not Karaoke_Kevin___AJ
        assert "Karaoke_Kevin," in ass_text


# ---------------------------------------------------------------------------
# Presence helpers
# ---------------------------------------------------------------------------


class TestPresenceHelpers:
    def test_dominant_speaker_presence_dedup(self):
        """Same dominant_speaker from different speaker labels → one entry."""
        lines = [
            _make_line("Solo", "Kevin", dominant_speaker="Kevin"),
            _make_line("Duet", "Kevin & AJ", dominant_speaker="Kevin"),
        ]
        present, has_ensemble = _dominant_speaker_presence(lines)
        assert present == ["Kevin"]
        assert has_ensemble is False

    def test_speaker_presence_distinct(self):
        """Different speaker labels → separate entries."""
        lines = [
            _make_line("Solo", "Kevin", dominant_speaker="Kevin"),
            _make_line("Duet", "Kevin & AJ", dominant_speaker="Kevin"),
        ]
        present, _ = _speaker_presence(lines)
        assert present == ["Kevin", "Kevin & AJ"]
