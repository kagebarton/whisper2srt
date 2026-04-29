"""Unit tests for run.py helpers: assign_speakers_from_genius."""

import pytest

from genius_diarize.run import assign_speakers_from_genius


def _make_line_obj(text, words=None, speaker=None):
    """Build a minimal line_obj for testing."""
    if words is None:
        words = [
            {
                "word": w,
                "start": 0.0,
                "end": 0.1,
                "speaker": None,
                "dominant_speaker": None,
                "is_segment_first": i == 0,
            }
            for i, w in enumerate(text.split())
        ]
    return {"text": text, "words": words, "start": 0.0, "end": 1.0}


# ---------------------------------------------------------------------------
# assign_speakers_from_genius
# ---------------------------------------------------------------------------


class TestAssignSpeakersFromGenius:
    def test_single_name(self):
        """Single-name speaker_label propagates to line and words."""
        genius_lines = [
            {
                "text": "You are my fire",
                "section": "Verse 1",
                "speaker_label": "Brian",
                "dominant_speaker": "Brian",
                "is_ensemble": False,
            }
        ]
        line_obj = _make_line_obj("You are my fire")
        assign_speakers_from_genius([line_obj], genius_lines)

        assert line_obj["speaker"] == "Brian"
        assert line_obj["dominant_speaker"] == "Brian"
        for w in line_obj["words"]:
            assert w["speaker"] == "Brian"
            assert w["dominant_speaker"] == "Brian"

    def test_duet(self):
        """Named-duet label and dominant_speaker propagate to words."""
        genius_lines = [
            {
                "text": "Now I can see",
                "section": "Bridge",
                "speaker_label": "Kevin & AJ",
                "dominant_speaker": "Kevin",
                "is_ensemble": False,
            }
        ]
        line_obj = _make_line_obj("Now I can see")
        assign_speakers_from_genius([line_obj], genius_lines)

        assert line_obj["speaker"] == "Kevin & AJ"
        assert line_obj["dominant_speaker"] == "Kevin"
        for w in line_obj["words"]:
            assert w["speaker"] == "Kevin & AJ"
            assert w["dominant_speaker"] == "Kevin"

    def test_ensemble_none(self):
        """Ensemble lines get speaker=None, dominant_speaker=None."""
        genius_lines = [
            {
                "text": "Tell me why",
                "section": "Chorus",
                "speaker_label": None,
                "dominant_speaker": None,
                "is_ensemble": True,
            }
        ]
        line_obj = _make_line_obj("Tell me why")
        assign_speakers_from_genius([line_obj], genius_lines)

        assert line_obj["speaker"] is None
        assert line_obj["dominant_speaker"] is None
        for w in line_obj["words"]:
            assert w["speaker"] is None
            assert w["dominant_speaker"] is None

    def test_mixed_sections(self):
        """Multiple lines with different attribution."""
        genius_lines = [
            {
                "text": "Yeah",
                "section": "Intro",
                "speaker_label": "AJ",
                "dominant_speaker": "AJ",
                "is_ensemble": False,
            },
            {
                "text": "Everyone sings",
                "section": "Chorus",
                "speaker_label": None,
                "dominant_speaker": None,
                "is_ensemble": True,
            },
            {
                "text": "Now I can see",
                "section": "Bridge",
                "speaker_label": "Kevin & AJ",
                "dominant_speaker": "Kevin",
                "is_ensemble": False,
            },
        ]
        line_objs = [
            _make_line_obj("Yeah"),
            _make_line_obj("Everyone sings"),
            _make_line_obj("Now I can see"),
        ]
        assign_speakers_from_genius(line_objs, genius_lines)

        assert line_objs[0]["speaker"] == "AJ"
        assert line_objs[0]["dominant_speaker"] == "AJ"
        assert line_objs[1]["speaker"] is None
        assert line_objs[1]["dominant_speaker"] is None
        assert line_objs[2]["speaker"] == "Kevin & AJ"
        assert line_objs[2]["dominant_speaker"] == "Kevin"


