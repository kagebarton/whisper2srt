"""Unit tests for genius_diarize/genius.py."""

import pytest

from genius_diarize.genius import (
    genius_singer_mode,
    parse_genius_sections,
    split_groups,
)


# ---------------------------------------------------------------------------
# parse_genius_sections
# ---------------------------------------------------------------------------


class TestParseGeniusSections:
    """Test parse_genius_sections against plan §5 / §10 spec."""

    def test_single_name(self):
        text = "[Verse 1: Brian]\nYou are my fire\nThe one desire"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] == "Brian"
            assert gl["dominant_speaker"] == "Brian"
            assert gl["is_ensemble"] is False
            assert gl["section"] == "Verse 1"

    def test_named_duet(self):
        text = "[Bridge: Kevin & AJ]\nNow I can see\nShow me the meaning"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] == "Kevin & AJ"
            assert gl["dominant_speaker"] == "Kevin"
            assert gl["is_ensemble"] is False
            assert gl["section"] == "Bridge"

    def test_all_only(self):
        text = "[Chorus: All]\nEveryone sings\nTogether now"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] is None
            assert gl["dominant_speaker"] is None
            assert gl["is_ensemble"] is True

    def test_multi_group_with_all(self):
        text = "[Chorus: Nick, All]\nTell me why\nAin't nothin' but a heartache"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] == "Nick"
            assert gl["dominant_speaker"] == "Nick"
            assert gl["is_ensemble"] is False

    def test_multi_group_named(self):
        """Two named groups (no All) → colored by first group."""
        text = "[Bridge: Brian, AJ]\nHello\nGoodbye"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] == "Brian"
            assert gl["dominant_speaker"] == "Brian"
            assert gl["is_ensemble"] is False

    def test_multi_group_with_pair(self):
        """First group is pair, second group exists → colored by first group."""
        text = "[Verse: Brian & AJ, Nick]\nLine one\nLine two"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] == "Brian & AJ"
            assert gl["dominant_speaker"] == "Brian"
            assert gl["is_ensemble"] is False

    def test_no_attribution(self):
        text = "[Verse 1]\nJust a verse\nNo attribution"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] is None
            assert gl["dominant_speaker"] is None
            assert gl["is_ensemble"] is True

    def test_blank_lines_skipped(self):
        text = "[Verse 1: Brian]\n\nYou are my fire\n\nThe one desire\n"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        assert lines[0]["text"] == "You are my fire"
        assert lines[1]["text"] == "The one desire"



    def test_section_field_kept(self):
        text = "[Verse 1: Brian]\nYou are my fire"
        lines = parse_genius_sections(text)
        assert lines[0]["section"] == "Verse 1"

    def test_section_changes_between_headers(self):
        text = "[Verse 1: Brian]\nLine A\n[Chorus: Nick]\nLine B"
        lines = parse_genius_sections(text)
        assert lines[0]["section"] == "Verse 1"
        assert lines[0]["speaker_label"] == "Brian"
        assert lines[1]["section"] == "Chorus"
        assert lines[1]["speaker_label"] == "Nick"

    def test_no_headers_at_all(self):
        text = "Just plain lyrics\nNo headers here"
        lines = parse_genius_sections(text)
        assert len(lines) == 2
        for gl in lines:
            assert gl["speaker_label"] is None
            assert gl["is_ensemble"] is True

    def test_header_with_trailing_whitespace(self):
        text = "[Verse 1: Brian]  \nYou are my fire"
        lines = parse_genius_sections(text)
        assert len(lines) == 1
        assert lines[0]["speaker_label"] == "Brian"

    def test_three_group_header(self):
        """Multi-group with 3 groups → colored by first group."""
        text = "[Chorus: AJ, All, Brian]\nTell me why"
        lines = parse_genius_sections(text)
        assert lines[0]["speaker_label"] == "AJ"
        assert lines[0]["is_ensemble"] is False

    def test_back_txt_integration(self):
        """Spot-check back.txt headers against expected output."""
        text = (
            "[Intro: AJ]\n"
            "Yeah\n"
            "\n"
            "[Verse 1: Brian]\n"
            "You are my fire\n"
            "\n"
            "[Chorus: Nick, All]\n"
            "Tell me why\n"
            "\n"
            "[Bridge: Kevin & AJ]\n"
            "Now I can see\n"
            "\n"
            "[Break: All, Nick]\n"
            "Ain't nothin' but a heartache\n"
        )
        lines = parse_genius_sections(text)
        # Intro: AJ → single name
        assert lines[0]["speaker_label"] == "AJ"
        assert lines[0]["dominant_speaker"] == "AJ"
        # Verse 1: Brian → single name
        assert lines[1]["speaker_label"] == "Brian"
        assert lines[1]["dominant_speaker"] == "Brian"
        # Chorus: Nick, All → multi-group → first group
        assert lines[2]["speaker_label"] == "Nick"
        assert lines[2]["dominant_speaker"] == "Nick"
        assert lines[2]["is_ensemble"] is False
        # Bridge: Kevin & AJ → named pair
        assert lines[3]["speaker_label"] == "Kevin & AJ"
        assert lines[3]["dominant_speaker"] == "Kevin"
        # Break: All, Nick → multi-group where first is All → ensemble
        assert lines[4]["speaker_label"] is None
        assert lines[4]["is_ensemble"] is True


# ---------------------------------------------------------------------------
# align_text field
# ---------------------------------------------------------------------------


class TestAlignText:
    def test_no_parens_unchanged(self):
        text = "[Verse 1: Brian]\nYou are my fire"
        lines = parse_genius_sections(text)
        assert lines[0]["align_text"] == "You are my fire"
        assert lines[0]["text"] == "You are my fire"

    def test_text_field_preserves_parens(self):
        """The display text field always retains the original parens."""
        text = "[Chorus: All]\nEveryone sings (yeah)\nTogether now"
        lines = parse_genius_sections(text)
        assert lines[0]["text"] == "Everyone sings (yeah)"
        assert lines[0]["align_text"] == "Everyone sings (yeah)"
        assert lines[1]["text"] == "Together now"
        assert lines[1]["align_text"] == "Together now"


# ---------------------------------------------------------------------------
# genius_singer_mode
# ---------------------------------------------------------------------------


class TestGeniusSingerMode:
    def test_solo_no_headers(self):
        lines = parse_genius_sections("Just lyrics\nNo headers")
        assert genius_singer_mode(lines) == "solo"

    def test_solo_no_attribution(self):
        lines = parse_genius_sections("[Verse 1]\nLyrics here")
        assert genius_singer_mode(lines) == "solo"

    def test_multi_single_name(self):
        lines = parse_genius_sections("[Verse 1: Brian]\nLyrics here")
        assert genius_singer_mode(lines) == "multi"

    def test_multi_named_duet(self):
        lines = parse_genius_sections("[Bridge: Kevin & AJ]\nLyrics here")
        assert genius_singer_mode(lines) == "multi"

    def test_all_ensemble_is_solo(self):
        """Only [Chorus: All] headers → solo (nothing to label)."""
        lines = parse_genius_sections("[Chorus: All]\nEveryone sings")
        assert genius_singer_mode(lines) == "solo"

    def test_multi_group_is_multi(self):
        """Multi-group assigned to first speaker → multi."""
        lines = parse_genius_sections("[Chorus: Nick, All]\nTell me why")
        assert genius_singer_mode(lines) == "multi"

    def test_mixed_labeled_and_unlabeled_is_multi(self):
        text = "[Verse 1: Brian]\nSolo line\n[Chorus: Nick, All]\nEnsemble line"
        lines = parse_genius_sections(text)
        assert genius_singer_mode(lines) == "multi"


# ---------------------------------------------------------------------------
# split_groups
# ---------------------------------------------------------------------------


class TestSplitGroups:
    def test_single_name(self):
        assert split_groups("Brian") == [["Brian"]]

    def test_duet(self):
        assert split_groups("Kevin & AJ") == [["Kevin", "AJ"]]

    def test_multiple_groups(self):
        assert split_groups("Nick, All") == [["Nick"], ["All"]]

    def test_pair_and_single(self):
        assert split_groups("Brian & AJ, Nick") == [["Brian", "AJ"], ["Nick"]]

    def test_all_only(self):
        assert split_groups("All") == [["All"]]

    def test_three_groups(self):
        assert split_groups("AJ, All, Brian") == [["AJ"], ["All"], ["Brian"]]

    def test_whitespace_trimming(self):
        assert split_groups("  Brian  &  AJ  ,  Nick  ") == [
            ["Brian", "AJ"],
            ["Nick"],
        ]

    def test_trailing_comma(self):
        """Trailing comma produces empty group, which is skipped."""
        assert split_groups("Brian,") == [["Brian"]]


# ---------------------------------------------------------------------------
# Section carry-forward
# ---------------------------------------------------------------------------


class TestSectionCarryForward:
    def test_bare_chorus_inherits_labeled_chorus(self):
        """[Chorus: Rumi] then [Chorus] → second chorus labeled Rumi."""
        text = (
            "[Chorus: Rumi]\nLine one\n"
            "[Chorus]\nLine two\n"
        )
        lines = parse_genius_sections(text)
        assert lines[0]["speaker_label"] == "Rumi"
        assert lines[1]["speaker_label"] == "Rumi"
        assert lines[1]["dominant_speaker"] == "Rumi"

    def test_bare_chorus_inherits_ensemble_chorus(self):
        """[Chorus: All] then [Chorus] → both ensemble."""
        text = (
            "[Chorus: All]\nLine one\n"
            "[Chorus]\nLine two\n"
        )
        lines = parse_genius_sections(text)
        assert lines[0]["is_ensemble"] is True
        assert lines[1]["is_ensemble"] is True

    def test_bare_section_no_prior_stays_ensemble(self):
        """[Verse 3] inherits from [Verse 1] via section family carry-forward."""
        text = (
            "[Verse 1: Brian]\nLine one\n"
            "[Verse 3]\nLine two\n"
        )
        lines = parse_genius_sections(text)
        assert lines[0]["speaker_label"] == "Brian"
        assert lines[1]["speaker_label"] == "Brian"
        assert lines[1]["is_ensemble"] is False

    def test_carry_forward_updates_with_later_header(self):
        """Second attribution for same section name updates carry-forward."""
        text = (
            "[Chorus: Rumi]\nFirst chorus\n"
            "[Chorus: Jinu]\nSecond chorus\n"
            "[Chorus]\nThird chorus\n"
        )
        lines = parse_genius_sections(text)
        assert lines[0]["speaker_label"] == "Rumi"
        assert lines[1]["speaker_label"] == "Jinu"
        assert lines[2]["speaker_label"] == "Jinu"  # inherits most recent

    def test_carry_forward_named_duet(self):
        """[Bridge: Kevin & AJ] then [Bridge] → both labeled Kevin & AJ."""
        text = (
            "[Bridge: Kevin & AJ]\nLine one\n"
            "[Bridge]\nLine two\n"
        )
        lines = parse_genius_sections(text)
        assert lines[1]["speaker_label"] == "Kevin & AJ"
        assert lines[1]["dominant_speaker"] == "Kevin"

    def test_different_sections_dont_cross_carry(self):
        """Carry-forward is per section name, not global."""
        text = (
            "[Chorus: Rumi]\nChorus line\n"
            "[Bridge]\nBridge line\n"
        )
        lines = parse_genius_sections(text)
        assert lines[0]["speaker_label"] == "Rumi"
        assert lines[1]["speaker_label"] is None  # Bridge has no prior attribution
