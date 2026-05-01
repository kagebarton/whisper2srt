"""Unit tests for word_extraction.py: NW alignment and helpers."""

import pytest

from genius_diarize.word_extraction import (
    _levenshtein,
    _normalize_token,
    _score,
    match_words_to_lines,
)


def _word(text, start, end):
    return {
        "word": text,
        "start": start,
        "end": end,
        "is_segment_first": False,
        "speaker": None,
        "dominant_speaker": None,
    }


# ---------------------------------------------------------------------------
# _normalize_token
# ---------------------------------------------------------------------------


class TestNormalizeToken:
    def test_lowercase(self):
        assert _normalize_token("Brian") == "brian"

    def test_strips_punctuation(self):
        assert _normalize_token("fire,") == "fire"
        assert _normalize_token("fire!") == "fire"

    def test_apostrophe_stripped(self):
        assert _normalize_token("I'm") == "im"
        assert _normalize_token("ain't") == "aint"

    def test_nfkc(self):
        # Full-width A → regular A
        assert _normalize_token("Ａ") == "a"


# ---------------------------------------------------------------------------
# _levenshtein
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_equal(self):
        assert _levenshtein("fire", "fire") == 0

    def test_one_substitution(self):
        assert _levenshtein("fire", "fare") == 1

    def test_one_insertion(self):
        assert _levenshtein("fire", "fires") == 1

    def test_one_deletion(self):
        assert _levenshtein("fires", "fire") == 1

    def test_early_exit(self):
        assert _levenshtein("a", "abcd") == 2  # length diff > 1


# ---------------------------------------------------------------------------
# _score
# ---------------------------------------------------------------------------


class TestScore:
    def test_exact_match(self):
        assert _score("fire", "fire") == 2

    def test_contraction_expansion(self):
        assert _score("im", "i") == 2  # "i" is part of contraction expansion "i am"
        assert _score("dont", "do") == 2
        # "wanna" → "want to"; if whisper says "wanna" and lyric says "wanna" → exact
        assert _score("wanna", "wanna") == 2

    def test_fuzzy_one_edit(self):
        assert _score("fire", "fires") == 1  # one insertion, both ≥3 chars

    def test_fuzzy_too_short(self):
        assert _score("im", "is") == 0  # both < 3 chars, fuzzy disabled

    def test_mismatch(self):
        assert _score("fire", "water") == 0

    def test_contraction_match(self):
        assert _score("cant", "cannot") == 2  # "cant" → "cannot" in table
        assert _score("dont", "do") == 2       # "do" is part of "do not"


# ---------------------------------------------------------------------------
# match_words_to_lines — basic alignment
# ---------------------------------------------------------------------------


def _make_words(tokens_with_times):
    """Build whisper word list from [(text, start, end), ...]."""
    return [_word(t, s, e) for t, s, e in tokens_with_times]


class TestMatchWordsToLines:
    def test_perfect_match(self):
        """Whisper words match lyrics exactly — correct line boundaries."""
        words = _make_words([
            ("You", 0.0, 0.3), ("are", 0.3, 0.5), ("my", 0.5, 0.7), ("fire", 0.7, 1.0),
            ("The", 1.1, 1.3), ("one", 1.3, 1.5), ("desire", 1.5, 2.0),
        ])
        lines = ["You are my fire", "The one desire"]
        result = match_words_to_lines(words, lines)
        assert len(result) == 2
        assert result[0]["text"] == "You are my fire"
        assert len(result[0]["words"]) == 4
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"] == pytest.approx(1.0)
        assert result[1]["text"] == "The one desire"
        assert len(result[1]["words"]) == 3
        assert result[1]["start"] == pytest.approx(1.1)

    def test_punctuation_difference(self):
        """Lyric has 'fire,' but whisper says 'fire' — still matches."""
        words = _make_words([("fire", 0.0, 0.5), ("the", 0.6, 0.8)])
        lines = ["fire, the"]
        result = match_words_to_lines(words, lines)
        assert len(result[0]["words"]) == 2

    def test_extra_whisper_word(self):
        """Whisper hallucination between two real tokens is dropped."""
        words = _make_words([
            ("you", 0.0, 0.3), ("are", 0.3, 0.5),
            ("yeah", 0.5, 0.6),        # hallucination — no lyric match
            ("my", 0.6, 0.8), ("fire", 0.8, 1.0),
        ])
        lines = ["you are my fire"]
        result = match_words_to_lines(words, lines)
        word_texts = [w["word"] for w in result[0]["words"]]
        assert "yeah" not in word_texts
        assert len(result[0]["words"]) == 4

    def test_missing_whisper_word(self):
        """Lyric word absent from whisper output — remaining words still align."""
        words = _make_words([
            ("you", 0.0, 0.3),
            # "are" missing from whisper
            ("my", 0.4, 0.6), ("fire", 0.6, 1.0),
        ])
        lines = ["you are my fire"]
        result = match_words_to_lines(words, lines)
        word_texts = [w["word"] for w in result[0]["words"]]
        assert "you" in word_texts
        assert "my" in word_texts
        assert "fire" in word_texts

    def test_contraction_split(self):
        """'I'm' in lyrics aligns to 'Im' in whisper (both normalize to 'im')."""
        words = _make_words([("Im", 0.0, 0.3), ("gonna", 0.3, 0.6), ("go", 0.6, 1.0)])
        lines = ["I'm gonna go"]
        result = match_words_to_lines(words, lines)
        assert len(result[0]["words"]) == 3

    def test_two_lines_no_cascade(self):
        """Drift on line 1 does not displace line 2 words."""
        words = _make_words([
            ("you", 0.0, 0.3), ("fire", 0.3, 0.6),   # "are my" missing
            ("the", 1.0, 1.2), ("one", 1.2, 1.5), ("desire", 1.5, 2.0),
        ])
        lines = ["you are my fire", "the one desire"]
        result = match_words_to_lines(words, lines)
        assert result[1]["text"] == "the one desire"
        assert len(result[1]["words"]) == 3
        # "the" belongs to line 2, not line 1
        first_line_texts = [w["word"] for w in result[0]["words"]]
        assert "the" not in first_line_texts

    def test_align_lines_param(self):
        """align_lines (paren-stripped) used for alignment; display text kept."""
        words = _make_words([
            ("bye", 0.0, 0.3), ("bye", 0.3, 0.6), ("bye", 0.6, 1.0),
        ])
        lines = ["Bye, bye, bye (Bye, bye)"]
        align_lines = ["Bye, bye, bye"]
        result = match_words_to_lines(words, lines, align_lines)
        assert result[0]["text"] == "Bye, bye, bye (Bye, bye)"
        assert len(result[0]["words"]) == 3

    def test_unaligned_line_gets_empty_words(self):
        """A lyric line whisper entirely skipped gets empty words, no cascade."""
        words = _make_words([
            ("the", 1.0, 1.3), ("one", 1.3, 1.6), ("desire", 1.6, 2.0),
        ])
        lines = ["you are my fire", "the one desire"]
        result = match_words_to_lines(words, lines)
        assert len(result) == 2
        assert result[0]["words"] == []
        assert result[1]["text"] == "the one desire"
        assert len(result[1]["words"]) == 3
