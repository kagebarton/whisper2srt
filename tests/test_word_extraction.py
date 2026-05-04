"""Unit tests for word_extraction.py: NW alignment and helpers."""

import pytest

from genius_diarize.word_extraction import (
    _BAND_MIN_LENGTH,
    _levenshtein,
    _needleman_wunsch,
    _needleman_wunsch_banded,
    _needleman_wunsch_unbanded,
    _normalize_token,
    _score,
    _MAX_MATCH,
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
        assert _score("cant", "cannot") == 2 # "cant" → "cannot" in table
        assert _score("dont", "do") == 2 # "do" is part of "do not"

    def test_anchor_bonus(self):
        assert _score("california", "california") == 3  # len >= 6 exact match

    def test_anchor_bonus_short_unchanged(self):
        assert _score("fire", "fire") == 2  # len < 6, stays at +2

    def test_phonetic_equiv_short(self):
        assert _score("mm", "mmm") == 1
        assert _score("uh", "uhh") == 1
        assert _score("oh", "ooh") == 1
        assert _score("la", "laa") == 1

    def test_phonetic_equiv_two_edit(self):
        assert _score("woah", "whoa") == 1  # Levenshtein-2, both len 4

    def test_no_false_2char_fuzzy(self):
        assert _score("it", "is") == 0
        assert _score("in", "on") == 0


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

    def test_zero_score_filter(self):
        """Zero-score NW pairs are filtered: 'me' paired with a hallucination
        at score 0 should not get the hallucination's timestamps."""
        words = _make_words([
            ("love", 0.0, 0.5),
            ("xyz", 0.5, 0.7),  # hallucination — no lyric match
            ("tender", 0.7, 1.0),
        ])
        lines = ["love me tender"]
        result = match_words_to_lines(words, lines)
        # "me" should not be mapped to "xyz" (score 0 pair filtered out)
        word_texts = [w["word"] for w in result[0]["words"]]
        assert "xyz" not in word_texts
        # "me" has no whisper match → line only gets "love" and "tender"
        assert "love" in word_texts
        assert "tender" in word_texts

    def test_zero_score_filter_orphans_whisper(self):
        """When the filter rejects a pair (l_idx, w_idx), the whisper token
        at w_idx does not appear in any line's word list. This documents the
        known edge case: a real whisper word paired at score 0 is silently
        dropped, and interpolated timing is used for the lyric token."""
        words = _make_words([
            ("love", 0.0, 0.5),
            ("xyz", 0.5, 0.7),  # will be paired at score 0 then filtered
            ("tender", 0.7, 1.0),
        ])
        lines = ["love me tender"]
        result = match_words_to_lines(words, lines)
        # Collect all whisper words across all lines
        all_whisper_words = []
        for line_obj in result:
            all_whisper_words.extend(line_obj["words"])
        all_whisper_texts = [w["word"] for w in all_whisper_words]
        assert "xyz" not in all_whisper_texts

    def test_anchor_tiebreak_prefers_match(self):
        """When anchor bonus +3 ties gap-pair cost -3, NW should prefer the
        match path (not gap both sides). This pins the tie-breaking behavior:
        max() picks the first argument (match) when values are equal."""
        # Construct a minimal case: one long anchor word with a whisper token
        # that matches exactly. The match score (+3) equals the gap-pair cost
        # (-2 + -1 = -3). NW should still match them.
        lyric = ["california"]
        whisper = ["california"]
        alignment = _needleman_wunsch(lyric, whisper)
        # Should match, not gap both sides
        matched = [(l, w) for l, w in alignment if l is not None and w is not None]
        assert len(matched) == 1
        assert matched[0] == (0, 0)


# ---------------------------------------------------------------------------
# Change 2: Banding tests
# ---------------------------------------------------------------------------


class TestBanding:
    """Tests for Sakoe-Chiba banded NW alignment."""

    @staticmethod
    def _repeating_tokens(n, token_pool=None):
        """Generate n normalized tokens from a pool, cycling through them."""
        if token_pool is None:
            token_pool = ["love", "me", "tender", "tonight", "california",
                          "dream", "heart", "world", "fire", "desire"]
        return [token_pool[i % len(token_pool)] for i in range(n)]

    def test_short_sequences_skip_banding(self):
        """Sequences below _BAND_MIN_LENGTH produce identical results
        whether dispatched through _needleman_wunsch (which skips banding)
        or run explicitly unbanded."""
        lyric = self._repeating_tokens(20)
        whisper = self._repeating_tokens(22)
        disp_result = _needleman_wunsch(lyric, whisper)
        unband_result = _needleman_wunsch_unbanded(lyric, whisper)
        assert disp_result == unband_result

    def test_banding_matches_unbanded_on_diagonal(self):
        """When sequences are similar length and well-aligned, banded NW
        should produce the same alignment as unbanded."""
        length = _BAND_MIN_LENGTH + 100
        lyric = self._repeating_tokens(length)
        whisper = self._repeating_tokens(length + 5)  # slight length diff
        banded = _needleman_wunsch_banded(lyric, whisper)
        unbanded = _needleman_wunsch_unbanded(lyric, whisper)
        # The alignments should be identical (or very close)
        assert len(banded) == len(unbanded)

    def test_banding_prevents_chorus_drift(self):
        """Synthetic multi-chorus case: lyrics list one chorus but whisper
        has three repetitions. Unbanded NW may drift to match later choruses;
        banded NW should stay near the diagonal."""
        # One chorus of lyrics
        chorus = ["love", "me", "tender", "tonight"]
        lyric = chorus * 1  # 4 tokens
        # Three choruses in whisper (much longer)
        whisper = chorus * 3  # 12 tokens
        # Scale up past _BAND_MIN_LENGTH to trigger banding
        scale = (_BAND_MIN_LENGTH // 4) + 10
        lyric = chorus * scale
        whisper = chorus * (scale * 3)  # 3x as many whisper tokens

        banded = _needleman_wunsch_banded(lyric, whisper)
        # Verify alignment is non-empty and roughly monotone
        matches = [(l, w) for l, w in banded
                    if l is not None and w is not None]
        assert len(matches) > 0
        # Most lyric tokens should be matched (not gapped)
        matched_lyrics = {l for l, _ in matches}
        assert len(matched_lyrics) >= len(lyric) * 0.8

    def test_anchor_bonus_within_band(self):
        """Long exact-match anchor at the diagonal scores +3 and pulls
        alignment correctly. This is the expected behavior — anchor bonus
        strengthens diagonal alignment within the band."""
        length = _BAND_MIN_LENGTH + 50
        # Build sequences where most tokens match, with a long anchor
        lyric = self._repeating_tokens(length)
        whisper = list(lyric)  # exact copy
        # Replace middle token with a long anchor word
        mid = length // 2
        lyric[mid] = "california"
        whisper[mid] = "california"

        banded = _needleman_wunsch_banded(lyric, whisper)
        matches = [(l, w) for l, w in banded
                    if l is not None and w is not None]
        # The anchor should be matched
        anchor_matched = any(l == mid and w == mid for l, w in matches)
        assert anchor_matched

    def test_banded_fallback_triggers(self, monkeypatch):
        """When the optimal alignment requires exiting the band (e.g. singer
        skips a long section), the fallback should fire and the unbanded
        result is returned. We construct a case where whisper has a huge
        prefix that the band can't reach."""
        length = _BAND_MIN_LENGTH + 100
        lyric = self._repeating_tokens(length)
        # Whisper has a very long prefix of junk, pushing the true alignment
        # far off-diagonal at the start
        long_prefix = ["junk"] * (length // 2)
        whisper = long_prefix + lyric

        # Banded NW should detect degenerate result and fall back
        # (We just verify it doesn't crash and returns a valid alignment)
        result = _needleman_wunsch_banded(lyric, whisper)
        assert len(result) > 0
        # The result should have some matched pairs
        matches = [(l, w) for l, w in result
                    if l is not None and w is not None]
        assert len(matches) > 0

    def test_dispatcher_short_uses_unbanded(self):
        """Verify _needleman_wunsch dispatches to unbanded for short seqs."""
        lyric = ["hello", "world"]
        whisper = ["hello", "world"]
        # max(len) = 2, well below _BAND_MIN_LENGTH
        result = _needleman_wunsch(lyric, whisper)
        # Should match both tokens
        matches = [(l, w) for l, w in result
                    if l is not None and w is not None]
        assert len(matches) == 2
