"""Unit tests for genius_align.tiling_match: parenthetical-phrase splitting
into match units and the order-independent tiling matcher."""

from genius_align.tiling_match import (
    _split_paren_units,
    match_words_to_lines_tiling,
)


def _words(seq, t0=0.0):
    """Build a whisper word list from a sequence of tokens."""
    return [
        {"word": w, "start": t0 + i, "end": t0 + i + 0.5}
        for i, w in enumerate(seq)
    ]


# ---------------------------------------------------------------------------
# _split_paren_units
# ---------------------------------------------------------------------------


class TestSplitParenUnits:
    def test_no_parens_single_unit(self):
        assert _split_paren_units("All the voices in my mind") == [
            "All the voices in my mind"
        ]

    def test_trailing_paren_splits(self):
        assert _split_paren_units(
            "All the voices in my mind (Tell me when it kicks in)"
        ) == ["All the voices in my mind", "Tell me when it kicks in"]

    def test_multiple_parens(self):
        assert _split_paren_units("foo (bar) baz (qux)") == [
            "foo baz",
            "bar",
            "qux",
        ]

    def test_paren_only_line(self):
        assert _split_paren_units("(Brokenhearted)") == ["Brokenhearted"]

    def test_empty_paren_ignored(self):
        assert _split_paren_units("hello ()") == ["hello"]


# ---------------------------------------------------------------------------
# match_words_to_lines_tiling — paren split integration
# ---------------------------------------------------------------------------


class TestTilingParenSplit:
    def test_paren_phrase_matches_independently(self):
        # Main phrase and parenthetical appear as separate runs in the
        # token stream, never contiguous. Splitting lets both match.
        words = _words([
            "all", "the", "voices", "in", "my", "mind",
            "xx", "yy",
            "tell", "me", "when", "it", "kicks", "in",
        ])
        lines = ["All the voices in my mind (Tell me when it kicks in)"]
        objs = match_words_to_lines_tiling(words, lines)
        assert sorted(o["text"] for o in objs) == [
            "All the voices in my mind",
            "Tell me when it kicks in",
        ]
        # Both sub-objects back-reference the single original line.
        assert all(o["line_id"] == 0 for o in objs)

    def test_unsplit_line_unaffected(self):
        words = _words(["hello", "world", "foo", "bar"])
        objs = match_words_to_lines_tiling(words, ["hello world"])
        assert len(objs) == 1
        assert objs[0]["text"] == "hello world"
        assert objs[0]["line_id"] == 0

    def test_only_main_phrase_present(self):
        # The parenthetical is absent from the audio — only the main
        # phrase matches; the paren unit is silently dropped.
        words = _words(["all", "the", "voices", "in", "my", "mind"])
        lines = ["All the voices in my mind (Tell me when it kicks in)"]
        objs = match_words_to_lines_tiling(words, lines)
        assert [o["text"] for o in objs] == ["All the voices in my mind"]
        assert objs[0]["line_id"] == 0
