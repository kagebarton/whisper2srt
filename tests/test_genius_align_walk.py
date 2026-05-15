"""Unit tests for genius_align.word_extraction: the two-pointer walk matcher.

Covers _walk_align's anchor-preserving biases (asymmetric lookahead,
confirmed re-sync, whisper-skip budget) and match_words_to_lines_walk's
interp cap (long unmatched runs dropped instead of interpolated).
"""

from genius_align.word_extraction import _walk_align, match_words_to_lines_walk


def _words(tokens_with_times):
    """Build a whisper word list from [(text, start, end), ...]."""
    return [
        {"word": t, "start": s, "end": e} for t, s, e in tokens_with_times
    ]


# ---------------------------------------------------------------------------
# _walk_align
# ---------------------------------------------------------------------------


class TestWalkAlign:
    def test_perfect_match(self):
        lyric = ["a", "b", "c"]
        whisper = ["a", "b", "c"]
        assert _walk_align(lyric, whisper) == [0, 1, 2]

    def test_whisper_lookahead_absorbs_long_hallucination_stretch(self):
        # 5 hallucinated whisper words before the real tokens — within the
        # default whisper_lookahead=10 even with confirmation.
        lyric = ["a", "b", "c", "d", "e"]
        whisper = ["x1", "x2", "x3", "x4", "x5", "a", "b", "c", "d", "e"]
        assert _walk_align(lyric, whisper) == [5, 6, 7, 8, 9]

    def test_desync_past_whisper_lookahead_is_lossy(self):
        # Beyond whisper_lookahead, the asymmetric blind-step kicks in
        # (j-only advance up to whisper_skip_budget, then i++). Lyric
        # tokens are preserved as None rather than mis-anchored.
        lyric = ["a", "b", "c"]
        whisper = ["x" + str(k) for k in range(40)] + ["a", "b", "c"]
        mapping = _walk_align(lyric, whisper)
        assert mapping.count(None) > 0

    def test_confirmed_resync_rejects_false_short_word_anchor(self):
        # During a desync, whisper has an unrelated "the" earlier than the
        # real one. Without confirmation the walker jumps to that spurious
        # "the"; with confirmation it requires the following token to also
        # match, forcing it to the correct "the" later in the stream.
        lyric = ["hello", "world", "the", "song", "ends"]
        whisper = [
            "hello",
            "noise1",
            "noise2",
            "the",
            "noise3",
            "world",
            "the",
            "song",
            "ends",
        ]
        mapping = _walk_align(lyric, whisper)
        assert mapping[0] == 0
        assert mapping[1] == 5  # "world" anchored after the noise
        assert mapping[2] == 6  # not the false "the" at whisper[3]

    def test_whisper_skip_budget_advances_past_missing_lyric_section(self):
        # Two lyric tokens have no audio counterpart (skipped section).
        # The whisper stream between "intro" and "ending" is longer than
        # whisper_lookahead, so the budget triggers and walks the lyric
        # pointer forward instead of bleeding lyric anchors blindly.
        lyric = ["intro", "miss1", "miss2", "ending"]
        whisper = ["intro"] + [f"x{k}" for k in range(20)] + ["ending"]
        mapping = _walk_align(lyric, whisper)
        assert mapping[0] == 0
        assert mapping[-1] == len(whisper) - 1
        assert mapping[1] is None
        assert mapping[2] is None


# ---------------------------------------------------------------------------
# match_words_to_lines_walk — interp cap
# ---------------------------------------------------------------------------


class TestMatchWordsToLinesWalk:
    def test_perfect_match(self):
        words = _words([
            ("you", 0.0, 0.3), ("are", 0.3, 0.5), ("my", 0.5, 0.7),
            ("fire", 0.7, 1.0),
        ])
        result = match_words_to_lines_walk(words, ["you are my fire"])
        assert len(result) == 1
        assert [w["word"] for w in result[0]["words"]] == [
            "you", "are", "my", "fire"
        ]

    def test_long_unmatched_run_is_dropped(self):
        # 3 unmatched tokens > max_interp_run=2 → drop the whole run.
        words = _words([("anchor1", 0.0, 1.0), ("anchor2", 10.0, 11.0)])
        lines = ["anchor1 g1 g2 g3 anchor2"]
        result = match_words_to_lines_walk(
            words, lines, lyric_lookahead=5, max_interp_run=2
        )
        emitted = [w["word"] for w in result[0]["words"]]
        assert emitted == ["anchor1", "anchor2"]

    def test_short_unmatched_run_still_interpolated(self):
        # 4 unmatched tokens <= cap=5 → still interpolated.
        words = _words([("anchor1", 0.0, 1.0), ("anchor2", 5.0, 6.0)])
        lines = ["anchor1 g1 g2 g3 g4 anchor2"]
        result = match_words_to_lines_walk(words, lines, lyric_lookahead=10)
        emitted = [w["word"] for w in result[0]["words"]]
        assert emitted == ["anchor1", "g1", "g2", "g3", "g4", "anchor2"]

    def test_line_with_all_tokens_dropped_is_suppressed(self):
        # A whole line falls inside a long unmatched run → words=[],
        # start=None. SRT/ASS generators skip lines with start=None.
        words = _words([("before", 0.0, 1.0), ("after", 20.0, 21.0)])
        lines = ["before", "lost1 lost2 lost3 lost4 lost5 lost6", "after"]
        result = match_words_to_lines_walk(words, lines, lyric_lookahead=10)
        assert result[1]["words"] == []
        assert result[1]["start"] is None
        assert result[1]["end"] is None

    def test_paren_only_line_still_inherits_after_cap(self):
        # The cap suppresses *lines that had tokens*; a paren-only line
        # (no normalizable tokens) keeps its neighbor-inherited timing.
        words = _words([("hello", 0.0, 1.0), ("world", 2.0, 3.0)])
        lines = ["hello", "(instrumental)", "world"]
        align_lines = ["hello", "", "world"]
        result = match_words_to_lines_walk(words, lines, align_lines)
        assert result[1]["words"] == []
        assert result[1]["start"] == 1.0
        assert result[1]["end"] == 2.0

    def test_interp_cap_can_be_disabled(self):
        # A huge cap restores the old gap-free behavior.
        words = _words([("anchor1", 0.0, 1.0), ("anchor2", 10.0, 11.0)])
        lines = ["anchor1 g1 g2 g3 g4 g5 g6 anchor2"]
        result = match_words_to_lines_walk(
            words, lines, lyric_lookahead=10, max_interp_run=1000
        )
        assert len(result[0]["words"]) == 8

    def test_crammed_interp_run_is_dropped(self):
        # Anchors nearly coincident in time with a short unmatched run
        # between them — well under max_interp_run, but interpolating would
        # give each token a ~0.03s slot. Drop instead of cramming.
        words = _words([("anchor1", 10.0, 10.1), ("anchor2", 10.2, 10.3)])
        lines = ["anchor1 g1 g2 g3 anchor2"]
        result = match_words_to_lines_walk(words, lines, lyric_lookahead=5)
        emitted = [w["word"] for w in result[0]["words"]]
        assert emitted == ["anchor1", "anchor2"]

    def test_crammed_check_can_be_disabled(self):
        # min_interp_slot=0.0 disables the degenerate-slot drop.
        words = _words([("anchor1", 10.0, 10.1), ("anchor2", 10.2, 10.3)])
        lines = ["anchor1 g1 g2 g3 anchor2"]
        result = match_words_to_lines_walk(
            words, lines, lyric_lookahead=5, min_interp_slot=0.0
        )
        emitted = [w["word"] for w in result[0]["words"]]
        assert emitted == ["anchor1", "g1", "g2", "g3", "anchor2"]

    def test_collapsed_matched_run_is_demoted_and_dropped(self):
        # align() couldn't locate a section: 10 matched words all stamped
        # at one instant. They pair 1:1 so the interp caps can't see them
        # — the collapsed-run pass demotes them, then max_interp_run drops
        # them. The line falls inside the run, so it's suppressed.
        collapsed = [(f"c{i}", 50.0, 50.0) for i in range(10)]
        words = _words(
            [("before", 0.0, 1.0)] + collapsed + [("after", 90.0, 91.0)]
        )
        lines = [
            "before",
            " ".join(f"c{i}" for i in range(10)),
            "after",
        ]
        result = match_words_to_lines_walk(words, lines, lyric_lookahead=20)
        assert [w["word"] for w in result[0]["words"]] == ["before"]
        assert result[1]["words"] == []
        assert result[1]["start"] is None
        assert [w["word"] for w in result[2]["words"]] == ["after"]

    def test_short_collapsed_run_is_kept(self):
        # 4 words sharing a timestamp is normal fast singing, not align()
        # failure — below max_collapsed_run, so left matched.
        words = _words([
            ("a", 0.0, 1.0),
            ("f1", 5.0, 5.0), ("f2", 5.0, 5.0), ("f3", 5.0, 5.0),
            ("f4", 5.0, 5.0),
            ("b", 9.0, 10.0),
        ])
        lines = ["a f1 f2 f3 f4 b"]
        result = match_words_to_lines_walk(words, lines, lyric_lookahead=10)
        emitted = [w["word"] for w in result[0]["words"]]
        assert emitted == ["a", "f1", "f2", "f3", "f4", "b"]

    def test_collapsed_run_check_can_be_disabled(self):
        # A huge max_collapsed_run keeps the old behavior: a long collapsed
        # run stays matched at its degenerate timestamps.
        collapsed = [(f"c{i}", 50.0, 50.0) for i in range(10)]
        words = _words(
            [("before", 0.0, 1.0)] + collapsed + [("after", 90.0, 91.0)]
        )
        lines = ["before " + " ".join(f"c{i}" for i in range(10)) + " after"]
        result = match_words_to_lines_walk(
            words, lines, lyric_lookahead=20, max_collapsed_run=1000
        )
        assert len(result[0]["words"]) == 12
