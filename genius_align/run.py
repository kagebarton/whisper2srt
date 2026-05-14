"""genius_align — unified CLI entry point.

Usage:
    python -m genius_align <vocal_audio> [<lyrics_file>]

Modes:
    Alignment (default when lyrics_file supplied):
        --match-method auto   (default): run the walk path, but if stable-ts
            align() reports too many failed segments
            (> cfg.align_failure_escalation), discard it and re-run with the
            tiling matcher on an honest transcription.
        --match-method walk:   stable-ts align() + two-pointer walk matcher
            with gap interpolation. Trusts word order; covers every line.
        --match-method tiling: stable-ts transcribe() + order-independent
            fuzzy candidate + interval-scheduling DP. Resilient to
            remixes/repeats/drift; may drop unmatched lines.
    Transcription (when lyrics_file omitted):
        Open-ended transcription with regrouping for natural line breaks.

Output files are written next to the vocal stem:
    <vocal_stem>.diarized.srt
    <vocal_stem>.diarized.ass
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# --- Bootstrap: sys.path before any package imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genius_align.config import GeniusAlignConfig
from genius_align.caption import generate_srt, generate_ass
from genius_align.genius import genius_singer_mode
from genius_align.word_extraction import (
    extract_words,
    load_genius_lyrics,
    match_words_to_lines_walk,
)
from genius_align.tiling_match import match_words_to_lines_tiling
from genius_align.workers.whisper_worker import WhisperWorker

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genius_align.run")


def assign_speakers_from_genius(line_objects, genius_lines):
    """Copy speaker_label and dominant_speaker from each genius_line onto
    the matching line_obj (and onto every word). Modifies in place.

    The walk matcher emits line_objects 1:1 with genius_lines, so a
    positional index works. The tiling matcher may drop or repeat lines
    and tags each object with a ``line_id`` back-reference — map by that
    when present.
    """
    for idx, line_obj in enumerate(line_objects):
        gl = genius_lines[line_obj.get("line_id", idx)]
        line_obj["speaker"] = gl["speaker_label"]
        line_obj["dominant_speaker"] = gl["dominant_speaker"]
        for word in line_obj["words"]:
            word["speaker"] = gl["speaker_label"]
            word["dominant_speaker"] = gl["dominant_speaker"]


def _segments_to_line_objects(result, min_prob: float = 0.0001) -> list:
    """Transcription mode: build one line_object per stable-ts segment."""
    line_objects = []
    dropped = 0
    for segment in result.segments:
        words = []
        for w in segment.words:
            prob = getattr(w, "probability", None)
            if prob is not None and prob < min_prob:
                dropped += 1
                continue
            words.append({
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "speaker": None,
                "dominant_speaker": None,
            })
        if words:
            start = words[0]["start"]
            end = words[-1]["end"]
        else:
            start = segment.start
            end = segment.end
        line_objects.append({
            "text": segment.text.strip(),
            "words": words,
            "start": start,
            "end": end,
        })
    if dropped:
        logger.info(
            "Dropped %d low-probability whisper words (< %.4f)",
            dropped, min_prob,
        )
    return line_objects


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate diarized .srt + .ass captions from Genius headers."
    )
    parser.add_argument(
        "vocal_audio",
        type=Path,
        help="Path to vocal stem audio file (.wav, .m4a, etc.)",
    )
    parser.add_argument(
        "lyrics_file",
        nargs="?",
        type=Path,
        default=None,
        help="Genius-formatted lyrics file (.txt) with section headers. Omit for transcription mode.",
    )
    parser.add_argument(
        "--match-method",
        choices=["auto", "walk", "tiling"],
        default="auto",
        help=(
            "Lyric→token matcher (alignment mode only). "
            "'auto' (default): run 'walk', but escalate to 'tiling' if "
            "stable-ts align() fails more than cfg.align_failure_escalation "
            "of its segments. "
            "'walk': runs stable-ts align() then a two-pointer lockstep "
            "matcher that trusts word order and covers every lyric line. "
            "'tiling': runs stable-ts transcribe() then an order-independent "
            "fuzzy candidate + interval-scheduling DP — resilient to "
            "remixes/repeats/drift, may drop unmatched lines."
        ),
    )
    parser.add_argument(
        "--save-whisper",
        action="store_true",
        help="Save raw Whisper alignment result as <stem>.whisper.json next to the audio file.",
    )
    args = parser.parse_args()

    vocal_path = args.vocal_audio
    lyrics_path = args.lyrics_file
    is_transcribe_mode = lyrics_path is None

    if not vocal_path.exists():
        logger.error(f"Vocal file not found: {vocal_path}")
        sys.exit(1)
    if not is_transcribe_mode and not lyrics_path.exists():
        logger.error(f"Lyrics file not found: {lyrics_path}")
        sys.exit(1)

    cfg = GeniusAlignConfig()
    logger.info(
        f"Mode: {'transcription' if is_transcribe_mode else args.match_method}"
    )

    # --- Parse lyrics (single source of truth) ---
    if is_transcribe_mode:
        genius_lines = []
        plain_text = ""
        mode = "solo"
        logger.info("Transcription mode — no lyrics file supplied")
    else:
        lyrics_text = lyrics_path.read_text(encoding="utf-8")
        genius_lines, plain_text = load_genius_lyrics(lyrics_text)
        mode = genius_singer_mode(genius_lines)
        logger.info(f"Genius singer mode: {mode}")

    # --- Load whisper model ---
    logger.info("Loading whisper model...")
    whisper_worker = WhisperWorker(cfg.whisper)
    t0 = time.time()
    whisper_worker.load_model()
    logger.info(f"Whisper model loaded in {time.time() - t0:.1f}s")

    try:
        if is_transcribe_mode:
            # --- Transcribe pipeline: transcribe → regroup → refine ---
            logger.info(f"Transcribing: {vocal_path.name}")
            result = whisper_worker.transcribe(vocal_path)
            whisper_worker.regroup(result)
            result = whisper_worker.refine(vocal_path, result)
            line_objects = _segments_to_line_objects(
                result, min_prob=cfg.whisper.post_process.min_word_probability
            )
        else:
            # --- Align ---
            lyrics_lines = [g["text"] for g in genius_lines]
            align_lines = [g["align_text"] for g in genius_lines]
            min_prob = cfg.whisper.post_process.min_word_probability

            use_tiling = args.match_method == "tiling"

            if not use_tiling:
                # walk / auto: run align() first and gate on its failure
                # ratio *before* refine — refine is the expensive stage and
                # is wasted work if we're about to discard the align result.
                logger.info(
                    f"Aligning lyrics: {lyrics_path.name} → {vocal_path.name}"
                )
                result = whisper_worker.align(vocal_path, plain_text)
                fail_ratio = whisper_worker.last_align_failure_ratio
                if (
                    args.match_method == "auto"
                    and fail_ratio > cfg.align_failure_escalation
                ):
                    logger.warning(
                        "align() failed %.0f%% of segments (> %.0f%% threshold) "
                        "— escalating to tiling matcher (skipping refine)",
                        fail_ratio * 100,
                        cfg.align_failure_escalation * 100,
                    )
                    use_tiling = True
                else:
                    if args.match_method == "auto":
                        logger.info(
                            "align() failure ratio %.0f%% within threshold "
                            "— keeping walk match",
                            fail_ratio * 100,
                        )
                    # walk pipeline: refine → postprocess → walk match
                    result = whisper_worker.refine(vocal_path, result)
                    result = whisper_worker.postprocess(result)
                    words = extract_words(result, min_prob=min_prob)
                    line_objects = match_words_to_lines_walk(
                        words, lyrics_lines, align_lines
                    )

            if use_tiling:
                # Tiling is order-independent and drops unmatched lines, so it
                # runs on an honest transcription rather than align() output.
                # align() force-places every reference token, collapsing
                # unalignable lyrics into zero-duration slots — feeding that
                # into the tiling matcher would just launder corrupted timing.
                # tiling pipeline: transcribe → regroup → refine → tiling match
                logger.info(f"Transcribing for tiling match: {vocal_path.name}")
                result = whisper_worker.transcribe(vocal_path)
                whisper_worker.regroup(result)
                result = whisper_worker.refine(vocal_path, result)
                words = extract_words(result, min_prob=min_prob)
                line_objects = match_words_to_lines_tiling(
                    words, lyrics_lines, align_lines
                )

            # --- original_split=True path (preserved for reference) ---
            # Requires strict 1:1 segment→line mapping; uses
            # ``align_for_strict_split`` to skip merge/adjust post-processing.
            # from genius_align.segment_lines import (
            #     align_for_strict_split,
            #     segments_to_line_objects_strict,
            # )
            # result = align_for_strict_split(whisper_worker, vocal_path, plain_text)
            # line_objects = segments_to_line_objects_strict(
            #     result,
            #     genius_lines,
            #     min_prob=cfg.whisper.post_process.min_word_probability,
            # )

            if mode == "multi":
                assign_speakers_from_genius(line_objects, genius_lines)
                logger.info("Speaker assignment complete (Genius headers only)")
            else:
                logger.info(
                    "Solo mode — no labeled sections. "
                    "Output will be plain karaoke (no speaker labels)."
                )

        if args.save_whisper:
            whisper_json = vocal_path.with_name(vocal_path.stem + ".whisper.json")
            result.save_as_json(str(whisper_json))
            logger.info(f"Whisper result saved: {whisper_json}")

        # --- Write outputs ---
        srt_out = vocal_path.with_suffix(".diarized.srt")
        ass_out = vocal_path.with_suffix(".diarized.ass")

        srt_content = generate_srt(line_objects, cfg)
        srt_out.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT written: {srt_out}")

        ass_content = generate_ass(line_objects, cfg)
        ass_out.write_text(ass_content, encoding="utf-8")
        logger.info(f"ASS written: {ass_out}")

    finally:
        whisper_worker.unload_model()
        logger.info("Models unloaded.")


if __name__ == "__main__":
    main()
