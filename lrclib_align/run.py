"""LRCLIB-anchored alignment orchestrator.

Flow:

  1. Interactive Genius search (top-10 picker) → lyrics + metadata.
  2. ffprobe the vocal stem for duration.
  3. LRCLIB lookup + duration gate; save sidecar JSON.
  4. **Phase 2 path** — LRCLIB match with synced lyrics:
       reconcile lines, derive section chunks, run per-chunk
       `align()` (with per-chunk auto-escalation to tiling),
       stitch back to absolute time.
  5. **Tier-2 fallback** — no usable match (or `--match-method tiling`):
       run the whole-song alignment, identical in behavior to
       `python -m genius_align`.

Phase 2 imports `genius_align` modules (whisper worker, matchers,
caption emitters) directly. The "no edits to genius_align" boundary
still holds — we only *import* it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from genius_align.caption import generate_ass, generate_srt
from genius_align.config import GeniusAlignConfig
from genius_align.genius import genius_singer_mode
from genius_align.run import assign_speakers_from_genius
from genius_align.tiling_match import match_words_to_lines_tiling
from genius_align.word_extraction import (
    extract_words,
    load_genius_lyrics,
    match_words_to_lines_walk,
)
from genius_align.workers.whisper_worker import WhisperWorker

from lrclib_align.chunk import Chunk, cut_audio, offset_line_objects, plan_chunks
from lrclib_align.lrc import parse_lrc
from lrclib_align.lrclib import LrclibMatch, find_match, probe_duration
from lrclib_align.reconcile import reconcile
from lrclib_align.search import GeniusSelection, interactive_search

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lrclib_align.run")


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


def _save_lyrics(selection: GeniusSelection, vocal_path: Path) -> Path:
    lyrics_path = vocal_path.with_name(f"{vocal_path.stem}.genius.txt")
    lyrics_path.write_text(selection.lyrics, encoding="utf-8")
    logger.info("Lyrics saved: %s", lyrics_path)
    return lyrics_path


def _save_lrclib_sidecar(
    match: LrclibMatch | None,
    selection: GeniusSelection,
    file_duration: float,
    vocal_path: Path,
) -> Path:
    sidecar = vocal_path.with_name(f"{vocal_path.stem}.lrclib.json")
    payload = {
        "genius": {
            "song_id": selection.hit.song_id,
            "title": selection.hit.title,
            "artist": selection.hit.artist,
            "url": selection.hit.url,
        },
        "file_duration": file_duration,
        "lrclib": dataclasses.asdict(match) if match is not None else None,
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("LRCLIB sidecar saved: %s", sidecar)
    return sidecar


# ---------------------------------------------------------------------------
# Shared align helper: align/tiling switch — used by chunked and whole-song
# ---------------------------------------------------------------------------


def _align_segment(
    whisper_worker: WhisperWorker,
    cfg: GeniusAlignConfig,
    audio_path: Path,
    plain_text: str,
    lyrics_lines: list[str],
    align_lines: list[str],
    match_method: str,
    label: str,
) -> list[dict]:
    """Mirror of genius_align.run alignment switch, returning line_objects.

    Side effects: prints align-failure-ratio decisions.

    Behavior is identical to genius_align — only difference is that
    when called per chunk, `audio_path` is a bounded temp WAV.
    """
    min_prob = cfg.whisper.post_process.min_word_probability
    use_tiling = match_method == "tiling"

    if not use_tiling:
        logger.info("[%s] align()", label)
        result = whisper_worker.align(audio_path, plain_text)
        fail_ratio = whisper_worker.last_align_failure_ratio
        if (
            match_method == "auto"
            and fail_ratio > cfg.align_failure_escalation
        ):
            logger.warning(
                "[%s] align() failed %.0f%% (> %.0f%%) — escalating to tiling",
                label, fail_ratio * 100, cfg.align_failure_escalation * 100,
            )
            use_tiling = True
        else:
            if match_method == "auto":
                logger.info(
                    "[%s] align() failure ratio %.0f%% within threshold",
                    label, fail_ratio * 100,
                )
            result = whisper_worker.refine(audio_path, result)
            result = whisper_worker.postprocess(result)
            words = extract_words(result, min_prob=min_prob)
            return match_words_to_lines_walk(words, lyrics_lines, align_lines)

    logger.info("[%s] transcribe → tiling", label)
    result = whisper_worker.transcribe(audio_path)
    whisper_worker.regroup(result)
    result = whisper_worker.refine(audio_path, result)
    words = extract_words(result, min_prob=min_prob)
    return match_words_to_lines_tiling(words, lyrics_lines, align_lines)


# ---------------------------------------------------------------------------
# Phase 2: chunked alignment
# ---------------------------------------------------------------------------


def _run_chunked(
    whisper_worker: WhisperWorker,
    cfg: GeniusAlignConfig,
    vocal_path: Path,
    genius_lines: list[dict],
    match: LrclibMatch,
    file_duration: float,
    match_method: str,
) -> list[dict]:
    lrc_lines = parse_lrc(match.synced_lyrics or "")
    reconciled = reconcile(genius_lines, lrc_lines)
    chunks = plan_chunks(genius_lines, reconciled, file_duration)
    if not chunks:
        logger.warning("Phase 2: no usable chunks after reconcile — falling back")
        return []

    all_line_objects: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="lrclib_align_") as tmpdir:
        tmp = Path(tmpdir)
        for i, chunk in enumerate(chunks):
            chunk_wav = tmp / f"chunk_{i:02d}.wav"
            cut_audio(vocal_path, chunk.chunk_start, chunk.chunk_end, chunk_wav)

            chunk_genius = [genius_lines[gi] for gi in chunk.genius_idxs]
            lyrics_lines = [g["text"] for g in chunk_genius]
            align_lines = [g["align_text"] for g in chunk_genius]
            plain_text = "\n".join(align_lines)

            label = f"chunk {i+1}/{len(chunks)} {chunk.section_name[:24]}"
            t0 = time.time()
            line_objects = _align_segment(
                whisper_worker, cfg, chunk_wav, plain_text,
                lyrics_lines, align_lines, match_method, label,
            )
            logger.info("[%s] aligned %d lines in %.1fs",
                        label, len(line_objects), time.time() - t0)

            offset_line_objects(line_objects, chunk.chunk_start)
            for j, lo in enumerate(line_objects):
                src_idx = lo.get("line_id", j)
                if 0 <= src_idx < len(chunk.genius_idxs):
                    lo["line_id"] = chunk.genius_idxs[src_idx]

            all_line_objects.extend(line_objects)

    logger.info("Phase 2: %d total line_objects across %d chunks",
                len(all_line_objects), len(chunks))
    return all_line_objects


# ---------------------------------------------------------------------------
# Tier 2 fallback: whole-song behavior
# ---------------------------------------------------------------------------


def _run_whole_song(
    whisper_worker: WhisperWorker,
    cfg: GeniusAlignConfig,
    vocal_path: Path,
    genius_lines: list[dict],
    plain_text: str,
    match_method: str,
) -> list[dict]:
    lyrics_lines = [g["text"] for g in genius_lines]
    align_lines = [g["align_text"] for g in genius_lines]
    return _align_segment(
        whisper_worker, cfg, vocal_path, plain_text,
        lyrics_lines, align_lines, match_method, "whole-song",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LRCLIB-anchored chunked alignment.",
    )
    parser.add_argument("vocal_audio", type=Path,
                        help="Path to vocal stem audio file.")
    parser.add_argument("--query", "-q", default=None,
                        help="Initial Genius search query.")
    parser.add_argument("--match-method", choices=["auto", "walk", "tiling"],
                        default="auto",
                        help="'tiling' bypasses the LRCLIB chunked path entirely.")
    parser.add_argument("--save-whisper", action="store_true",
                        help="(whole-song path only) dump raw whisper result.")
    parser.add_argument("--skip-align", action="store_true",
                        help="Stop after sidecar; don't run alignment.")
    parser.add_argument("--force-whole-song", action="store_true",
                        help="Skip Phase 2 chunking even if LRCLIB matched.")
    args = parser.parse_args()

    vocal_path: Path = args.vocal_audio
    if not vocal_path.exists():
        logger.error("Vocal file not found: %s", vocal_path)
        sys.exit(1)

    selection = interactive_search(initial_query=args.query)
    lyrics_path = _save_lyrics(selection, vocal_path)

    try:
        file_duration = probe_duration(vocal_path)
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError) as exc:
        logger.error("ffprobe failed on %s: %s", vocal_path, exc)
        sys.exit(1)
    logger.info("File duration: %.2fs", file_duration)

    match = find_match(
        title=selection.hit.title,
        artist=selection.hit.artist,
        file_duration=file_duration,
    )
    if match is None:
        logger.info("No usable LRCLIB match — using whole-song fallback.")
    else:
        logger.info(
            "LRCLIB match via %s: id=%d synced=%s duration=%.2fs (Δ=%+.2fs)",
            match.source, match.track_id,
            "yes" if match.synced_lyrics else "no",
            match.duration, match.duration_delta,
        )

    _save_lrclib_sidecar(match, selection, file_duration, vocal_path)

    if args.skip_align:
        logger.info("--skip-align set; stopping after gate.")
        return

    use_chunked = (
        match is not None
        and bool(match.synced_lyrics)
        and not match.instrumental
        and args.match_method != "tiling"
        and not args.force_whole_song
    )
    if args.match_method == "tiling":
        logger.info("--match-method=tiling overrides chunked path.")
    if args.force_whole_song:
        logger.info("--force-whole-song set — chunked path bypassed.")

    cfg = GeniusAlignConfig()
    lyrics_text = lyrics_path.read_text(encoding="utf-8")
    genius_lines, plain_text = load_genius_lyrics(lyrics_text)
    mode = genius_singer_mode(genius_lines)
    logger.info("Genius singer mode: %s", mode)

    logger.info("Loading whisper model...")
    whisper_worker = WhisperWorker(cfg.whisper)
    t0 = time.time()
    whisper_worker.load_model()
    logger.info("Whisper model loaded in %.1fs", time.time() - t0)

    try:
        if use_chunked:
            line_objects = _run_chunked(
                whisper_worker, cfg, vocal_path, genius_lines,
                match, file_duration, args.match_method,
            )
            if not line_objects:
                logger.warning("Chunked path produced nothing — falling back to whole-song")
                line_objects = _run_whole_song(
                    whisper_worker, cfg, vocal_path, genius_lines,
                    plain_text, args.match_method,
                )
        else:
            line_objects = _run_whole_song(
                whisper_worker, cfg, vocal_path, genius_lines, plain_text,
                args.match_method,
            )

        if mode == "multi":
            assign_speakers_from_genius(line_objects, genius_lines)
            logger.info("Speaker assignment complete (Genius headers only).")
        else:
            logger.info("Solo mode — output will be plain karaoke (no speaker labels).")

        srt_out = vocal_path.with_suffix(".diarized.srt")
        ass_out = vocal_path.with_suffix(".diarized.ass")
        srt_out.write_text(generate_srt(line_objects, cfg), encoding="utf-8")
        logger.info("SRT written: %s", srt_out)
        ass_out.write_text(generate_ass(line_objects, cfg), encoding="utf-8")
        logger.info("ASS written: %s", ass_out)
    finally:
        whisper_worker.unload_model()
        logger.info("Models unloaded.")


if __name__ == "__main__":
    main()
