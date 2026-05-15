"""Phase 1 orchestrator.

End-to-end flow:

  1. Interactive Genius search (top-10 picker) → lyrics + metadata.
  2. ffprobe the vocal stem for duration.
  3. LRCLIB /api/get → /api/search → duration gate.
  4. Save fetched lyrics next to the audio.
  5. Run the existing `genius_align` whole-song alignment as-is.

Phase 1 does **not** modify alignment behavior. The LRCLIB result is
fetched, logged, and saved as a sidecar JSON; Phase 2 will consume it
for chunking. Whether LRCLIB matches strongly, acceptably, or not at
all, the alignment that runs in Phase 1 is the same.

We invoke `genius_align` as a subprocess (`python -m genius_align ...`)
to honor the plan's "no edits to genius_align" boundary cleanly. Phase 2
will replace this with direct module calls when chunking lands.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from lrclib_align.lrclib import LrclibMatch, find_match, probe_duration
from lrclib_align.search import GeniusSelection, interactive_search

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lrclib_align.run")


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _save_lyrics(selection: GeniusSelection, vocal_path: Path) -> Path:
    """Write fetched lyrics next to the audio; return the path."""
    stem = vocal_path.stem
    lyrics_path = vocal_path.with_name(f"{stem}.genius.txt")
    lyrics_path.write_text(selection.lyrics, encoding="utf-8")
    logger.info("Lyrics saved: %s", lyrics_path)
    return lyrics_path


def _save_lrclib_sidecar(
    match: LrclibMatch | None,
    selection: GeniusSelection,
    file_duration: float,
    vocal_path: Path,
) -> Path:
    """Persist the Phase 1 bundle for Phase 2 to consume."""
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


def _run_genius_align(
    vocal_path: Path,
    lyrics_path: Path,
    match_method: str,
    save_whisper: bool,
) -> int:
    """Invoke `python -m genius_align <vocal> <lyrics>` as a subprocess.

    Reuses genius_align verbatim — Phase 1 boundary.
    """
    cmd = [
        sys.executable, "-m", "genius_align",
        str(vocal_path), str(lyrics_path),
        "--match-method", match_method,
    ]
    if save_whisper:
        cmd.append("--save-whisper")
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LRCLIB-anchored alignment (Phase 1: gate + sidecar; runs "
            "the existing genius_align whole-song alignment unchanged)."
        ),
    )
    parser.add_argument(
        "vocal_audio",
        type=Path,
        help="Path to vocal stem audio file (.wav, .m4a, etc.)",
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Initial Genius search query (else prompt interactively).",
    )
    parser.add_argument(
        "--match-method",
        choices=["auto", "walk", "tiling"],
        default="auto",
        help="Forwarded to genius_align (default: auto).",
    )
    parser.add_argument(
        "--save-whisper",
        action="store_true",
        help="Forwarded to genius_align: dump raw whisper result as JSON.",
    )
    parser.add_argument(
        "--skip-align",
        action="store_true",
        help="Stop after the gate + sidecar; don't run alignment.",
    )
    args = parser.parse_args()

    vocal_path: Path = args.vocal_audio
    if not vocal_path.exists():
        logger.error("Vocal file not found: %s", vocal_path)
        sys.exit(1)

    # 1. Genius search + lyrics fetch (interactive).
    selection = interactive_search(initial_query=args.query)

    # 2. Save lyrics for downstream consumers.
    lyrics_path = _save_lyrics(selection, vocal_path)

    # 3. Probe file duration.
    try:
        file_duration = probe_duration(vocal_path)
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError) as exc:
        logger.error("ffprobe failed on %s: %s", vocal_path, exc)
        sys.exit(1)
    logger.info("File duration: %.2fs", file_duration)

    # 4. LRCLIB lookup + duration gate.
    match = find_match(
        title=selection.hit.title,
        artist=selection.hit.artist,
        file_duration=file_duration,
    )
    if match is None:
        logger.info(
            "No usable LRCLIB match — Phase 2 would fall through to tier 2 "
            "(current whole-song behavior). Phase 1 always does that anyway."
        )
    else:
        synced = "yes" if match.synced_lyrics else "no"
        logger.info(
            "LRCLIB match via %s: id=%d synced=%s duration=%.2fs (Δ=%+.2fs)",
            match.source, match.track_id, synced, match.duration,
            match.duration_delta,
        )

    _save_lrclib_sidecar(match, selection, file_duration, vocal_path)

    if args.skip_align:
        logger.info("--skip-align set; stopping after gate.")
        return

    # 5. Hand off to genius_align unchanged.
    rc = _run_genius_align(
        vocal_path=vocal_path,
        lyrics_path=lyrics_path,
        match_method=args.match_method,
        save_whisper=args.save_whisper,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
