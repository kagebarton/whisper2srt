"""LRCLIB client + duration gate.

Inline `requests`-based wrapper for the two LRCLIB endpoints we need:

  - GET /api/get?track_name=&artist_name=&album_name=&duration=
  - GET /api/search?track_name=&artist_name=

We don't use the published `lrclib-python` library because all released
versions use Python 3.12+ f-string syntax that fails to import on 3.11
(the `pik` env), even though the package metadata claims ≥3.8.

`find_match()` runs a **duration sweep** over `/api/get`. The endpoint
is duration-keyed with ~±2s buckets per call: probing different
`duration` values returns *different* indexed recordings. Sweeping a
small window around the file's true duration turns `/api/get` into a
range query and surfaces live/remix/cut mixes that the `/api/search`
endpoint buries (it caps at 20 results ranked by popularity and
silently ignores any `duration` param). `/api/search` is reserved as a
backstop for the case where the sweep finds nothing at all under the
Genius-picked title + artist.

Returns:
  - LrclibMatch — closest swept (or backstop) row, Δ ≤ ACCEPTABLE_DELTA
  - None — sweep + backstop produced nothing within tolerance; caller
    falls through to tier 2 (current genius_align whole-song behavior)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


# Duration sweep parameters (seconds).
#
# `/api/get` bucket width is ~±2s, so a 2s step samples every distinct
# bucket inside the window. ±10s with step 2 = 11 probes per sweep,
# wide enough to catch live cuts / radio edits that differ from the
# canonical recording by ≤10s of intro/outro/applause.
SWEEP_WINDOW = 10
SWEEP_STEP = 2
ACCEPTABLE_DELTA = 10.0  # |row.duration - file.duration| ≤ this = match

# Artist-variant escalation. When the primary sweep under the
# Genius-picked artist returns no rows, search by title-only and re-sweep
# each related-artist variant (e.g. "Ed Sheeran" → "Ed Sheeran &
# Rudimental"). Capped to keep HTTP volume bounded.
MAX_ARTIST_VARIANTS = 4

_BASE_URL = "https://lrclib.net/api"
_USER_AGENT = "lrclib_align/0.1 (whisper2srt; https://github.com/)"
# LRCLIB can be slow under load (observed 10–15s response times); a tight
# timeout starves all probes when the server is under stress, so allow
# headroom even though that means a stuck probe blocks the sweep.
_HTTP_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------

# Trailing parenthetical or bracketed group that's metadata, not part of the
# canonical title (feat. X, with Y, ...Remaster, Live, Acoustic, Remix, etc.).
_PAREN_META_RE = re.compile(
    r"""\s*[\(\[]\s*
        (?:feat\.?|featuring|ft\.?|with|
           live(?:\s+at\b[^)\]]*)?|
           acoustic|remix(?:ed)?|
           remaster(?:ed)?(?:\s*\d{2,4})?|
           \d{2,4}\s*remaster(?:ed)?|
           single\s+version|radio\s+edit|version|edit)\b
        [^)\]]*[\)\]]\s*""",
    re.IGNORECASE | re.VERBOSE,
)

# Trailing " - <metadata>" tail (Spotify-style).
_DASH_META_RE = re.compile(
    r"""\s*-\s*
        (?:remaster(?:ed)?(?:\s*\d{2,4})?|
           \d{2,4}\s*remaster(?:ed)?|
           live(?:\s+at\b.*)?|single\s+version|
           radio\s+edit|acoustic|remix(?:ed)?)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Inline "feat. X" / "ft. X" anywhere in the string.
_INLINE_FEAT_RE = re.compile(
    r"\s*\b(?:feat\.?|ft\.?|featuring)\b[^,(\[]*",
    re.IGNORECASE,
)


def normalize_title(title: str) -> str:
    """Strip release-variant noise from a track title.

    Applied to BOTH the Genius title and any LRCLIB candidate title
    before comparing as strings; duration is the disambiguator when
    strings still disagree.
    """
    s = title.strip()
    # Apply repeatedly — a title like "Foo (feat. A) - 2009 Remaster" has
    # both a paren group and a dash tail.
    for _ in range(3):
        new = _PAREN_META_RE.sub(" ", s)
        new = _DASH_META_RE.sub("", new)
        new = _INLINE_FEAT_RE.sub("", new)
        if new == s:
            break
        s = new
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_artist(artist: str) -> str:
    """Light artist normalization: drop 'feat.'/'ft.' tails."""
    s = _INLINE_FEAT_RE.sub("", artist)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# Split a collab artist string into individual lowercase substrings,
# e.g. "Ed Sheeran & Rudimental" → ["ed sheeran", "rudimental"].
_ARTIST_SPLIT_RE = re.compile(r"\s*(?:&|,|;|\+|/|\bvs\.?\b|\bx\b)\s*", re.IGNORECASE)


def _artist_pieces(artist: str) -> list[str]:
    s = _INLINE_FEAT_RE.sub("", artist)
    return [p.strip().lower() for p in _ARTIST_SPLIT_RE.split(s) if p.strip()]


def _is_related_artist(candidate: str, base_pieces: list[str]) -> bool:
    """True if `candidate` shares any of the base artist's pieces."""
    c = candidate.lower()
    return any(p in c for p in base_pieces)


# ---------------------------------------------------------------------------
# Duration probe
# ---------------------------------------------------------------------------


def probe_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(audio_path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(out.stdout)
    return float(data["format"]["duration"])


# ---------------------------------------------------------------------------
# LRCLIB API
# ---------------------------------------------------------------------------


@dataclass
class LrclibMatch:
    """A successful LRCLIB lookup.

    `synced_lyrics` is the raw LRC string (lines of `[mm:ss.xx] text`)
    or None if the track has only plain lyrics. Phase 2 needs the synced
    timestamps; Phase 1 just logs whether they're available.
    """

    track_id: int
    track_name: str
    artist_name: str
    album_name: str | None
    duration: float
    synced_lyrics: str | None
    plain_lyrics: str | None
    instrumental: bool
    duration_delta: float  # signed: file - lrclib
    source: str  # "get" or "search"


def _request_get(path: str, params: dict) -> dict | None:
    """GET path?params → JSON dict, or None on 404."""
    url = f"{_BASE_URL}{path}"
    r = requests.get(
        url,
        params=params,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=_HTTP_TIMEOUT,
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _request_search(params: dict) -> list[dict]:
    url = f"{_BASE_URL}/search"
    r = requests.get(
        url,
        params=params,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=_HTTP_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def lrclib_get(
    *,
    track_name: str,
    artist_name: str,
    duration: float,
    album_name: str | None = None,
) -> dict | None:
    """GET /api/get — exact match by signature. None on 404."""
    params = {
        "track_name": track_name,
        "artist_name": artist_name,
        "duration": int(round(duration)),
    }
    if album_name:
        params["album_name"] = album_name
    return _request_get("/get", params)


def lrclib_search(
    *,
    track_name: str | None = None,
    artist_name: str | None = None,
    query: str | None = None,
) -> list[dict]:
    """GET /api/search — fuzzy search returning a list of candidates."""
    params: dict[str, str] = {}
    if query:
        params["q"] = query
    if track_name:
        params["track_name"] = track_name
    if artist_name:
        params["artist_name"] = artist_name
    return _request_search(params)


def _to_match(row: dict, file_duration: float, source: str) -> LrclibMatch:
    return LrclibMatch(
        track_id=int(row.get("id", 0)),
        track_name=row.get("trackName") or row.get("name") or "",
        artist_name=row.get("artistName") or "",
        album_name=row.get("albumName"),
        duration=float(row.get("duration") or 0.0),
        synced_lyrics=row.get("syncedLyrics"),
        plain_lyrics=row.get("plainLyrics"),
        instrumental=bool(row.get("instrumental", False)),
        duration_delta=file_duration - float(row.get("duration") or 0.0),
        source=source,
    )


def _sweep_get(
    *,
    track_name: str,
    artist_name: str,
    file_duration: float,
    album_name: str | None,
) -> list[dict]:
    """Probe `/api/get` across a duration window; return unique rows.

    Each `/api/get` call returns the indexed recording whose duration is
    closest to the queried value, within a narrow (~±2s) bucket. Walking
    the duration param therefore enumerates the duration-adjacent
    recordings LRCLIB has under this title/artist.
    """
    center = int(round(file_duration))
    probes = range(center - SWEEP_WINDOW, center + SWEEP_WINDOW + 1, SWEEP_STEP)
    seen: dict[int, dict] = {}
    for d in probes:
        try:
            row = lrclib_get(
                track_name=track_name,
                artist_name=artist_name,
                duration=float(d),
                album_name=album_name,
            )
        except requests.RequestException as exc:
            logger.warning("LRCLIB /api/get d=%d failed: %s", d, exc)
            continue
        if row and row.get("id") is not None:
            seen.setdefault(int(row["id"]), row)
    return list(seen.values())


def _pick_closest(
    rows: list[dict], file_duration: float
) -> tuple[float, dict] | None:
    """Return (|Δ|, row) for the row with duration closest to file."""
    scored = [
        (abs(file_duration - float(r.get("duration") or 0.0)), r)
        for r in rows
        if r.get("duration")
    ]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0]


def _discover_artist_variants(
    *, track_name: str, base_artist: str
) -> list[str]:
    """Title-only search → distinct artist strings related to `base_artist`.

    "Related" means the candidate artist contains any piece of the base
    (e.g. base "Ed Sheeran" → matches "Ed Sheeran & Rudimental",
    "Rudimental feat. Ed Sheeran", "Sheeran, Ed"). Capped at
    MAX_ARTIST_VARIANTS.
    """
    base_pieces = _artist_pieces(base_artist)
    if not base_pieces:
        return []
    try:
        hits = lrclib_search(track_name=track_name)
    except requests.RequestException as exc:
        logger.warning("LRCLIB variant discovery failed: %s", exc)
        return []
    seen: set[str] = {base_artist.lower()}
    variants: list[str] = []
    for h in hits:
        name = (h.get("artistName") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        if _is_related_artist(key, base_pieces):
            seen.add(key)
            variants.append(name)
            if len(variants) >= MAX_ARTIST_VARIANTS:
                break
    return variants


def find_match(
    *,
    title: str,
    artist: str,
    file_duration: float,
    album: str | None = None,
) -> LrclibMatch | None:
    """Duration-sweep `/api/get` under the Genius-picked artist; on empty,
    discover related artist variants via title-only search and sweep each;
    on still-empty, backstop with `/api/search`. Returns the
    closest-duration row within ACCEPTABLE_DELTA, else None.
    """
    norm_title = normalize_title(title)
    norm_artist = normalize_artist(artist)

    rows = _sweep_get(
        track_name=norm_title,
        artist_name=norm_artist,
        file_duration=file_duration,
        album_name=album,
    )
    source = "sweep"

    def have_within_tolerance() -> bool:
        p = _pick_closest(rows, file_duration)
        return p is not None and p[0] <= ACCEPTABLE_DELTA

    # Escalate to variants whenever the primary sweep didn't land a row
    # inside the tolerance band — an empty result and an out-of-tolerance
    # result both mean "didn't find the right recording under this artist."
    if not have_within_tolerance():
        primary_best = _pick_closest(rows, file_duration)
        logger.info(
            "LRCLIB primary sweep miss under '%s' / '%s' (%s) — probing artist variants",
            norm_title, norm_artist,
            "empty" if primary_best is None
            else f"closest Δ={primary_best[0]:.2f}s",
        )
        variants = _discover_artist_variants(
            track_name=norm_title, base_artist=norm_artist
        )
        rows_before = len(rows)
        for variant in variants:
            v_rows = _sweep_get(
                track_name=norm_title,
                artist_name=variant,
                file_duration=file_duration,
                album_name=album,
            )
            logger.info(
                "  variant '%s' → %d row%s",
                variant, len(v_rows), "" if len(v_rows) == 1 else "s",
            )
            rows.extend(v_rows)
            if have_within_tolerance():
                logger.info(
                    "  within-tolerance row found — skipping remaining variants",
                )
                break
        if len(rows) > rows_before:
            source = "variant-sweep"

    # Final backstop: free-text search under the primary title/artist.
    if not have_within_tolerance():
        logger.info(
            "Still no within-tolerance row — trying /api/search backstop"
        )
        try:
            backstop = lrclib_search(
                track_name=norm_title, artist_name=norm_artist
            )
        except requests.RequestException as exc:
            logger.warning("LRCLIB /api/search failed: %s", exc)
            backstop = []
        if backstop:
            rows.extend(backstop)
            source = "search"

    picked = _pick_closest(rows, file_duration)
    if picked is None:
        logger.info("LRCLIB: no candidates found")
        return None
    best_delta, best = picked

    if best_delta > ACCEPTABLE_DELTA:
        logger.info(
            "LRCLIB %s best candidate off by %.2fs (>%.1fs) — weak match, "
            "falling through to tier 2",
            source, best_delta, ACCEPTABLE_DELTA,
        )
        return None

    m = _to_match(best, file_duration, source=source)
    logger.info(
        "LRCLIB %s hit: %s — %s (Δ=%+.2fs, %d candidate%s scanned)",
        source, m.artist_name, m.track_name, m.duration_delta,
        len(rows), "" if len(rows) == 1 else "s",
    )
    return m
