from __future__ import annotations

import configparser
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pikaraoke.lib.metadata_parser import regex_tidy  # noqa: E402

import lyricsgenius
from flask import Flask, jsonify, render_template, request

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
app = Flask(__name__)

_YT_DLP = [sys.executable, "-m", "yt_dlp"]

_config = configparser.ConfigParser()
_config.read(Path(__file__).parent / "config.ini")
_GENIUS_TOKEN = _config.get("genius", "api_token", fallback=None)
if _GENIUS_TOKEN and _GENIUS_TOKEN != "YOUR_GENIUS_API_TOKEN_HERE":
    _genius = lyricsgenius.Genius(_GENIUS_TOKEN, timeout=10)
    _genius.remove_section_headers = False
    _genius.skip_non_songs = True
else:
    _genius = None


def get_search_results(query: str) -> list[tuple[str, str, str, str, str]]:
    """Search YouTube via yt-dlp.

    Returns list of (title, url, video_id, channel, duration_str).
    """
    cmd = _YT_DLP + ["-j", "--no-playlist", "--flat-playlist", f'ytsearch10:"{query}"']
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8", "ignore")
    except subprocess.CalledProcessError as e:
        logging.warning(f"yt-dlp search failed: {e}")
        return []

    results = []
    for line in output.splitlines():
        if len(line) <= 2:
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "title" not in j or "url" not in j:
            continue
        channel = j.get("channel") or j.get("uploader") or ""
        duration_raw = j.get("duration")
        duration_str = ""
        if isinstance(duration_raw, (int, float)):
            s = int(duration_raw)
            duration_str = f"{s // 60}:{s % 60:02d}"
        results.append((j["title"], j["url"], j["id"], channel, duration_str))
    return results


def get_stream_url(video_url: str) -> str | None:
    """Return a direct playable stream URL without downloading."""
    cmd = _YT_DLP + ["-g", "-f", "18/worst[ext=mp4][protocol*=http]/worst[protocol*=http]", video_url]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0:
            return None
        output = result.stdout.decode("utf-8").strip()
        return output.splitlines()[0] if output else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/")
@app.route("/search")
def search():
    query = request.args.get("search_string", "")
    search_results = []
    if query:
        # Append None for existing_path (no local library in prototype)
        search_results = [(*r, None) for r in get_search_results(query)]
    return render_template(
        "search.html",
        search_string=query,
        search_results=search_results,
    )


@app.route("/autocomplete")
def autocomplete():
    # No local library in prototype; extend here to add local song matching.
    return jsonify([])


@app.route("/preview")
def preview():
    url = request.args.get("url", "")
    if not url:
        return jsonify({"error": "No URL"}), 400
    stream_url = get_stream_url(url)
    if stream_url:
        return jsonify({"stream_url": stream_url})
    return jsonify({"error": "Could not fetch stream URL"}), 500


@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()
    logging.info(f"[MOCK DOWNLOAD] {data}")
    # To enable real downloads: wire up pikaraoke's DownloadManager here.
    return jsonify({"status": "ok"})


@app.route("/enqueue", methods=["GET", "POST"])
def enqueue():
    return jsonify({"success": [True, "Added to queue (mock)"]})


@app.route("/lyrics_search")
def lyrics_search():
    if _genius is None:
        return jsonify([])
    query = regex_tidy(request.args.get("q", "").strip()) or request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    try:
        data = _genius.search(query, per_page=20, type_="song")
        query_has_genius = "genius" in query.lower()
        blocked_term = "translation" if query_has_genius else "genius"
        hits = [
            {"id": h["result"]["id"], "title": h["result"]["title"], "artist": h["result"]["primary_artist"]["name"]}
            for h in (data.get("sections") or [{}])[0].get("hits", [])
            if h.get("type") == "song"
            and blocked_term not in h["result"]["primary_artist"]["name"].lower()
        ][:8]
        return jsonify(hits)
    except Exception as e:
        logging.warning(f"Genius search failed: {e}")
        return jsonify([])


@app.route("/lyrics")
def lyrics():
    """Return lyrics text for a given Genius title + artist, or empty for raw."""
    raw = request.args.get("raw", "")
    if raw:
        return jsonify({"lyrics": ""})
    title = request.args.get("title", "").strip()
    artist = request.args.get("artist", "").strip()
    if not title or not artist or _genius is None:
        return jsonify({"lyrics": "", "error": "Missing title/artist or Genius not configured"}), 400
    try:
        song = _genius.search_song(title, artist)
        if song and song.lyrics:
            return jsonify({"lyrics": song.lyrics, "title": song.title, "artist": song.artist})
        return jsonify({"lyrics": "", "error": "Lyrics not found"}), 404
    except Exception as e:
        logging.warning(f"Genius lyrics fetch failed: {e}")
        return jsonify({"lyrics": "", "error": str(e)}), 500


@app.route("/lyrics_download", methods=["POST"])
def lyrics_download():
    """Fetch lyrics from Genius and save as a headered .txt file next to app.py."""
    data = request.get_json(force=True)
    raw = data.get("raw", False) if data else False
    if raw:
        return jsonify({"status": "ok", "file": None})
    song_id = data.get("id")
    yt_title = data.get("yt_title", "").strip()
    yt_id = data.get("yt_id", "").strip()
    if not song_id or _genius is None:
        return jsonify({"error": "Missing song id or Genius not configured"}), 400
    try:
        # search_song(song_id=) scrapes the lyrics page for the exact selected song.
        # _genius.song(id) only hits the API metadata endpoint and never includes lyrics.
        song = _genius.search_song(song_id=int(song_id))
        if not song or not song.lyrics:
            return jsonify({"error": "Lyrics not found"}), 404
        _bad = str.maketrans({c: "_" for c in r'/\:*?"<>|'})
        base = yt_title.translate(_bad) if yt_title else song.title.translate(_bad)
        suffix = f"---{yt_id}" if yt_id else ""
        out_path = Path(__file__).parent / f"{base}{suffix}.txt"
        out_path.write_text(song.lyrics, encoding="utf-8")
        logging.info(f"Lyrics saved to {out_path}")
        return jsonify({"status": "ok", "file": str(out_path.name)})
    except Exception as e:
        logging.warning(f"Genius lyrics download failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")
