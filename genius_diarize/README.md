# Genius Diarize

Speaker-aware karaoke subtitle generation from Genius.com section headers — no pyannote, no overlap detection. One label per section, conservatively assigned.

## Usage

```bash
python -m genius_diarize.run <vocal_audio> <lyrics_file>
```

## Example

```bash
python -m genius_diarize.run song.m4a song.txt
# Produces: song.diarized.srt, song.diarized.ass
```

## Lyrics format

Lyrics files must be in Genius format with section headers:

```
[Verse 1: Brian]
You are my fire
The one desire

[Bridge: Kevin & AJ]
Now I can see

[Chorus: Nick, All]
Tell me why
```

### Attribution rules

| Header                                   | Groups | Result                                   |
| ---------------------------------------- | ------ | ---------------------------------------- |
| `[Verse 1: Brian]`                       | 1 group, single name    | Labeled **Brian**                        |
| `[Bridge: Kevin & AJ]`                   | 1 group, named pair     | Labeled **Kevin & AJ**                   |
| `[Chorus: All]`                          | 1 group, "All"           | Unlabeled (ensemble)                     |
| `[Chorus: Nick, All]`                    | 2+ groups               | Colored by first group (**Nick**)        |
| `[Chorus: Brian, AJ]`                    | 2+ named groups         | Colored by first group (**Brian**)       |
| `[Chorus]` after `[Chorus: Rumi]`        | Bare repeat             | Inherits **Rumi**                        |
| `[Verse 3]` with no prior "Verse 3"      | No attribution, no prior exact match | Falls back to any prior **Verse** section (e.g., **Verse 1**) |

- **Multi-group headers (2+ groups)** are attributed to the first group because line-to-group alignment within a section is unknown.
- **Bare section headers** (`[Chorus]` without `: <name>`) first attempt to inherit the attribution from the last time that exact section name appeared. If no exact match exist but another section in the same family (e.g., `Verse 1`) was seen, the family carry-forward is used. If neither is available, the section is unlabeled.

## How it works

1. **Parse Genius headers** into per-line attribution.
2. **Single-name** -> label as that name. **Named duet** -> label as `"A & B"`. `"All"`, multi-group, or no attribution -> unlabeled.
3. **Run Whisper (stable-ts)** to align words to lyric lines.
4. **Generate outputs:**
   - **SRT** — plain timed lyric text (no speaker labels).
   - **ASS** — per-speaker karaoke styles with `\kf` fill-sweep coloring keyed by the dominant speaker.

## CLI options

Run `python -m genius_diarize.run --help` for full usage.

| Flag / Option | Description |
| ------------- | ----------- |
| `vocal_audio` | Vocal stem audio file (positional) |
| `lyrics_file` | Genius-formatted lyrics file (positional) |
| `--save-whisper` | Save raw Whisper alignment as `<stem>.whisper.json` |
| `--no-vad` | Disable Silero VAD (forces alignment of silent regions) |
| `--match-method` | Word-to-lyric matching algorithm: `nw` (default), `walk`, or `count` |

### `--match-method` algorithms

| Method | Description |
|--------|-------------|
| `nw` (default) | **Needleman-Wunsch global sequence alignment**. Long anchors (≥6 chars) score 3; exact matches / contractions score 2; Levenshtein-1 / phonetic equivalent pairs score 1. Automatically uses banded alignment for very long sequences. |
| `walk` | **Two-pointer lockstep walk** with bounded lookahead and gap interpolation. Matches every lyric token (no dropped tokens), synthesizing timestamps where gaps occur. |
| `count` | **Legacy count-based positional slicing** (brittle, kept for comparison). |

## ASS output

Each unique **dominant speaker** (first individual name in a header) gets a dedicated karaoke style with a muted speaker color from the palette in first-appearance order. Lines share a style between solo and duet appearances of the same singer.

```
Style: Karaoke_Brian     <- color[0] (goldenrod) for solo and duet lines where Brian is first
Style: Karaoke_Kevin    <- color[1] (cyan)
Style: Karaoke_ensemble  <- white (unlabeled sections)
```

## Solo mode

If no header has single-name or named-duet attribution (e.g., all headers are `[Chorus: All]` or `[Verse 1]`), the tool runs in **solo mode** — output is plain karaoke with no speaker labels and a single goldenrod style. The same `.diarized.srt` and `.diarized.ass` files are written regardless of mode.

## Configuration

`GeniusDiarizeConfig` (defined in `config.py`) controls ASS styling and Whisper behavior.

### Whisper sub-config (imported from `pipeline.config`)

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | `"base"` | Whisper model size or local path |
| `device` | `"cuda"` | Inference device (`cuda` / `cpu`) |
| `suppress_silence` | `True` | Suppress non-speech segments |
| `vad` | `"silero"` | VAD backend (`"silero"` / `None`) |
| `temperature` | `0` | Whisper sampling temperature |
| `condition_on_previous_text` | `True` | Condition on prior transcription |
| `initial_prompt` | `"lyrics"` | Prompt prefix for alignment |
| `regroup` | `True` | Regroup words into sentences after alignment |

### ASS styling

| Field | Default | Description |
|-------|---------|-------------|
| `font_name` | `"Arial"` | ASS font name |
| `font_size` | `60` | ASS font size |
| `secondary_color` | `"&H00FFFFFF&"` | Not-yet-sung (white) |
| `outline_color` | `"&H00000000&"` | Black outline |
| `back_color` | `"&H80000000&"` | 50% translucent shadow |
| `outline_width` | `3` | Outline width |
| `shadow_offset` | `2` | Shadow offset |
| `margin_left` | `50` | Left margin |
| `margin_right` | `50` | Right margin |
| `margin_vertical` | `150` | Vertical margin |
| `line_lead_in_cs` | `80` | Lead-in (centiseconds) |
| `line_lead_out_cs` | `20` | Lead-out (centiseconds) |
| `first_word_nudge_cs` | `0` | Nudge first word if it starts immediately after previous segment |
| `ensemble_color` | `"&H0000D7FF&"` | Goldenrod for unlabeled sections |
| `speaker_colors` | ... | Per-speaker palette in first-appearance order |

## File structure

```
genius_diarize/
├── __init__.py          # Empty module init
├── __main__.py          # `python -m genius_diarize` entry point
├── config.py            # GeniusDiarizeConfig (flat caption + whisper sub-config)
├── caption.py           # SRT + ASS generation (styles keyed by dominant_speaker)
├── genius.py            # Header parsing, mode detection, group splitting, carry-forward
├── word_extraction.py   # Whisper word extraction, lyrics loading, line matching (NW / walk / count)
├── run.py               # CLI orchestration + helpers (assign_speakers_from_genius, etc.)
└── workers/
    └── whisper_worker.py  # stable-ts worker with per-encoder-pass cancellation via cancel_event
```

## Tests

Tests live at the repo root under `tests/` (not under `genius_diarize/`):

```bash
python -m pytest tests/test_genius.py tests/test_assign.py tests/test_caption.py tests/test_word_extraction.py -v
```

- `test_genius.py` — header parsing, attribution rules, mode detection, group splitting, bare/family carry-forward
- `test_assign.py` — speaker assignment (single name, duet, ensemble, mixed sections)
- `test_caption.py` — plain SRT generation, ASS style/color keying, solo mode, duet style resolution
- `test_word_extraction.py` — token normalization, Levenshtein scoring, Needleman-Wunsch alignment, walk matcher, anchor bonus, banding, zero-score filtering
