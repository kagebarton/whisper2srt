# Genius Diarize

Speaker attribution from Genius.com section headers — no pyannote, no overlap
detection. One label per section, conservatively assigned.

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

Lyrics file must be in Genius format with section headers:

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

| Header | Groups | Result |
|--------|--------|--------|
| `[Verse 1: Brian]` | 1 group, single name | Labeled **Brian** |
| `[Bridge: Kevin & AJ]` | 1 group, named pair | Labeled **Kevin & AJ** |
| `[Chorus: All]` | 1 group, "All" | Unlabeled (ensemble) |
| `[Chorus: Nick, All]` | 2+ groups | Colored by first group (**Nick**) |
| `[Chorus: Brian, AJ]` | 2+ named groups | Colored by first group (**Brian**) |
| `[Chorus]` after `[Chorus: Rumi]` | Bare repeat | Inherits **Rumi** |
| `[Verse 3]` with no prior "Verse 3" | No attribution, no history | Unlabeled (ensemble) |

Multi-group headers (2+) are attributed to the first group because line-to-group
alignment within a section is unknown. Bare section headers (no `:`)
carry forward the attribution from the last time that exact section
name appeared with explicit attribution.

## How it works

1. Parse Genius headers into per-line attribution.
2. Single-name → label as that name. Named duet → label as `"A & B"`.
   `"All"`, multi-group, or no attribution → unlabeled.
3. Run Whisper to align words to lyric lines.
4. Generate SRT (full label first time, initials thereafter) + ASS
   (colors keyed by the first individual name in the header).

### SRT output

First appearance of each speaker uses the full name; subsequent
appearances use initials:

```
1
00:00:01,000 --> 00:00:03,000
Brian: You are my fire

2
00:00:03,200 --> 00:00:05,000
B: The one desire
```

Named duets are truncated to initials joined with `&`:
`Kevin & AJ` → `K & A`.

### ASS output

Each unique **dominant speaker** (first individual name) gets a color
from the palette in first-appearance order. A solo line and a duet line
for the same singer share one color slot:

```
Style: Karaoke_Brian   ← color[0] (goldenrod)
Style: Karaoke_Kevin   ← color[1] (cyan) — shared by "Kevin" solo and "Kevin & AJ" duet
Style: Karaoke_ensemble ← white (unlabeled sections)
```

### Solo mode

If no header has single-name or named-duet attribution (e.g., all
headers are `[Chorus: All]` or `[Verse 1]`), the tool runs in **solo
mode** — output is plain karaoke with no speaker labels and a single
goldenrod style. The same `.diarized.srt` and `.diarized.ass` files are
written regardless of mode.

## File structure

```
genius_diarize/
├── config.py            # GeniusDiarizeConfig (no DiarizeConfig / pyannote)
├── caption.py           # SRT + ASS generation (dominant_speaker for styles, display_label for SRT)
├── genius.py            # Header parsing, mode detection, truncation
├── word_extraction.py   # Whisper word extraction, lyrics loading, line matching
├── run.py               # CLI orchestration + helpers
└── workers/
    └── whisper_worker.py  # stable-ts worker (unchanged from diarized_captions)
```

## Tests

```bash
python -m pytest tests/test_genius.py tests/test_assign.py tests/test_caption.py -v
```

57 tests covering: header parsing, attribution rules, mode detection,
group splitting, label truncation, speaker assignment, display label
annotation, SRT prefixes, ASS style/color keying, and presence helpers.
