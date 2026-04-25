# Diarized Captions Prototype

A new independent prototype that takes a vocal stem (and optionally a confirmed
lyrics file) and produces `.srt` + `.ass` caption files where every line is
labeled and colored by speaker.

## 1. Goal & scope

- **Input**: vocal stem audio file (`.wav`, `.m4a`, etc.); optional lyrics file
  (`.txt` or `.srt`).
- **Output**: two files written next to the vocal stem:
  - `<vocal_stem>.diarized.srt` — each line prefixed `A:`, `B:`, `C:`, …
  - `<vocal_stem>.diarized.ass` — same prefix; each speaker has a unique color;
    ASS still uses karaoke `\kf` word-fill timing.
- **Single-speaker fallback**: if pyannote detects only one speaker, omit
  the `A:` prefix entirely and use the standard goldenrod color
  (`&H0000D7FF&`, the pipeline default) for all lines — output is visually
  identical to a non-diarized karaoke file.
- **Independence**: self-contained folder. Copies what it needs from
  `pipeline/` (whisper worker + caption generation logic) and from
  `cancel_tests/diarize/` (pyannote worker + HF cache setup). No imports from
  either prototype at runtime.

Out of scope:
- Stem separation (caller must already have a vocal stem).
- Loudnorm / transcoding.
- Cancellation hooks. The copied workers retain their `cancel_event=None`
  fast paths; this prototype always passes `None`.

---

## 2. Folder structure

```
diarized_captions/
├── __init__.py                 # MUST stay empty — see "Import hygiene" below
├── README.md
├── requirements.txt
├── run.py                      # CLI entry point
├── config.py                   # DiarizedCaptionsConfig + WhisperModelConfig + DiarizeConfig
├── word_extraction.py          # load_lyrics, extract_words, match_words_to_lines, segments_to_line_objects
├── speaker_labels.py           # appearance-order remap, word→speaker assignment, line splitting
├── caption.py                  # SRT + per-speaker karaoke ASS generators
└── workers/
    ├── __init__.py             # MUST stay empty
    ├── whisper_worker.py       # copied from pipeline/workers/whisper_worker.py
    └── diarize_worker.py       # copied from cancel_tests/diarize/workers/cancelable_diarize_worker.py
```

The copied workers are dropped in as-is (they already support
`cancel_event=None`). Imports inside them are rewritten to point at
`diarized_captions.config` instead of `pipeline.config` /
`diarize.config`.

### Import hygiene

`__init__.py` files **must stay empty**. The diarize worker imports `torch`
and `torchaudio` at module top, and pyannote/HF model loading inside
`load_model()` reads `HF_HOME`. The eager torch imports themselves don't
read `HF_HOME` (so timing is forgiving in practice), but eagerly importing
the worker through `__init__.py` would still pull torch into every
sub-module load and lengthen startup. Keeping `__init__.py` empty also
sidesteps any future case where pyannote moves env-var reads to import
time.

---

## 3. Data structures

### Word
```python
{
  "word": str,           # token text, stripped
  "start": float,        # seconds
  "end": float,          # seconds
  "is_segment_first": bool,
  "speaker": str | None, # initialized to None by extract_words/segments_to_line_objects;
                         # filled by assign_speakers_to_words: "A" / "B" / ...
}
```

`extract_words` and `segments_to_line_objects` both initialize
`"speaker": None` explicitly so the type annotation is honest pre-assignment.

### Diarization turn
```python
{
  "speaker": str,        # raw pyannote label, e.g. "SPEAKER_00"
  "letter": str,         # remapped, e.g. "A"
  "start": float,
  "end": float,
}
```

### Line object
```python
{
  "text": str,
  "words": list[Word],
  "start": float,
  "end": float,
  "speaker": str,        # "A" / "B" / "C"
}
```

### Speaker color map
```python
{ "A": "&H0000D7FF&", "B": "&H00FFFF00&", ... }   # ASS &HBBGGRR& format
```

---

## 4. Configuration — `config.py`

Three dataclasses, mirroring the source prototypes so the copied workers don't
need rewiring beyond their import lines.

### `WhisperModelConfig`
Identical to `pipeline.config.WhisperModelConfig`. Defaults:

| field | default |
|---|---|
| `model_path` | `<repo>/models/large-v3-turbo.pt` |
| `device` | `"auto"` |
| `compute_type` | `"int8"` |
| `language` | `"en"` |
| `vad` | `True` |
| `vad_threshold` | `0.25` |
| `suppress_silence` | `True` |
| `suppress_word_ts` | `True` |
| `only_voice_freq` | `True` |
| `refine_steps` | `"s"` |
| `refine_word_level` | `False` |
| `regroup` | `""` |

### `DiarizeConfig`
Identical to `cancel_tests/diarize/config.py` minus the ASS styling fields
(those move into `DiarizedCaptionsConfig`) **and** minus
`speaker_label_format` (which was only consumed by the source caption
module, not by the worker — verified safe to drop):

| field | default |
|---|---|
| `hf_token_path` | `""` |
| `device` | `"auto"` |
| `num_speakers` | `0` *(0 = auto)* |
| `min_speakers` | `0` |
| `max_speakers` | `0` |

### `DiarizedCaptionsConfig`
Top-level. Owns whisper + diarize sub-configs and the caption styling.

```python
@dataclass
class DiarizedCaptionsConfig:
    whisper: WhisperModelConfig = field(default_factory=WhisperModelConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)

    # ASS karaoke styling (cloned from pipeline.config.PipelineConfig).
    # All colors normalized to 8-hex `&HAABBGGRR&` form with trailing `&`.
    font_name: str = "Arial"
    font_size: int = 60
    secondary_color: str = "&H00FFFFFF&"   # not-yet-sung (white)
    outline_color: str = "&H00000000&"     # black outline
    back_color: str = "&H80000000&"        # 50% translucent shadow
    outline_width: int = 3
    shadow_offset: int = 2
    margin_left: int = 50
    margin_right: int = 50
    margin_vertical: int = 150

    # Karaoke timing (centiseconds)
    line_lead_in_cs: int = 80
    line_lead_out_cs: int = 20
    first_word_nudge_cs: int = 0

    # Per-speaker colors. Index 0 → speaker A, index 1 → B, etc.
    # Format: `&HAABBGGRR&` (8 hex digits, alpha + reversed RGB, trailing `&`).
    speaker_colors: list[str] = field(default_factory=lambda: [
        "&H0000D7FF&",  # A — goldenrod (alpha=00 form of pipeline's &H00D7FF&)
        "&H00FFFF00&",  # B — cyan
        "&H00B469FF&",  # C — pink
        "&H0000FF00&",  # D — green
        "&H000080FF&",  # E — orange
        "&H00FA82FA&",  # F — lavender
    ])

    # Speaker letters. Mapped 1:1 to speaker_colors by index.
    speaker_letters: str = "ABCDEFGHIJ"
```

> **Note on color format**: ASS accepts both 6-hex (`&HBBGGRR&`, alpha
> implicit) and 8-hex (`&HAABBGGRR&`, alpha explicit) forms — they render
> identically when alpha is `00`. The pipeline's `primary_color` is the
> 6-hex `&H00D7FF&`; this prototype uses the 8-hex `&H0000D7FF&` for
> consistency with the alpha-bearing `back_color`. Visually identical,
> different string.

---

## 5. CLI — `run.py`

Mirrors `pipeline/run_pipeline.py`. Positional args only.

```bash
python -m diarized_captions.run <vocal_audio> [lyrics_file]
```

### Invocation & sys.path

The intended invocation is `python -m diarized_captions.run` from the repo
root, which puts the repo root on `sys.path` automatically. To stay robust
against direct invocation (`python diarized_captions/run.py`), `run.py`
also does:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

(matching the pattern in `cancel_tests/diarize/run_test.py`). This must
happen **before** `from diarized_captions.workers...` imports, alongside
the `HF_HOME` setup.

### Flow

```
main()
 ├── set sys.path  = repo root            ── BEFORE any package imports
 ├── set HF_HOME   = <repo>/models        ── BEFORE pyannote import (lazy inside load_model)
 ├── parse args: vocal_path, lyrics_path?
 ├── configure logging
 ├── cfg = DiarizedCaptionsConfig()
 ├── whisper_worker = WhisperWorker(cfg.whisper)
 ├── diarize_worker = CancelableDiarizeWorker(cfg.diarize)
 ├── whisper_worker.load_model()
 ├── diarize_worker.load_model()
 │
 ├── # 1. transcribe or align via whisper (functions live in word_extraction.py)
 │   if lyrics_path is None:
 │       result = whisper_worker.transcribe_and_refine(vocal_path)
 │       line_objects = segments_to_line_objects(result)
 │   else:
 │       lyrics_text, lyrics_format = load_lyrics(lyrics_path)
 │       result = whisper_worker.align_and_refine(vocal_path, lyrics_text)
 │       words = extract_words(result)
 │       lyric_lines = [l.strip() for l in lyrics_text.split("\n") if l.strip()]
 │       line_objects = match_words_to_lines(words, lyric_lines)
 │       reset_segment_first_flags(line_objects)   # ← see §6.5
 │
 ├── # 2. diarize the same vocal stem
 │   turns = diarize_worker.diarize(vocal_path)
 │
 ├── # 3. remap pyannote labels → A/B/C by appearance order
 │   turns, label_map = remap_speakers_by_appearance(turns, cfg)
 │
 ├── # 4. attach speaker letter to every word
 │   line_objects = assign_speakers_to_words(line_objects, turns)
 │
 ├── # 5. split lines wherever speaker changes mid-line
 │   line_objects = split_lines_at_speaker_boundaries(line_objects)
 │
 ├── # 6. write outputs alongside stem
 │   srt_out = vocal_path.with_suffix(".diarized.srt")
 │   ass_out = vocal_path.with_suffix(".diarized.ass")
 │   srt_out.write_text(generate_srt(line_objects, cfg))
 │   ass_out.write_text(generate_ass(line_objects, cfg))
 │
 └── unload models, log paths.
```

Output naming uses `vocal_path.with_suffix(...)`, which replaces the last
suffix. So `title---vocal.m4a` → `title---vocal.diarized.srt` /
`.ass`, sitting in the same directory as the input. (A no-suffix input
like `vocal` becomes `vocal.diarized.srt` — `with_suffix` appends when
there's nothing to replace.)

---

## 6. Speaker assignment — `speaker_labels.py`

### `remap_speakers_by_appearance(turns, cfg) -> (turns, label_map)`

Walk turns sorted by `start`; the first-seen pyannote label is mapped to
`cfg.speaker_letters[0]` (= `"A"`), second to `"B"`, etc. Fails loudly if the
detected speaker count exceeds `len(cfg.speaker_letters)` — caller should
either bump the list or set `max_speakers` in config.

```python
def remap_speakers_by_appearance(turns, cfg):
    label_map = {}
    for t in sorted(turns, key=lambda t: t["start"]):
        raw = t["speaker"]
        if raw not in label_map:
            idx = len(label_map)
            if idx >= len(cfg.speaker_letters):
                raise ValueError(
                    f"More speakers than configured letters "
                    f"({len(cfg.speaker_letters)})"
                )
            label_map[raw] = cfg.speaker_letters[idx]
    for t in turns:
        t["letter"] = label_map[t["speaker"]]
    return turns, label_map
```

### `assign_speakers_to_words(line_objects, turns) -> line_objects`

For each word, find the diarization turn with the largest temporal overlap.
If the word falls entirely outside any turn (gap), pick the closest turn by
midpoint distance — keeps every word labeled.

If `turns` is empty (e.g. pyannote returned nothing for very short or silent
audio), `assign_speakers_to_words` raises a clear error before the
per-word loop. The caller can decide whether to bail or fall back to a
single-speaker output. v1: bail with a `RuntimeError("Diarization produced
zero turns")`.

```python
def assign_speakers_to_words(line_objects, turns):
    if not turns:
        raise RuntimeError("Diarization produced zero turns — cannot label words")
    for line in line_objects:
        for w in line["words"]:
            w["speaker"] = _word_speaker(w, turns)
    return line_objects

def _word_speaker(word, turns):
    best_turn, best_overlap = None, 0.0
    for t in turns:
        overlap = max(0.0, min(word["end"], t["end"]) - max(word["start"], t["start"]))
        if overlap > best_overlap:
            best_turn, best_overlap = t, overlap
    if best_turn is not None:
        return best_turn["letter"]
    # gap fallback: nearest turn by midpoint (turns guaranteed non-empty above)
    mid = 0.5 * (word["start"] + word["end"])
    nearest = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
    return nearest["letter"]
```

### `reset_segment_first_flags(line_objects) -> None`

Called from `run.py` immediately after `match_words_to_lines` (alignment
path only). The whisper alignment marks `is_segment_first=True` on the
first word of each *whisper* segment, but those boundaries don't match
the lyric-line boundaries we just regrouped to. Stray `True` flags in
the middle of a line would cause `first_word_nudge_cs` to fire at the
wrong position. After regrouping, only the first word of each line
should carry the flag.

```python
def reset_segment_first_flags(line_objects):
    for line in line_objects:
        for i, w in enumerate(line["words"]):
            w["is_segment_first"] = (i == 0)
```

Not needed in the transcription path — `segments_to_line_objects` builds
each line from one whisper segment, so the flags already line up.

`split_lines_at_speaker_boundaries` (below) re-asserts
`is_segment_first=True` on the first word of every sub-line it emits, so
flags stay correct after that step too.

### `split_lines_at_speaker_boundaries(line_objects) -> list`

User-confirmed behavior: option (b). Walk each line's words; whenever the
speaker letter changes between consecutive words, close the current sub-line
and start a new one. The first word of every emitted sub-line gets
`is_segment_first=True` so the existing `first_word_nudge_cs` logic still
applies inside the karaoke generator.

```python
def split_lines_at_speaker_boundaries(line_objects):
    out = []
    for line in line_objects:
        words = line["words"]
        if not words:
            continue
        cur_words = [dict(words[0], is_segment_first=True)]
        cur_speaker = words[0]["speaker"]
        for w in words[1:]:
            if w["speaker"] != cur_speaker:
                out.append(_make_sub(cur_words, cur_speaker))
                cur_words = [dict(w, is_segment_first=True)]
                cur_speaker = w["speaker"]
            else:
                cur_words.append(w)
        out.append(_make_sub(cur_words, cur_speaker))
    return out

def _make_sub(words, speaker):
    return {
        "text": " ".join(w["word"] for w in words),
        "words": words,
        "speaker": speaker,
        "start": words[0]["start"],
        "end": words[-1]["end"],
    }
```

---

## 7. Caption generation — `caption.py`

Cloned and adapted from `pipeline/stages/lyric_align.py`. Two functions,
no class.

### `generate_srt(line_objects, cfg) -> str`

```python
subs = []
for i, line in enumerate(line_objects, start=1):
    content = f"{line['speaker']}: {line['text']}"
    subs.append(srt.Subtitle(
        index=i,
        start=timedelta(seconds=line["start"]),
        end=timedelta(seconds=line["end"]),
        content=content,
    ))
return srt.compose(subs)
```

### `generate_ass(line_objects, cfg) -> str`

Header gets **one Style per speaker letter that actually appears** in the
line objects — keeps the file lean if only A/B are present.

```
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, ...
Style: Karaoke_A,Arial,60,&H0000D7FF&,&H00FFFFFF,...
Style: Karaoke_B,Arial,60,&H00FFFF00&,&H00FFFFFF,...
```

For each line, emit one Dialogue using `Karaoke_<letter>`, with the speaker
prefix outside the karaoke fill (so the prefix is steady text, the lyric
text sweeps as the words are sung):

```
Dialogue: 0,0:00:01.23,0:00:04.56,Karaoke_A,,0,0,0,,A: {\k80}{\kf45}Hello {\kf30}world
```

#### Single-speaker fallback
Before building the header, check `len(present) == 1`. If so:
- Emit a single `Style: Karaoke` (no letter suffix) with `PrimaryColour = &H0000D7FF&`
  (goldenrod, the pipeline default).
- Skip the `A:` prefix in every Dialogue line — the body is just the karaoke
  text, indistinguishable from a standard non-diarized karaoke file.
- `generate_srt` also omits the prefix (content = `line['text']` not
  `f"{line['speaker']}: {line['text']}"`).

The check is on `present` (letters that actually appear in the line
objects), not on `len(label_map)`. So if pyannote detects two speakers
but only one of them ends up with words assigned (e.g. speaker B's turns
fall entirely in instrumental gaps with no transcribed words), the output
collapses to single-speaker form. That is the desired behavior — a
caption file with B never appearing would be misleading.

#### Build steps (multi-speaker)
1. Determine which speaker letters appear: `present = sorted({l["speaker"] for l in line_objects})`.
2. If `len(present) == 1`: single-speaker fallback (see above).
3. Otherwise emit `[Script Info]` + `[V4+ Styles]` + a `Style:` row per `present` letter,
   pulling color from `cfg.speaker_colors[index_of(letter)]`.
4. Emit `[Events]` then one Dialogue per line.

#### Dialogue body
Same word-fill construction as `pipeline/stages/lyric_align.py:_generate_ass`:

```python
event_start = max(0.0, words[0]["start"] - cfg.line_lead_in_cs / 100.0)
event_end   = words[-1]["end"] + cfg.line_lead_out_cs / 100.0
prev_end = event_start
parts = [f"{line['speaker']}: "]   # plain prefix, no \kf
for i, w in enumerate(words):
    word_start = w["start"]
    if (
        w.get("is_segment_first")
        and abs(word_start - prev_end - cfg.line_lead_in_cs / 100.0) < 0.05
    ):
        word_start += cfg.first_word_nudge_cs / 100.0
    word_dur_cs = max(10, round((w["end"] - word_start) * 100))
    gap_cs = max(0, round((word_start - prev_end) * 100))
    if gap_cs > 0:
        parts.append(f"{{\\k{gap_cs}}}")
    parts.append(f"{{\\kf{word_dur_cs}}}{w['word']}")
    prev_end = w["end"]
    if i < len(words) - 1:
        parts.append(" ")
karaoke_text = "".join(parts)
```

`_seconds_to_ass_time()` is the same `H:MM:SS.cc` helper as the pipeline.

> **Note**: the `A: ` prefix is plain text outside the karaoke run; it shows
> for the entire line duration in the speaker's color, while the lyric text
> sweeps left-to-right with `\kf`. Because the prefix is part of the same
> Dialogue, it inherits `Karaoke_A`'s `PrimaryColour` — already the speaker
> color, no extra override needed.

---

## 8. Worker integration

### Whisper worker
Copy `pipeline/workers/whisper_worker.py` verbatim. Change one import line:

```python
from diarized_captions.config import WhisperModelConfig
```

Used methods: `load_model`, `transcribe_and_refine`, `align_and_refine`,
`unload_model`. `cancel_event` argument is always `None` from this prototype.

### Diarize worker
Copy `cancel_tests/diarize/workers/cancelable_diarize_worker.py` verbatim.
Change one import:

```python
from diarized_captions.config import DiarizeConfig
```

Used methods: `load_model`, `diarize(vocal_path, cancel_event=None)`,
`unload_model`. The pyannote pipeline name (`pyannote/speaker-diarization-3.1`)
and HF token resolution carry over unchanged.

### HF cache priming
`run.py` sets `os.environ["HF_HOME"] = <repo>/models` **before** importing the
diarize worker — same pattern as `cancel_tests/diarize/run_test.py`. This
ensures the cached `pyannote/speaker-diarization-3.1` is picked up without a
network call.

---

## 9. Output examples

Vocal stem `~/song/title---vocal.m4a` → outputs:

`~/song/title---vocal.diarized.srt`:
```
1
00:00:01,200 --> 00:00:04,500
A: We were both young when I first saw you

2
00:00:04,500 --> 00:00:07,800
B: I close my eyes and the flashback starts
```

`~/song/title---vocal.diarized.ass` (excerpt):
```
[V4+ Styles]
Format: Name, Fontname, ...
Style: Karaoke_A,Arial,60,&H0000D7FF&,&H00FFFFFF&,&H00000000&,&H80000000&,...
Style: Karaoke_B,Arial,60,&H00FFFF00&,&H00FFFFFF&,&H00000000&,&H80000000&,...

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.12,0:00:04.70,Karaoke_A,,0,0,0,,A: {\kf30}We {\kf25}were {\kf45}both ...
Dialogue: 0,0:00:04.42,0:00:08.00,Karaoke_B,,0,0,0,,B: {\kf28}I {\kf30}close ...
```

---

## 10. requirements.txt

```
stable-ts
faster-whisper
srt
pyannote.audio
torch
torchaudio
```

(`audio-separator` is intentionally omitted — this prototype consumes a
pre-separated stem.)

---

## 11. Test plan

1. **Single-speaker stem, no lyrics** — transcription path. Verify all output
   lines prefixed `A:`, ASS has only `Karaoke_A` style.
2. **Single-speaker stem, `.txt` lyrics** — alignment path. Lyric line breaks
   preserved (no splits, since one speaker).
3. **Two-speaker duet stem, no lyrics** — verify A/B labeling, color
   distinction in ASS, and that line splits occur where pyannote says the
   speaker changes.
4. **Two-speaker duet stem, `.txt` lyrics with one line spanning a hand-off** —
   verify that single lyric line is split into two sub-lines (option b),
   each with the correct prefix.
5. **`.srt` lyrics input** — alignment runs but no SRT is written? *Decision
   point:* this prototype **always writes both** SRT and ASS, even when input
   was `.srt` (the pipeline behavior of skipping SRT in that case doesn't
   apply here — the diarized SRT carries new information regardless of input
   format).
6. **Speaker count > letter list** — `remap_speakers_by_appearance` raises;
   user is told to bump `speaker_letters` / `speaker_colors` in config.

---

## 12. Open follow-ups (not in v1)

- Per-word color toggling inside a line (showing speaker change without
  splitting the line). Discarded for v1 in favor of split lines.
- Configurable letter prefix format (e.g. `[A]` vs `A:`).
- Confidence-weighted speaker assignment (currently uses raw overlap).
- Cancel-event wiring (workers support it; CLI never sets it).

---

## 13. Revisions (post-review)

Addressed against `plans/diarized-captions-review.md`:

- **C2 fixed**: §6.5 added `reset_segment_first_flags` step in the
  alignment path. §5 flow calls it after `match_words_to_lines`.
- **C3 fixed**: §6 `assign_speakers_to_words` now raises `RuntimeError` on
  empty `turns` before `_word_speaker` runs.
- **S2 fixed**: §5 documents the `python -m diarized_captions.run`
  invocation and adds a defensive `sys.path.insert(...)` in `run.py`.
- **S3 fixed**: §2 folder structure adds `word_extraction.py` (home for
  `load_lyrics`, `extract_words`, `match_words_to_lines`,
  `segments_to_line_objects`).
- **S4 fixed**: §4 `DiarizeConfig` table notes `speaker_label_format` is
  intentionally dropped (verified the worker never reads it).
- **S5 fixed**: §7 single-speaker fallback documents the
  >1-detected-but-only-1-with-words edge case as desired behavior.
- **M4 fixed**: §3 Word data structure clarifies `"speaker"` is initialized
  to `None` in `extract_words` / `segments_to_line_objects`.
- **C1 / M1 fixed (downgraded to Minor)**: §4 normalizes all ASS color
  fields to 8-hex `&HAABBGGRR&` form with trailing `&`. The
  `&H0000D7FF&` ↔ `&H00D7FF&` divergence from the pipeline is documented
  inline as visually identical (alpha=00 default) but textually different.
  §9 example updated to match.
- **S1 partially addressed**: §2 adds an "Import hygiene" note. The
  review's diagnosis (torch imports affect HF_HOME timing) doesn't hold
  — `HF_HOME` is read by pyannote's `from_pretrained` inside
  `load_model()`, not by torch — but the practical advice (keep
  `__init__.py` empty, avoid eager worker imports) is sound and
  documented.
- **M3, M6 left as-is**: `with_suffix` is safe for the inputs we expect;
  no `__main__.py` for `python -m diarized_captions` (consistent with
  `pipeline/`).
