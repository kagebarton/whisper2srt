# Diarized Captions Prototype — Review Findings

Review of `plans/diarized-captions-prototype.md` against the existing codebase
(`pipeline/`, `cancel_tests/diarize/`). Findings are grouped by severity.

---

## Critical (will cause bugs)

### C1. `primary_color` format mismatch in speaker color A

The plan's `DiarizedCaptionsConfig.speaker_colors[0]` is `&H0000D7FF&`, claiming
it "reuses the existing pipeline `primary_color`" (soft yellow). But the actual
`PipelineConfig.primary_color` default is `&H00D7FF&` — **7 characters, not 8**.
The plan adds a leading `00` (alpha byte) that doesn't exist in the original.

In ASS `&HBBGGRR&` format, `&H00D7FF&` is the 7-char shorthand where the alpha
byte defaults to `00` (fully opaque). Writing `&H0000D7FF&` is 8 chars and
technically valid, but it's not the same string as the pipeline. If the intent
is visual parity with the existing pipeline output, the color should be
`&H00D7FF&` — or, if the plan intentionally wants the explicit 8-char form,
all colors and the `secondary_color` / `outline_color` / `back_color` fields
need the same treatment for consistency. Currently the plan mixes formats:
`secondary_color` is `&H00FFFFFF` (7 chars, no trailing `&`), while
`speaker_colors[0]` is `&H0000D7FF&` (8 chars, trailing `&`).

**Resolution needed**: Decide on a consistent ASS color format. The existing
`PipelineConfig` uses 7-char form without trailing `&` for most fields
(`&H00D7FF&` is the one exception — it has a trailing `&`). The
`DiarizedCaptionsConfig` in the plan should either match the pipeline exactly
or document the deliberate format change.

### C2. `match_words_to_lines` drops `is_segment_first` semantics on aligned words

When the alignment path is used (`lyrics_path` provided), `run.py` calls
`extract_words(result)` which sets `is_segment_first=True` only for the first
word of each whisper **segment**. But then `match_words_to_lines(words,
lyric_lines)` re-groups words into lyric lines, which can span multiple
segments. After this regrouping, the `is_segment_first` flags no longer align
with the new line boundaries — a word that was `is_segment_first=True` in the
whisper segment might be in the middle of a lyric line.

This is the **exact same pattern** as `lyric_align.py:_match_words_to_lines`,
which also has this issue. In the pipeline it's mostly harmless because
`first_word_nudge_cs` defaults to `0`. But the plan copies this bug forward,
and if anyone sets `first_word_nudge_cs > 0`, the nudge will fire at wrong word
positions in the alignment path.

**Fix**: After `match_words_to_lines`, reset `is_segment_first=True` only on
the first word of each line object and `False` on all others. This should be
done in `run.py` before the lines are passed to the caption generator.

### C3. Diarize worker returns no `"letter"` field — `remap_speakers_by_appearance` adds it, but the assignment step's gap fallback may fail on empty turns

The `CancelableDiarizeWorker.diarize()` returns turns as:
```python
{"speaker": "SPEAKER_00", "start": float, "end": float}
```
No `"letter"` key. The plan's `remap_speakers_by_appearance` correctly adds
`"letter"` to each turn dict. However, the `_word_speaker` gap-fallback logic
uses `min(turns, key=...)` — if the diarization returns zero turns (possible
for very short or silent audio), this raises `ValueError: min() arg is an
empty sequence`.

**Fix**: Guard `assign_speakers_to_words` against empty turns list — either
return all words with a default speaker or raise a clear error.

---

## Significant (design issues / incorrect assumptions)

### S1. The diarize worker imports `torch` and `torchaudio` at module top level

`cancelable_diarize_worker.py` has:
```python
import torch
import torchaudio
```
at lines 34–35 (top-level). The plan says these workers will be "dropped in
as-is" with only the import line changed. But these top-level imports mean
**torch and torchaudio are imported before `run.py` sets `HF_HOME`** if the
worker is imported at module top of any file in `diarized_captions/`.

In the existing `cancel_tests/diarize/run_test.py`, `sys.path` and `HF_HOME`
are set **before** `from diarize.workers.cancelable_diarize_worker import ...`.
The plan correctly notes setting `HF_HOME` before importing, but doesn't
mention that the worker itself imports `torch`/`torchaudio` eagerly. As long as
`run.py` sets `HF_HOME` before `from diarized_captions.workers.diarize_worker
import CancelableDiarizeWorker`, this works. But if anyone imports the worker
indirectly (e.g., `from diarized_captions import ...` in `__init__.py`), the
timing breaks.

**Fix**: Ensure `__init__.py` does **not** eagerly import the workers. Add a
note in the plan that `torch`/`torchaudio` are top-level imports in the
diarize worker and `HF_HOME` must be set before any import of that module.

### S2. The diarize worker uses `sys.path` and `diarize.config` — not just one import to change

The plan says: "Change one import" for the diarize worker. But the existing
`cancelable_diarize_worker.py` imports `from diarize.config import DiarizeConfig`
(line 37). The `diarize` package is only on `sys.path` because `run_test.py`
does `sys.path.insert(0, ...)`. When the worker is copied into
`diarized_captions/workers/`, the import becomes `from
diarized_captions.config import DiarizeConfig`. This is the one change the plan
mentions.

However, the whisper worker imports `from pipeline.config import
WhisperModelConfig`. If `pipeline` is not on `sys.path`, this also needs
changing. The plan mentions this but doesn't address how `sys.path` will be
set up so that `diarized_captions` is importable without the parent `pipeline/`
or `cancel_tests/` packages being on the path.

The `run_test.py` pattern uses:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
to make `diarize` importable. The plan's `run.py` needs a similar `sys.path`
insertion to make `diarized_captions` importable as a package (since it uses
`python -m diarized_captions.run`, Python auto-adds the parent directory, but
only if invoked as a module from the repo root).

**Fix**: The plan should specify that `run.py` is invoked from the repo root as
`python -m diarized_captions.run <vocal>`, which automatically puts the repo
root on `sys.path`. If invoked directly (`python diarized_captions/run.py`),
the import chain breaks. Either document the invocation clearly or add a
`sys.path` insertion like the existing prototypes do.

### S3. `_extract_words` and `_match_words_to_lines` are methods on `LyricAlignStage`, not standalone functions

The plan references these as top-level functions in `run.py`'s flow:
```
words = extract_words(result)
line_objects = match_words_to_lines(words, lyric_lines)
```
But in the existing codebase, they're instance methods on `LyricAlignStage`:
- `_extract_words(self, result)`
- `_match_words_to_lines(self, words, lines)`
- `_segments_to_line_objects(self, result)`
- `_load_lyrics(self, lyrics_path)`

The plan correctly shows standalone versions of these in the flow diagram, but
doesn't specify which file they live in. Since `caption.py` is for ASS/SRT
generation and `speaker_labels.py` is for speaker assignment, there's a gap —
these word-extraction and line-building functions need a home.

**Fix**: Add a module (e.g., `word_extraction.py` or put them in `run.py`)
for `_extract_words`, `_match_words_to_lines`, `_segments_to_line_objects`,
and `_load_lyrics`. Update the folder structure in §2 accordingly.

### S4. `DiarizeConfig` in `cancel_tests/diarize/config.py` has `speaker_label_format` field — plan omits it

The plan's `DiarizeConfig` for `diarized_captions/config.py` lists only 5 fields
(`hf_token_path`, `device`, `num_speakers`, `min_speakers`, `max_speakers`),
omitting the existing `speaker_label_format: str = "{speaker}"`. While the
plan correctly removes the 13 ASS styling fields (they move to
`DiarizedCaptionsConfig`), `speaker_label_format` is a diarization-label
concern, not a styling concern. The diarize worker itself doesn't use it
directly (it's only consumed by `cancel_tests/diarize/caption.py`), but if
the copied `cancelable_diarize_worker.py` ever references `self._config`
for this field, it would fail.

Verified: the worker doesn't reference `speaker_label_format`. It only reads
`hf_token_path`, `device`, `num_speakers`, `min_speakers`, `max_speakers`. So
omitting it is safe for the worker, but the plan should note the omission
explicitly.

### S5. Single-speaker fallback check should use `label_map` size, not `present` set

The plan says (§7): "Before building the header, check `len(present) == 1`."
Where `present = sorted({l["speaker"] for l in line_objects})`. But after
`split_lines_at_speaker_boundaries`, a line object's `speaker` field is a
letter like `"A"`. If only one speaker exists, all line objects have
`speaker="A"`, so `len(present) == 1` works. This is correct.

However, there's a subtle issue: what if pyannote detects two speakers but
only one of them actually has words assigned? (E.g., speaker B appears only
in a brief silence gap and no words fall into their turns.) In that case,
`remap_speakers_by_appearance` would still create two entries in `label_map`,
but `present` in the caption generator would only have one letter. The ASS
output would use single-speaker fallback, but the SRT output would also skip
the prefix. This is arguably the right behavior, but it's worth documenting.

---

## Minor (clarity / consistency / polish)

### M1. Inconsistent trailing `&` in ASS color strings

The plan's `DiarizedCaptionsConfig` mixes formats:
- `speaker_colors[0]` = `"&H0000D7FF&"` (8 chars + `&`)
- `secondary_color` = `"&H00FFFFFF"` (8 chars, no `&`)
- `outline_color` = `"&H00000000"` (8 chars, no `&`)
- `back_color` = `"&H80000000&"` (8 chars + `&`)

The existing `PipelineConfig` is similarly inconsistent:
- `primary_color` = `"&H00D7FF&"` (7 chars + `&`)
- `secondary_color` = `"&H00FFFFFF"` (8 chars, no `&`)
- `outline_color` = `"&H00000000"` (8 chars, no `&`)
- `back_color` = `"&H80000000&"` (8 chars + `&`)

ASS is lenient about trailing `&`, but the plan should either match the
pipeline exactly or normalize. This matters for string-comparison tests.

### M2. `regroup` default is `""` — empty string is falsy, worker skips regroup

The plan shows `regroup: str = ""`. In `transcribe_and_refine`:
```python
if self._config.regroup:
    result.regroup(self._config.regroup)
```
Empty string is falsy, so `regroup` is skipped. This matches the pipeline
default. Just confirming the behavior is intentional.

### M3. Output naming uses `with_suffix("").with_suffix(...)` — fragile for double-extension stems

The plan shows:
```python
srt_out = vocal_path.with_suffix("").with_suffix(".diarized.srt")
```
For `title---vocal.m4a`, this strips `.m4a` then adds `.diarized.srt` →
`title---vocal.diarized.srt`. But `with_suffix("")` strips only the last
suffix. For a file like `title.v2.m4a`, this would produce `title.v2.diarized.srt`
— which is fine. The real risk is for files with no suffix: `vocal` →
`vocal.diarized.srt`. But `Path.with_suffix("")` on a no-suffix path returns
the path unchanged, then `.with_suffix(".diarized.srt")` would produce
`vocal.diarized.srt`. This actually works correctly because `.with_suffix()`
replaces the last suffix, and a path without a suffix effectively gets one
appended. So this is safe, but worth a comment in the code.

### M4. The plan's `Word` dataclass includes `"speaker": str | None` — but `assign_speakers_to_words` always fills it

The plan's Word structure has `speaker: str | None`, implying it can be unset.
But after `assign_speakers_to_words`, every word has a speaker letter (the gap
fallback ensures this). The `None` type is only valid for words *before* speaker
assignment. This isn't a bug, but the pipeline from whisper → extract_words →
line_objects doesn't set `speaker` at all — the plan needs to add `"speaker":
None` to the word dicts returned by `_extract_words` and
`_segments_to_line_objects`.

**Fix**: Ensure `_extract_words` and `_segments_to_line_objects` initialize each
word dict with `"speaker": None`, so the type annotation matches reality.

### M5. `_generate_ass` in `caption.py` should be a standalone function, not a method

The plan correctly shows it as a standalone function `generate_ass(line_objects,
cfg)`. The existing pipeline's `_generate_ass` is a method on `LyricAlignStage`
and reads `self._config`. The plan's version takes `cfg` explicitly, which is
clean. Just confirming the adaptation is intentional and correct.

### M6. The `srt` library `compose` function adds blank lines between entries

The plan's `generate_srt` uses `srt.compose(subs)`. The `srt` library adds
blank lines between subtitle entries by default. This matches the pipeline
behavior. Just confirming this produces the expected output format shown in §9.

### M7. No `__main__.py` for `python -m diarized_captions` support

The plan specifies `python -m diarized_captions.run`. This works because
`run.py` is executed as `__main__`. But if someone tries `python -m
diarized_captions` (without `.run`), they'll get an error unless an
`__main__.py` is provided. The existing `pipeline/` also lacks this, so it's
consistent, but worth noting.

---

## Observations (not issues, but useful context)

### O1. `diarized_captions/` directory does not yet exist

The folder has not been created. All source files exist and match the plan's
descriptions.

### O2. Worker constructors both accept `Optional[Config] = None`

Both `WhisperWorker(Optional[WhisperModelConfig])` and
`CancelableDiarizeWorker(Optional[DiarizeConfig])` default to constructing
their own config if none is passed. The plan's `run.py` correctly passes
explicit configs: `WhisperWorker(cfg.whisper)`,
`CancelableDiarizeWorker(cfg.diarize)`.

### O3. Both workers have `model_loaded` property

Useful for pre-flight checks. The plan doesn't use it but could.

### O4. The diarize worker pre-loads audio via `torchaudio.load()`

This avoids a pyannote bug with M4A duration headers. The plan doesn't mention
this but it's internal to the worker and doesn't affect the prototype.

### O5. The `cancel_tests/diarize/caption.py` is completely different from what the plan builds

The existing `caption.py` generates speaker-label-only subtitles (no lyric
text). The plan's `caption.py` generates full lyric-text + karaoke subtitles.
They share only the `_seconds_to_ass_time` helper. This is expected — the plan
is building new functionality, not copying the existing caption module.

### O6. `load_lyrics` handles `.srt` input by parsing and concatenating text

The plan's flow (§5) shows `lyrics_text, lyrics_format = load_lyrics(lyrics_path)`
but doesn't specify that `.srt` input is supported. The existing
`_load_lyrics` in `lyric_align.py` does parse `.srt` into plain text. The plan
should clarify whether `.srt` lyrics input is supported (it is in the pipeline).

---

## Summary

| ID | Severity | Title |
|----|----------|-------|
| C1 | Critical | `primary_color` format mismatch — `&H00D7FF&` vs `&H0000D7FF&` |
| C2 | Critical | `is_segment_first` wrong after `match_words_to_lines` regrouping |
| C3 | Critical | Empty turns list crashes `_word_speaker` gap fallback |
| S1 | Significant | Top-level `torch`/`torchaudio` imports in diarize worker |
| S2 | Significant | `sys.path` setup for `python -m` vs direct invocation |
| S3 | Significant | Word extraction / line building functions have no home in folder structure |
| S4 | Significant | `speaker_label_format` field omitted from `DiarizeConfig` (safe but undocumented) |
| S5 | Significant | Edge case: pyannote detects >1 speaker but only 1 has words assigned |
| M1 | Minor | Inconsistent `&` trailing in ASS color strings |
| M3 | Minor | `with_suffix("").with_suffix(...)` — works but fragile-looking |
| M4 | Minor | Word dicts need explicit `"speaker": None` initialization |
| M6 | Minor | No `__main__.py` for `python -m diarized_captions` |
