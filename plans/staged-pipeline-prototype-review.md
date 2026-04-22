# Review: Staged Processing Manager Prototype

## Blocking Issues

### 1. `cancel_whisper/config.py` does not exist — import will fail at runtime

The plan's `run_pipeline.py` (line 641) imports:

```python
from cancel_whisper.config import WhisperModelConfig
```

But `cancel_whisper/config.py` **does not exist on disk**. The `cancelable_whisper_worker.py` (line 82) also imports from it, so the entire `cancel_whisper` package is currently broken — you can't even `import cancel_whisper.workers.cancelable_whisper_worker` without a `ModuleNotFoundError`. This is a pre-existing issue in the codebase, not introduced by the plan, but the plan depends on it without acknowledging it.

**Fix:** `config.py` must be created in `cancel_whisper/` before the pipeline can run. Based on the worker's usage (`self._config.model_path`, `.device`, `.compute_type`, `.language`, `.vad`, `.vad_threshold`, `.suppress_silence`, `.suppress_word_ts`, `.only_voice_freq`, `.refine_steps`, `.refine_word_level`), the dataclass fields are inferrable, but the plan should state that this file needs to be created first or include its definition.

### 2. All `cancel_separator/` source files still use `cancel_test` in imports

Every file in `cancel_separator/` still imports from `cancel_test.*`:

| File | Import |
|------|--------|
| `context.py` | `from cancel_test.context import ...` |
| `stages/base.py` | `from cancel_test.context import StageContext` |
| `stages/ffmpeg_extract.py` | `from cancel_test.context import CancelledError, StageContext` |
| `stages/stem_separation.py` | `from cancel_test.context import CancelledError, StageContext` |
| `stages/ffmpeg_transcode.py` | `from cancel_test.context import CancelledError, StageContext` |
| `orchestrator.py` | `from cancel_test.context import CancelledError, StageContext` |
| `run_test.py` | `from cancel_test.context import CancelledError, StageContext` |

The user renamed the folder to `cancel_separator`, but the Python package name inside all files is still `cancel_test`. This means the `cancel_separator` package is also broken at import time. The plan (step 5) says "Copy `workers/stem_worker.py` from `cancel_test/`" — but the actual source folder is `cancel_separator/` and all its internal imports use the old `cancel_test` package name.

**Fix:** When copying workers/stages, the plan should either:
- Rename all `cancel_test` → `cancel_separator` in the source files first, or
- Note that a find-and-replace of `cancel_test` → `cancel_separator` in the source tree is a prerequisite step, or
- Accept that the copy step must also rewrite imports (which is the de facto requirement)

### 3. `requirements.txt` is missing `audio-separator` and `srt`

The plan's "Dependencies" section (line 807) claims these are already in `requirements.txt`:

> Already in `requirements.txt`:
> - `stable-ts` ✓
> - `faster-whisper` ✓
> - `srt` ✗ (not present)
> - `audio-separator` ✗ (not present)

Actual `requirements.txt` contents:
```
stable-ts
faster-whisper
lyricsgenius
#torch
```

`srt` and `audio-separator` are missing. The `lyric_align.py` stage imports `srt`, and the stem worker depends on `audio-separator`. Both will fail at runtime if not installed.

**Fix:** Add `srt` and `audio-separator` to `requirements.txt`, or note in the plan that they must be installed manually.

---

## Design Issues

### 4. `artifacts` typing is `dict[str, Any]` but existing code uses `dict[str, Path]`

The plan changes `artifacts: dict[str, Path]` (in `cancel_separator/context.py`) to `artifacts: dict[str, Any]` because values are now a mix of `Path`, `str`, and `float`. This is a reasonable loosening for the pipeline, but it removes type safety. Consider using a `TypedDict` or `dataclass` for the artifacts schema — the plan already documents a complete schema table (lines 159–176), so the types are known. A `TypedDict` would give IDE autocomplete and catch key typos at static-analysis time without changing runtime behavior.

This is a suggestion, not a blocker — `dict[str, Any]` works fine for a prototype.

### 5. `lyric_align.py` SRT handling has an edge case: `.srt` lyrics input produces only `.ass`

When the input lyrics is `.srt`, the plan says (line 490):

> Only generated when the input lyrics was `.txt` — if the user provided an `.srt`, they already have one and we don't overwrite it.

But the plan's `_load_lyrics` for `.srt` input (line 404) says:

> parse with `srt.parse(...)`, concatenate `sub.content` with newlines, `lyrics_format = "srt"`.

This concatenation **destroys the original SRT timestamps** and reduces the content to plain text. If the user's original `.srt` had different segment boundaries than what stable-ts produces, the new `.ass` file's karaoke timing will not match the original `.srt` structure — it will reflect stable-ts's segmentation instead. This is by design (the whole point is to re-align), but it's worth documenting that `.srt` input is treated as raw text with no structural preservation. Users expecting their `.srt` timing to influence alignment will be surprised.

### 6. Worker subprocess logger uses hardcoded `cancel_test.worker` name

In `cancel_separator/workers/cancelable_stem_worker.py` line 519:

```python
processing_logger = logging.getLogger("cancel_test.worker")
```

When copied to `pipeline/workers/stem_worker.py`, this logger name will still say `cancel_test.worker`. It should be updated to `pipeline.workers.stem_worker` (or similar) for consistent log output. Minor, but would confuse debugging.

### 7. `StemWorker` constructor doesn't accept `model_name` as a parameter

The plan's `PipelineConfig` has `separator_model_name` (line 84), but `CancelableStemWorker.__init__()` hardcodes `MODEL_NAME` as a module-level constant (line 50). The `run_pipeline.py` CLI code (line 674–677) constructs the worker with only `temp_dir` and `model_dir` — `model_name` is never passed.

If a user changes `PipelineConfig.separator_model_name`, it will have no effect because the worker ignores it.

**Fix:** Add a `model_name` parameter to `StemWorker.__init__()` (or rename the constant to a constructor parameter), and pass `cfg.separator_model_name` from `run_pipeline.py`.

### 8. `WhisperWorker.align_and_refine()` clears the cancel event between align and refine

In `cancelable_whisper_worker.py` lines 520–521:

```python
if cancel_event is not None:
    cancel_event.clear()
```

This is problematic if the pipeline later adds cancellation support. The method clears a shared event that the **caller** owns. If the orchestrator sets the event to cancel the pipeline, `align_and_refine` would clear it between the align and refine phases, losing the cancellation signal. In the current prototype this is moot (`cancel_event=None`), but the plan says "can be wired up later without touching the workers themselves" (line 814) — this clearing behavior contradicts that claim.

**Fix:** If preserving future cancellation compatibility matters, `align_and_refine` should **not** clear the caller's event. Instead, it should check the event state at the start of `refine()` and skip if already set.

---

## Minor Issues

### 9. `FFmpegExtractStage` no longer needs `__init__`

The plan removes `self._proc` and `cancel()`, making `FFmpegExtractStage.__init__()` a no-op. The class can drop the explicit `__init__` entirely — inherited `BaseStage.__init__` (which also doesn't exist; `BaseStage` has no constructor) is sufficient. Including an empty `__init__` is harmless but adds noise.

### 10. `FFmpegTranscodeStage._transcode` signature drops `ctx` but plan code still passes it

The plan's `_transcode` signature (line 336) is:

```python
def _transcode(self, wav_path: Path, output_path: Path) -> None:
```

But the existing code passes `ctx` as a third argument (for `ctx.check_cancelled()`). After removing cancellation, `ctx` is no longer needed — the plan is correct to drop it. Just make sure the implementation matches the plan's signature and doesn't accidentally pass `ctx`.

### 11. Plan says `StemWorker` class is renamed from `CancelableStemWorker` but doesn't mention the error classes

Step 5 says "rename class `CancelableStemWorker` → `StemWorker`" but the worker module also exports `WorkerCancelledError` and `WorkerDiedError`. The `StemSeparationStage` (plan line 296–314) uses `WorkerDiedError` in its `except` clause, so these error classes must also be imported/exported from the new `pipeline/workers/stem_worker.py`. The plan doesn't mention whether to rename these (they have "Worker" not "Cancelable" in the name, so they're fine as-is), but they need to be included in the copy.

Similarly, `CancelableWhisperWorker` → `WhisperWorker` but `AlignmentCancelledError` must also be copied. It's not used by the stage (the plan drops cancellation handling), but the worker's internal code raises it — it must be present for the module to be self-consistent.

### 12. `WhisperModelConfig` constructor: plan's `build_whisper_config()` may not match actual fields

The plan's `build_whisper_config()` (lines 644–657) constructs a `WhisperModelConfig` with 11 keyword arguments. Since `config.py` doesn't exist, we can't verify these are the correct field names. Based on the worker's usage they appear correct, but any mismatch (e.g., `vad` vs `vad_options`, or `refine_steps` vs `steps`) will cause a runtime `TypeError`.

### 13. Loudnorm JSON parsing: the regex may match the wrong block

The plan (line 270) suggests:

```python
re.search(r"\{[^{}]+\}\s*$", stderr, re.DOTALL)
```

FFmpeg loudnorm with `print_format=json` outputs a JSON block to stderr. However, ffmpeg may also emit other `{`/`}` characters in log lines before the JSON block (e.g., filter descriptions). The regex `\{[^{}]+\}` matches any single-level `{...}` block with no nested braces — this should work for loudnorm's flat JSON output, but if any earlier stderr line contains `{...}`, the regex could match the wrong block. The `re.DOTALL` + `\s*$` anchor helps, but `[^{}]+` is greedy so it could over-match across multiple `{...}` blocks.

A safer approach: split stderr into lines, walk backward from the end, and find the last contiguous JSON block (as the plan also mentions as an alternative). This is what most ffmpeg loudnorm wrappers do in practice.

### 14. `ffmpeg -threads` is applied to transcode but not to extract or loudnorm

`FFmpegTranscodeStage` uses `-threads <ffmpeg_threads>` (from config), but `FFmpegExtractStage` and `LoudnormAnalyzeStage` don't pass any `-threads` flag. For consistency and performance control, all three ffmpeg-invoking stages should probably respect the same thread setting (or the config should have per-stage thread settings if different behavior is desired).

### 15. No validation that `extracted_wav` is 44.1 kHz stereo 16-bit before loudnorm

`LoudnormAnalyzeStage` depends on `extracted_wav` being 44.1 kHz stereo 16-bit (as produced by `FFmpegExtractStage`). If someone bypasses the extract stage or provides a pre-existing WAV, the loudnorm analysis might run on a different sample format/channel count. This is fine for the prototype (the pipeline always runs extract first), but worth noting if stages are ever run independently.

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | **Blocker** | `cancel_whisper/config.py` doesn't exist — worker import fails |
| 2 | **Blocker** | All `cancel_separator/` imports still use `cancel_test` — package is broken |
| 3 | **Blocker** | `srt` and `audio-separator` missing from `requirements.txt` |
| 4 | Low | `dict[str, Any]` loses type safety (consider `TypedDict`) |
| 5 | Medium | `.srt` lyrics input: original timestamps discarded, not documented |
| 6 | Low | Subprocess logger name hardcoded to `cancel_test.worker` |
| 7 | Medium | `separator_model_name` config field has no effect (worker hardcodes model name) |
| 8 | Medium | `align_and_refine()` clears shared cancel event — breaks future cancellation |
| 9 | Low | `FFmpegExtractStage.__init__()` becomes unnecessary |
| 10 | Low | `_transcode` signature change — ensure implementation matches |
| 11 | Low | Error classes (`WorkerDiedError`, `AlignmentCancelledError`) must be included in copies |
| 12 | Medium | `WhisperModelConfig` fields are unverified (source file missing) |
| 13 | Medium | Loudnorm JSON regex may match wrong `{...}` block in noisy stderr |
| 14 | Low | `-threads` flag inconsistent across ffmpeg stages |
| 15 | Low | No validation of WAV format before loudnorm stage |
