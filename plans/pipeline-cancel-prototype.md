# Pipeline Cancel Prototype

Add hook-based, phase-targeted cancellation to the `pipeline/` prototype.
The cancel mechanism is modelled after pikaraoke's `ProcessingManager`
(`_JobState` + per-step subprocess targeting), extended to cover the
extra pipeline stages (loudnorm, lyric_align) and the in-process Whisper
worker. A new CLI test driver fires cancellation at a chosen phase and
verifies that (a) the cancel lands cleanly, (b) the model-bearing
workers are still loaded afterwards. The pipeline does not re-run
after a cancel — production behaviour ends the current job and leaves
the workers ready for the *next* enqueued song.

## 1. Goal & scope

**Goal.** Make the pipeline cancellable at any time, at the granularity
of one "phase". A phase is either a stage (ffmpeg-based stages) or a
sub-step inside a stage (the three model passes inside `lyric_align`).
After cancellation:
- the pipeline aborts cleanly, no further stages run,
- per-job temp dir is removed,
- **no partial outputs are left in the song's output directory**
  (transcoded m4a files, ASS, SRT). Stages that write to the song's
  parent directory write to `tmp_dir` first and only move/rename to
  the final location after the stage's work fully completes. See
  §6.4.1 (transcode) and §6.4.2 (lyric_align).
- model-bearing workers (`StemWorker`, `WhisperWorker`) stay loaded.

**Scope.**
- Wire cancellation hooks into all five pipeline stages.
- Add a phase tracker / cancel mechanism on the orchestrator, modelled
  after `pikaraoke.lib.processing_manager.ProcessingManager`.
- Add `--phase` and `--cancel-after` CLI flags to `run_pipeline.py` for
  testing. Pipeline behaviour without those flags is unchanged.
- After a cancel fires, assert the workers are still loaded by
  inspecting their state (`stem_worker.is_alive()`,
  `whisper_worker.model_loaded`). The pipeline then exits — no
  re-run, matching production behaviour.

**Out of scope.**
- Diarization (not yet part of the pipeline).
- Job queue / multiple concurrent songs. The prototype still processes
  one song per invocation.
- Persisting partial outputs on cancel — temp dir is removed, no stems
  written.

---

## 2. Phases (the cancellation grain)

```python
class Phase(enum.Enum):
    EXTRACT          = "extract"           # ffmpeg_extract stage
    LOUDNORM         = "loudnorm"          # loudnorm_analyze stage
    STEM_SEPARATION  = "stem_separation"   # stem_separation stage (per-chunk)
    TRANSCODE        = "transcode"         # ffmpeg_transcode stage (both stems)
    ALIGN            = "align"             # lyric_align: model.align()
    TRANSCRIBE       = "transcribe"        # lyric_align: model.transcribe()
    REFINE           = "refine"            # lyric_align: model.refine()
```

Notes on `lyric_align` sub-phases:
- Alignment mode runs ALIGN → REFINE.
- Transcription mode runs TRANSCRIBE → REFINE.
- The stage opens a fresh `ctx.cancel.activity(Phase.ALIGN, ...)`
  scope per call; that's where the sub-phase tracking lives.

Cancellation mechanism by phase:

| Phase            | Mechanism                                                 |
|------------------|-----------------------------------------------------------|
| EXTRACT          | `Popen.kill()` of the ffmpeg subprocess                   |
| LOUDNORM         | `Popen.kill()` of the ffmpeg subprocess                   |
| STEM_SEPARATION  | `cancel_event.set()` → forwarded to worker via Pipe       |
| TRANSCODE        | `Popen.kill()` of the active ffmpeg subprocess            |
| ALIGN            | `cancel_event.set()` → forward pre-hook on `model.encoder` raises |
| TRANSCRIBE       | `cancel_event.set()` → forward pre-hook on `model.encoder` raises |
| REFINE           | `cancel_event.set()` → forward pre-hook on `model.encoder` raises |

The `cancel_event` is **shared** across stem and whisper workers — the
orchestrator owns one `threading.Event` per job. Stem worker bridges it
to a Pipe internally (already implemented in
`pipeline/workers/stem_worker.py:152-158`).

---

## 3. Files affected

```
pipeline/
├── context.py              MOD: add Phase, Cancellable + KillProcess + SetEvent,
│                                CancelToken (with activity() contextmanager),
│                                PipelineCancelled; extend StageContext.cancel
├── orchestrator.py         MOD: split run() → start/run_one/run_one_async/join/stop;
│                                add cancel_active(); expose stem_worker + whisper_worker
│                                properties; between-stage check_cancelled()
├── run_pipeline.py         MOD: add --cancel-after / --phase, embed run_cancel_test()
├── stages/
│   ├── _ffmpeg_helpers.py  NEW: run_ffmpeg(cmd, ctx, phase, capture_stderr=False)
│   ├── base.py             NO CHANGE
│   ├── ffmpeg_extract.py   MOD: subprocess.run → run_ffmpeg under activity scope
│   ├── loudnorm_analyze.py MOD: same, capture_stderr=True; JSON parsing unchanged
│   ├── stem_separation.py  MOD: wrap worker.separate() in activity(SetEvent(event));
│   │                            REMOVE stale auto-start path (§6.3, issue #8)
│   ├── ffmpeg_transcode.py MOD: run_ffmpeg per transcode; check_cancelled() between;
│   │                            write to tmp_dir, shutil.move() on success (§6.6.1)
│   └── lyric_align.py      MOD: split align_and_refine; one activity scope per
│                                Phase.ALIGN / Phase.TRANSCRIBE / Phase.REFINE;
│                                write ASS/SRT to tmp_dir, shutil.move() on success (§6.6.2)
└── workers/
    ├── stem_worker.py      MOD: drain cancel pipe at START of separate() (§6.3.1)
    └── whisper_worker.py   MOD: replace model.encode() monkey-patch with
                                 register_forward_pre_hook on model.encoder (§6.7)
```

One new file (`_ffmpeg_helpers.py`); the CLI driver lives inside
`run_pipeline.py` — same approach as `cancel_tests/*/run_test.py`.

---

## 4. New data structures

### 4.1. `Phase` enum
Defined in `pipeline/context.py` (so both orchestrator and stages can
import it without a circular dep on `orchestrator.py`).

```python
import enum

class Phase(enum.Enum):
    EXTRACT          = "extract"
    LOUDNORM         = "loudnorm"
    STEM_SEPARATION  = "stem_separation"
    TRANSCODE        = "transcode"
    ALIGN            = "align"
    TRANSCRIBE       = "transcribe"
    REFINE           = "refine"
```

### 4.2. `Cancellable` protocol
Defined in `pipeline/context.py`. Stages produce a `Cancellable` for
each in-flight operation; the orchestrator only knows the abstraction.

```python
class Cancellable(Protocol):
    def cancel(self) -> None: ...

class KillProcess:
    """Cancel an ffmpeg subprocess by SIGKILL."""
    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
    def cancel(self) -> None:
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass

class SetEvent:
    """Cancel a model worker call by setting its threading.Event."""
    def __init__(self, event: threading.Event) -> None:
        self._event = event
    def cancel(self) -> None:
        self._event.set()
```

Adding a new cancellation mechanism = a new `Cancellable` subclass.
Nothing in the orchestrator changes.

### 4.3. `CancelToken`
Defined in `pipeline/context.py`. Owned by the orchestrator, referenced
from `StageContext`. Holds the active `Cancellable` and the sticky
`cancelled` flag — that's it.

```python
# eq=False/repr=False: threading.Lock is not equality-comparable and its
# default repr is unhelpful. Plain class works too; keeping @dataclass
# only for the field declarations.
@dataclass(eq=False, repr=False)
class CancelToken:
    event:     threading.Event                # per-job model-worker event (see §5.2)
    cancelled: bool = False                   # sticky flag, set by cancel()
    phase:     Optional[Phase] = None         # current phase (None = between activities)
    active:    Optional[Cancellable] = None   # registered cancel target
    lock:      threading.Lock = field(default_factory=threading.Lock)

    # Stage-facing -----------------------------------------------------------
    @contextmanager
    def activity(self, phase: Phase, target: Cancellable):
        """Open an activity for `phase`, with `target` as the cancel mechanism.

        Atomically sets phase + active under the lock. On exit (normal,
        exception, or cancel), atomically clears them. If a cancel
        arrived before the activity opened, raises PipelineCancelled
        immediately so the work inside never starts.

        On exit, only synthesises PipelineCancelled if the work itself
        did NOT raise — otherwise we'd mask the original exception
        (real RuntimeError, AlignmentCancelledError already translated
        by the stage, etc.). Python's exception chaining preserves
        __context__ either way, but masking the displayed exception
        makes debugging painful.
        """
        with self.lock:
            if self.cancelled:
                raise PipelineCancelled(phase)
            self.phase = phase
            self.active = target
        try:
            yield
        finally:
            with self.lock:
                self.active = None
            # Only raise if the work itself completed without an exception.
            # If yield raised (PipelineCancelled, RuntimeError, anything),
            # let it propagate untouched.
            if self.cancelled and sys.exc_info()[0] is None:
                raise PipelineCancelled(phase)

    def is_cancelled(self) -> bool:
        with self.lock:
            return self.cancelled

    def check_cancelled(self) -> None:
        """Raise PipelineCancelled if cancelled is set. Used between stages."""
        with self.lock:
            if self.cancelled:
                raise PipelineCancelled(self.phase)

    # Driver-facing ----------------------------------------------------------
    def get_phase(self) -> Optional[Phase]:
        with self.lock:
            return self.phase

    def cancel(self) -> None:
        """Mark cancelled, signal whichever Cancellable is currently active.

        Sets cancelled=True FIRST (under the lock), THEN calls
        target.cancel(). This ordering means by the time the cancelled
        work observes the cancel (Popen wakes up from SIGKILL, worker
        sees the event), `cancelled` is already True — so the
        activity()'s exit check reliably sees it and raises
        PipelineCancelled. Do not reorder.

        - If an activity is open: call its `.cancel()` (kills Popen,
          sets event, etc., depending on the Cancellable subclass).
        - If no activity is open (between stages): just set the flag;
          the next `check_cancelled()` raises.
        """
        with self.lock:
            self.cancelled = True
            target = self.active
        if target is not None:
            target.cancel()
```

Key invariants:
- **phase + active are always set/cleared together, under the lock.**
  No way to register a Popen without setting the phase, no way to set a
  phase without registering a target.
- **`cancelled` is set before any `target.cancel()` call.** The
  activity's exit check then reliably sees it.

The four `set_phase`/`clear_phase`/`register_process`/`clear_process`
methods of the previous design are replaced by one `activity()` context
manager.

### 4.4. `PipelineCancelled` exception
Defined in `pipeline/context.py`. Single exception type the orchestrator
catches; stages translate worker-specific cancel errors to it.

```python
class PipelineCancelled(Exception):
    """Raised when the active job has been cancelled."""
    def __init__(self, phase: Optional[Phase] = None):
        self.phase = phase
```

### 4.5. `StageContext` extension

```python
@dataclass
class StageContext:
    song_path: Path
    tmp_dir:   Path
    config:    PipelineConfig
    artifacts: dict[str, Any] = field(default_factory=dict)
    cancel:    Optional[CancelToken] = None    # NEW: None when run without cancel
```

When the orchestrator runs without cancel support (which is still the
default — `--cancel-after` not provided), `cancel` stays `None` and
stages take their existing fast paths.

---

## 5. Orchestrator changes

`pipeline/orchestrator.py` gets the pikaraoke pattern grafted on, but
single-job (no queue).

### 5.1. New fields on `PipelineOrchestrator`

```python
self._cancel_token: Optional[CancelToken] = None
self._pipeline_thread: Optional[threading.Thread] = None
self._result: dict[str, Any] = {}    # {"ctx": ..., "exception": ...}
```

### 5.2. New methods

Naming convention: `run_one*` = one song, sync or async. `start`/`stop`
manage worker lifecycle independently of any single job.

```python
def start(self) -> None:
    """Spawn workers (idempotent). Driver calls this once before any run."""

def stop(self) -> None:
    """Tear down workers (idempotent). Driver calls this once at end."""

def run_one(self, song_path, lyrics_path=None) -> StageContext:
    """Run one song synchronously. Convenience wrapper around
    run_one_async() + join(). Workers must already be started."""

def run_one_async(self, song_path, lyrics_path=None) -> CancelToken:
    """Start the pipeline on a background thread, return its
    CancelToken. Each call creates a FRESH CancelToken and a FRESH
    threading.Event — never reused across jobs. The driver uses the
    returned token to observe phase and call cancel_active().
    Workers must already be started.
    """

def join(self, timeout: Optional[float] = None) -> StageContext:
    """Wait for the pipeline thread to finish. Re-raises whatever
    exception the thread caught (PipelineCancelled, FileNotFoundError,
    RuntimeError, etc.) — the caller is responsible for distinguishing
    cancellation from other failures."""

def cancel_active(self) -> None:
    """Request cancellation of the currently-running pipeline.
    Delegates to self._cancel_token.cancel()."""

# Read-only access for the test driver (avoids passing workers through
# main → run_cancel_test as separate args):
@property
def stem_worker(self) -> StemWorker: ...
@property
def whisper_worker(self) -> WhisperWorker: ...
```

**Per-job freshness rule (closes review issue #3).** Each
`run_one_async` call constructs a *new* `CancelToken(event=threading.Event())`.
The previous job's token (if any) is discarded; its event keeps the
default unset state forever. This isolates jobs from each other —
in particular, the StemWorker's leaked cancel-forwarder daemon thread
from a prior job is waiting on a stale event that nobody can set, so
it cannot inject a spurious cancel into the next job.

### 5.3. Modified methods

The existing `run()` is folded into `run_one()`:

```python
def run(self, song_path, lyrics_path=None) -> StageContext:
    """Back-compat: start workers, run one song, stop workers."""
    self.start()
    try:
        return self.run_one(song_path, lyrics_path)
    finally:
        self.stop()
```

`_run_pipeline()` accepts a `CancelToken` and wires it onto the
`StageContext`. Phase tracking is owned entirely by stages via
`activity()` — the orchestrator's only cancel-aware responsibility is
checking between stages:

```python
def _run_pipeline(self, song_path, lyrics_path, cancel_token):
    ...
    ctx = StageContext(
        song_path=song_path,
        tmp_dir=tmp_dir,
        config=self._config,
        cancel=cancel_token,
    )
    ctx.artifacts["lyrics_path"] = lyrics_path
    try:
        for stage in self._stages:
            if cancel_token is not None:
                cancel_token.check_cancelled()       # cancel between stages
            stage.run(ctx)
        return ctx
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

No `_STAGE_TO_PHASE` map, no `set_phase`/`clear_phase` calls. The
phase is whatever activity is currently open inside whichever stage is
running. When no activity is open (between stages, or before the first
one), `cancel_token.phase` retains its last value — that value gets
attached to `PipelineCancelled` if a cancel races with the stage
boundary.

### 5.4. Worker lifecycle stays as is

`_start_workers()` / `_stop_workers()` are unchanged — workers are
loaded before the pipeline thread starts and unloaded after it finishes.

---

## 6. Stage changes

### 6.1. `stages/base.py`
Unchanged signature: `run(ctx) -> None`. The `BaseStage` keeps no new
state; cancel plumbing flows through `ctx.cancel`.

### 6.2. FFmpeg stages — `Popen` + `activity()`

A free function in `pipeline/stages/_ffmpeg_helpers.py`:

```python
def run_ffmpeg(cmd: list[str], ctx: StageContext, phase: Phase,
               *, capture_stderr: bool = False) -> str:
    """Spawn ffmpeg under an activity scope, wait, raise on cancel/error.

    The Popen is wrapped in a KillProcess Cancellable so the orchestrator
    can SIGKILL it via cancel_token.cancel(). The activity() contextmanager
    ensures phase + cancellable are registered atomically and always cleared.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )

    stderr_data = b""
    if ctx.cancel is not None:
        with ctx.cancel.activity(phase, KillProcess(proc)):
            if capture_stderr:
                stderr_data, _ = proc.communicate()
            else:
                proc.wait()
        # activity() re-raises PipelineCancelled on exit if cancelled
    else:
        # No cancel support — straight wait, no activity bookkeeping.
        if capture_stderr:
            stderr_data, _ = proc.communicate()
        else:
            proc.wait()

    if proc.returncode != 0:
        # If activity() raised PipelineCancelled, we never reach here —
        # CancelToken.cancel() sets cancelled=True before SIGKILLing, so the
        # activity's exit check sees it. The check below is defensive against
        # any future refactor that breaks that ordering.
        if ctx.cancel is not None and ctx.cancel.is_cancelled():
            raise PipelineCancelled(phase)
        raise RuntimeError(f"ffmpeg failed (exit code {proc.returncode})")

    return stderr_data.decode("utf-8", errors="replace") if capture_stderr else ""
```

Per stage:

- **`ffmpeg_extract.py`**: `run_ffmpeg(cmd, ctx, Phase.EXTRACT)`.
- **`loudnorm_analyze.py`**: `run_ffmpeg(cmd, ctx, Phase.LOUDNORM,
  capture_stderr=True)`. Existing `_extract_json_from_stderr` parses
  the returned string unchanged.
- **`ffmpeg_transcode.py`**: two `run_ffmpeg(... Phase.TRANSCODE ...)`
  calls (vocal + instrumental), each opens its own activity. Between
  them, call `ctx.cancel.check_cancelled()` if `ctx.cancel is not
  None` — a cancel arriving after the first transcode finishes (so
  outside any activity scope) is caught here.

### 6.3. `stages/stem_separation.py`

Today ([pipeline/stages/stem_separation.py:48](pipeline/stages/stem_separation.py#L48)):
```python
vocal_wav, instrumental_wav = self._worker.separate(
    wav_path=extracted_wav,
    output_dir=ctx.tmp_dir,
    cancel_event=None,
)
```

Change to use an activity scope around the worker call:
```python
if ctx.cancel is not None:
    cancel_event = ctx.cancel.event
    try:
        with ctx.cancel.activity(Phase.STEM_SEPARATION, SetEvent(cancel_event)):
            vocal_wav, instrumental_wav = self._worker.separate(
                wav_path=extracted_wav,
                output_dir=ctx.tmp_dir,
                cancel_event=cancel_event,
            )
    except WorkerCancelledError:
        # Worker raised before activity() noticed cancelled flag; translate.
        raise PipelineCancelled(Phase.STEM_SEPARATION)
else:
    vocal_wav, instrumental_wav = self._worker.separate(
        wav_path=extracted_wav,
        output_dir=ctx.tmp_dir,
        cancel_event=None,
    )
```

`WorkerDiedError` keeps its current treatment (raise `RuntimeError`).

**Remove the stale auto-start (closes review issue #8).** The current
code at [pipeline/stages/stem_separation.py:39-41](pipeline/stages/stem_separation.py#L39-L41) auto-starts the worker if it
appears dead:

```python
if not self._worker.is_alive():
    logger.info(f"[{self.name}] Worker not alive — starting")
    self._worker.start()
```

Delete these lines. With orchestrator-managed lifecycle, a dead worker
mid-pipeline is a real failure — silently restarting masks it from the
test driver's liveness check (the new subprocess passes `is_alive()`
but lost the loaded model). Let `worker.separate()` raise
`WorkerDiedError`; the stage already translates that to `RuntimeError`,
which surfaces cleanly through `_run_pipeline`.

### 6.3.1. `workers/stem_worker.py` — drain stale cancel signal at job start

Closes review issue #4. Add one line at the top of `StemWorker.separate()`:

```python
def separate(self, wav_path, output_dir, cancel_event=None):
    ...
    self._drain_cancel_pipe()    # NEW: clear any stale signal from prior job
    js.send((str(wav_path), str(output_dir)))
    ...
```

Why: a leaked cancel-forwarder daemon from a prior cancelled job can
race with the main-side drain in that job's `finally`. If the
forwarder's `send(1)` lands AFTER the drain, the signal sits in the
pipe and gets consumed by the next `separate()` as a spurious cancel.
Per-job `CancelToken` (§5.2) prevents the forwarder from firing on a
*different* job's event, but doesn't help if the forwarder was already
in flight when the prior job ended. Draining at the start of each
`separate()` is idempotent and cheap.

### 6.4. `stages/lyric_align.py` — sub-phase tracking

Each sub-phase opens its own activity scope. The worker's
`align_and_refine` / `transcribe_and_refine` wrappers are too coarse
(they fold two model calls into one), so the stage calls `align()`,
`transcribe()`, `refine()` directly.

```python
def _model_call(ctx, phase: Phase, fn):
    """Wrap a single model call in an activity scope (or run it bare
    when ctx.cancel is None)."""
    if ctx.cancel is None:
        return fn()
    cancel_event = ctx.cancel.event
    try:
        with ctx.cancel.activity(phase, SetEvent(cancel_event)):
            return fn()
    except AlignmentCancelledError:
        raise PipelineCancelled(phase)

if lyrics_path is not None:
    # Alignment mode: ALIGN → REFINE
    result = _model_call(ctx, Phase.ALIGN,
        lambda: self._worker.align(vocal_wav, lyrics_text,
                                   ctx.cancel.event if ctx.cancel else None))
    result = _model_call(ctx, Phase.REFINE,
        lambda: self._worker.refine(vocal_wav, result,
                                    ctx.cancel.event if ctx.cancel else None))
else:
    # Transcription mode: TRANSCRIBE → (regroup) → REFINE
    result = _model_call(ctx, Phase.TRANSCRIBE,
        lambda: self._worker.transcribe(vocal_wav,
                                        ctx.cancel.event if ctx.cancel else None))
    # NOTE: regroup is fast (CPU-bound, milliseconds) and runs OUTSIDE any
    # activity scope. It's not cancellable. A cancel arriving during regroup
    # is caught by REFINE's activity() on entry via the sticky `cancelled`
    # flag, so it lands at the next phase boundary. Acceptable.
    # FIELD NAME: PipelineConfig has `whisper_regroup`, WhisperModelConfig
    # has `regroup` — they map via build_whisper_config(). The stage reads
    # the PipelineConfig field.
    if self._config.whisper_regroup:
        result.regroup(self._config.whisper_regroup)
    result = _model_call(ctx, Phase.REFINE,
        lambda: self._worker.refine(vocal_wav, result,
                                    ctx.cancel.event if ctx.cancel else None))
```

The `event` is reused across activity scopes within one job. The
worker's `_terminate_orphaned_audioloaders()` is the existing cleanup
path inside `whisper_worker.py` — no change there. Note: `event` is
**not** cleared between activities within one job. If a cancel arrives
during ALIGN, it sets the event; the ALIGN activity raises
`PipelineCancelled` on exit, which short-circuits before REFINE ever
opens its scope.

Edge case: cancel arrives between activities (after ALIGN succeeded but
before REFINE opens). `cancelled` is sticky → REFINE's `activity()`
sees it on entry and raises immediately, before any model call.

Everything else in `lyric_align.py` (line matching, ASS/SRT generation)
runs unchanged after the model calls return — those code paths do no
heavy work and don't need cancellation.

### 6.5. `stages/loudnorm_analyze.py` — stderr capture
Uses the same `run_ffmpeg(cmd, ctx, Phase.LOUDNORM, capture_stderr=True)`
helper. Existing `_extract_json_from_stderr` parses the returned string
unchanged.

### 6.6. Orphan-output prevention (closes review issue #6)

`ffmpeg_transcode` and `lyric_align` write final outputs to
`song_path.parent` — outside `tmp_dir`, so the orchestrator's
`shutil.rmtree(tmp_dir)` won't clean them up. A cancel partway through
either stage would leave orphan files (e.g. vocal m4a without
nonvocal, or a partial ASS file).

**Strategy: write to `tmp_dir`, atomically rename only after the stage
fully succeeds.** `os.replace()` is atomic on POSIX when source and
destination are on the same filesystem. Since `tmp_dir` is created
under `intermediate_dir` (or system tmp), the rename may cross
filesystems; in that case use `shutil.move()` which falls back to
copy+delete. Both stages write a small handful of files, so the
copy-fallback cost is negligible.

#### 6.6.1. `stages/ffmpeg_transcode.py` — write to tmp, rename on success

```python
def run(self, ctx: StageContext) -> None:
    vocal_wav      = ctx.artifacts["vocal_wav"]
    instrumental_wav = ctx.artifacts["instrumental_wav"]

    # Final destinations (created lazily; only after both transcodes succeed)
    song = ctx.song_path
    final_vocal    = song.parent / "vocal"    / f"{song.stem}---vocal.m4a"
    final_nonvocal = song.parent / "nonvocal" / f"{song.stem}---nonvocal.m4a"

    # Write transcoded output INSIDE tmp_dir
    tmp_vocal    = ctx.tmp_dir / final_vocal.name
    tmp_nonvocal = ctx.tmp_dir / final_nonvocal.name

    self._transcode(vocal_wav, tmp_vocal, ctx)
    if ctx.cancel is not None:
        ctx.cancel.check_cancelled()      # cancel between vocal and instrumental
    self._transcode(instrumental_wav, tmp_nonvocal, ctx)

    # Both transcodes succeeded — promote tmp files to final locations.
    # mkdir destinations, then move atomically (on same FS) or fall back.
    final_vocal.parent.mkdir(exist_ok=True)
    final_nonvocal.parent.mkdir(exist_ok=True)
    shutil.move(str(tmp_vocal),    str(final_vocal))
    shutil.move(str(tmp_nonvocal), str(final_nonvocal))

    ctx.artifacts["vocal_m4a"]    = final_vocal
    ctx.artifacts["nonvocal_m4a"] = final_nonvocal
```

A cancel during either `_transcode()` raises `PipelineCancelled` from
inside `run_ffmpeg`'s activity scope; the partial m4a sits in
`tmp_dir` and is wiped by the orchestrator's `shutil.rmtree(tmp_dir)`.
A cancel between the two transcodes is caught by the explicit
`check_cancelled()` between them — first transcode's tmp file is
likewise wiped by `tmp_dir` cleanup. The final destinations are never
written to until both transcodes have succeeded.

#### 6.6.2. `stages/lyric_align.py` — write ASS/SRT to tmp, move on success

```python
# (after _generate_ass / _generate_srt have produced the strings)
tmp_ass = ctx.tmp_dir / f"{ctx.song_path.stem}.ass"
tmp_ass.write_text(ass_content, encoding="utf-8")
if write_srt:
    tmp_srt = ctx.tmp_dir / f"{ctx.song_path.stem}.srt"
    tmp_srt.write_text(srt_content, encoding="utf-8")

# Both writes succeeded — promote to final locations.
karaoke_dir   = ctx.song_path.parent / "karaoke"
subtitles_dir = ctx.song_path.parent / "subtitles"
karaoke_dir.mkdir(exist_ok=True)
final_ass = karaoke_dir / tmp_ass.name
shutil.move(str(tmp_ass), str(final_ass))
ctx.artifacts["ass_file"] = final_ass

if write_srt:
    subtitles_dir.mkdir(exist_ok=True)
    final_srt = subtitles_dir / tmp_srt.name
    shutil.move(str(tmp_srt), str(final_srt))
    ctx.artifacts["srt_file"] = final_srt
```

ASS/SRT generation is purely CPU-bound — it's not cancellable in
practice (no checkpoints, no subprocesses). The only risk is a cancel
arriving during the move itself, which is too short to matter. If that
happens, one of the two files might already be moved; the next
`check_cancelled()` (between this stage and any downstream stage)
catches it. In the current pipeline `lyric_align` is the last stage,
so this is a non-issue.

### 6.7. `workers/whisper_worker.py` — hook-based cancellation

`stable_whisper.load_model()` returns `whisper.model.Whisper` — a PyTorch
`nn.Module`. This class has no `.encode()` method; the monkey-patch approach
from the older `cancel_whisper` prototype (which targeted faster-whisper's
CTranslate2 wrapper) cannot work here.

Instead, use `register_forward_pre_hook()` on the encoder `nn.Module`:

1. Cache `self._encoder_module = self._model.encoder` at `load_model()` time.
   Log a warning if it is not an `nn.Module` (graceful degradation).
2. In `align()`, `refine()`, and `transcribe()`, when `cancel_event` is
   provided, register a pre-hook that raises `_CancelledInsideEncoder` if
   the event is set before each forward pass.
3. In a `finally` block, call `handle.remove()` (wrapped in a bare `except`
   to avoid masking the in-flight exception).
4. Clear `self._encoder_module = None` in `unload_model()`.

The hook fires through `nn.Module.__call__` at the same call sites the old
monkey-patch targeted. Exception unwinding is identical: weights survive,
orphaned AudioLoaders are cleaned up, and `AlignmentCancelledError` is raised.

Adapted from `cancel_tests/whisper/whisper_worker.py`.

---

## 7. CLI changes — `run_pipeline.py`

Add two flags:

```python
parser.add_argument("--cancel-after", type=float, default=None,
                    help="Seconds to wait after --phase begins before cancelling.")
parser.add_argument("--phase",
                    choices=[p.value for p in Phase],
                    default=None,
                    help="Phase at which to fire the cancel timer.")
```

Behaviour matrix:

| `--cancel-after` | `--phase` | Behaviour                                              |
|------------------|-----------|--------------------------------------------------------|
| absent           | absent    | Run pipeline normally (current behaviour, unchanged).  |
| present          | absent    | Error — must specify a phase.                          |
| absent           | present   | Error — must specify a delay.                          |
| present          | present   | Run in test mode (see §8).                             |

### 7.1. Test driver — `run_cancel_test()`

The driver does **not** re-run the pipeline after cancel. It just
verifies the workers are still alive, then tears them down. This
matches production: pikaraoke's `ProcessingManager` cancels the
current job, leaves the worker ready for the next enqueued song, and
moves on.

Pseudocode (closes review issues #15, #16, #19):

```python
def run_cancel_test(orchestrator, song_path, lyrics_path,
                    target_phase, cancel_after) -> int:
    """Driver pulls workers off the orchestrator; no extra args needed."""
    log.info(f"=== Cancel test: target phase = {target_phase.value}, "
             f"cancel-after = {cancel_after}s ===")

    cancel_token = orchestrator.run_one_async(song_path, lyrics_path)

    # Wait for the target phase to start. 120s is plenty: extract is
    # ~2s, loudnorm ~10s, stem_separation ~30s on GPU. If we don't see
    # the target phase in 120s, something is wrong (or the phase is
    # already past).
    if not _wait_for_phase(cancel_token, target_phase, timeout=120):
        log.warning(
            f"Phase {target_phase.value} never reached within 120s — "
            f"pipeline may have completed first. Joining without cancel."
        )
        try:
            orchestrator.join()
        except Exception as e:
            log.error(f"Pipeline failed during fallback join: {e}")
            return 1
        return 0

    log.info(f"Phase {target_phase.value} entered — sleeping {cancel_after}s")
    time.sleep(cancel_after)

    # Confirm we are still in the target phase before firing.
    current = cancel_token.get_phase()
    if current != target_phase:
        log.warning(
            f"Pipeline left {target_phase.value} (now {current}) before "
            f"cancel timer fired. Try a smaller --cancel-after."
        )

    log.info(">>> SETTING CANCEL <<<")
    orchestrator.cancel_active()

    # Wait for the pipeline thread to finish.
    try:
        orchestrator.join(timeout=120)
        log.error("Pipeline completed instead of cancelling — "
                  "raise --cancel-after")
        return 1
    except PipelineCancelled as e:
        cancelled_phase = e.phase.value if e.phase else "?"
        log.info(f"Pipeline cancelled at phase {cancelled_phase}")
    except Exception as e:
        # Any non-cancel failure: report cleanly instead of letting it
        # propagate as an unhandled exception.
        log.error(f"Pipeline failed with unexpected error: {type(e).__name__}: {e}")
        return 1

    # --- Worker liveness assertion (no re-run) ---
    log.info("=== Verifying workers survived cancellation ===")
    stem_alive    = orchestrator.stem_worker.is_alive()
    whisper_loaded = orchestrator.whisper_worker.model_loaded
    log.info(f"  StemWorker subprocess alive: {stem_alive}")
    log.info(f"  WhisperWorker model loaded:  {whisper_loaded}")

    if not stem_alive:
        log.error("FAIL: StemWorker subprocess died — model would need reload.")
        return 1
    if not whisper_loaded:
        log.error("FAIL: WhisperWorker model unloaded — would need reload.")
        return 1

    log.info("PASS: both workers still loaded after cancellation.")
    return 0
```

`_wait_for_phase` polls `cancel_token.get_phase()` every 100 ms; uses
`is_cancelled()` (lock-acquired) to be free-threading-safe:

```python
def _wait_for_phase(token, target, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if token.get_phase() == target:
            return True
        if token.is_cancelled():
            return False
        time.sleep(0.1)
    return False
```

Driver lifecycle (called from `main()`):

```python
orchestrator = PipelineOrchestrator(...)
orchestrator.start()                    # load workers once
try:
    if args.phase and args.cancel_after is not None:
        target_phase = Phase(args.phase)   # enum lookup by value
        rc = run_cancel_test(orchestrator, song_path, lyrics_path,
                             target_phase, args.cancel_after)
    else:
        try:
            orchestrator.run_one(song_path, lyrics_path)
            rc = 0
        except Exception as e:
            log.error(f"Pipeline failed: {e}")
            rc = 1
finally:
    orchestrator.stop()                 # unload workers
sys.exit(rc)
```

The driver does not pass workers as separate args — it reads them off
the orchestrator's read-only properties (§5.2).

This requires splitting the existing `run()` into:
- `start()` — spawn workers (idempotent).
- `run_one(song, lyrics)` — single-song pipeline, synchronous.
- `run_one_async(song, lyrics)` — single-song pipeline on a thread,
  returns a fresh `CancelToken` with a fresh `threading.Event`.
- `stop()` — tear down workers.

`run()` (the existing public method) becomes
`start()` + `run_one()` + `stop()` for backwards compatibility. The
split is needed because the driver inspects worker liveness *after*
the pipeline thread exits but *before* workers are stopped.

---

## 8. End-to-end flow (cancel scenario)

Timeline for `python -m pipeline.run_pipeline song.mp4 --phase stem_separation --cancel-after 5`:

```
main thread                        pipeline thread                worker procs
-----------                        ---------------                ------------
run_pipeline.main()
  orchestrator.start()             [workers preloaded]
  orchestrator.run_one_async()
                                   _run_pipeline()
                                     stage: ffmpeg_extract
                                       with activity(EXTRACT, KillProcess(p)):
                                         Popen ffmpeg; proc.wait() → ok
                                       (activity exits, active=None)
                                     stage: loudnorm_analyze
                                       with activity(LOUDNORM, KillProcess(p)):
                                         proc.communicate() → ok
                                     stage: stem_separation
                                       with activity(STEM_SEPARATION, SetEvent(ev)):
  _wait_for_phase(STEM_SEPARATION)      worker.separate(event=ev)
  → returns True                                                  StemWorker subproc:
  time.sleep(5)                                                     model_run.forward
                                                                    [chunk 1, 2, ...]
  orchestrator.cancel_active()
    token.cancel()
    → event.set()
                                                                    forward() polls pipe
                                                                    → _CancelledInsideDemix
                                                                    → ("cancelled",)
                                       worker.separate raises
                                         WorkerCancelledError
                                       stage translates →
                                         PipelineCancelled
                                       _run_pipeline finally:
                                         shutil.rmtree(tmp_dir)
                                       thread stores exception
  orchestrator.join() → re-raises
    PipelineCancelled
  assert stem_worker.is_alive()
  assert whisper_worker.model_loaded
  orchestrator.stop()              [unload workers; exit]
```

For ffmpeg phases (EXTRACT/LOUDNORM/TRANSCODE) the cancel kills the
Popen; in the normal path the activity's `__exit__` raises
`PipelineCancelled` because `cancelled` was set before the SIGKILL.
The defensive `is_cancelled()` check after `proc.wait()` (§6.2) is a
safety net — if any future refactor breaks the cancel→kill ordering,
it still surfaces as `PipelineCancelled` rather than `RuntimeError`.

---

## 9. Worker liveness check (post-cancel assertion)

After `PipelineCancelled` is caught, the driver inspects the workers
without running another job:

| Check                                  | Implementation                                  |
|----------------------------------------|-------------------------------------------------|
| `StemWorker` subprocess still alive    | `stem_worker.is_alive()` (existing API)         |
| `WhisperWorker` model still loaded     | `whisper_worker.model_loaded` (existing prop)   |

A failure on either check means the cancel mechanism damaged the
worker, which is the bug we're trying to avoid. There is no automatic
re-run — confirming the worker is *capable* of taking another job is
sufficient evidence for the prototype.

## 10. Cancel ordering & race considerations

The `activity()` contextmanager closes most race windows by
construction. The remaining cases:

1. **Cancel between stages (no activity open).**
   Caught by `cancel_token.check_cancelled()` at the top of each
   loop iteration in `_run_pipeline()`. `phase` retains its last
   value, so `PipelineCancelled.phase` reports whichever stage just
   finished. Acceptable.

2. **Cancel between the two transcode calls in `ffmpeg_transcode`.**
   First `run_ffmpeg(... TRANSCODE ...)` exits its activity; second
   hasn't opened yet. Bridge with an explicit
   `ctx.cancel.check_cancelled()` between them.

3. **Cancel between ALIGN and REFINE in `lyric_align`.**
   ALIGN's `activity()` exits cleanly. The sticky `cancelled` flag is
   still set. REFINE's `activity()` checks `self.cancelled` on entry
   and raises `PipelineCancelled(Phase.REFINE)` before any model call.

4. **Cancel arriving *during* `activity()` registration.**
   The `with self.lock:` block at the top of `activity()` orders this:
   either `cancel()` won the lock (sets `cancelled=True`, sees no
   `active` yet, just sets the flag) and `activity()` then raises on
   entry, or `activity()` won the lock first (sets `active`) and
   `cancel()` then calls `target.cancel()`. No way for both to
   "miss" each other.

5. **Cancel arriving after pipeline already completed.**
   `_run_pipeline` has finished, the thread is exiting. `cancel()`
   sets a flag that nobody reads; harmless. `join()` returns the
   completed `ctx`.

6. **Race between cancel and worker raising naturally.**
   If a worker raises `WorkerCancelledError` /
   `AlignmentCancelledError` of its own accord (not because we
   cancelled), the stage translates it to `PipelineCancelled`
   regardless. `cancelled` may be False in that case — fine, the
   exception is the source of truth.

---

## 11. Edge cases & failure modes

| Situation                                | Handling                                                                 |
|------------------------------------------|--------------------------------------------------------------------------|
| `--phase` value but pipeline finishes target phase before timer fires | Log warning, do not error. Let pipeline finish normally; print "cancel timer never armed". |
| Pipeline completes before `_wait_for_phase` ever sees the target phase | Same — driver logs warning, exits 0 with completion summary.            |
| Stem worker dies mid-cancel              | `WorkerDiedError` is raised; driver treats as test failure (exit 1).     |
| Whisper alignment cancelled but model not loaded (config error) | Existing `RuntimeError("Model not loaded")` propagates; not a cancel.    |
| User passes `--cancel-after` but no `--phase` (or vice versa) | argparse-level `parser.error()` with a helpful message.                 |
| Worker liveness check fails after cancel | Driver logs FAIL and returns non-zero exit code.                         |

---

## 12. Implementation order

1. **`pipeline/context.py`**: add `Phase`, `Cancellable` protocol,
   `KillProcess`, `SetEvent`, `CancelToken` (with `activity()`
   contextmanager that uses `sys.exc_info()` to avoid masking
   in-flight exceptions, plus `is_cancelled()` lock-acquired helper),
   `PipelineCancelled`. Use `@dataclass(eq=False, repr=False)` to
   avoid `threading.Lock` equality/repr issues. Extend `StageContext`
   with `cancel` field.
2. **`pipeline/workers/stem_worker.py`**: add
   `self._drain_cancel_pipe()` at the top of `separate()` to clear
   any stale signal from a leaked prior-job forwarder thread.
3. **`pipeline/stages/_ffmpeg_helpers.py`** (new): `run_ffmpeg(cmd,
   ctx, phase, capture_stderr=False)`. Defensive `is_cancelled()`
   check on non-zero exit before raising `RuntimeError`.
4. **`pipeline/stages/ffmpeg_extract.py`**: switch to
   `run_ffmpeg(cmd, ctx, Phase.EXTRACT)`.
5. **`pipeline/stages/loudnorm_analyze.py`**: switch to
   `run_ffmpeg(cmd, ctx, Phase.LOUDNORM, capture_stderr=True)`.
6. **`pipeline/stages/ffmpeg_transcode.py`**: switch to `run_ffmpeg`
   for both transcodes, add `check_cancelled()` between them. Write
   to `ctx.tmp_dir`; `shutil.move()` to final destinations only after
   both transcodes succeed.
7. **`pipeline/stages/stem_separation.py`**: wrap worker call in
   `ctx.cancel.activity(Phase.STEM_SEPARATION, SetEvent(...))`,
   translate `WorkerCancelledError` → `PipelineCancelled`. **Remove
   the stale auto-start path** (current lines 39-41).
8. **`pipeline/stages/lyric_align.py`**: open
   `align_and_refine`/`transcribe_and_refine`, wrap each model call in
   its own activity scope (`Phase.ALIGN` / `Phase.TRANSCRIBE` /
   `Phase.REFINE`), translate `AlignmentCancelledError` →
   `PipelineCancelled`. Use `self._config.whisper_regroup` (not
   `regroup`). Write ASS/SRT to `ctx.tmp_dir`; `shutil.move()` to
   final destinations after both writes succeed.
9. **`pipeline/orchestrator.py`**: add `start`/`stop`/`run_one`/
   `run_one_async`/`join`/`cancel_active` with the per-job-fresh
   `CancelToken` rule. Expose `stem_worker` and `whisper_worker` as
   read-only properties for the test driver. Wire `CancelToken`
   through `_run_pipeline` (between-stage `check_cancelled()` only —
   no `set_phase` calls). Keep `run()` as a back-compat wrapper.
10. **`pipeline/run_pipeline.py`**: add `--phase`/`--cancel-after`,
    validate combination, embed `run_cancel_test()` and
    `_wait_for_phase()`. Driver catches `Exception` broadly to
    surface non-cancel errors cleanly.
11. **Smoke test (manual)**: run normally, then run with each phase
    (`extract`, `loudnorm`, `stem_separation`, `transcode`, `align`,
    `refine`, `transcribe`) and verify (a) cancel lands during the
    target phase and the pipeline exits, (b) post-cancel worker
    liveness check passes for every phase
    (`stem_worker.is_alive()` and `whisper_worker.model_loaded`
    both true), (c) no orphan files in `vocal/`, `nonvocal/`,
    `karaoke/`, or `subtitles/` after cancel. No re-run.

---

## 13. What stays unchanged

- `pipeline/workers/whisper_worker.py` — cancel mechanism updated: replaced
  the `model.encode()` monkey-patch (which only works on faster-whisper's
  CTranslate2 wrapper) with `register_forward_pre_hook` on `model.encoder`
  (an `nn.Module`). `stable_whisper.load_model()` returns `whisper.model.Whisper`,
  which has no `.encode()` method. The encoder reference is cached at
  `load_model()` time as `self._encoder_module` and cleared in `unload_model()`.
  All three methods (`align`, `refine`, `transcribe`) use the hook approach.
  See `cancel_tests/whisper/whisper_worker.py` for the prototype this was
  adapted from.
- `pipeline/workers/stem_worker.py` — only the one-line drain at the
  top of `separate()` is added; the cancel infrastructure (forwarder
  thread, monkey-patched `model_run.forward`, `_CancelledInsideDemix`)
  is untouched.
- `pipeline/config.py` — no new config knobs. Cancel is purely
  runtime/CLI.
- `pipeline/__init__.py` — empty, stays empty.
- Final output paths and formats (ASS, SRT, M4A) — unchanged.
  Outputs are still written to `vocal/`, `nonvocal/`, `karaoke/`,
  `subtitles/` next to the source song. The only change is *when*
  they appear there: only after their producing stage fully succeeds
  (they're staged in `tmp_dir` first). On cancel, those directories
  are unchanged.
- Backwards compat: `python -m pipeline.run_pipeline song.mp4
  lyrics.txt` (no flags) does exactly what it did before — synchronous,
  no cancel infrastructure exercised, `ctx.cancel is None` so all
  stages take their existing fast paths.
