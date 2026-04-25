# Pipeline Cancel Prototype — Issue Review

Review of `plans/pipeline-cancel-prototype.md` against the current codebase.
Each issue is classified as **bug** (will cause incorrect runtime behaviour),
**race** (concurrency hazard), **design** (architectural concern), or **nit**
(minor discrepancy).

---

## 1. `run_ffmpeg` does not distinguish SIGKILL'd Popen from real ffmpeg failures  *(bug, critical)*

**Plan §6.2** — `run_ffmpeg` helper (new file `_ffmpeg_helpers.py`):

```python
if proc.returncode != 0:
    raise RuntimeError(f"ffmpeg failed (exit code {proc.returncode})")
```

When `KillProcess.cancel()` calls `proc.kill()`, the Popen receives SIGKILL
(exit code −9 on Linux). After the `activity()` context exits and re-raises
`PipelineCancelled`, the stage returns. But there is a race window: the
`activity()` `__exit__` checks `self.cancelled` **only at the end**. If the
SIGKILL causes `proc.wait()` to return *before* `cancel()` sets the
`cancelled` flag (or if `PipelineCancelled` is not raised because the cancel
hasn't been flagged yet), execution falls through to the `returncode != 0`
check and raises `RuntimeError` instead of `PipelineCancelled`.

In the timeline the plan acknowledges this at §8:

> For ffmpeg phases the cancel kills the Popen; the stage sees a non-zero
> exit code, looks at `ctx.cancel.cancelled`, and raises `PipelineCancelled`
> instead of `RuntimeError`.

But the **pseudocode for `run_ffmpeg` does not implement this check**. It
blindly raises `RuntimeError` on any non-zero exit code. The fix:

```python
if proc.returncode != 0:
    if ctx.cancel is not None and ctx.cancel.cancelled:
        raise PipelineCancelled(phase)
    raise RuntimeError(f"ffmpeg failed (exit code {proc.returncode})")
```

Without this, a cancel during EXTRACT/LOUDNORM/TRANSCODE surfaces as an
unhandled `RuntimeError` in `_run_pipeline`, which does **not** trigger the
`PipelineCancelled` catch path in `join()`. The pipeline reports a generic
failure instead of a clean cancel.

---

## 2. `CancelToken.activity()` double-raises on cancel exit  *(bug, medium)*

**Plan §4.3** — `activity()` contextmanager:

```python
@contextmanager
def activity(self, phase: Phase, target: Cancellable):
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
        if self.cancelled:
            raise PipelineCancelled(phase)
```

Consider the scenario where a `Cancellable.cancel()` call inside the
`activity` scope causes the work to raise `PipelineCancelled` *on its own*
(e.g. `SetEvent.cancel()` sets the event, the worker detects it and raises
`WorkerCancelledError`, the stage translates to `PipelineCancelled`). The
exception propagates through `yield`, hits the `finally` block, and then
`if self.cancelled:` is True, so `activity()` raises a **second**
`PipelineCancelled`. Python contextmanager semantics mean the second raise
**replaces** the first exception (the original traceback is lost). This
doesn't cause a crash (same exception type), but:

- The `phase` in the replacement exception might differ from the original
  (if the stage's own raise used a different `phase` argument, though in
  practice they match).
- Debugging becomes harder — the original traceback from the worker is
  silently discarded.
- If any `except PipelineCancelled` block between the stage and the
  `finally` had side effects (e.g. logging, cleanup), they are lost.

**Fix options:**
- (a) Suppress the re-raise if we're already unwinding for
  `PipelineCancelled`:
  ```python
  finally:
      with self.lock:
          self.active = None
      if self.cancelled and not isinstance(sys.exc_info()[1], PipelineCancelled):
          raise PipelineCancelled(phase)
  ```
- (b) Only re-raise if the `yield` completed normally (no exception in
  flight):
  ```python
  exc_info = sys.exc_info()
  if self.cancelled and exc_info[0] is None:
      raise PipelineCancelled(phase)
  ```
  Option (b) is cleaner — it preserves the original exception when there
  is one and only synthesises `PipelineCancelled` when the cancel arrived
  but no exception was raised by the work itself.

---

## 3. `cancel_event` is never reset between pipeline jobs  *(bug, medium)*

**Plan §4.3** — `CancelToken`:

```python
@dataclass
class CancelToken:
    event: threading.Event  # shared model-worker event (reused per phase)
    cancelled: bool = False
    ...
```

The `event` is a shared `threading.Event` that is **never cleared** between
jobs. The plan states in §6.4:

> Note: `event` is **not** cleared between activities. If a cancel arrives
> during ALIGN, it sets the event; the ALIGN activity raises
> `PipelineCancelled` on exit, which short-circuits before REFINE ever opens
> its scope.

This is correct within a single job. But the `CancelToken` is created per
job in `run_async()` (§5.2). The question is: is a **new** `CancelToken`
(and thus a **new** `threading.Event`) created for each job?

The plan doesn't show the `run_async()` implementation clearly. If a new
`CancelToken` with a fresh `threading.Event` is created each time, this is
fine. But if the `CancelToken` or its `event` is reused across jobs (as the
field name "shared" implies), a cancelled job's event stays set and poisons
the next job immediately.

**Recommendation:** Explicitly state that `run_async()` creates a **new**
`CancelToken` (and a new `threading.Event`) for each job invocation. The
current `_start_workers()` only runs once, but the event is per-job, not
per-worker.

---

## 4. `StemWorker` cancel-forwarder daemon thread leaks across jobs  *(race, medium)*

**Current code** — `StemWorker.separate()` (line 152–158):

```python
if cancel_event is not None and self._cancel_send is not None:
    cancel_forwarder = threading.Thread(
        target=_forward_cancel,
        args=(cancel_event, self._cancel_send),
        daemon=True,
    )
    cancel_forwarder.start()
```

The `_forward_cancel` daemon thread calls `cancel_event.wait()` and then
`cancel_send.send(1)`. After a job completes or is cancelled, the plan
reuses the same `threading.Event` for the next activity scope (§6.3,
§6.4). If the forwarder thread from the **previous** job hasn't exited yet
(e.g. the event was set but `send()` hasn't completed, or the event was
never set and the thread is still waiting), the old forwarder will react to
the **new** job's event being set and send a spurious cancel signal on the
pipe.

This is partially mitigated by `_drain_cancel_pipe()` (called in the
`finally` block of `separate()`), which drains any leftover signals. But
there is a timing gap: the drain happens immediately after `separate()`
returns, and the old forwarder might not have executed `send()` yet. If the
next `separate()` call starts quickly, the old forwarder's signal could
arrive during the new job.

**Mitigation:** The plan should either:
- (a) `join()` the cancel forwarder thread before returning from
  `separate()` (it's a daemon thread and `cancel_event` is set or the
  pipe is closed, so the join should be near-instant), or
- (b) Use a per-job `threading.Event` (consistent with issue #3 above),
  so the old forwarder is waiting on a stale event that will never be
  set again.

---

## 5. `lyric_align.py` splits `align_and_refine`/`transcribe_and_refine` but the worker already provides them  *(design, medium)*

**Plan §6.4** calls `self._worker.align()`, `self._worker.transcribe()`,
`self._worker.refine()` directly instead of using the existing
`align_and_refine()` and `transcribe_and_refine()` wrappers. The plan
justifies this:

> The worker's `align_and_refine` / `transcribe_and_refine` wrappers are
> too coarse (they fold two model calls into one), so the stage calls
> `align()`, `transcribe()`, `refine()` directly.

This is correct — the wrappers bundle two calls, and the stage needs per-call
activity scopes. However, there are two issues:

**(a) Regroup logic duplication.** The current `transcribe_and_refine()`
(whisper_worker.py:557–575) includes a `regroup` step between transcribe
and refine:

```python
def transcribe_and_refine(self, vocal_path, cancel_event=None):
    result = self.transcribe(vocal_path, cancel_event)
    if self._config.regroup:
        result.regroup(self._config.regroup)
    refined = self.refine(vocal_path, result, cancel_event)
    return refined
```

The plan's pseudocode (§6.4) also includes `regroup` between TRANSCRIBE and
REFINE. But `regroup` is a **CPU-bound, fast** operation that runs between
the two activity scopes. If a cancel arrives during regroup, it's not
inside any activity scope. The sticky `cancelled` flag handles this
(REFINE's `activity()` will raise on entry), so this is correct. But it's
worth noting that `regroup` is not cancellable — it completes instantly and
then the cancel is caught at the next activity boundary. This is acceptable
behaviour but should be documented.

**(b) The `_config.regroup` access is on the worker's config, not the
pipeline's.** The plan's pseudocode shows `self._config.regroup` inside the
stage, but `self._config` in `LyricAlignStage` is a `PipelineConfig`, which
has `whisper_regroup: str = ""`. The worker's `WhisperModelConfig` has
`regroup: str = ""`. The plan must ensure the stage uses the correct field
(`self._config.whisper_regroup`), or that the two configs are kept in sync.
Currently `build_whisper_config()` in `run_pipeline.py` maps
`cfg.whisper_regroup` → `WhisperModelConfig.regroup`, so they should match —
but the stage is reading from `PipelineConfig`, not `WhisperModelConfig`, so
the field name is different.

---

## 6. `ffmpeg_transcode` writes output files to `song_path.parent`, not `tmp_dir`  *(bug, medium)*

**Current code** — `ffmpeg_transcode.py`:

```python
vocal_dir = video.parent / "vocal"
nonvocal_dir = video.parent / "nonvocal"
vocal_dir.mkdir(exist_ok=True)
nonvocal_dir.mkdir(exist_ok=True)
vocal_out = vocal_dir / f"{video.stem}---vocal.m4a"
nonvocal_out = nonvocal_dir / f"{video.stem}---nonvocal.m4a"
```

The transcode stage writes final output files to the **source file's parent
directory**, not to `ctx.tmp_dir`. The plan's §5.3 shows:

```python
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
```

This cleanup only removes `tmp_dir`. Transcoded M4A files in `video.parent`
survive cleanup — that's intentional (they're the final outputs).

**Issue on cancel:** If the pipeline is cancelled **after** the first
transcode (vocal M4A written) but **before** the second (instrumental M4A),
the vocal M4A is left on disk as a partial output. The plan states in §1:

> Per-job temp dir is removed. No partial outputs written.

But the transcode stage writes to a **non-temp** directory. A cancelled
transcode could leave an orphan vocal M4A with no corresponding nonvocal
M4A. This isn't a crash, but it contradicts the stated invariant. The plan
should either:
- (a) Write transcode outputs to `tmp_dir` first, then move them to the
  final location only after both complete successfully, or
- (b) Acknowledge that cancel during transcode may leave one orphan M4A,
  and document this as a known limitation.

The same concern applies to `lyric_align.py` — ASS/SRT files are written to
`song_path.parent / "karaoke"` and `song_path.parent / "subtitles"`, which
are outside `tmp_dir`. A cancel during ASS generation could leave a partial
ASS file on disk.

---

## 7. `_wait_for_phase` spin-poll on `token.cancelled` without the lock  *(race, low)*

**Plan §7.1**:

```python
def _wait_for_phase(token, target, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if token.get_phase() == target:
            return True
        if token.cancelled:
            return False
        time.sleep(0.1)
    return False
```

`token.cancelled` is read without the lock. This is a benign data race
(booleans are atomic on CPython), but it's inconsistent with `get_phase()`,
which **does** acquire the lock. If the code is ever ported to a
free-threading Python build (PEP 703), this becomes a real data race.

**Fix:** Use `token.get_phase() is not None and token.cancelled` behind the
lock, or add a `is_cancelled()` method that acquires the lock.

---

## 8. `StemSeparationStage.run()` has a stale auto-start path  *(bug, low)*

**Current code** — `stem_separation.py:31–33`:

```python
if not self._worker.is_alive():
    logger.info(f"[{self.name}] Worker not alive — starting")
    self._worker.start()
```

The plan's §6.3 pseudocode for the cancel-aware path doesn't show this
auto-start. Under the new architecture, the orchestrator calls `start()`
once before any pipeline run. If the stage still has this auto-start, and
the worker died due to cancellation (which shouldn't happen but is a
failure mode), the stage would silently restart the worker mid-pipeline.
This would:
- Create a new subprocess (expensive, ~10s model load)
- Not be reflected in the `stem_worker` reference held by the test driver
  for liveness checking

The plan should explicitly remove this auto-start path, or at minimum
guard it behind `ctx.cancel is None` so it only triggers in the
non-cancel path.

---

## 9. `CancelToken.event` field name clashes with `threading.Event`  *(nit, low)*

The `CancelToken` dataclass has a field named `event` of type
`threading.Event`. Python's `threading.Event` is itself a class, and the
field name `event` is ambiguous — it could be confused with the
`Cancellable` pattern. The plan should consider renaming to `cancel_event`
for consistency with the worker APIs (`cancel_event` parameter in
`StemWorker.separate()`, `WhisperWorker.align()`, etc.).

---

## 10. `run_pipeline.py` driver lifecycle doesn't match the `run_async` → `join` API  *(bug, medium)*

**Plan §7.1** — the test driver calls:

```python
cancel_token = orchestrator.run_one_async(song_path, lyrics_path)
```

But §5.2 defines the method as `run_async()`, and §5.3's driver lifecycle
calls `run_one_async()`. The naming is inconsistent across the plan.
§5.2 says:

> `run_async()` — Start the pipeline on a background thread

§7.1 says:

> `orchestrator.run_one_async()`

The driver lifecycle in §7.1 shows:

```python
orchestrator.run_one(song_path, lyrics_path)  # synchronous
```

These are three different names for potentially the same thing. The plan
should pick one name and use it consistently. Suggested:
- `run_one()` — synchronous, blocks until complete
- `run_one_async()` — returns `CancelToken` immediately, runs on a thread

---

## 11. `check_cancelled()` implementation reads `phase` under the lock but raises outside it  *(race, low)*

**Plan §4.3**:

```python
def check_cancelled(self) -> None:
    if self.cancelled:
        with self.lock:
            phase = self.phase
        raise PipelineCancelled(phase)
```

The `if self.cancelled` check is outside the lock. Between the check and
the `with self.lock:`, `cancel()` could be called, setting `cancelled=True`
and changing `phase`. The snapshot of `phase` taken under the lock might not
match the `cancelled` flag that triggered the check. This is a benign
inconsistency (the phase is for diagnostics only), but if exact consistency
matters:

```python
def check_cancelled(self) -> None:
    with self.lock:
        if self.cancelled:
            raise PipelineCancelled(self.phase)
```

---

## 12. Missing `PipelineCancelled` import in stage files  *(nit, low)*

The plan shows stages raising `PipelineCancelled` (§6.3, §6.4), but doesn't
show the import. `PipelineCancelled` is defined in `pipeline/context.py`,
which is already imported by all stages (for `StageContext`). This should
work, but the implementation should verify the import is added explicitly.

Similarly, `Phase`, `KillProcess`, `SetEvent`, and `CancelToken` all live
in `pipeline/context.py`. Stages that use them (e.g. `stem_separation.py`
uses `Phase.STEM_SEPARATION` and `SetEvent`) will need new imports from
`pipeline.context`. The plan doesn't call this out explicitly.

---

## 13. `loudnorm_analyze.py` currently uses `subprocess.run(capture_output=True)`, not `Popen`  *(design, medium)*

**Current code** — `loudnorm_analyze.py:48–51`:

```python
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
)
```

The plan proposes switching to `Popen` + `run_ffmpeg()` with
`capture_stderr=True`. But `subprocess.run()` with `capture_output=True`
captures **both** stdout and stderr, while the proposed `run_ffmpeg()` with
`stderr=subprocess.PIPE` and `stdout=subprocess.DEVNULL` discards stdout.

Currently the loudnorm stage doesn't use stdout (ffmpeg loudnorm writes
everything to stderr), so this is fine. But the semantic change from
`capture_output=True` to `stderr=PIPE, stdout=DEVNULL` should be noted —
any future use of stdout from this command would silently lose data.

---

## 14. `ffmpeg_extract.py` currently uses `subprocess.run()`, not `Popen`  *(nit, low)*

Same as #13 but for `ffmpeg_extract.py`, which uses:

```python
result = subprocess.run(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
```

The plan's `run_ffmpeg()` uses `Popen` instead, which changes how the
process is managed. The current code uses `subprocess.run()` which blocks
and returns a `CompletedProcess`. The switch to `Popen` + `proc.wait()`
is functionally equivalent for the non-cancel path, but the plan should
note that `subprocess.run()` also checks for `CalledProcessError` via
`check=True` (not used here, but the mental model differs).

---

## 15. `_wait_for_phase` timeout of 600s is excessive  *(nit, low)*

**Plan §7.1**:

```python
if not _wait_for_phase(cancel_token, target_phase, timeout=600):
```

A 10-minute timeout for phase detection is very long. If the target phase
is never reached (e.g. the pipeline fails early with a different error),
the test driver hangs for 10 minutes before reporting. The timeout should
be configurable or reduced (e.g. 120s), with the rationale documented.

---

## 16. Orchestrator `join()` re-raises arbitrary exceptions  *(design, medium)*

**Plan §5.2**:

```python
def join(self, timeout=None) -> StageContext:
    """Wait for the pipeline thread to finish.
    Re-raises whatever exception the thread caught (PipelineCancelled,
    FileNotFoundError, etc.).
    """
```

The `run_cancel_test()` function in §7.1 catches `PipelineCancelled`:

```python
try:
    orchestrator.join(timeout=120)
    log.error("Pipeline completed instead of cancelling — raise --cancel-after")
    return 1
except PipelineCancelled as e:
    ...
```

But if the pipeline fails with a **different** exception (e.g.
`FileNotFoundError`, `RuntimeError` from a stage), `join()` re-raises it,
and the test driver doesn't catch it — it propagates as an unhandled
exception and the process exits with a non-zero code (likely 1 from the
Python interpreter). The error message would be confusing ("pipeline failed
with FileNotFoundError" instead of a clear test failure message).

**Fix:** Catch `Exception` broadly in the test driver, distinguish
`PipelineCancelled` from other errors:

```python
try:
    orchestrator.join(timeout=120)
    log.error("Pipeline completed instead of cancelling — raise --cancel-after")
    return 1
except PipelineCancelled as e:
    ...  # expected
except Exception as e:
    log.error(f"Pipeline failed with unexpected error: {e}")
    return 1
```

---

## 17. `activity()` contextmanager acquires `self.lock` twice on exit  *(bug, low)*

**Plan §4.3**:

```python
try:
    yield
finally:
    with self.lock:         # ← acquire #1
        self.active = None
    if self.cancelled:     # ← outside lock
        raise PipelineCancelled(phase)
```

The `finally` block acquires the lock once to clear `active`, then drops
it, then checks `cancelled`. If `cancel()` runs between dropping the lock
and checking `cancelled`:

1. `cancel()` acquires the lock, sets `cancelled=True`, calls
   `target.cancel()` — but `active` is already `None`, so `target` is
   `None`, and `cancel()` only sets the flag.
2. Back in `activity()`, `self.cancelled` is `True`, so it raises
   `PipelineCancelled`.

This is actually **correct** — the cancel that arrived after clearing
`active` is caught by the `cancelled` check. But there is a subtle issue:
`cancel()` saw `active=None` and did not invoke any `Cancellable.cancel()`.
If the kill was supposed to reach a Popen, the Popen was already
unregistered (its activity exited normally), and the cancel arrives too
late — it just sets the flag. This is the intended "between activities"
behaviour. No bug, but worth noting that the cancel *mechanism* (SIGKILL,
event set) is only invoked when `active` is not `None`.

---

## 18. `run_ffmpeg` uses `stdin=subprocess.DEVNULL` but current `loudnorm_analyze` doesn't  *(nit, low)*

The proposed `run_ffmpeg()` always passes `stdin=subprocess.DEVNULL`. The
current `loudnorm_analyze.py` uses `subprocess.run(capture_output=True)`
which does **not** set `stdin=DEVNULL` — it inherits the parent's stdin.
This is unlikely to matter (ffmpeg doesn't read stdin unless interactive),
but the behavioural change should be noted.

---

## 19. Test driver accesses `stem_worker` and `whisper_worker` that are private orchestrator fields  *(design, low)*

**Plan §7.1** — `run_cancel_test()` takes `stem_worker` and
`whisper_worker` as parameters, but the driver lifecycle (§7.1) shows:

```python
orchestrator = PipelineOrchestrator(...)
orchestrator.start()
rc = run_cancel_test(orchestrator, ...)
```

The `stem_worker` and `whisper_worker` are constructed in `main()` before
the orchestrator, so they're available. But the plan doesn't show how they
reach `run_cancel_test()`. The function signature takes them as parameters,
so the driver caller must pass them explicitly. This is fine but should be
shown in the `main()` pseudocode:

```python
rc = run_cancel_test(orchestrator, song_path, lyrics_path,
                     target_phase, cancel_after,
                     stem_worker, whisper_worker)
```

The plan's driver lifecycle snippet omits these arguments.

---

## 20. `CancelToken` is a `@dataclass` with a `threading.Lock` field  *(nit, low)*

```python
@dataclass
class CancelToken:
    event: threading.Event
    cancelled: bool = False
    phase: Optional[Phase] = None
    active: Optional[Cancellable] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
```

Python `@dataclass` generates `__eq__` and `__repr__` by default.
`threading.Lock` does not support equality comparison and its `repr()`
is unhelpful. This will cause errors if anyone accidentally compares two
`CancelToken` instances or logs them. Fix:

```python
@dataclass(eq=False, repr=False)
class CancelToken:
    ...
```

Or make it a regular class (not a dataclass) — there's no benefit to
dataclass here since the class has methods and mutable state.

---

## Summary

| # | Severity | Category | Short description |
|---|----------|----------|-------------------|
| 1 | **critical** | bug | `run_ffmpeg` doesn't check `ctx.cancel.cancelled` on non-zero exit — raises `RuntimeError` instead of `PipelineCancelled` after SIGKILL |
| 2 | medium | bug | `activity()` double-raises `PipelineCancelled`, losing original traceback |
| 3 | medium | bug | `event` not reset between jobs — must create new `CancelToken` per job |
| 4 | medium | race | StemWorker cancel-forwarder daemon thread can leak into next job |
| 5 | medium | design | `lyric_align` regroup runs between activity scopes; `PipelineConfig.whisper_regroup` vs `WhisperModelConfig.regroup` field name mismatch |
| 6 | medium | bug | Cancel during transcode/lyric_align leaves orphan output files outside `tmp_dir` |
| 7 | low | race | `_wait_for_phase` reads `token.cancelled` without lock |
| 8 | low | bug | `StemSeparationStage` has stale auto-start path that conflicts with orchestrator-managed lifecycle |
| 9 | low | nit | `CancelToken.event` field name is ambiguous |
| 10 | medium | bug | Inconsistent method naming: `run_async()` vs `run_one_async()` vs `run_one()` |
| 11 | low | race | `check_cancelled()` reads `cancelled` outside lock, snapshots `phase` inside |
| 12 | low | nit | Missing import declarations for new context symbols in stage files |
| 13 | medium | design | `loudnorm_analyze` switches from `subprocess.run(capture_output=True)` to `Popen(stderr=PIPE)` — stdout silently discarded |
| 14 | low | nit | `ffmpeg_extract` switches from `subprocess.run()` to `Popen` |
| 15 | low | nit | 600s `_wait_for_phase` timeout is excessive |
| 16 | medium | design | `join()` re-raises arbitrary exceptions; test driver doesn't catch non-cancel errors |
| 17 | low | — | `activity()` lock/cancel ordering is correct but non-obvious; worth a comment |
| 18 | low | nit | `run_ffmpeg` adds `stdin=DEVNULL` that current loudnorm doesn't set |
| 19 | low | design | Test driver needs explicit `stem_worker`/`whisper_worker` params not shown in pseudocode |
| 20 | low | nit | `@dataclass` with `threading.Lock` breaks `__eq__`/`__repr__` |

**Top 3 issues to fix before implementation:**

1. **#1** — Without the `ctx.cancel.cancelled` check in `run_ffmpeg`, all
   ffmpeg-phase cancellations will surface as `RuntimeError` instead of
   `PipelineCancelled`, breaking the entire cancel flow for EXTRACT,
   LOUDNORM, and TRANSCODE phases.

2. **#2** — The `activity()` double-raise will lose original exception
   context and traceback, making post-cancel debugging significantly harder.
   Fix is a one-liner (`sys.exc_info()` check in the `finally` block).

3. **#6** — Orphan output files after cancel violate the stated invariant
   ("per-job temp dir is removed, no stems written"). Needs either a
   write-to-tmpdir-then-atomic-rename strategy or an explicit documented
   exception for these stages.
