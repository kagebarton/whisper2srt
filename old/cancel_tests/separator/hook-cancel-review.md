# Hook-based Cancellation Plan: Review Findings

Reviewed against `pipeline/workers/stem_worker.py` (539 lines) and
`cancel_separator/workers/cancelable_stem_worker.py` (531 lines).

---

## Issue 1 — Counter increment semantics differ from production (Bug)

**Severity:** Medium (logging inconsistency)

The plan's `cancel_pre_hook` increments `chunk_counter[0]` **before** the
forward runs (plan line 129: "incremented BEFORE forward"). The production
monkey-patch increments it **after** the forward returns successfully
(`stem_worker.py` line 431: `chunk_counter[0] += 1` is reached only when
`cancel_recv` is empty *and* the forward completes).

Consequences:

- The log message `"Cancel detected before chunk #N"` will report a
  different N than the production code for the same cancel point.
- The final `"processed N chunks before exit"` debug log in `finally`
  will count one extra chunk — the hook increments the counter for the
  last chunk that completes, but the production code only increments
  after a successful forward return. The hook version will report `N+1`
  for the same workload.

**Fix:** Remove the `chunk_counter[0] += 1` line from inside
`cancel_pre_hook`. The counter should only be incremented after a
successful forward pass. Options:

1. Use `register_forward_hook` (post-hook) for counting only, and keep
   `register_forward_pre_hook` for the cancel check.
2. Wrap the hook around the forward call manually.
3. Accept the semantic difference and document it explicitly, since the
   counter is only used for logging.

---

## Issue 2 — Hook does NOT call the real forward; architecture is sound but plan under-explains it

**Severity:** Low (clarity)

In the production monkey-patch:

```python
model_run.forward = cancelable_forward
# model_run(inputs) → nn.Module.__call__ → cancelable_forward(inputs)
# cancelable_forward: poll cancel → original_forward(*args, **kwargs)
```

The patched `forward()` is called by `nn.Module.__call__`, which then
calls the real `forward()` internally via `original_forward`.

With `register_forward_pre_hook`:

```python
handle = model_run.register_forward_pre_hook(cancel_pre_hook)
# model_run(inputs) → nn.Module.__call__ → pre_hook fires → self.forward(inputs) [real]
```

The pre_hook fires **before** `self.forward()`, then `self.forward()`
runs unmodified. This is architecturally cleaner — **however**, the plan
doesn't mention that `nn.Module.__call__` also fires post-hooks and runs
`__call__` bookkeeping (`_backward_hooks`, `_forward_hooks`, tracking
etc.).

The plan's §Risks #1 acknowledges hook ordering but understates the
implication: if audio-separator adds a **pre-hook** that modifies inputs,
our cancel pre-hook runs first and sees the *unmodified* inputs. This is
fine for a cancel-check (we don't inspect inputs), but worth noting
explicitly.

**Fix:** Add an explicit note in the plan that `cancel_pre_hook` must
**not** inspect or modify `inputs` — it only polls `cancel_recv`. This
guarantees it is order-independent with respect to any future hooks.

---

## Issue 3 — `_CancelledInsideDemix` must remain at module scope

**Severity:** Low (implementation clarity)

In the production code, `_CancelledInsideDemix` is a module-level class
(`stem_worker.py` line 275). The plan lists it under "Data structures"
which is correct, but the pseudocode for `_separate_with_cancel_check`
just shows `raise _CancelledInsideDemix()` inside the hook closure
without clarifying where the class lives.

Since the hook is a closure inside `_separate_with_cancel_check`, it
captures `_CancelledInsideDemix` from the enclosing module scope. This
works, but the plan should be explicit about keeping it module-level
rather than nesting it inside a function.

**Fix:** Add a note: "Keep `_CancelledInsideDemix` at module scope, same
as production. The closure inside `_separate_with_cancel_check` captures
it from the enclosing module scope."

---

## Issue 4 — Stem identification logic must come from production, not cancel_separator

**Severity:** Medium (correctness — could cause runtime failures on non-karaoke models)

The plan states output-path identification is "unchanged" (plan line
148). The `cancel_separator` prototype uses simpler stem identification
**without** the `(other)` pattern check. The plan says it's replicating
`pipeline/workers/stem_worker.py`, so the prototype should use the
**production** stem identification logic which includes:

```python
is_other = "(other)" in lower
if "instrumental" in lower or no_vocal or is_other:
    instrumental_wav = full_path
elif "vocal" in lower:
    vocals_wav = full_path
```

Someone implementing from the plan might copy from
`cancel_separator/workers/cancelable_stem_worker.py` instead, which
lacks the `(other)` pattern and will fail on non-karaoke MelBand
Roformer models.

**Fix:** Add an explicit note: "Stem identification must be copied from
`pipeline/workers/stem_worker.py` (which includes `(other)` pattern
matching), not from `cancel_separator/workers/cancelable_stem_worker.py`."

---

## Issue 5 — `model_name` parameter must be a constructor arg forwarded to subprocess

**Severity:** Medium (API compatibility — drop-in claim would break)

The plan says `__init__` signatures are "unchanged" from
`pipeline/workers/stem_worker.py` but doesn't mention that the
production code passes `model_name` through to `_worker_main` via the
`Process(args=...)` tuple. The `cancel_separator` prototype hardcodes
`MODEL_NAME` and doesn't pass it through `Process(args=...)`.

Since the plan states "drop-in for testing without renaming" and "class
names, exception types, and method signatures are kept identical," the
prototype must include `model_name` in both `__init__` and
`_worker_main` signatures, and forward it through `Process(args=...)`.

Production code (`stem_worker.py` lines 72-118):

```python
def __init__(self, ..., model_name: str = DEFAULT_MODEL_NAME) -> None:
    self._model_name = model_name
    ...

self._process = Process(
    target=_worker_main,
    args=(..., self._model_name),
)

def _worker_main(..., model_name: str = DEFAULT_MODEL_NAME):
    separator.load_model(model_filename=model_name)
```

**Fix:** Explicitly note that `model_name` must be a constructor
parameter forwarded to the subprocess, matching production. The
`cancel_separator` prototype's approach of hardcoding `MODEL_NAME` is
incorrect for a drop-in replacement.

---

## Issue 6 — `separator.output_dir` assignment missing from pseudocode

**Severity:** Medium (correctness — separation outputs go to wrong directory)

The production code (`stem_worker.py` lines 386-388) sets both
`separator.output_dir` and `separator.model_instance.output_dir` before
calling `separator.separate()`:

```python
separator.output_dir = str(tmp_dir)
if separator.model_instance:
    separator.model_instance.output_dir = str(tmp_dir)
```

The plan's pseudocode for `_separate_with_cancel_check` jumps straight
to the hook registration without showing this. While the plan says
"unchanged" for the function overall, the pseudocode is the implementor's
primary reference and omits this critical step.

**Fix:** Add the `output_dir` assignment lines to the pseudocode, before
the hook registration.

---

## Issue 7 — `handle.remove()` in `finally` could mask `_CancelledInsideDemix`

**Severity:** Low (robustness — edge case)

In the cancel flow, the production code's `finally` block restores
`model_run.forward` *before* the exception unwinds to `_worker_main`'s
`except _CancelledInsideDemix` block which calls `_clear_gpu_state`.

With hooks, the `finally: handle.remove()` in
`_separate_with_cancel_check` also runs before `_worker_main`'s except
block. The ordering is the same, so this is fine — but there's a
subtlety: if `handle.remove()` itself raises (e.g., if the handle was
already removed by some other code), the `_CancelledInsideDemix`
exception would be replaced by the `handle.remove()` exception. The
original cancel exception would be lost, and `_worker_main` would fall
through to the generic `except Exception` handler, sending `("error",`
instead of `("cancelled",)`.

**Fix:** Wrap `handle.remove()` in its own try/except inside the
`finally` block:

```python
finally:
    try:
        handle.remove()
    except Exception:
        pass
    worker_log.debug(f"Removed hook (processed {chunk_counter[0]} chunks before exit)")
```

This mirrors the defensive pattern used throughout the production code
(e.g., `_close_all_connections`, `kill`, `stop` all wrap close/join/kill
in try/except).

---

## Issue 8 — Logger name must be consistent for drop-in compatibility

**Severity:** Low (logging consistency)

The production code uses `"pipeline.workers.stem_worker"` as the logger
name (`stem_worker.py` line 48 and line 527). The plan doesn't specify
the logger name. If the prototype uses a different logger name (e.g.,
`"cancel_tests.separator.stem_worker"`), any logging configuration that
targets `"pipeline.workers.stem_worker"` will miss the prototype's
output.

This matters for the drop-in claim: if someone swaps
`pipeline/workers/stem_worker.py` with the prototype, log filtering
shouldn't break.

**Fix:** Either use the same logger name `"pipeline.workers.stem_worker"`
for drop-in compatibility, or explicitly document the logger name
difference as a known deviation from the drop-in claim.

---

## Summary

| # | Issue | Severity | Type |
|---|-------|----------|------|
| 1 | Counter increment semantics differ | Medium | Bug |
| 2 | Plan under-explains hook vs monkey-patch architecture | Low | Clarity |
| 3 | `_CancelledInsideDemix` scope must be explicit | Low | Clarity |
| 4 | Stem identification must include `(other)` pattern | Medium | Correctness |
| 5 | `model_name` must be constructor arg forwarded to subprocess | Medium | API compat |
| 6 | `separator.output_dir` assignment missing from pseudocode | Medium | Correctness |
| 7 | `handle.remove()` could mask cancel exception | Low | Robustness |
| 8 | Logger name must be consistent for drop-in | Low | Logging |

The core architectural decision — `register_forward_pre_hook` instead of
monkey-patching `forward()` — is sound. The issues above are
implementation details the plan omits or gets wrong, not fundamental
design flaws.
