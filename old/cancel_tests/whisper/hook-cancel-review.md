# Hook-based Cancellation Plan: Review Findings

Reviewed against:
- `cancel_whisper/workers/cancelable_whisper_worker.py` (554 lines)
- `pipeline/workers/whisper_worker.py` (605 lines)
- `cancel_whisper/config.py` (35 lines)
- `pipeline/config.py` (131 lines, WhisperModelConfig section)
- Installed packages: `stable_whisper` (site-packages), `whisper` (site-packages),
  `faster_whisper` (site-packages)
- `cancel_tests/separator/hook-cancel-review.md` (8 issues found there;
  cross-referenced for analogous problems)

---

## Issue 1 — Counter increment in pre_hook disagrees with production semantics (Bug)

**Severity:** Medium (logging inconsistency, same class as separator Issue 1)

The plan's `cancel_pre_hook` increments `encode_counter[0]` **before** the
encoder forward runs (plan lines 156-159):

```python
def cancel_pre_hook(module, inputs):
    if cancel_event.is_set():
        log.info(f"Cancel detected before encode pass #{encode_counter[0] + 1}")
        raise _Cancelled()
    encode_counter[0] += 1  # <-- incremented BEFORE forward runs
```

The production monkey-patch increments it **after** the forward returns
successfully (`cancelable_whisper_worker.py` line 377):
`encode_counter[0] += 1` is reached only when `original_encode(*a, **kw)`
completes without raising.

Consequences:

- The "cancelled after N encode passes" message reports N+1 compared to the
  production code for the same cancel point (the counter includes the pass
  that was about to start but never completed).
- The `finally` log "processed N encode passes" is similarly inflated.
- If two implementations are compared side-by-side (e.g., during A/B
  testing), the counts won't match.

This is the **exact same bug** found in the separator review (Issue 1).

**Fix:** Remove `encode_counter[0] += 1` from `cancel_pre_hook`. A
`register_forward_pre_hook` cannot increment after the forward completes --
that requires a `register_forward_hook` (post-hook). Options:

1. Use a separate `register_forward_hook` for counting only, and keep the
   `register_forward_pre_hook` for the cancel check. Two hooks, two handles.
2. Wrap the hook around the forward call manually (defeats the purpose of
   using the clean PyTorch API).
3. Accept the semantic difference and document it explicitly. The counter is
   only used for logging, not correctness -- but the plan claims
   "drop-in compatibility" with cancel_whisper test code, which expects the
   post-increment semantics.

Option 1 is recommended: it preserves both the clean API and the correct
semantics.

---

## Issue 2 — `model.model.encoder` attribute path is wrong for `load_model()` (Bug)

**Severity:** High (core architectural assumption is incorrect)

The plan states (line 35-36):

> `model.encode(mel)` is a thin wrapper that calls `model.model.encoder(mel)`,
> so a `register_forward_pre_hook` on `model.model.encoder` fires at the same
> points as the existing monkey-patch.

This is **false** for the `stable_whisper.load_model()` return type.

Verification against installed packages:

| Model API | Returns | Has `.encode`? | Type |
|---|---|---|---|
| `load_faster_whisper()` | `faster_whisper.WhisperModel` | **Yes** (transcribe.py:1391) | Python method wrapping CTranslate2 |
| `load_model()` | `whisper.model.Whisper` (nn.Module) | **No** | Pure nn.Module |

`stable_whisper.load_model()` calls `modify_model(model)` (original_whisper.py
line 935), which patches on `align`, `refine`, `transcribe`, etc. as
`MethodType` -- but it does **not** add an `encode` method.

The correct attribute path is:

```
stable_whisper.load_model("turbo")
  -> returns whisper.model.Whisper (nn.Module, IS the model itself)
     .encoder = whisper.model.AudioEncoder (nn.Module)  <-- hook target
     .decoder = whisper.model.TextDecoder  (nn.Module)
```

**Not** `model.model.encoder` -- because the `Whisper` nn.Module IS the model.
There is no wrapper. The plan's diagram (lines 29-33) shows a `WhisperModel`
wrapper that doesn't exist for `load_model()`. That diagram describes the
`load_faster_whisper()` architecture (`faster_whisper.WhisperModel` wrapping
CTranslate2), not the `load_model()` architecture.

Furthermore, `model.encode(mel)` does **not** exist on
`whisper.model.Whisper`. The existing monkey-patch in cancel_whisper patched a
method that only exists on `faster_whisper.WhisperModel`. The `alignment.py`
code that calls `model.encode(features)` (line 490) is inside a
`is_faster_whisper_on_pt()` branch -- it's **faster-whisper-specific**.

For the pure PyTorch model, the encoder is called directly as
`model.encoder(mel_segment.unsqueeze(0))` (alignment.py:931, :976), or
indirectly through the `non_whisper/alignment.py` `Aligner` class's
`inference_func` closure.

**Fix:**

1. Correct the attribute path: `self._encoder_module = self._model.encoder`
   (not `self._model.model.encoder`). The model returned by
   `stable_whisper.load_model()` IS `whisper.model.Whisper`, which has
   `.encoder` as a direct attribute.

2. Correct the architecture diagram:
   ```
   whisper.model.Whisper (nn.Module -- returned directly by load_model())
   +-- .encoder = AudioEncoder (nn.Module)  <-- hook target
   +-- .decoder = TextDecoder  (nn.Module)
   ```

3. Re-evaluate the `isinstance(self._encoder_module, torch.nn.Module)` check
   -- it still works, but the warning message should reference the correct
   path.

4. The plan needs to acknowledge that the hook approach is not just "cleaner"
   but fundamentally **different** in where it intercepts -- it hooks the
   actual `nn.Module.__call__` dispatch on `model.encoder`, not a Python
   wrapper method `model.encode()`.

---

## Issue 3 — Encoder call path for PyTorch model needs runtime verification (Gap)

**Severity:** High (could invalidate the entire hook interception strategy)

The `alignment.py:align()` function constructs a `compute_timestamps` closure
that calls `model.encode(features)` (line 490). As established in Issue 2,
`whisper.model.Whisper` does not have an `encode` method. That call site is
inside a `is_faster_whisper_on_pt()` branch (faster-whisper-specific).

For the pure PyTorch model, the alignment call path goes through
`non_whisper/alignment.py:Aligner`, which takes an `inference_func` callable.
The `inference_func` is constructed in `alignment.py:align()` and its encoder
calls need runtime verification:

- Does `inference_func` call `model.encoder(mel)` directly? (The `locate`
  function does -- alignment.py:931, :976 use `model.encoder(mel_segment...)`)
- Does it call something else entirely?

The `model.align()` method is bound via `modify_model`:
```python
model.align = MethodType(align, model)
```

This binds `stable_whisper.alignment.align` as a method. Whether that function
delegates to `non_whisper/alignment.py:Aligner` or takes a different path for
the pure PyTorch model is **not documented in the plan** and cannot be
determined from static analysis alone.

**Fix:** Before implementing, **runtime-verify** the call chain by running
`stable_whisper.load_model("turbo").align(...)` with a breakpoint or
`sys.settrace` to confirm exactly which function calls the encoder. Two
questions need answering:

1. Does `model.encoder(...)` get called during `model.align()` for the
   PyTorch model? (Almost certainly yes, but the exact call site matters.)
2. How many times is `model.encoder(...)` called per alignment? This
   determines cancel granularity and whether the hook fires at the expected
   points.

If the encoder is called through a path that doesn't go through
`nn.Module.__call__` (e.g., via `F.linear` directly), the pre-hook would
not fire and cancellation would silently degrade.

---

## Issue 4 — `compute_type` config field is silently ignored by `load_model()`

**Severity:** Low (behavioral -- no runtime error, but misleading config)

The plan says `config.py` is "copied from cancel_whisper/config.py"
(plan line 60). That config includes `compute_type: str = "int8"`, which is
meaningful for `load_faster_whisper()` (CTranslate2 supports int8/float16
quantization) but **ignored** by `stable_whisper.load_model()` (PyTorch
Whisper doesn't take a `compute_type` parameter).

The pipeline's `WhisperModelConfig` already handles this by setting
`compute_type: str = ""` (empty string) since it uses `load_model()`.

**Fix:** Change the default to `compute_type: str = ""` in the prototype's
`config.py`, matching the pipeline convention. Add a comment noting it's
unused with `load_model()` but kept for interface compatibility with
cancel_whisper.

---

## Issue 5 — `handle.remove()` in `finally` could mask `_Cancelled` (Robustness)

**Severity:** Low (edge case -- same as separator Issue 7)

If `handle.remove()` itself raises (e.g., the handle was already removed by
some other code, or a stable-ts internal cleanup), the `_Cancelled` exception
would be replaced by the `handle.remove()` exception. The
`except _Cancelled` handler would never run, and
`AlignmentCancelledError` would not be raised.

The production monkey-patch has the same theoretical risk with
`model.encode = original_encode` in `finally`, but `handle.remove()` is more
likely to fail because PyTorch hook removal involves internal bookkeeping
that can raise if the hook was already removed.

**Fix:** Wrap `handle.remove()` in its own try/except inside the `finally`
block:

```python
finally:
    try:
        handle.remove()
    except Exception:
        pass
    log.debug(f"Removed encoder hook (processed {encode_counter[0]} encode passes)")
```

This mirrors the defensive pattern recommended in the separator review
(Issue 7).

---

## Issue 6 — `_Cancelled` exception name is overly generic for module scope

**Severity:** Low (clarity -- same class as separator Issue 3)

The plan renames `_CancelledInsideEncode` to `_Cancelled` because "it's now
raised from a PyTorch hook on the encoder, not from inside `encode()`"
(plan line 80). The reasoning is sound, but `_Cancelled` is extremely generic
for a module-level exception class. If this module is ever imported alongside
another module that also defines `_Cancelled`, `except _Cancelled` in the
wrong module would silently catch the wrong exception.

The underscore prefix signals "internal", but the name doesn't indicate
*what* was cancelled. `_CancelledInsideEncoder` or `_EncoderCancelled` would
be more descriptive while still reflecting the new hook-based origin.

**Fix:** Rename to `_CancelledInsideEncoder` or keep `_CancelledInsideEncode`.
The name change is cosmetic and shouldn't be a priority, but if changing it,
pick a name that identifies the source (encoder hook) rather than a generic
`_Cancelled`.

---

## Issue 7 — Missing `RuntimeError("Model not loaded")` guard

**Severity:** Medium (correctness -- would crash with `AttributeError` instead of clear error)

Both `cancelable_whisper_worker.py` (line 347) and `pipeline/workers/whisper_worker.py`
(line 303) guard `align()` and `refine()` with:

```python
if self._model is None:
    raise RuntimeError("Model not loaded -- call load_model() first")
```

The plan's pseudocode for `align()` and `refine()` omits this guard. The
early return for `cancel_event is None` would crash with `AttributeError`
on `self._model.align(...)` if the model was never loaded.

The plan's pseudocode also accesses `self._encoder_module` in the
cancel_event branch without guarding against it being `None` -- but the plan
does handle that case (line 149-151). Still, the top-level model-loaded
guard is missing.

**Fix:** Add the `RuntimeError` guard at the top of both `align()` and
`refine()`, before the `cancel_event is None` check.

---

## Issue 8 — `align_and_refine` contract should be explicitly noted in file list

**Severity:** Low (documentation -- the plan correctly describes the divergence
but doesn't flag it prominently)

The plan (lines 191-206) correctly states that `align_and_refine` clears
`cancel_event` between phases, matching cancel_whisper's standalone-test
semantics. The pipeline intentionally does NOT clear -- a cancelled song is
discarded entirely.

This is correct as documented. However, the plan's "Files to create" section
(line 58) says `whisper_worker.py` without noting which `align_and_refine`
contract it implements. Someone reading just the file list might copy from
`pipeline/workers/whisper_worker.py` which uses the no-clear contract.

**Fix:** Add an explicit comment in the file list:
`whisper_worker.py -- the prototype worker (cancel_whisper align_and_refine
contract with cancel_event.clear())`

---

## Issue 9 — `_clear_gpu_state` docstring references CTranslate2

**Severity:** Low (documentation -- stale comment)

The existing `_clear_gpu_state()` in both `cancelable_whisper_worker.py`
(line 540) and `pipeline/workers/whisper_worker.py` (line 591) says:

> "After _CancelledInsideEncode unwinds, there should be no leftover GPU state
> from the interrupted encoder pass (CTranslate2 manages its own memory
> internally)."

The prototype uses PyTorch Whisper, not CTranslate2. After a `_Cancelled`
exception unwinds from an interrupted `AudioEncoder.__call__`, there **may**
be leftover intermediate tensors on the GPU from the interrupted forward pass
(unlike CTranslate2 which manages memory internally). `torch.cuda.empty_cache()`
only releases unused cached memory -- it does NOT free tensors still
referenced by the interrupted call's stack frames.

However, since the exception unwinds the entire call stack, those stack
frames are gone and their tensors should be eligible for collection. The
`empty_cache()` call is still appropriate as a safety net, but the docstring
should not reference CTranslate2.

**Fix:** Update the docstring to reference PyTorch instead of CTranslate2:
"After `_Cancelled` unwinds from the encoder hook, intermediate tensors from
the interrupted forward pass should be freed as their stack frames are
unwound. We clear the PyTorch cache as a safety net."

---

## Issue 10 — `_encoder_module` must be explicitly cleared in `unload_model()`

**Severity:** Low (plan is self-consistent; flagging for implementation)

The plan (line 209) says `unload_model()` adds `self._encoder_module = None`
alongside clearing `_model`. This is correct -- but the existing
`pipeline/workers/whisper_worker.py:unload_model()` (lines 577-584) doesn't
have this field. The plan says it's a "new field" (line 96), so this is
fine -- just flagging that the implementor must add this line and not copy
`unload_model()` verbatim from the pipeline.

**Fix:** No fix needed in the plan; this is an implementation note. The
`unload_model()` method must be written to include `self._encoder_module = None`
rather than copied verbatim from the pipeline.

---

## Issue 11 — Concurrent `align()` calls: hooks coexist (unlike monkey-patch)

**Severity:** Low (same contract as existing code -- flagged in plan Risks #4)

The plan acknowledges (lines 322-325) that concurrent `align()` calls would
install two hooks on the same encoder module. With the monkey-patch approach,
the second `model.encode = cancelable_encode` silently replaces the first.
With hooks, both coexist -- the second call's hook would fire alongside the
first's.

The plan doesn't mention that `handle.remove()` in the first call's
`finally` removes only the first hook -- the second hook remains installed
until the second call's `finally` runs. This is correct behavior (each
handle removes only its own hook), but it's a different failure mode than
the monkey-patch approach (where the second `finally` restores the first
call's already-gone patch).

**Fix:** Add a brief note in Risks #4 explaining the different concurrent
failure mode: "With hooks, concurrent calls install independent hooks that
coexist on the encoder. Each `handle.remove()` in `finally` removes only its
own hook. With the monkey-patch, the second call overwrites the first call's
patch, and the first call's `finally` restores the original method -- leaving
the second call unpatched. Neither approach supports concurrent use; the
hook version fails more gracefully (both hooks run) while the monkey-patch
version silently drops the second call's cancel check."

---

## Summary

| # | Issue | Severity | Type |
|---|-------|----------|------|
| 1 | Counter increment in pre_hook disagrees with production | Medium | Bug |
| 2 | `model.model.encoder` path is wrong -- should be `model.encoder` | **High** | Bug |
| 3 | Encoder call path for PyTorch model needs runtime verification | **High** | Gap |
| 4 | `compute_type` config field silently ignored by `load_model()` | Low | Behavioral |
| 5 | `handle.remove()` could mask `_Cancelled` exception | Low | Robustness |
| 6 | `_Cancelled` exception name overly generic | Low | Clarity |
| 7 | Missing `RuntimeError("Model not loaded")` guard | Medium | Correctness |
| 8 | `align_and_refine` contract not noted in file list | Low | Documentation |
| 9 | `_clear_gpu_state` docstring references CTranslate2 | Low | Documentation |
| 10 | `_encoder_module` must be cleared in `unload_model()` | Low | Implementation |
| 11 | Concurrent hooks coexist vs. monkey-patch overwrites | Low | Clarity |

Issues 2 and 3 are the most critical. Issue 2 means the implementation would
crash (`AttributeError: 'Whisper' object has no attribute 'model'`) if
someone follows the plan literally. Issue 3 means the entire hook interception
strategy needs runtime verification to confirm the encoder is called through
`nn.Module.__call__` (which fires pre-hooks) rather than through some other
path that bypasses hooks.

The core architectural decision -- `register_forward_pre_hook` instead of
monkey-patching -- is sound, and the plan's overall structure is well thought
out. The issues above are correctable implementation details, not fundamental
design flaws -- with the exception of Issue 2, which reflects a
misunderstanding of the `load_model()` return type that affects the central
claim of the plan.
