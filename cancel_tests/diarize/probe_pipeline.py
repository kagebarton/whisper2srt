#!/usr/bin/env python3
"""Pre-implementation probe for the unified cancel architecture.

Validates four assumptions the two-layer hook design rests on before we
commit to implementation.  Expected runtime: ~30s on top of one short
diarization call.

Checks:
  1. Embedding model is a torch.nn.Module (or wraps one we can reach).
  2. pyannote pipeline.__call__ accepts a hook= kwarg.
  3. pyannote does NOT swallow exceptions raised from the hook callback.
  4. PyTorch forward_pre_hook exceptions propagate cleanly through the
     pipeline and leave the model usable.

Usage:
  python probe_pipeline.py <vocal_audio> [--hf-token /path/to/token.txt]
"""

import inspect
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# Point HF cache at the local pre-downloaded model folder BEFORE importing
# anything that pulls in huggingface_hub or pyannote.
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def resolve_hf_token():
    """Try to find a HuggingFace token."""
    token_path = Path(__file__).resolve().parent.parent / "hf_token.txt"
    if token_path.exists():
        return token_path.read_text().strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def probe_embedding_model(pipeline) -> dict:
    """Check #1: Embedding model is a torch.nn.Module."""
    import torch.nn as nn

    result = {"pass": False, "details": []}

    # Try pipeline._embedding directly
    embedding = getattr(pipeline, "_embedding", None)
    result.details.append(f"pipeline._embedding type: {type(embedding)}")
    result.details.append(
        f"pipeline._embedding isinstance(nn.Module): {isinstance(embedding, nn.Module)}"
    )

    # Try pipeline._embedding.model_
    if embedding is not None:
        inner = getattr(embedding, "model_", None)
        result.details.append(f"pipeline._embedding.model_ type: {type(inner)}")
        result.details.append(
            f"pipeline._embedding.model_ isinstance(nn.Module): {isinstance(inner, nn.Module)}"
        )
        if inner is None:
            # Try .model as well
            inner = getattr(embedding, "model", None)
            result.details.append(f"pipeline._embedding.model type: {type(inner)}")
            result.details.append(
                f"pipeline._embedding.model isinstance(nn.Module): {isinstance(inner, nn.Module)}"
            )

    # Walk named_modules for embedding/wespeaker
    embed_modules = []
    for name, module in pipeline.named_modules():
        name_lower = name.lower()
        if "embed" in name_lower or "wespeaker" in name_lower:
            embed_modules.append(
                (name, type(module).__name__, isinstance(module, nn.Module))
            )
    result.details.append(f"Named modules with 'embed' or 'wespeaker': {embed_modules}")

    # Determine pass/fail: we need at least one nn.Module we can hook
    usable = False
    if isinstance(embedding, nn.Module):
        usable = True
    elif embedding is not None:
        for attr in ("model_", "model"):
            inner = getattr(embedding, attr, None)
            if isinstance(inner, nn.Module):
                usable = True
                break
    if not usable and embed_modules:
        usable = any(m[2] for m in embed_modules)

    result["pass"] = usable
    return result


def probe_hook_kwarg(pipeline) -> dict:
    """Check #2: pipeline.__call__ accepts hook= kwarg."""
    result = {"pass": False, "details": []}

    sig = inspect.signature(pipeline.__call__)
    params = list(sig.parameters.keys())
    result.details.append(f"pipeline.__call__ params: {params}")
    result.details.append(f"'hook' in params: {'hook' in params}")

    result["pass"] = "hook" in params
    return result


def probe_hook_exception(pipeline, vocal_path) -> dict:
    """Check #3: pyannote does NOT swallow hook-raised exceptions."""
    result = {"pass": False, "details": []}

    class _Sentinel(Exception):
        pass

    call_count = [0]

    def raising_hook(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 1:
            raise _Sentinel("hook exception probe")

    try:
        pipeline(str(vocal_path), hook=raising_hook)
        result.details.append("pipeline() returned normally — exception was swallowed")
        result["pass"] = False
    except _Sentinel:
        result.details.append("_Sentinel reached caller — hook exceptions propagate")
        result["pass"] = True
    except Exception as e:
        # The sentinel may be wrapped in another exception
        if isinstance(e.__cause__, _Sentinel) or isinstance(e.__context__, _Sentinel):
            result.details.append(
                f"_Sentinel reached caller wrapped in {type(e).__name__} — still propagates"
            )
            result["pass"] = True
        else:
            result.details.append(
                f"Got {type(e).__name__}: {e} — NOT our sentinel, exception was swallowed or replaced"
            )
            result["pass"] = False

    return result


def probe_prehook_exception(pipeline, vocal_path) -> dict:
    """Check #4: PyTorch forward_pre_hook exceptions propagate cleanly."""
    import torch.nn as nn

    result = {"pass": False, "details": []}

    segmentation_model = getattr(
        getattr(pipeline, "_segmentation", None), "model", None
    )
    if segmentation_model is None:
        result.details.append("Could not find segmentation model — cannot test")
        return result

    class _PreHookSentinel(Exception):
        pass

    call_count = [0]

    def raising_pre_hook(module, inputs):
        call_count[0] += 1
        if call_count[0] >= 1:
            raise _PreHookSentinel("pre_hook exception probe")

    handle = segmentation_model.register_forward_pre_hook(raising_pre_hook)
    result.details.append(f"Registered pre_hook on {type(segmentation_model).__name__}")

    try:
        pipeline(str(vocal_path))
        result.details.append(
            "pipeline() returned normally — pre_hook exception was swallowed"
        )
        result["pass"] = False
    except _PreHookSentinel:
        result.details.append(
            "_PreHookSentinel reached caller — pre_hook exceptions propagate"
        )
        result["pass"] = True
    except Exception as e:
        if isinstance(e.__cause__, _PreHookSentinel) or isinstance(
            e.__context__, _PreHookSentinel
        ):
            result.details.append(
                f"_PreHookSentinel reached caller wrapped in {type(e).__name__} — still propagates"
            )
            result["pass"] = True
        else:
            result.details.append(f"Got {type(e).__name__}: {e} — NOT our sentinel")
            result["pass"] = False
    finally:
        handle.remove()

    # Verify model still works after the hook-induced exception
    if result["pass"]:
        result.details.append(
            "Verifying model is still usable after pre_hook exception..."
        )
        try:
            pipeline(str(vocal_path))
            result.details.append("Model still usable after pre_hook exception ✓")
        except Exception as e:
            result.details.append(f"Model NOT usable after pre_hook exception: {e}")
            result["pass"] = False

    return result


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Probe pyannote pipeline for cancel architecture assumptions"
    )
    parser.add_argument(
        "vocal_audio", help="Path to a short vocal audio file for probing"
    )
    parser.add_argument("--hf-token", default="", help="Path to HuggingFace token file")
    args = parser.parse_args()

    vocal_path = Path(args.vocal_audio)
    if not vocal_path.exists():
        print(f"ERROR: {vocal_path} not found")
        sys.exit(1)

    # Load pipeline
    from pyannote.audio import Pipeline

    hf_token = None
    if args.hf_token:
        hf_token = Path(args.hf_token).read_text().strip()
    else:
        hf_token = resolve_hf_token()

    print("Loading pyannote pipeline...")
    start = time.time()
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
    elapsed = time.time() - start
    print(f"Pipeline loaded in {elapsed:.1f}s")

    # Run probes
    checks = [
        ("1. Embedding model is nn.Module", lambda: probe_embedding_model(pipeline)),
        ("2. hook= kwarg accepted", lambda: probe_hook_kwarg(pipeline)),
        (
            "3. Hook exceptions propagate",
            lambda: probe_hook_exception(pipeline, vocal_path),
        ),
        (
            "4. Pre-hook exceptions propagate",
            lambda: probe_prehook_exception(pipeline, vocal_path),
        ),
    ]

    results = {}
    all_pass = True

    print(f"\n{'=' * 70}")
    print("PROBE RESULTS")
    print(f"{'=' * 70}")

    for name, probe_fn in checks:
        print(f"\n--- {name} ---")
        try:
            r = probe_fn()
            results[name] = r
            status = "PASS" if r["pass"] else "FAIL"
            if not r["pass"]:
                all_pass = False
            print(f"  Status: {status}")
            for detail in r["details"]:
                print(f"  {detail}")
        except Exception as e:
            results[name] = {"pass": False, "details": [str(e)]}
            all_pass = False
            print(f"  Status: ERROR")
            print(f"  {e}")

    # Summary
    print(f"\n{'=' * 70}")
    if all_pass:
        print("ALL CHECKS PASSED — safe to implement two-layer hook architecture")
    else:
        failed = [name for name, r in results.items() if not r["pass"]]
        print(f"FAILURES: {failed}")
        print("Revise the architecture plan before implementing")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
