"""Cancelable pyannote diarization worker: two-layer hook cancellation.

Architecture: in-process (like cancel_whisper), no subprocess needed.
Cancel checks are installed at two complementary layers, both driving the
same threading.Event and raising the same internal _Cancelled exception.

Layer 1 — pyannote hook= callback on pipeline(...):
  Fires at stage boundaries and progress events. Covers clustering and
  provides progress logging for free. Cancel check is guarded on
  cancel_event so the same callback works with or without cancellation.

Layer 2 — PyTorch register_forward_pre_hook on each inference nn.Module:
  Fires before every batch forward pass of segmentation AND embedding
  models. This is what gives "cancel within embedding" behavior — the
  missing feature in the old monkey-patch design. Only installed when
  cancel_event is provided (no per-batch overhead otherwise).

If pyannote swallows hook exceptions (detected by probe check #3), the
Layer 1 callback also sets self._cancel_requested as a fallback flag,
checked after _run_diarization() returns.

Cancel granularity:
  Layer 2: per batch (~0.1–1s GPU time)
  Layer 1: end of current pyannote stage (typically <1s for clustering)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from cancel_diarize.config import DiarizeConfig

logger = logging.getLogger(__name__)


class _Cancelled(Exception):
    """Internal. Raised by any cancel check to unwind the pipeline.

    Unwinds through: PyTorch forward dispatch OR pyannote hook callback
    → pipeline.__call__() → diarize() except clause.
    Model weights survive because they're GPU/CPU attributes on the
    internal pyannote model, not Python stack locals.
    """


class DiarizationCancelledError(Exception):
    """Raised when diarization was cancelled mid-computation (model still loaded)."""


class CancelableDiarizeWorker:
    """In-process pyannote worker with two-layer hook cancellation.

    Loads a pyannote SpeakerDiarization pipeline once and keeps it loaded
    across jobs. If diarization is cancelled, the exception unwinds cleanly
    and the model weights survive — no reload needed for the next job.

    Cancellation uses PyTorch register_forward_pre_hook on both the
    segmentation and embedding models (Layer 2) plus a pyannote hook=
    callback for stage-boundary coverage and progress logging (Layer 1).
    """

    def __init__(self, config: Optional[DiarizeConfig] = None) -> None:
        self._config = config or DiarizeConfig()
        self._pipeline = None
        self._model_loaded = False
        # Persistent model references (set by _find_inference_models)
        self._segmentation_model = None  # nn.Module from pipeline._segmentation.model
        self._embedding_model = None  # nn.Module from pipeline._embedding(.model_)
        # Per-call state (reset each diarize())
        self._current_stage = None  # last stage name from pyannote hook
        self._batch_counter = {}  # {stage_label: forward_count}
        self._hook_handles = []  # PyTorch RemovableHandle list
        self._cancel_requested = (
            False  # fallback flag if pyannote swallows hook exceptions
        )

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None and self._model_loaded

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the pyannote speaker diarization pipeline.

        This is expensive (~10-30s) and should be called once at startup.
        Requires a HuggingFace token with pyannote model access, or a
        local HF cache (see pyann-models/).
        """
        if self._pipeline is not None:
            logger.info("Pipeline already loaded — skipping")
            return

        from pyannote.audio import Pipeline

        # Resolve HuggingFace token
        hf_token = self._resolve_hf_token()

        device = self._config.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info(f"Loading pyannote diarization pipeline (device={device})")
        start = time.time()

        from_pretrained_kwargs = {}
        if hf_token:
            from_pretrained_kwargs["token"] = hf_token
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **from_pretrained_kwargs,
        )

        if device != "cpu":
            import torch

            self._pipeline = self._pipeline.to(
                torch.device(device) if device != "auto" else None
            )

        # Find both inference models for Layer 2 hooks
        self._find_inference_models()

        # Log what we found
        seg_cls = (
            type(self._segmentation_model).__name__
            if self._segmentation_model
            else "NOT FOUND"
        )
        emb_cls = (
            type(self._embedding_model).__name__
            if self._embedding_model
            else "NOT FOUND"
        )
        logger.info(f"Segmentation: {seg_cls}, Embedding: {emb_cls}")

        elapsed = time.time() - start
        self._model_loaded = True
        logger.info(f"Pyannote pipeline loaded in {elapsed:.1f}s")

    def unload_model(self) -> None:
        """Unload the pipeline and free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._segmentation_model = None
            self._embedding_model = None
            self._model_loaded = False
            _clear_gpu_cache()
            logger.info("Pyannote pipeline unloaded")

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def _resolve_hf_token(self) -> Optional[str]:
        """Resolve HuggingFace token from config path or environment."""
        import os

        # Try config path first
        if self._config.hf_token_path:
            token_path = Path(self._config.hf_token_path)
            if token_path.exists():
                token = token_path.read_text().strip()
                if token:
                    logger.debug(f"Loaded HF token from {token_path}")
                    return token

        # Fall back to environment variable
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            logger.debug("Loaded HF token from environment")
            return token

        logger.info(
            "No HuggingFace token found — relying on local HF cache (HF_HOME). "
            "Set hf_token_path or HF_TOKEN if the model isn't already cached."
        )
        return None

    def _find_inference_models(self) -> None:
        """Find both inference models inside the pyannote pipeline.

        Populates self._segmentation_model and self._embedding_model.
        If a model cannot be resolved, the ref stays None and that stage
        degrades to Layer-1-only cancel granularity.
        """
        import torch.nn as nn

        pipeline = self._pipeline
        if pipeline is None:
            return

        # --- Segmentation ---
        seg_model = None
        seg_path = "pipeline._segmentation.model"

        # Try the standard path: pipeline._segmentation.model
        segmentation = getattr(pipeline, "_segmentation", None)
        if segmentation is not None:
            model = getattr(segmentation, "model", None)
            if model is not None and isinstance(model, nn.Module):
                seg_model = model
                logger.debug(
                    f"Found segmentation model at {seg_path} ({type(model).__name__})"
                )

        # Fallback: walk named_modules for an nn.Module whose name contains "segmentation"
        if seg_model is None:
            for name, module in pipeline.named_modules():
                if module is pipeline:
                    continue
                if "segmentation" in name.lower() and isinstance(module, nn.Module):
                    seg_model = module
                    seg_path = f"named_modules['{name}']"
                    logger.debug(
                        f"Found segmentation model at {seg_path} ({type(module).__name__})"
                    )
                    break

        self._segmentation_model = seg_model
        if seg_model is None:
            logger.warning(
                "Could not find segmentation model — "
                "segmentation-stage cancel degrades to Layer 1 only"
            )

        # --- Embedding ---
        emb_model = None
        emb_path = "pipeline._embedding.model_"

        embedding = getattr(pipeline, "_embedding", None)
        if embedding is not None:
            # Try .model_ first (pyannote wraps some embedders with this attribute)
            inner = getattr(embedding, "model_", None)
            if inner is not None and isinstance(inner, nn.Module):
                emb_model = inner
                logger.debug(
                    f"Found embedding model at {emb_path} ({type(inner).__name__})"
                )

            # Try .model
            if emb_model is None:
                inner = getattr(embedding, "model", None)
                if inner is not None and isinstance(inner, nn.Module):
                    emb_model = inner
                    emb_path = "pipeline._embedding.model"
                    logger.debug(
                        f"Found embedding model at {emb_path} ({type(inner).__name__})"
                    )

            # Try the embedding object itself
            if emb_model is None and isinstance(embedding, nn.Module):
                emb_model = embedding
                emb_path = "pipeline._embedding"
                logger.debug(
                    f"Found embedding model at {emb_path} ({type(embedding).__name__})"
                )

        # Fallback: walk named_modules for "embedding" or "wespeaker"
        if emb_model is None:
            for name, module in pipeline.named_modules():
                if module is pipeline:
                    continue
                name_lower = name.lower()
                if (
                    "embedding" in name_lower or "wespeaker" in name_lower
                ) and isinstance(module, nn.Module):
                    emb_model = module
                    emb_path = f"named_modules['{name}']"
                    logger.debug(
                        f"Found embedding model at {emb_path} ({type(module).__name__})"
                    )
                    break

        self._embedding_model = emb_model
        if emb_model is None:
            logger.warning(
                "Could not find embedding model — "
                "embedding-stage cancel degrades to Layer 1 only"
            )

    # ------------------------------------------------------------------
    # Diarization
    # ------------------------------------------------------------------

    def diarize(
        self,
        vocal_path: Path,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[dict]:
        """Run speaker diarization with two-layer hook cancellation.

        Args:
            vocal_path: Path to the vocal stem audio file.
            cancel_event: Optional threading.Event. When set, diarization
                aborts at the next cancel check point (within-stage via
                Layer 2 PyTorch hooks, or at stage boundaries via Layer 1
                pyannote callback). If None, diarization runs to completion
                with progress logging only.

        Returns:
            List of turn dicts: [{"speaker": str, "start": float, "end": float}, ...]

        Raises:
            DiarizationCancelledError: If diarization was cancelled mid-computation.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded — call load_model() first")

        # Build pipeline kwargs from config
        kwargs = {}
        if self._config.num_speakers > 0:
            kwargs["num_speakers"] = self._config.num_speakers
        if self._config.min_speakers > 0:
            kwargs["min_speakers"] = self._config.min_speakers
        if self._config.max_speakers > 0:
            kwargs["max_speakers"] = self._config.max_speakers

        # Reset per-call state
        self._current_stage = None
        self._batch_counter = {}
        self._hook_handles = []
        self._cancel_requested = False

        # Install hooks (always installs pyannote progress hook;
        # installs PyTorch pre_hooks only when cancel_event is provided)
        pyannote_hook = self._install_hooks(cancel_event)

        try:
            turns = self._run_diarization(vocal_path, hook=pyannote_hook, **kwargs)

            # Fallback: if pyannote swallows hook exceptions, the callback
            # sets _cancel_requested instead of raising. Check it here.
            if self._cancel_requested:
                stage = self._current_stage or "pre-start"
                total_batches = sum(self._batch_counter.values())
                logger.info(
                    f"Diarization cancel detected post-hoc at stage={stage} "
                    f"after {total_batches} forward passes — model still loaded"
                )
                raise DiarizationCancelledError(
                    f"Cancelled during stage={stage} after {total_batches} forward passes"
                )

            return turns

        except _Cancelled:
            _clear_gpu_state()
            stage = self._current_stage or "pre-start"
            total_batches = sum(self._batch_counter.values())
            logger.info(
                f"Diarization cancelled during stage={stage} "
                f"after {total_batches} forward passes — model still loaded"
            )
            raise DiarizationCancelledError(
                f"Cancelled during stage={stage} after {total_batches} forward passes"
            )
        finally:
            self._remove_cancel_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _install_hooks(
        self, cancel_event: Optional[threading.Event] = None
    ) -> Callable:
        """Install both cancel hook layers and return the pyannote callback.

        Layer 1 (pyannote hook callback) is always installed so callers get
        stage-boundary progress for free.

        Layer 2 (PyTorch forward_pre_hook) is only installed when
        cancel_event is provided — no per-batch overhead otherwise.

        Returns:
            The pyannote-style hook callback to pass as hook= to pipeline(...).
        """
        # --- Layer 2: PyTorch forward_pre_hook on each inference model ---
        if cancel_event is not None:

            def make_pre_hook(stage_label: str):
                def pre_hook(module, inputs):
                    if cancel_event.is_set():
                        raise _Cancelled()
                    self._batch_counter[stage_label] = (
                        self._batch_counter.get(stage_label, 0) + 1
                    )

                return pre_hook

            models_to_hook = [
                (self._segmentation_model, "segmentation"),
                (self._embedding_model, "embedding"),
            ]
            for model, label in models_to_hook:
                if model is not None:
                    handle = model.register_forward_pre_hook(make_pre_hook(label))
                    self._hook_handles.append(handle)

            n_hooks = len(self._hook_handles)
            logger.debug(f"Installed {n_hooks} forward_pre_hooks + pyannote callback")
        else:
            logger.debug("Installed pyannote callback only (no cancel_event)")

        # --- Layer 1: pyannote hook callback (always installed) ---
        # The cancel check is guarded on cancel_event so the same callback
        # works with or without cancellation.
        def pyannote_hook(step_name, *args, completed=None, total=None, **kwargs):
            self._current_stage = step_name

            if cancel_event is not None and cancel_event.is_set():
                # If pyannote swallows this exception, the fallback
                # _cancel_requested flag will catch it in diarize().
                self._cancel_requested = True
                raise _Cancelled()

            if completed is not None and total:
                logger.info(f"Progress: {step_name} {completed}/{total}")
            # else: stage-start events — don't spam the log

        return pyannote_hook

    def _remove_cancel_hooks(self) -> None:
        """Remove all PyTorch forward_pre_hooks. Called from finally block."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        n_removed = len(self._hook_handles)
        self._hook_handles = []
        if n_removed:
            logger.debug(f"Removed {n_removed} hook handles")

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _run_diarization(
        self, vocal_path: Path, hook: Optional[Callable] = None, **kwargs
    ) -> list[dict]:
        """Run the pyannote pipeline and convert to turn dicts.

        Args:
            vocal_path: Path to the vocal stem audio file.
            hook: Optional pyannote-style callback for progress/cancel.
            **kwargs: Pipeline kwargs (num_speakers, min_speakers, max_speakers).
        """
        logger.info(f"Diarization started on {vocal_path.name}")

        # Build pipeline call kwargs
        call_kwargs = dict(kwargs)
        if hook is not None:
            call_kwargs["hook"] = hook

        # pyannote pipeline() accepts a file path or dict
        result = self._pipeline(str(vocal_path), **call_kwargs)

        # pyannote 4.x returns DiarizeOutput(speaker_diarization=Annotation, ...)
        # pyannote 3.x returns Annotation directly
        annotation = getattr(result, "speaker_diarization", result)

        # Convert Annotation to list of turn dicts
        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                }
            )

        speakers = sorted(set(t["speaker"] for t in turns))
        logger.info(
            f"Diarization complete: {len(turns)} turns, {len(speakers)} speakers"
        )
        return turns


# ------------------------------------------------------------------
# GPU cleanup helpers
# ------------------------------------------------------------------


def _clear_gpu_state() -> None:
    """Clear intermediate GPU state after a cancelled operation."""
    _clear_gpu_cache()


def _clear_gpu_cache() -> None:
    """Clear PyTorch GPU cache on worker shutdown."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
