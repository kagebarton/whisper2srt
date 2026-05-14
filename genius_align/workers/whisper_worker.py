"""Whisper worker: in-process stable-ts worker with per-encoder-pass cancellation.

Forked from pipeline.workers.whisper_worker for genius_align — extended to
wire all tuning fields exposed by ``genius_align.config.WhisperModelConfig``.
This is the canonical version; port additions back to pipeline when proven.

Key differences from the pipeline variant:
- Uses register_forward_pre_hook() on model.encoder instead of monkey-patching
  model.encode() — whisper.model.Whisper has no .encode() method.
- Exposes each stage individually (align, transcribe, regroup, refine,
  postprocess) rather than combined align_and_refine/transcribe_and_refine
  wrappers — run.py composes the stages it needs per match-method, and the
  --match-method=auto gate gates between align() and refine(). Callers pass
  the same cancel_event to each stage, so a cancel propagates across all of
  them (production: whole song is discarded).

Unlike the StemWorker (which runs audio-separator in a subprocess), this
worker runs stable-ts in the same process. Cancellation is done by
registering a forward pre-hook on the encoder nn.Module to check a
threading.Event before each encoder forward pass.

The model is loaded once and stays loaded across jobs. If alignment is
cancelled, the exception unwinds cleanly and the model weights survive.

AudioLoader FFmpeg subprocess cleanup:
On cancellation, the Aligner's while loop exits via exception, skipping
the normal audio_loader.terminate() cleanup. The orphaned FFmpeg process
would produce "Broken pipe" stderr messages when eventually killed by GC.
We prevent this by:
1. Monkey-patching AudioLoader._audio_loading_process() to redirect
   FFmpeg stderr to /dev/null (harmless muxer errors never reach terminal).
2. Explicitly terminating orphaned AudioLoaders in the cancel handler.
"""

import gc
import logging
import os
import re
import subprocess
import threading
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

from genius_align.config import WhisperModelConfig

logger = logging.getLogger(__name__)

# stable-ts emits "<n>/<m> segments failed to align." as a UserWarning at the
# end of Aligner.align(). The align() wrapper captures and parses it so run.py
# can auto-escalate to the tiling matcher: a high failure ratio means align()'s
# forced word placement is untrustworthy.
_ALIGN_FAILURE_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+segments failed to align")


def _splat(section) -> dict[str, Any]:
    """Convert a dataclass section to kwargs, dropping None values.

    None means 'unset' — fall back to stable-ts's own default.
    """
    return {k: v for k, v in asdict(section).items() if v is not None}


def _extract_align_failure_ratio(caught_warnings) -> float:
    """Scan captured warnings for stable-ts's 'N/M segments failed to align'
    message and return N/M. Every captured warning is re-logged so capturing
    them here doesn't hide them from the user. Returns 0.0 if not found.
    """
    ratio = 0.0
    for w in caught_warnings:
        msg = str(w.message)
        m = _ALIGN_FAILURE_RE.search(msg)
        if m:
            failed, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                ratio = failed / total
        logger.warning("%s: %s", w.category.__name__, msg)
    return ratio


class _CancelledInsideEncoder(Exception):
    """Raised by the forward pre-hook when cancel is detected.

    This exception unwinds through:
    pre_hook() → nn.Module.__call__() → encoder.forward() → inference_func() →
    _compute_timestamps() → while loop → Aligner.align() → model.align()
    or:
    pre_hook() → nn.Module.__call__() → encoder.forward() → inference_func() →
    Refiner.get_prob() → while loop → Refiner._refine() → model.refine()

    The model weights survive because they're nn.Parameter attributes on the
    Whisper nn.Module (stored on GPU/CPU), not stack locals that get destroyed
    during unwinding.
    """


class AlignmentCancelledError(Exception):
    """Raised when alignment was cancelled mid-computation (model still loaded)."""


class WhisperWorker:
    """In-process stable-ts worker with per-encoder-pass cancellation via hooks.

    Unlike StemWorker (which runs audio-separator in a subprocess), this
    worker runs stable-ts in the same process. Cancellation is done by
    registering a forward pre-hook on model.encoder (an nn.Module) to check
    a threading.Event before each encoder forward pass.

    The model is loaded once and stays loaded across jobs. If alignment is
    cancelled, the exception unwinds cleanly and the model weights survive.

    AudioLoader FFmpeg subprocess cleanup:
    On cancellation, the Aligner's while loop exits via exception, skipping
    the normal audio_loader.terminate() cleanup. The orphaned FFmpeg process
    would produce "Broken pipe" stderr messages when eventually killed by GC.
    We prevent this by:
    1. Monkey-patching AudioLoader._audio_loading_process() to redirect
       FFmpeg stderr to /dev/null (harmless muxer errors never reach terminal).
    2. Explicitly terminating orphaned AudioLoaders in the cancel handler.
    """

    def __init__(self, config: Optional[WhisperModelConfig] = None) -> None:
        self._config = config or WhisperModelConfig()
        self._model = None
        self._model_loaded = False
        self._encoder_module = None
        self._audioloader_patched = False
        self._last_align_failure_ratio: float = 0.0

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._model_loaded

    @property
    def last_align_failure_ratio(self) -> float:
        """Fraction of segments stable-ts reported as failed in the most
        recent ``align()`` call. 0.0 if align hasn't run, none failed, or
        the warning could not be parsed.
        """
        return self._last_align_failure_ratio

    def load_model(self) -> None:
        """Load the PyTorch whisper model via stable-ts.

        This is expensive (~10-30s) and should be called once at startup.
        Also caches the encoder nn.Module reference and patches AudioLoader
        to suppress FFmpeg broken-pipe stderr on cancellation.
        """
        if self._model is not None:
            logger.info("Model already loaded — skipping")
            return

        import stable_whisper

        load_kwargs = _splat(self._config.load_model)
        model_source = load_kwargs.pop("model_path")
        device = load_kwargs.pop("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        load_kwargs["device"] = device

        logger.info(f"Loading whisper model: {model_source} on {device}")
        start = time.time()

        self._model = stable_whisper.load_model(model_source, **load_kwargs)

        # Cache the encoder nn.Module for hook registration.
        # stable_whisper.load_model() returns whisper.model.Whisper directly —
        # the encoder is at self._model.encoder, not self._model.model.encoder.
        self._encoder_module = self._model.encoder
        if not isinstance(self._encoder_module, torch.nn.Module):
            logger.warning(
                f"self._model.encoder is {type(self._encoder_module).__name__}, "
                f"not nn.Module — cancellation will degrade to waiting for the "
                f"current encode pass to return"
            )

        elapsed = time.time() - start
        self._model_loaded = True
        logger.info(f"Whisper model loaded in {elapsed:.1f}s (device={device})")

        # Patch AudioLoader to redirect FFmpeg stderr to /dev/null so that
        # killed FFmpeg processes don't produce "Broken pipe" messages.
        self._patch_audioloader_stderr()

    def _patch_audioloader_stderr(self) -> None:
        """Monkey-patch AudioLoader._audio_loading_process() to suppress
        FFmpeg broken-pipe stderr messages on cancellation.

        When alignment is cancelled, the AudioLoader's FFmpeg subprocess gets
        killed mid-stream. FFmpeg writes muxer errors to stderr when this
        happens ("Error submitting a packet", "Broken pipe", etc.). These are
        harmless but noisy. By redirecting FFmpeg's stderr to /dev/null, these
        messages never reach the terminal.

        The original _audio_loading_process() launches FFmpeg with
        `-loglevel error` but no stderr redirect — FFmpeg errors go to the
        inherited stderr. We patch it to pass stderr=subprocess.DEVNULL.
        """
        if self._audioloader_patched:
            return

        try:
            from stable_whisper.audio import AudioLoader
        except ImportError:
            logger.debug("Could not import AudioLoader — skipping stderr patch")
            return

        original_audio_loading_process = AudioLoader._audio_loading_process

        def _quiet_audio_loading_process(self_loader):
            """Patched _audio_loading_process that redirects FFmpeg stderr."""
            # The original method returns None when self.source is not a string
            # or self._stream is False (i.e. non-streaming mode). In that case
            # there's no FFmpeg process to quiet.
            if not isinstance(self_loader.source, str) or not self_loader._stream:
                return original_audio_loading_process(self_loader)

            # Replicate the original _audio_loading_process logic but with
            # stderr redirected to DEVNULL instead of inheriting the parent's
            # stderr. This suppresses the "Broken pipe" / muxer error messages
            # that appear when the FFmpeg process is killed on cancellation.
            from stable_whisper.audio.utils import load_source

            only_ffmpeg = False
            source = load_source(
                self_loader.source,
                verbose=self_loader.verbose,
                only_ffmpeg=only_ffmpeg,
                return_dict=True,
            )
            if isinstance(source, dict):
                info = source
                source = info.pop("popen")
            else:
                info = None

            if info and info.get("duration"):
                self_loader._duration_estimation = info["duration"]
            if not self_loader._stream and info and info.get("is_live"):
                import warnings

                warnings.warn(
                    "The audio appears to be a continuous stream but "
                    "setting was set to `stream=False`."
                )

            if isinstance(source, subprocess.Popen):
                self_loader._extra_process = source
                stdin = source.stdout
            else:
                stdin = None

            try:
                cmd = [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-threads",
                    "0",
                    "-i",
                    self_loader.source if stdin is None else "pipe:",
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(self_loader._sr),
                    "-",
                ]
                out = subprocess.Popen(
                    cmd,
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # ← THE KEY CHANGE
                )
            except subprocess.SubprocessError as e:
                raise RuntimeError(f"Failed to load audio: {e}") from e

            return out

        AudioLoader._audio_loading_process = _quiet_audio_loading_process
        self._audioloader_patched = True
        logger.debug(
            "Patched AudioLoader._audio_loading_process() to suppress "
            "FFmpeg broken-pipe stderr"
        )

    def _terminate_orphaned_audioloaders(self) -> None:
        """Terminate any orphaned AudioLoader FFmpeg subprocesses.

        When _CancelledInsideEncoder unwinds through Aligner.align(), the
        normal cleanup path (self.audio_loader.terminate()) is skipped. The
        AudioLoader still holds a reference to the FFmpeg subprocess. We walk
        the gc to find these orphaned instances and terminate them explicitly.

        This is a belt-and-suspenders approach alongside the stderr redirect
        patch: even if stderr is suppressed, we still want to kill the FFmpeg
        process promptly to free its resources.
        """
        try:
            from stable_whisper.audio import AudioLoader
        except ImportError:
            return

        terminated_count = 0
        for obj in gc.get_objects():
            if isinstance(obj, AudioLoader):
                process = getattr(obj, "_process", None)
                if process is not None and process.poll() is None:
                    # FFmpeg subprocess is still running — kill it
                    try:
                        process.terminate()
                        process.wait(timeout=2)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                    terminated_count += 1

                extra_process = getattr(obj, "_extra_process", None)
                if extra_process is not None and extra_process.poll() is None:
                    try:
                        extra_process.terminate()
                        extra_process.wait(timeout=2)
                    except Exception:
                        try:
                            extra_process.kill()
                        except Exception:
                            pass
                    terminated_count += 1

        if terminated_count > 0:
            logger.debug(
                f"Terminated {terminated_count} orphaned FFmpeg subprocess(es)"
            )

    def align(
        self,
        vocal_path: Path,
        lyrics_text: str,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run align() and capture stable-ts's segment-failure warning.

        Thin wrapper over ``_align()`` that records the
        "N/M segments failed to align" UserWarning into
        ``last_align_failure_ratio``, so callers can decide whether align()'s
        forced word placement is trustworthy *before* spending time on
        refine().
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self._align(vocal_path, lyrics_text, cancel_event)
        self._last_align_failure_ratio = _extract_align_failure_ratio(caught)
        return result

    def _align(
        self,
        vocal_path: Path,
        lyrics_text: str,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run model.align() with per-encode-pass cancellation.

        Args:
            vocal_path: Path to the vocal stem WAV file.
            lyrics_text: Plain-text lyrics to align against.
            cancel_event: Optional threading.Event. When set, alignment
                aborts between encoder passes. If None, alignment runs to
                completion (no cancellation).

        Returns:
            WhisperResult on success.

        Raises:
            AlignmentCancelledError: If alignment was cancelled mid-computation.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        align_kwargs = _splat(self._config.align)

        if cancel_event is None:
            return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)

        if self._encoder_module is None:
            logger.warning("Encoder module not cached — running without cancel check")
            return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)

        # --- Register forward pre-hook on encoder for cancel check ---
        encode_counter = [0]

        def cancel_pre_hook(module, inputs):
            if cancel_event.is_set():
                logger.info(
                    f"Cancel detected before align pass #{encode_counter[0] + 1} — aborting"
                )
                raise _CancelledInsideEncoder()
            encode_counter[0] += 1

        handle = self._encoder_module.register_forward_pre_hook(cancel_pre_hook)
        logger.debug("Registered forward pre-hook on encoder for align")

        try:
            return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)
        except _CancelledInsideEncoder:
            logger.info(
                f"Alignment cancelled after {encode_counter[0]} encode passes "
                f"— model still loaded"
            )
            self._terminate_orphaned_audioloaders()
            _clear_gpu_state()
            raise AlignmentCancelledError(
                f"Alignment cancelled after {encode_counter[0]} encode passes "
                f"(model still loaded)"
            )
        finally:
            try:
                handle.remove()
            except Exception:
                pass
            logger.debug(f"Removed encoder hook ({encode_counter[0]} encode passes)")

    def refine(
        self,
        vocal_path: Path,
        result,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run model.refine() with per-encode-pass cancellation.

        Args:
            vocal_path: Path to the vocal stem WAV file.
            result: WhisperResult from a previous align() call.
            cancel_event: Optional threading.Event. When set, refinement
                aborts between encoder passes.

        Returns:
            WhisperResult on success.

        Raises:
            AlignmentCancelledError: If refinement was cancelled mid-computation.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        refine_kwargs = _splat(self._config.refine)

        if cancel_event is None:
            return self._model.refine(str(vocal_path), result, **refine_kwargs)

        if self._encoder_module is None:
            logger.warning("Encoder module not cached — running without cancel check")
            return self._model.refine(str(vocal_path), result, **refine_kwargs)

        # --- Register forward pre-hook on encoder for cancel check ---
        encode_counter = [0]

        def cancel_pre_hook(module, inputs):
            if cancel_event.is_set():
                logger.info(
                    f"Cancel detected before refine pass #{encode_counter[0] + 1} — aborting"
                )
                raise _CancelledInsideEncoder()
            encode_counter[0] += 1

        handle = self._encoder_module.register_forward_pre_hook(cancel_pre_hook)
        logger.debug("Registered forward pre-hook on encoder for refine")

        try:
            return self._model.refine(str(vocal_path), result, **refine_kwargs)
        except _CancelledInsideEncoder:
            logger.info(
                f"Refinement cancelled after {encode_counter[0]} encode passes "
                f"— model still loaded"
            )
            self._terminate_orphaned_audioloaders()
            _clear_gpu_state()
            raise AlignmentCancelledError(
                f"Refinement cancelled after {encode_counter[0]} encode passes "
                f"(model still loaded)"
            )
        finally:
            try:
                handle.remove()
            except Exception:
                pass
            logger.debug(f"Removed encoder hook ({encode_counter[0]} encode passes)")

    def postprocess(self, result):
        """Apply config-driven WhisperResult post-processing.

        Each ``PostProcessKwargs`` field drives a separate WhisperResult
        call. A discrete stage (not folded into ``refine()``) so the
        transcribe pipeline can refine without it: ``run.py`` composes the
        stages it needs per match-method.
        """
        pp = self._config.post_process
        if pp.adjust_gaps_threshold is not None:
            logger.debug(
                f"Post-process: adjust_gaps(duration_threshold={pp.adjust_gaps_threshold})"
            )
            result = result.adjust_gaps(duration_threshold=pp.adjust_gaps_threshold)

        if pp.merge_by_gap_min is not None:
            logger.debug(f"Post-process: merge_by_gap(min_gap={pp.merge_by_gap_min})")
            result = result.merge_by_gap(min_gap=pp.merge_by_gap_min)

        return result

    def transcribe(
        self,
        vocal_path: Path,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run model.transcribe() with per-encode-pass cancellation.

        Args:
            vocal_path: Path to the vocal stem WAV file.
            cancel_event: Optional threading.Event. When set, transcription
                aborts between encoder passes.

        Returns:
            WhisperResult on success.

        Raises:
            AlignmentCancelledError: If transcription was cancelled mid-computation.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        transcribe_kwargs = _splat(self._config.transcribe)

        if cancel_event is None:
            return self._model.transcribe(str(vocal_path), **transcribe_kwargs)

        if self._encoder_module is None:
            logger.warning("Encoder module not cached — running without cancel check")
            return self._model.transcribe(str(vocal_path), **transcribe_kwargs)

        # --- Register forward pre-hook on encoder for cancel check ---
        encode_counter = [0]

        def cancel_pre_hook(module, inputs):
            if cancel_event.is_set():
                logger.info(
                    f"Cancel detected before transcribe pass #{encode_counter[0] + 1} — aborting"
                )
                raise _CancelledInsideEncoder()
            encode_counter[0] += 1

        handle = self._encoder_module.register_forward_pre_hook(cancel_pre_hook)
        logger.debug("Registered forward pre-hook on encoder for transcribe")

        try:
            return self._model.transcribe(str(vocal_path), **transcribe_kwargs)
        except _CancelledInsideEncoder:
            logger.info(
                f"Transcription cancelled after {encode_counter[0]} encode passes "
                f"— model still loaded"
            )
            self._terminate_orphaned_audioloaders()
            _clear_gpu_state()
            raise AlignmentCancelledError(
                f"Transcription cancelled after {encode_counter[0]} encode passes "
                f"(model still loaded)"
            )
        finally:
            try:
                handle.remove()
            except Exception:
                pass
            logger.debug(f"Removed encoder hook ({encode_counter[0]} encode passes)")

    def regroup(self, result):
        """Apply the configured stable-ts regroup expression in place.

        Used by the transcribe pipeline only — the align pipeline keeps
        reference-driven segmentation. No-op if ``cfg.regroup`` is empty.
        Returns ``result`` for call-site composability (stable-ts'
        ``regroup()`` mutates in place and returns self).
        """
        if self._config.regroup:
            logger.info(f"Regrouping transcription segments: {self._config.regroup}")
            result.regroup(self._config.regroup)
        return result

    def unload_model(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._encoder_module = None
            self._model_loaded = False
            _clear_gpu_cache()
            logger.info("Whisper model unloaded")


def _clear_gpu_state() -> None:
    """Clear intermediate GPU state after a cancelled operation.

    After _CancelledInsideEncoder unwinds, there should be no leftover GPU
    state from the interrupted encoder pass (CTranslate2 manages its own
    memory internally). But we clear the PyTorch cache as a safety net.
    """
    _clear_gpu_cache()


def _clear_gpu_cache() -> None:
    """Clear PyTorch GPU cache on worker shutdown."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
