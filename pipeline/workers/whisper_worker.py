"""Whisper worker: in-process stable-ts worker with per-encoder-pass cancellation.

Adapted from cancel_tests/whisper/whisper_worker.py (hook-based approach).
Key differences from the pipeline variant:
- Uses register_forward_pre_hook() on model.encoder instead of monkey-patching
  model.encode() — whisper.model.Whisper has no .encode() method.
- align_and_refine() does NOT clear the cancel_event between phases — a single
  cancel signal propagates through both (production: whole song is discarded).

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
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import torch

from pipeline.config import WhisperModelConfig

logger = logging.getLogger(__name__)


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

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._model_loaded

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

        device = self._config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_source = self._config.model_path
        logger.info(f"Loading whisper model: {model_source} on {device}")
        start = time.time()

        self._model = stable_whisper.load_model(
            model_source,
            device=device,
        )

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

        align_kwargs = dict(
            language=self._config.language,
            vad=self._config.vad,
            vad_threshold=self._config.vad_threshold,
            suppress_silence=self._config.suppress_silence,
            suppress_word_ts=self._config.suppress_word_ts,
            only_voice_freq=self._config.only_voice_freq,
        )

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

        refine_kwargs = dict(
            steps=self._config.refine_steps,
            word_level=self._config.refine_word_level,
        )

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

    def align_and_refine(
        self,
        vocal_path: Path,
        lyrics_text: str,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run align() then refine() with cancellation support.

        Both operations share the same cancel_event. In the pipeline,
        a cancelled song is discarded entirely — no partial outputs are
        kept and no further stages run — so a cancel signal during align
        must propagate through refine rather than being swallowed.

        Returns:
            WhisperResult on success.

        Raises:
            AlignmentCancelledError: If either align or refine was cancelled.
        """
        result = self.align(vocal_path, lyrics_text, cancel_event)

        # NOTE: No cancel_event.clear() between align and refine.
        # In the pipeline, a cancel during align should propagate through
        # refine — the whole song is discarded on cancellation.

        refined = self.refine(vocal_path, result, cancel_event)
        return refined

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

        transcribe_kwargs = dict(
            language=self._config.language,
            vad=self._config.vad,
            vad_threshold=self._config.vad_threshold,
            suppress_silence=self._config.suppress_silence,
            suppress_word_ts=self._config.suppress_word_ts,
            only_voice_freq=self._config.only_voice_freq,
            word_timestamps=True,
        )

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

    def transcribe_and_refine(
        self,
        vocal_path: Path,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run transcribe() then refine() with cancellation support.

        Returns:
            WhisperResult on success.

        Raises:
            AlignmentCancelledError: If either phase was cancelled.
        """
        result = self.transcribe(vocal_path, cancel_event)
        if self._config.regroup:
            logger.info(f"Regrouping transcription segments: {self._config.regroup}")
            result.regroup(self._config.regroup)
        refined = self.refine(vocal_path, result, cancel_event)
        return refined

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
