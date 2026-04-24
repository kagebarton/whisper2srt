"""Hook-based whisper worker: per-encoder-pass cancellation for stable-ts.

This is a hook-based prototype of cancel_whisper/workers/cancelable_whisper_worker.py.
The key change is the injection mechanism: instead of monkey-patching model.encode(),
we use PyTorch's register_forward_pre_hook() on the encoder nn.Module to inject a
per-encoder-pass cancellation check.

How it works:
1. The worker loads a PyTorch whisper model via stable_whisper.load_model().
2. At load_model() time, the encoder nn.Module is cached as self._encoder_module.
3. When align() is called with a cancel_event, the worker registers a forward
   pre-hook on the encoder that checks cancel_event.is_set() before each encoder
   forward pass. If set, the hook raises _CancelledInsideEncoder.
4. The exception unwinds through:
   pre_hook() → nn.Module.__call__() → encoder.forward() → inference_func() →
   _compute_timestamps() → while loop → Aligner.align() → model.align()
5. The worker catches _CancelledInsideEncoder, terminates any orphaned AudioLoader
   FFmpeg subprocesses, clears GPU state, and re-raises as AlignmentCancelledError.
6. The hook handle is removed in a finally block (defensive try/except around
   handle.remove() to avoid masking the _CancelledInsideEncoder exception).
7. The model weights survive the exception — they're nn.Parameter attributes on
   the Whisper nn.Module, not Python stack locals. The next align() call works
   immediately without reloading.

Why hooks instead of monkey-patching?
- The PyTorch model (whisper.model.Whisper) has no .encode() method — that method
  only exists on faster_whisper.WhisperModel (the CTranslate2 wrapper). So the
  monkey-patch approach from cancel_whisper literally cannot work here.
- The encoder (whisper.model.AudioEncoder) is a torch.nn.Module. PyTorch exposes
  a stable, documented hook API designed for exactly this purpose.
- register_forward_pre_hook(fn) returns a RemovableHandle — canonical cleanup,
  no save/restore of the original forward.
- The hook fires through nn.Module.__call__ at the same encoder call points that
  the monkey-patched encode() intercepted in cancel_whisper.
- Resilient to upstream changes: the hook is registered on the module instance,
  so it stays attached even if stable-ts rebinds methods.

Encoder call paths (verified against installed stable_whisper):
- model.align(...) → Aligner.align → compute_timestamps →
  add_word_timestamps_stable → model.encoder(mel.unsqueeze(0)) at
  stable_whisper/timing.py:60. This is nn.Module.__call__, so the pre-hook fires.
- model.refine(...) → refinement inference_func at
  stable_whisper/alignment.py:661 calls model(mel_segments, tokens),
  which dispatches to Whisper.forward → self.encoder(...) — also via
  nn.Module.__call__. Hook fires.
- model.transcribe(...) → transcribe_stable → DecodingTask._get_audio_features()
  → model.encoder(mel) at whisper/decoding.py:655 — nn.Module.__call__.
  Hook fires.

AudioLoader cleanup on cancellation:
When _CancelledInsideEncoder unwinds through Aligner.align(), the normal
cleanup path (self.audio_loader.terminate() at line 357 of non_whisper/
alignment.py) is SKIPPED — it's outside the while loop, so exception-based
exit never reaches it. The orphaned AudioLoader holds an FFmpeg subprocess
(self._process) that keeps writing raw PCM to stdout. When Python's GC
eventually calls AudioLoader.__del__() → terminate(), it kills FFmpeg
mid-stream, producing "Broken pipe" / "Error submitting a packet" messages
on stderr. These are harmless but noisy.

We fix this by:
a) Monkey-patching AudioLoader._audio_loading_process() to redirect FFmpeg
   stderr to /dev/null (harmless muxer errors never reach terminal).
b) In the _CancelledInsideEncoder handler, walking gc to find any orphaned
   AudioLoader instances and calling terminate() on them — before FFmpeg
   stderr can bleed through.

Cancel granularity:
- For align(): per-token-batch (~100 tokens, ~5-10s of audio, ~1-3s GPU time)
- For refine(): per-binary-search-iteration (~0.1-0.5s GPU time per step)
The hook fires before each encoder forward pass, giving cancellation between
passes — identical granularity to the monkey-patch approach in cancel_whisper.
"""

import gc
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import torch

from cancel_tests.whisper.config import WhisperModelConfig

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

    Uses register_forward_pre_hook() on the PyTorch encoder nn.Module instead of
    monkey-patching model.encode() (which doesn't exist on whisper.model.Whisper).

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

        model_source = self._config.model_path or str(
            Path(__file__).resolve().parent.parent.parent / "models" / "large-v3-turbo.pt"
        )
        logger.info(f"Loading whisper model: {model_source} on {device}")
        start = time.time()

        self._model = stable_whisper.load_model(
            model_source,
            device=device,
        )

        # Cache the encoder nn.Module reference.
        # stable_whisper.load_model() returns whisper.model.Whisper directly
        # (after modify_model() binds align/refine/transcribe as methods).
        # The encoder is at self._model.encoder — NOT self._model.model.encoder,
        # because the returned object IS the nn.Module (no wrapper).
        self._encoder_module = self._model.encoder
        if not isinstance(self._encoder_module, torch.nn.Module):
            logger.warning(
                f"self._model.encoder is {type(self._encoder_module).__name__}, "
                f"not nn.Module — cancellation will degrade to waiting for the "
                f"current encode call to return. "
                f"Attempted path: self._model.encoder"
            )

        elapsed = time.time() - start
        self._model_loaded = True
        logger.info(f"Whisper model loaded in {elapsed:.1f}s (PyTorch device={device})")

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
                    stderr=subprocess.DEVNULL,  # <- THE KEY CHANGE
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
        gc to find these orphaned instances and terminate them explicitly.

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
        """Run model.align() with per-encoder-pass cancellation.

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

        # Build align kwargs once (used in both cancel and no-cancel branches)
        align_kwargs = dict(
            language=self._config.language,
            vad=self._config.vad,
            vad_threshold=self._config.vad_threshold,
            suppress_silence=self._config.suppress_silence,
            suppress_word_ts=self._config.suppress_word_ts,
            only_voice_freq=self._config.only_voice_freq,
        )

        # If no cancel event, just run alignment normally
        if cancel_event is None:
            return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)

        # If the encoder module wasn't cached, we can't install a hook —
        # fall back to running without cancellation.
        if self._encoder_module is None:
            logger.warning("Encoder module not cached — running without cancel check")
            return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)

        # --- Register forward pre-hook on encoder for cancel check ---
        encode_counter = [0]
        align_start_time = time.time()

        def cancel_pre_hook(module, inputs):
            # Check cancel BEFORE running the encoder forward pass.
            # This means we cancel between encoder calls, not mid-inference.
            elapsed = time.time() - align_start_time
            is_set = cancel_event.is_set()
            logger.info(
                f"[diag] encoder hook fired #{encode_counter[0] + 1} "
                f"at t={elapsed:.2f}s (cancel_set={is_set})"
            )
            if is_set:
                logger.info(
                    f"Cancel detected before encode pass "
                    f"#{encode_counter[0] + 1} — aborting"
                )
                raise _CancelledInsideEncoder()
            encode_counter[0] += 1  # counts passes STARTED, not completed

        handle = self._encoder_module.register_forward_pre_hook(cancel_pre_hook)
        logger.info("[diag] Registered forward pre-hook on encoder")

        try:
            return self._model.align(str(vocal_path), lyrics_text, **align_kwargs)
        except _CancelledInsideEncoder:
            logger.info(
                f"Alignment cancelled after {encode_counter[0]} encode passes "
                f"started — model still loaded"
            )
            # Clean up orphaned AudioLoader FFmpeg subprocess
            self._terminate_orphaned_audioloaders()
            _clear_gpu_state()
            raise AlignmentCancelledError(
                f"Alignment cancelled after {encode_counter[0]} encode passes "
                f"started (model still loaded)"
            )
        finally:
            # Always remove the hook — even on cancel/error.
            # Defensive try/except: if handle.remove() raises (e.g., the hook
            # was already removed by some other code), we must not let that
            # exception replace a propagating _CancelledInsideEncoder or
            # AlignmentCancelledError.
            try:
                handle.remove()
            except Exception:
                pass  # defensive: if stable-ts internals already removed the hook
            logger.debug(
                f"Removed encoder hook ({encode_counter[0]} encode passes started)"
            )

    def refine(
        self,
        vocal_path: Path,
        result,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run model.refine() with per-encoder-pass cancellation.

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
                    f"Cancel detected before refine pass "
                    f"#{encode_counter[0] + 1} — aborting"
                )
                raise _CancelledInsideEncoder()
            encode_counter[0] += 1

        handle = self._encoder_module.register_forward_pre_hook(cancel_pre_hook)
        logger.debug(
            "Registered forward pre-hook on encoder for per-pass cancel check (refine)"
        )

        try:
            return self._model.refine(str(vocal_path), result, **refine_kwargs)
        except _CancelledInsideEncoder:
            logger.info(
                f"Refinement cancelled after {encode_counter[0]} encode passes "
                f"started — model still loaded"
            )
            # Note: refine() uses prep_audio() not AudioLoader, so no FFmpeg
            # subprocess to clean up. But we call it anyway for safety.
            self._terminate_orphaned_audioloaders()
            _clear_gpu_state()
            raise AlignmentCancelledError(
                f"Refinement cancelled after {encode_counter[0]} encode passes "
                f"started (model still loaded)"
            )
        finally:
            try:
                handle.remove()
            except Exception:
                pass
            logger.debug(
                f"Removed encoder hook ({encode_counter[0]} encode passes started)"
            )

    def align_and_refine(
        self,
        vocal_path: Path,
        lyrics_text: str,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run align() then refine() with cancellation support.

        This matches cancel_whisper's standalone-test contract: a single
        cancel_event only kills the in-flight phase. Between align and refine,
        the cancel_event is cleared so a cancel during align doesn't
        immediately abort refine. If the caller wants to cancel refine too,
        they set the event again.

        (The pipeline's variant intentionally does NOT clear between phases —
        a cancelled song is discarded entirely. This prototype mirrors
        cancel_whisper's contract as specified.)

        Returns:
            WhisperResult on success.

        Raises:
            AlignmentCancelledError: If either align or refine was cancelled.
        """
        result = self.align(vocal_path, lyrics_text, cancel_event)
        if result is None:
            return None

        # Clear the cancel event between align and refine so that
        # a cancel during align doesn't immediately abort refine.
        if cancel_event is not None:
            cancel_event.clear()

        refined = self.refine(vocal_path, result, cancel_event)
        return refined

    def transcribe(
        self,
        vocal_path: Path,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Run model.transcribe() with per-encoder-pass cancellation.

        Args:
            vocal_path: Path to the audio file.
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

        encode_counter = [0]

        def cancel_pre_hook(module, inputs):
            if cancel_event.is_set():
                logger.info(
                    f"Cancel detected before encode pass "
                    f"#{encode_counter[0] + 1} — aborting"
                )
                raise _CancelledInsideEncoder()
            encode_counter[0] += 1

        handle = self._encoder_module.register_forward_pre_hook(cancel_pre_hook)
        logger.debug(
            "Registered forward pre-hook on encoder for per-pass cancel check (transcribe)"
        )

        try:
            return self._model.transcribe(str(vocal_path), **transcribe_kwargs)
        except _CancelledInsideEncoder:
            logger.info(
                f"Transcription cancelled after {encode_counter[0]} encode passes "
                f"started — model still loaded"
            )
            self._terminate_orphaned_audioloaders()
            _clear_gpu_state()
            raise AlignmentCancelledError(
                f"Transcription cancelled after {encode_counter[0]} encode passes "
                f"started (model still loaded)"
            )
        finally:
            try:
                handle.remove()
            except Exception:
                pass
            logger.debug(
                f"Removed encoder hook ({encode_counter[0]} encode passes started)"
            )

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

    After _CancelledInsideEncoder unwinds from the encoder hook, intermediate
    tensors from the interrupted forward pass should be freed as their stack
    frames are unwound by normal Python exception propagation. We clear the
    PyTorch CUDA cache as a safety net — torch.cuda.empty_cache() only
    releases unused cached allocator blocks, it does not free tensors still
    referenced by live frames.
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
