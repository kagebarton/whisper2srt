"""Stem worker: persistent subprocess with mid-separation cancellation.

This is adapted from cancel_separator/workers/cancelable_stem_worker.py.
The worker runs audio-separator in a subprocess (model loaded once, stays
loaded), but injects a **per-chunk cancellation check** into the model's
demix loop.

How it works:
1. The worker process loads the audio-separator model at startup.
2. On each separate() call, the main process sends (wav_path, output_dir)
over a Pipe, and a cancel_event is forwarded via a dedicated cancel Pipe.
3. Before calling separator.separate(), the worker **monkey-patches** the
model's `model_run.forward()` to check the cancel pipe before each
forward pass. In the Roformer demix loop, `self.model_run(part.unsqueeze(0))[0]`
is called once per chunk — so the check runs between chunks.
4. When the caller sets the cancel event, the main process forwards
a signal on the cancel pipe. The patched forward() detects it and raises
_CancelledInsideDemix.
5. The exception unwinds through demix() → separate() → _separate_file().
Because we're in a subprocess, the exception stays local.
6. The worker catches _CancelledInsideDemix, clears GPU state, and sends
("cancelled",) back over the result Pipe.
7. The model weights (self.model_run) survive the exception — they're on
the GPU as class attributes, not on the Python stack. The next job
can call separate() immediately without reloading.

Why patch forward() not __call__?
- PyTorch nn.Module.__call__ is a class-level special method (dunder).
Python looks up dunder methods on the CLASS, not the instance, so
instance-level __call__ patches are silently ignored.
- forward() is a regular method. It IS looked up on the instance first,
so we can safely monkey-patch it on the model_run object.
- nn.Module.__call__ internally calls self.forward(...), so our patch
runs at the same point in the call stack — right before the actual
model computation for each chunk.

Pipe-based cancel signaling avoids shared-memory issues between processes.
"""

import logging
import os
import sys
import threading
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DEFAULT_MODEL_DIR = "./audio-separator/models"
SEPARATION_FORMAT = "wav"


class WorkerCancelledError(Exception):
    """Raised when the worker detects cancellation mid-separation."""


class WorkerDiedError(Exception):
    """Raised when the stem worker subprocess dies during a job."""


class StemWorker:
    """Persistent subprocess with per-chunk cancellation support.

    Unlike the production StemWorker (which can only cancel between
    full separations), this worker injects a cancellation check into
    audio-separator's model forward pass so separation can be aborted
    between chunks without killing the process or reloading the model.
    """

    def __init__(
        self,
        temp_dir: str = "",
        log_level: int = logging.INFO,
        model_dir: str = DEFAULT_MODEL_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self._temp_dir = temp_dir
        self._log_level = log_level
        self._model_dir = model_dir
        self._model_name = model_name
        self._process: Process | None = None
        self._job_send: Connection | None = None
        self._job_recv: Connection | None = None
        self._result_recv: Connection | None = None
        self._result_send: Connection | None = None
        self._cancel_send: Connection | None = None
        self._cancel_recv: Connection | None = None

    def start(self) -> None:
        """Spawn the subprocess with fresh IPC channels."""
        self._close_all_connections()

        job_recv, job_send = Pipe(duplex=False)
        self._job_send = job_send
        self._job_recv = job_recv

        result_recv, result_send = Pipe(duplex=False)
        self._result_recv = result_recv
        self._result_send = result_send

        cancel_recv, cancel_send = Pipe(duplex=False)
        self._cancel_send = cancel_send
        self._cancel_recv = cancel_recv

        self._process = Process(
            target=_worker_main,
            args=(
                self._job_recv,
                self._result_send,
                self._cancel_recv,
                self._temp_dir,
                self._log_level,
                self._model_dir,
                self._model_name,
            ),
            daemon=True,
        )
        self._process.start()
        logger.info("Stem worker started (PID %d)", self._process.pid)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def separate(
        self,
        wav_path: Path,
        output_dir: Path,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, Path]:
        """Submit a WAV for separation. Blocks until result or cancellation.

        Args:
            wav_path: Path to the input WAV file.
            output_dir: Directory for separation output.
            cancel_event: Optional threading.Event from the orchestrator.
                When set, the worker will abort separation between chunks.
                If None, the separation runs to completion (no cancellation).
        """
        rq = self._result_recv
        js = self._job_send
        proc = self._process

        if rq is None or js is None or proc is None:
            raise WorkerDiedError("Stem worker is not running")

        js.send((str(wav_path), str(output_dir)))

        # Forward threading.Event → cancel Pipe via a daemon thread
        cancel_forwarder: threading.Thread | None = None
        if cancel_event is not None and self._cancel_send is not None:
            cancel_forwarder = threading.Thread(
                target=_forward_cancel,
                args=(cancel_event, self._cancel_send),
                daemon=True,
            )
            cancel_forwarder.start()

        # Block until result
        try:
            while True:
                if rq.poll(0.5):
                    msg = rq.recv()
                    break
                if not proc.is_alive():
                    raise WorkerDiedError("Stem worker died during separation")
        finally:
            self._drain_cancel_pipe()

        tag = msg[0]
        if tag == "ok":
            return Path(msg[1]), Path(msg[2])
        if tag == "cancelled":
            raise WorkerCancelledError(
                "Separation cancelled between chunks (model still loaded)"
            )
        raise RuntimeError(f"Stem worker error: {msg[1]}")

    def kill(self) -> None:
        """SIGKILL the subprocess and discard IPC channels."""
        if self._process is not None and self._process.is_alive():
            try:
                self._process.kill()
            except OSError:
                pass
            try:
                self._process.join(timeout=3)
            except OSError:
                pass
        self._process = None
        self._close_all_connections()

    def stop(self) -> None:
        """Graceful shutdown: send sentinel, join, fall back to kill."""
        if self._process is None or not self._process.is_alive():
            return
        js = self._job_send
        if js is not None:
            try:
                js.send(None)
            except OSError:
                pass
        try:
            self._process.join(timeout=10)
        except OSError:
            pass
        if self._process is not None and self._process.is_alive():
            try:
                self._process.kill()
            except OSError:
                pass
            try:
                self._process.join(timeout=3)
            except OSError:
                pass
        self._close_all_connections()

    def _drain_cancel_pipe(self) -> None:
        """Drain the cancel pipe so it's clean for the next job.

        The worker subprocess already drains cancel_recv in its finally block,
        but we also drain from the main-process side as a safety net. If a
        cancel was sent but the worker finished before reading it, the signal
        would be left in the pipe and could trigger on the next job.
        """
        if self._cancel_recv is not None:
            while self._cancel_recv.poll(0):
                try:
                    self._cancel_recv.recv()
                except (EOFError, OSError):
                    break

    def _close_all_connections(self) -> None:
        for conn in (
            self._job_send,
            self._job_recv,
            self._result_recv,
            self._result_send,
            self._cancel_send,
            self._cancel_recv,
        ):
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
        self._job_send = None
        self._job_recv = None
        self._result_recv = None
        self._result_send = None
        self._cancel_send = None
        self._cancel_recv = None


def _forward_cancel(cancel_event: threading.Event, cancel_send: Connection) -> None:
    """Forward a threading.Event to a multiprocessing Pipe.

    Bridges the main-process threading world to the subprocess Pipe
    world. The thread exits after sending — the cancel signal only
    needs to be sent once per job.
    """
    cancel_event.wait()
    try:
        cancel_send.send(1)
    except (OSError, BrokenPipeError):
        pass


# ============================================================================
# Worker subprocess entry point
# ============================================================================


class _CancelledInsideDemix(Exception):
    """Raised inside the patched forward() when cancel is detected.

    This exception unwinds through:
    forward() → demix() → separate() → _separate_file()
    and is caught in the worker main loop. The model weights survive
    because they're GPU attributes on self.model_run, not stack locals.
    """


def _worker_main(
    job_recv: Connection,
    result_send: Connection,
    cancel_recv: Connection,
    temp_dir: str = "",
    log_level: int = logging.INFO,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """Entry point for the stem worker subprocess.

    Loads the audio-separator model, then loops on job_recv. For each
    job, it monkey-patches model_instance.model_run.forward() to add
    a per-chunk cancellation check that polls cancel_recv.

    Results sent on result_send:
    - ("ok", vocal_path, instrumental_path) on success
    - ("cancelled",) when separation was cancelled between chunks
    - ("error", message) on failure
    """
    worker_log = _setup_worker_logger(log_level)
    worker_log.info("Stem worker process started (PID %d)", os.getpid())

    from audio_separator.separator import Separator

    separator = Separator(
        model_file_dir=model_dir,
        output_format=SEPARATION_FORMAT,
    )
    separator.load_model(model_filename=model_name)
    worker_log.info("Audio separator model loaded — ready for jobs")

    try:
        while True:
            item = job_recv.recv()
            if item is None:
                break

            wav_path_str, output_dir_str = item
            wav_path = Path(wav_path_str)
            output_dir = Path(output_dir_str)
            worker_log.info(f"Separating: {wav_path.name}")

            try:
                vocal_wav, instrumental_wav = _separate_with_cancel_check(
                    wav_path, output_dir, separator, cancel_recv, worker_log
                )
                result_send.send(("ok", str(vocal_wav), str(instrumental_wav)))
            except _CancelledInsideDemix:
                worker_log.info(
                    "Separation cancelled between chunks — model still loaded"
                )
                _clear_gpu_state(separator, worker_log)
                result_send.send(("cancelled",))
            except Exception as e:
                worker_log.error(f"Stem separation failed for {wav_path}: {e}")
                result_send.send(("error", str(e)))
            finally:
                # Drain any remaining cancel signals so the pipe is clean
                while cancel_recv.poll(0):
                    try:
                        cancel_recv.recv()
                    except (EOFError, OSError):
                        break
    finally:
        del separator
        _clear_gpu_cache()
        worker_log.info("Audio separator model unloaded")


def _separate_with_cancel_check(
    audio_path: Path,
    tmp_dir: Path,
    separator,
    cancel_recv: Connection,
    worker_log: logging.Logger,
) -> tuple[Path, Path]:
    """Run separation with a monkey-patched model.forward() that checks for cancellation.

    Injection point: In the Roformer demix() loop, the model is called
    once per chunk:

    with torch.no_grad():
        for i in tqdm(range(0, mix.shape[1], step)):
            part = mix[:, i : i + chunk_size]
            ...
            x = self.model_run(part.unsqueeze(0))[0] # ← HERE
            ...

    nn.Module.__call__ internally calls self.forward(...), so patching
    forward() on the instance gives us a hook that runs right before each
    chunk's model computation.

    We patch model_run.forward() to:
    1. Poll cancel_recv (non-blocking, timeout=0)
    2. If data is available, raise _CancelledInsideDemix
    3. Otherwise, call the original forward() and return its result

    The patch is applied before separator.separate() and removed in a
    finally block, so it's always restored even on cancellation or error.
    """
    separator.output_dir = str(tmp_dir)
    if separator.model_instance:
        separator.model_instance.output_dir = str(tmp_dir)

    model_instance = separator.model_instance
    if model_instance is None:
        raise RuntimeError("No model instance loaded")

    # Get the model_run object (the nn.Module that gets called per chunk)
    model_run = getattr(model_instance, "model_run", None)
    if model_run is None:
        worker_log.warning("No model_run found — running without cancel check")
        return _run_separation_unpatched(audio_path, tmp_dir, separator, worker_log)

    # --- Monkey-patch model_run.forward() ---
    #
    # Why forward() not __call__?
    # Python's method resolution for dunder methods (like __call__)
    # always looks at the CLASS, not the instance. So setting
    # model_run.__call__ = ... is silently ignored by the interpreter.
    # But forward() is a regular method — instance attributes are
    # checked first, so our patch works correctly.
    #
    # nn.Module.__call__ flow:
    # model_run(inputs)
    # → nn.Module.__call__(self, inputs) # class-level, unpatchable
    # → hooks, then self.forward(inputs) # instance-level, patchable ✓

    original_forward = model_run.forward
    chunk_counter = [0]

    def cancelable_forward(*args, **kwargs):
        # Poll cancel_recv (non-blocking)
        if cancel_recv.poll(0):
            try:
                cancel_recv.recv()  # consume the signal
            except (EOFError, OSError):
                pass
            worker_log.info(
                f"Cancel detected before chunk #{chunk_counter[0] + 1} — aborting"
            )
            raise _CancelledInsideDemix()

        # Run the real forward pass
        result = original_forward(*args, **kwargs)
        chunk_counter[0] += 1
        return result

    model_run.forward = cancelable_forward
    worker_log.debug("Patched model_run.forward() with per-chunk cancel check")

    try:
        output_paths = separator.separate(str(audio_path))
    finally:
        # Always restore original forward() — even on cancel/error
        model_run.forward = original_forward
        worker_log.debug(
            f"Restored original model_run.forward() "
            f"(processed {chunk_counter[0]} chunks before exit)"
        )

    # Identify vocal/instrumental stems from output paths.
    # Handles both karaoke-model output ((vocals)/(instrumental)) and
    # non-karaoke MelBand Roformer output ((vocals)/(other)).
    vocals_wav = None
    instrumental_wav = None
    for p in output_paths:
        full_path = Path(tmp_dir) / Path(p).name
        lower = full_path.name.lower()
        no_vocal = "no vocal" in lower or "no_vocal" in lower
        is_other = "(other)" in lower
        if "instrumental" in lower or no_vocal or is_other:
            instrumental_wav = full_path
        elif "vocal" in lower:
            vocals_wav = full_path

    if not vocals_wav or not instrumental_wav:
        raise RuntimeError(
            f"Could not identify vocal/instrumental stems in output: {output_paths}"
        )

    return vocals_wav, instrumental_wav


def _run_separation_unpatched(
    audio_path: Path,
    tmp_dir: Path,
    separator,
    worker_log: logging.Logger,
) -> tuple[Path, Path]:
    """Fallback: run separation without cancel check (no model_run found)."""
    output_paths = separator.separate(str(audio_path))
    vocals_wav = None
    instrumental_wav = None
    for p in output_paths:
        full_path = Path(tmp_dir) / Path(p).name
        lower = full_path.name.lower()
        no_vocal = "no vocal" in lower or "no_vocal" in lower
        is_other = "(other)" in lower
        if "instrumental" in lower or no_vocal or is_other:
            instrumental_wav = full_path
        elif "vocal" in lower:
            vocals_wav = full_path
    if not vocals_wav or not instrumental_wav:
        raise RuntimeError(
            f"Could not identify vocal/instrumental stems: {output_paths}"
        )
    return vocals_wav, instrumental_wav


def _clear_gpu_state(separator, worker_log: logging.Logger) -> None:
    """Clear intermediate GPU state after a cancelled separation."""
    if separator.model_instance is not None:
        try:
            separator.model_instance.clear_gpu_cache()
        except Exception as e:
            worker_log.warning(f"Failed to clear GPU cache after cancel: {e}")
        try:
            separator.model_instance.clear_file_specific_paths()
        except Exception as e:
            worker_log.warning(f"Failed to clear file paths after cancel: {e}")


def _clear_gpu_cache() -> None:
    """Clear GPU cache on worker shutdown."""
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _setup_worker_logger(log_level: int = logging.INFO) -> logging.Logger:
    """Configure the worker subprocess logger."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    processing_logger = logging.getLogger("pipeline.workers.stem_worker")
    processing_logger.handlers = []
    processing_logger.addHandler(handler)
    processing_logger.propagate = False
    processing_logger.setLevel(log_level)

    sep_logger = logging.getLogger("audio_separator")
    sep_logger.handlers = []
    sep_logger.addHandler(handler)
    sep_logger.setLevel(log_level)
    sep_logger.propagate = False

    return processing_logger
