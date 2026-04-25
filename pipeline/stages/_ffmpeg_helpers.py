"""FFmpeg helper: spawn ffmpeg under a cancellation activity scope.

Wraps a Popen in a KillProcess Cancellable so the orchestrator can
SIGKILL it via CancelToken.cancel().  The activity() contextmanager
ensures phase + cancellable are registered atomically and always cleared.
"""

import subprocess

from pipeline.context import KillProcess, Phase, PipelineCancelled, StageContext


def run_ffmpeg(
    cmd: list[str],
    ctx: StageContext,
    phase: Phase,
    *,
    capture_stderr: bool = False,
) -> str:
    """Spawn ffmpeg under an activity scope, wait, raise on cancel/error.

    Returns captured stderr (as a string) when *capture_stderr* is True,
    otherwise returns an empty string.

    Raises:
        PipelineCancelled: if the job was cancelled (the Popen was SIGKILL'd
            by the cancel mechanism, or a cancel arrived before/during the
            activity).
        RuntimeError: if ffmpeg exits with a non-zero code and the job was
            NOT cancelled.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )

    stderr_data: bytes | None = None
    if ctx.cancel is not None:
        with ctx.cancel.activity(phase, KillProcess(proc)):
            if capture_stderr:
                _, stderr_data = proc.communicate()
            else:
                proc.wait()
        # activity() re-raises PipelineCancelled on exit if cancelled
    else:
        # No cancel support — straight wait, no activity bookkeeping.
        if capture_stderr:
            _, stderr_data = proc.communicate()
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

    if capture_stderr and stderr_data is not None:
        return stderr_data.decode("utf-8", errors="replace")
    return ""
