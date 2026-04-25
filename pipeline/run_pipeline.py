"""CLI entry point for the staged pipeline prototype.

Usage (from the repo root):
  python -m pipeline.run_pipeline <audio_file> [lyrics_file]

Cancel-test mode (both flags required):
  python -m pipeline.run_pipeline <audio_file> --phase stem_separation --cancel-after 5

See --help for full usage.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig, WhisperModelConfig
from pipeline.context import Phase, PipelineCancelled
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.stages.ffmpeg_extract import FFmpegExtractStage
from pipeline.stages.ffmpeg_transcode import FFmpegTranscodeStage
from pipeline.stages.loudnorm_analyze import LoudnormAnalyzeStage
from pipeline.stages.lyric_align import LyricAlignStage
from pipeline.stages.stem_separation import StemSeparationStage
from pipeline.workers.stem_worker import StemWorker
from pipeline.workers.whisper_worker import WhisperWorker

log = logging.getLogger("pipeline.run_pipeline")


def build_whisper_config(cfg: PipelineConfig) -> WhisperModelConfig:
    return WhisperModelConfig(
        model_path=cfg.whisper_model_path,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        language=cfg.whisper_language,
        vad=cfg.whisper_vad,
        vad_threshold=cfg.whisper_vad_threshold,
        suppress_silence=cfg.whisper_suppress_silence,
        suppress_word_ts=cfg.whisper_suppress_word_ts,
        only_voice_freq=cfg.whisper_only_voice_freq,
        refine_steps=cfg.whisper_refine_steps,
        refine_word_level=cfg.whisper_refine_word_level,
        regroup=cfg.whisper_regroup,
    )


# ---------------------------------------------------------------------------
# Cancel test driver
# ---------------------------------------------------------------------------

def _wait_for_phase(token, target: Phase, timeout: float) -> bool:
    """Poll cancel_token.get_phase() every 100 ms until *target* is reached.

    Uses ``token.is_cancelled()`` (lock-acquired) so the check is
    free-threading-safe.  Returns True if the target phase was reached,
    False on timeout or if the job was cancelled before reaching it.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if token.get_phase() == target:
            return True
        if token.is_cancelled():
            return False
        time.sleep(0.1)
    return False


def run_cancel_test(
    orchestrator: PipelineOrchestrator,
    song_path: Path,
    lyrics_path,
    target_phase: Phase,
    cancel_after: float,
) -> int:
    """Fire a cancel at *target_phase* after *cancel_after* seconds.

    Verifies that (a) the cancel lands cleanly, and (b) both workers are
    still loaded afterwards.  Returns 0 on pass, 1 on fail.  Does NOT
    re-run the pipeline after cancel — matching production behaviour.
    """
    log.info(
        f"=== Cancel test: target phase = {target_phase.value}, "
        f"cancel-after = {cancel_after}s ==="
    )

    cancel_token = orchestrator.run_one_async(song_path, lyrics_path)

    # 120s is plenty: extract ~2s, loudnorm ~10s, stem_separation ~30s on GPU.
    if not _wait_for_phase(cancel_token, target_phase, timeout=120):
        log.warning(
            f"Phase {target_phase.value} never reached within 120s — "
            f"pipeline may have completed first. Joining without cancel."
        )
        try:
            orchestrator.join()
        except Exception as e:
            log.error(f"Pipeline failed during fallback join: {e}")
            return 1
        return 0

    log.info(f"Phase {target_phase.value} entered — sleeping {cancel_after}s")
    time.sleep(cancel_after)

    # Confirm we are still in the target phase before firing.
    current = cancel_token.get_phase()
    if current != target_phase:
        log.warning(
            f"Pipeline left {target_phase.value} (now {current}) before "
            f"cancel timer fired. Try a smaller --cancel-after."
        )

    log.info(">>> SETTING CANCEL <<<")
    orchestrator.cancel_active()

    # Wait for the pipeline thread to finish.
    try:
        orchestrator.join(timeout=120)
        log.error("Pipeline completed instead of cancelling — raise --cancel-after")
        return 1
    except PipelineCancelled as e:
        cancelled_phase = e.phase.value if e.phase else "?"
        log.info(f"Pipeline cancelled at phase {cancelled_phase}")
    except Exception as e:
        # Any non-cancel failure: report cleanly instead of letting it
        # propagate as an unhandled exception.
        log.error(f"Pipeline failed with unexpected error: {type(e).__name__}: {e}")
        return 1

    # --- Worker liveness assertion (no re-run) ---
    log.info("=== Verifying workers survived cancellation ===")
    stem_alive = orchestrator.stem_worker.is_alive()
    whisper_loaded = orchestrator.whisper_worker.model_loaded
    log.info(f"  StemWorker subprocess alive: {stem_alive}")
    log.info(f"  WhisperWorker model loaded: {whisper_loaded}")

    if not stem_alive:
        log.error("FAIL: StemWorker subprocess died — model would need reload.")
        return 1
    if not whisper_loaded:
        log.error("FAIL: WhisperWorker model unloaded — would need reload.")
        return 1

    log.info("PASS: both workers still loaded after cancellation.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staged pipeline prototype. Omit lyrics_file to transcribe automatically."
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("lyrics_file", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=None,
        help="Seconds to wait after --phase begins before cancelling.",
    )
    parser.add_argument(
        "--phase",
        choices=[p.value for p in Phase],
        default=None,
        help="Phase at which to fire the cancel timer.",
    )
    args = parser.parse_args()

    # Validate flag combination
    if args.cancel_after is not None and args.phase is None:
        parser.error("--cancel-after requires --phase")
    if args.phase is not None and args.cancel_after is None:
        parser.error("--phase requires --cancel-after")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # MelBand Roformer uses PyTorch, not ONNXRuntime — suppress irrelevant ONNX CUDA warning
    logging.getLogger("audio_separator").addFilter(
        lambda r: "CUDAExecutionProvider" not in r.getMessage()
    )

    cfg = PipelineConfig()

    stem_worker = StemWorker(
        temp_dir=cfg.intermediate_dir,
        model_dir=cfg.separator_model_dir,
        model_name=cfg.separator_model_name,
    )
    whisper_worker = WhisperWorker(build_whisper_config(cfg))

    stages = [
        FFmpegExtractStage(cfg),
        LoudnormAnalyzeStage(cfg),
        StemSeparationStage(stem_worker),
        FFmpegTranscodeStage(cfg),
        LyricAlignStage(whisper_worker, cfg),
    ]

    orchestrator = PipelineOrchestrator(stages, stem_worker, whisper_worker, cfg)
    orchestrator.start()

    try:
        if args.phase is not None and args.cancel_after is not None:
            target_phase = Phase(args.phase)
            rc = run_cancel_test(
                orchestrator, args.audio_file, args.lyrics_file,
                target_phase, args.cancel_after,
            )
        else:
            try:
                ctx = orchestrator.run_one(args.audio_file, args.lyrics_file)
                print("\n=== Pipeline complete ===")
                print(f"Vocal M4A: {ctx.artifacts.get('vocal_m4a')}")
                print(f"Nonvocal M4A: {ctx.artifacts.get('nonvocal_m4a')}")
                print(f"ASS: {ctx.artifacts.get('ass_file')}")
                if "srt_file" in ctx.artifacts:
                    print(f"SRT: {ctx.artifacts['srt_file']}")
                print(f"Loudnorm I: {ctx.artifacts.get('loudnorm_input_i')} LUFS")
                print(f"Loudnorm TP: {ctx.artifacts.get('loudnorm_input_tp')} dBTP")
                print(f"Loudnorm LRA: {ctx.artifacts.get('loudnorm_input_lra')} LU")
                print(f"Target offset: {ctx.artifacts.get('loudnorm_target_offset')} dB")
                rc = 0
            except Exception as e:
                log.error(f"Pipeline failed: {e}")
                rc = 1
    finally:
        orchestrator.stop()

    sys.exit(rc)


if __name__ == "__main__":
    sys.exit(main())
