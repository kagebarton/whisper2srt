"""CLI entry point for the staged pipeline prototype.

Usage:
python -m pipeline.run_pipeline <audio_file> <lyrics_file>

Takes only positional args — no CLI flags. To change values, edit
pipeline/config.py or construct a PipelineConfig with overrides and
pass it to the orchestrator programmatically.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import PipelineConfig, WhisperModelConfig
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.stages.ffmpeg_extract import FFmpegExtractStage
from pipeline.stages.ffmpeg_transcode import FFmpegTranscodeStage
from pipeline.stages.loudnorm_analyze import LoudnormAnalyzeStage
from pipeline.stages.lyric_align import LyricAlignStage
from pipeline.stages.stem_separation import StemSeparationStage
from pipeline.workers.stem_worker import StemWorker
from pipeline.workers.whisper_worker import WhisperWorker


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staged pipeline prototype. Omit lyrics_file to transcribe automatically."
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("lyrics_file", type=Path, nargs="?", default=None)
    args = parser.parse_args()

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

    try:
        ctx = orchestrator.run(args.audio_file, args.lyrics_file)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return 1

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
