"""
================================================================================
 separate.py — Vocal / Non-Vocal stem splitter for karaoke use
================================================================================

 USAGE:
   python separate.py <input_video>

   Example:
     python separate.py "my_song.mp4"

 OUTPUT:
   Two .m4a files written next to the input file:
     my_song_(Vocals).m4a
     my_song_(Instrumental).m4a

 REQUIREMENTS:
   pip install audio-separator[gpu]   # or [cpu] if no CUDA GPU
   ffmpeg must be on PATH

 NOTES:
   - First run will download the model (~400 MB) into MODEL_DIR.
   - The instrumental stem is computed as (original − vocal mask), so both
     tracks played concurrently will closely reconstitute the original song.
   - The vocal stem is aggressively de-bled, making it suitable as input to
     WhisperX / ctc-forced-aligner as well as a low-volume karaoke guide track.
================================================================================
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# ==============================================================================
#  CONFIG
# ==============================================================================

# MelBand Roformer Karaoke — best single-model vocal clarity + complementary
# 2-stem output. Tops SDR benchmarks for karaoke use cases.
MODEL_NAME = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"

# Directory where audio-separator stores downloaded models.
# Change this if you want models on a different drive.
MODEL_DIR = os.path.expanduser("./audio-separator/models")

# Directory for intermediate temporary files during separation.
# Set to "" to use system temp (/tmp). Set to a path to keep files in a
# specific location (e.g., "./temp_audio" or "/mnt/scratch/karaoke_temp").
TEMP_DIR = "./temp_audio"

# Whether to delete temporary files after processing completes.
# Set to False to keep intermediate WAV files for debugging or archival.
CLEANUP_TEMP = True

# Intermediate separation format. WAV is lossless and avoids any decode/encode
# artefacts before the final AAC transcode. Don't change this.
SEPARATION_FORMAT = "wav"

# M4A encoding quality. "2" ≈ 128 kbps VBR AAC — transparent for karaoke.
# Lower number = higher quality (1 = ~180 kbps, 5 = ~48 kbps).
AAC_QUALITY = "2"

# Number of threads ffmpeg may use for the AAC transcode step.
FFMPEG_THREADS = "4"

# ==============================================================================


def extract_audio(video_path: Path, tmp_dir: str) -> Path:
    """Pull the audio stream from the video file into a temporary WAV.

    We ask ffmpeg for a 44.1 kHz stereo PCM WAV — the format audio-separator
    expects. If the source is already stereo 44.1 kHz this is a fast remux.
    """
    wav_path = Path(tmp_dir) / (video_path.stem + "_input.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                        # drop video stream
        "-ac", "2",                   # stereo
        "-ar", "44100",               # 44.1 kHz
        "-sample_fmt", "s16",         # 16-bit PCM
        str(wav_path),
    ]

    print(f"[1/3] Extracting audio from '{video_path.name}' ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:\n", result.stderr)
        sys.exit(1)

    return wav_path


def separate_stems(audio_path: Path, tmp_dir: str) -> tuple[Path, Path]:
    """Run audio-separator and return (vocals_wav, instrumental_wav) paths."""

    # Import here so the rest of the script can give a clean error if the
    # package is missing before we even touch ffmpeg.
    from audio_separator.separator import Separator

    print(f"[2/3] Separating stems with {MODEL_NAME} ...")
    print("      (model will be downloaded on first run — ~400 MB)")

    separator = Separator(
        output_dir=tmp_dir,
        model_file_dir=MODEL_DIR,
        output_format=SEPARATION_FORMAT,

        # Overlap between processing chunks — higher reduces boundary
        # artefacts at the cost of speed. 0.25 is a good balance.
        #overlap_InstVoc=0.25,
        #overlap_VitLarge=0.25,

        # Use half-precision on GPU to save VRAM (safe on RTX 2060 6 GB).
        #use_half_precision=True,

        # Normalise loudness of output stems to prevent clipping.
        #normalization_threshold=0.9,
    )

    separator.load_model(model_filename=MODEL_NAME)

    # Returns a list of output file paths in the order [Instrumental, Vocals]
    # (order may vary by model — we identify them by filename suffix below).
    output_paths = separator.separate(str(audio_path))

    # Resolve vocal and instrumental paths from whatever filenames the library
    # produced, rather than assuming a fixed order.
    vocals_wav = None
    instrumental_wav = None
    for p in output_paths:
        # audio-separator returns filenames without full path — prepend tmp_dir.
        full_path = Path(tmp_dir) / Path(p).name
        lower = full_path.name.lower()
        if "vocal" in lower:
            vocals_wav = full_path
        elif "instrumental" in lower or "no_vocals" in lower:
            instrumental_wav = full_path

    if not vocals_wav or not instrumental_wav:
        print("ERROR: Could not identify vocal/instrumental files in output:")
        for p in output_paths:
            print(" ", p)
        sys.exit(1)

    return vocals_wav, instrumental_wav


def wav_to_m4a(wav_path: Path, video_path: Path, stem_label: str) -> Path:
    """Transcode a WAV stem to AAC-in-M4A using ffmpeg.

    stem_label is appended to the base filename, e.g. '—vocal' or '—nonvocal'.
    """
    # Output file lands next to the input video with naming:
    # "Soda Pop—vocal.m4a" or "Soda Pop—nonvocal.m4a"
    m4a_name = video_path.parent / f"{video_path.stem}{stem_label}.m4a"

    cmd = [
        "ffmpeg", "-y",
        "-threads", FFMPEG_THREADS,
        "-i", str(wav_path),
        "-c:a", "aac",                # native ffmpeg AAC encoder
        "-q:a", AAC_QUALITY,          # VBR quality level
        str(m4a_name),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg transcode error for {stem_label}:\n", result.stderr)
        sys.exit(1)

    return m4a_name


def main():
    if len(sys.argv) < 2:
        print("Usage: python separate.py <input_video>")
        sys.exit(1)

    video_path = Path(sys.argv[1]).resolve()
    if not video_path.exists():
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    # Set up temp directory based on config.
    tmp_dir_context = None
    if TEMP_DIR:
        tmp_dir = Path(TEMP_DIR)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = str(tmp_dir)
    else:
        tmp_dir_context = tempfile.TemporaryDirectory(prefix="karaoke_sep_")
        tmp_dir = tmp_dir_context.name

    try:
        # Step 1 — extract audio from video to WAV
        audio_wav = extract_audio(video_path, tmp_dir)

        # Step 2 — separate into vocal + instrumental WAV stems
        vocals_wav, instrumental_wav = separate_stems(audio_wav, tmp_dir)

        # Step 3 — transcode both stems to M4A
        print("[3/3] Transcoding stems to M4A ...")

        vocals_m4a = wav_to_m4a(vocals_wav, video_path, "---vocal")
        instrumental_m4a = wav_to_m4a(instrumental_wav, video_path, "---nonvocal")

    finally:
        # Clean up temp files if configured to do so.
        if CLEANUP_TEMP:
            if tmp_dir_context:
                tmp_dir_context.cleanup()
            else:
                import shutil
                shutil.rmtree(tmp_dir)

    print("\nDone!")
    print(f"  Vocals:       {vocals_m4a}")
    print(f"  Instrumental: {instrumental_m4a}")


if __name__ == "__main__":
    main()
