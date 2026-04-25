# Diarized Captions

Generate synchronized captions (.srt and .ass) with speaker diarization from vocal stem audio files.

## Features

- **Speaker Diarization**: Automatically detects and labels different speakers in audio
- **Multiple Output Formats**: Generates both .srt (SubRip) and .ass (Advanced SubStation Alpha) caption files
- **Speaker Coloring**: ASS output includes color-coded speakers for easy visual identification
- **Flexible Input**: Works with vocal stem audio files (.wav, .m4a, etc.) and optional lyrics files for alignment
- **Offline Operation**: Uses locally cached models for privacy and efficiency

## Installation

This module is part of the whisper2srt project. Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Transcription Mode

Generate captions without pre-existing lyrics (Whisper will transcribe the audio):

```bash
python -m diarized_captions.run path/to/vocal_stem.wav
```

### Alignment Mode

Provide a lyrics file to improve timing accuracy:

```bash
python -m diarized_captions.run path/to/vocal_stem.wav path/to/lyrics.txt
```

Supported lyrics formats: `.txt` (plain text) or `.srt` (existing subtitles for alignment).

### Speaker Count Options

Control speaker diarization with:

```bash
# Exact number of speakers (disable auto-detection)
python -m diarized_captions.run path/to/vocal_stem.wav --num-speakers 2

# Range constraints
python -m diarized_captions.run path/to/vocal_stem.wav --min-speakers 2 --max-speakers 5

# Default: auto-detect (no constraints)
python -m diarized_captions.run path/to/vocal_stem.wav
```

## Output

Two files are generated alongside the input vocal stem:
- `<vocal_stem>.diarized.srt` - Standard SubRip subtitle file
- `<vocal_stem>.diarized.ass` - Advanced SubStation Alpha file with speaker coloring

The .ass file displays each speaker in a different color (using the first 10 letters of the alphabet: A-J).

## How It Works

1. **Transcription/Alignment**: Uses Whisper to either transcribe the audio directly or align provided lyrics to the audio
2. **Speaker Diarization**: Uses pyannote.audio to identify speaker segments in the audio
3. **Speaker Mapping**: Maps pyannote's speaker labels to simple letters (A, B, C, ...) based on appearance order
4. **Word Assignment**: Assigns each transcribed word to a speaker based on temporal overlap
5. **Line Generation**: Groups words into caption lines, splitting when speaker changes occur
6. **Output Generation**: Creates properly formatted .srt and .ass files with speaker labels and colors

## Configuration

Configuration options are available in `config.py` including:
- Whisper model settings (model path, device, compute type, etc.)
- Diarization settings (speaker count limits, device, **pipeline name**)
- ASS styling (fonts, colors, positioning, timing)
- Speaker color palette and letter mapping

### Diarization Pipeline

By default, uses `pyannote/speaker-diarization-community-1`. To switch pipelines, modify `DiarizeConfig.pipeline_name` in `config.py` (e.g., `pyannote/speaker-diarization-3.1`).

### HuggingFace Token

Community-1 is a gated model. To use it:
1. Accept terms at https://hf.co/pyannote/speaker-diarization-community-1
2. Provide your token via:
   - Environment variable: `export HF_TOKEN=hf_...`
   - Config file: set `hf_token_path` in `DiarizeConfig`
   - Or pre-cache the model by running `download_models.py` with `HF_TOKEN` set

After initial download, the model is cached locally and subsequent runs work offline.

## Requirements

See `requirements.txt` for specific dependency versions.

Key dependencies:
- faster-whisper
- pyannote.audio
- torch
- huggingface_hub

## Model Caching

Models are automatically cached in the project's `models/` directory. Set the `HF_HOME` environment variable to change this location.

## License

This module is part of the whisper2srt project. See the main project README for license information.