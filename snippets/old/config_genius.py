import os

# Whisper model name or path for stable_whisper
# Can be a model name (e.g., "faster-whisper-medium") or local folder path
WHISPER_MODEL = os.path.join(os.path.dirname(__file__), "whisper_model", "faster-whisper-large-v3")

# Device for inference: "cpu", "cuda", or "auto"
DEVICE = "auto"

# Compute type for faster-whisper: "float16", "int8", "int8_float16", or "float32"
# Use "int8" for CPU, "float16" for GPU with sufficient VRAM
COMPUTE_TYPE = "float16"

# Genius API token for lyricsgenius
# Get your token at: https://genius.com/api_clients
GENIUS_API_TOKEN = "OQNe-SALiHKew5tn4fwBEl5mcyiIBTiYS62tjWxhtiFQ2z7nvQcQJdEW05CZcdjB"
