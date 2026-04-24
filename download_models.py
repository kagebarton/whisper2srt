import os
import urllib.request

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR   = "./models"
HF_TOKEN    = ""
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)


# 1. Whisper large-v3-turbo (.pt)
print("Downloading Whisper large-v3-turbo...")
from whisper import _MODELS, _download
_download(_MODELS["large-v3-turbo"], MODEL_DIR, in_memory=False)
print("  Done:", os.path.join(MODEL_DIR, "large-v3-turbo.pt"))


# 2. Audio Separator - vocals_mel_band_roformer.ckpt
print("Downloading vocals_mel_band_roformer.ckpt...")
from audio_separator.separator import Separator
Separator(model_file_dir=MODEL_DIR).load_model("vocals_mel_band_roformer.ckpt")
print("  Done:", os.path.join(MODEL_DIR, "vocals_mel_band_roformer.ckpt"))


# 3. Pyannote speaker diarization (all required sub-models)
print("Downloading Pyannote models...")
from pyannote.audio import Pipeline
Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)

print("\nWhisper and Audio-Separator models downloaded to:", MODEL_DIR)
print("\nPyannote models are in .cache/huggingface/hub. Copy the 'hub' folder to the 'models' folder")