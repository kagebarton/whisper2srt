"""Configuration for genius_align — canonical stable-ts tuning surface.

Each section of ``WhisperModelConfig`` maps 1:1 to one stable-ts call.
The worker uses ``dataclasses.asdict()`` to splat the section's fields
straight into the corresponding method, filtering out ``None`` so unset
options fall back to stable-ts defaults.

Defaults on each dataclass *are* the configured values — there is no
separate "preset" layer. To tune the walk path, edit ``AlignKwargs``
defaults; for the transcribe path, edit ``TranscribeKwargs`` defaults.
``LoadModelKwargs``, ``RefineKwargs``, and ``PostProcessKwargs`` are
shared by both paths.

Sections:
  * ``load_model``    → ``stable_whisper.load_model()``    (shared)
  * ``align``         → ``model.align()``                  (walk path)
  * ``transcribe``    → ``model.transcribe()``             (transcribe path)
  * ``refine``        → ``model.refine()``                 (shared)
  * ``post_process``  → ``WhisperResult`` post-processing  (shared, one method per field)
  * ``regroup``       → ``WhisperResult.regroup()`` expression (transcribe path)
"""

from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = str(_REPO_ROOT / "models" / "large-v3-turbo.pt")


@dataclass
class LoadModelKwargs:
    """Splatted into ``stable_whisper.load_model(model_path, **rest)``.

    ``model_path`` is consumed by the worker as the positional model arg;
    everything else is passed as a kwarg.
    """
    model_path: str = _DEFAULT_MODEL
    device: str = "auto"  # 'auto' → 'cuda' if available else 'cpu'


@dataclass
class AlignKwargs:
    """Splatted into ``model.align(audio, text, **kwargs)`` — walk path config.

    Walk mode uses the two-pointer walk matcher with gap interpolation;
    it's resilient to stable-ts oversplitting, so we maximize anchor
    words (low ``min_word_dur``) and trust the matcher to interpolate.
    """

    language: str = "en"

    # Silero VAD pre-pass: gates whisper to voiced regions only.
    vad: bool = True
    vad_threshold: float = 0.05  # lower = more sensitive

    # Suppress timestamps in silent regions.
    suppress_silence: bool = True
    suppress_word_ts: bool = True

    # Restrict mel features to the human vocal range (~85–3000 Hz).
    only_voice_freq: bool = True  # cleaner mel → better anchors

    # Word duration floor / ceiling. None = stable-ts default.
    min_word_dur: float = 0.1  # more anchor words for walk matcher
    max_word_dur: float | None = 5.0  # trust walk matcher interpolation

    # Drop zero-duration words instead of leaving 0-cs entries.
    remove_instant_words: bool = True

    # Treat each '\\n' in alignment text as a segment boundary.
    original_split: bool = False

    # Abort alignment if zero-duration-word fraction exceeds this.
    failure_threshold: float | None = None

    # Skip non-speech regions (relies on VAD/suppress_silence accuracy).
    nonspeech_skip: float | None = None

    # Max tokens aligned per pass. Higher reduces misalignment risk.
    # None → stable-ts default (100).
    token_step: int | None = 150

@dataclass
class TranscribeKwargs:
    """Splatted into ``model.transcribe(audio, **kwargs)`` — transcribe path config.

    No lyrics are supplied, so the model transcribes from audio alone.
    Slightly more tolerant decoder; segment-level filters tuned to reject
    silent/repetitive hallucinations.
    """

    language: str = "en"

    vad: bool = True
    vad_threshold: float = 0.05

    suppress_silence: bool = True
    suppress_word_ts: bool = True

    only_voice_freq: bool = True

    min_word_dur: float = 0.1

    word_timestamps: bool = True

    # Decoding
    temperature: float = 0.0
    beam_size: int = 5
    patience: float | None = 1.0
    length_penalty: float | None = 1.0

    # Whisper segment-level filters — relaxed for sung vocals.
    # Singing has lower per-token logprobs and high compression ratios
    # (chorus repetition), so speech-tuned defaults reject valid content.
    no_speech_threshold: float | None = 0.3   # was 0.6 — keep more "uncertain" segments
    logprob_threshold: float | None = None    # was -1.0 — disable; trips on singing
    compression_ratio_threshold: float | None = 3.0  # was 2.4 — allow repeat-heavy choruses

    # False avoids hallucination/skip cascades when one segment goes wrong.
    condition_on_previous_text: bool = False

    # Optional text hint to guide style/vocab. Generic on purpose — a
    # full-lyrics prompt would bias the decoder toward the expected text
    # and undermine the tiling path's honest "what was actually sung"
    # account (remixes, ad-libs, dropped verses).
    initial_prompt: str | None = "Song lyrics with casual contractions."


@dataclass
class RefineKwargs:
    """Splatted into ``model.refine(audio, result, **kwargs)`` — shared by both paths."""

    steps: str = "se"  # 's' = starts, 'e' = ends, 'se' = both
    word_level: bool = True


@dataclass
class PostProcessKwargs:
    """``WhisperResult`` post-processing applied after refine() — shared by both paths.

    These are NOT splatted into one method — each field controls a
    separate ``WhisperResult`` call:
      * ``adjust_gaps_threshold`` → ``result.adjust_gaps(duration_threshold=)``
      * ``merge_by_gap_min``      → ``result.merge_by_gap(min_gap=)``
      * ``min_word_probability``  → consumed by ``extract_words()`` in run.py
    """

    # Word probability floor (used by run.py's word extractor; 0 disables).
    min_word_probability: float = 0

    # Merge words closer than this. None disables.
    adjust_gaps_threshold: float | None = None

    # Merge tiny adjacent segments. None disables.
    merge_by_gap_min: float | None = None


# Stable-ts regroup expression for the transcribe path.
# Methods chained with "_"; args follow "=". Shortcuts:
#   cm = clamp_max, sp = split_by_punctuation, sg = split_by_gap,
#   mg = merge_by_gap (min_gap+max_words).
_DEFAULT_REGROUP = "cm_sp=.* /,/?/!/。_sg=.3_mg=.2+5"


@dataclass
class WhisperModelConfig:
    """Top-level whisper config — one section per stable-ts call.

    All defaults are baked into the section dataclasses. Instantiating
    ``WhisperModelConfig()`` produces a fully-tuned config — the walk path
    reads ``align``, the transcribe path reads ``transcribe`` and
    ``regroup``, and both paths share the rest.

    To tune a value, edit the default on the relevant section dataclass.
    To add a new stable-ts kwarg, add a field to the matching section —
    no worker edit required.
    """

    load_model: LoadModelKwargs = field(default_factory=LoadModelKwargs)
    align: AlignKwargs = field(default_factory=AlignKwargs)
    transcribe: TranscribeKwargs = field(default_factory=TranscribeKwargs)
    refine: RefineKwargs = field(default_factory=RefineKwargs)
    post_process: PostProcessKwargs = field(default_factory=PostProcessKwargs)

    # Used by the transcribe path only; ignored by the walk path.
    regroup: str = _DEFAULT_REGROUP


@dataclass
class GeniusAlignConfig:
    """Top-level config: whisper sub-config + flat caption styling."""

    whisper: WhisperModelConfig = field(default_factory=WhisperModelConfig)

    # Auto match-method gate (used only by --match-method=auto). After the
    # walk path runs stable-ts align(), if the fraction of segments stable-ts
    # reported as "failed to align" exceeds this, run.py discards the walk
    # result and re-runs with the tiling matcher on an honest transcription.
    # A high failure rate means align()'s forced word placement is unreliable.
    # 0.1 → escalate once >10% of segments failed (e.g. 7/48 ≈ 0.15 escalates).
    align_failure_escalation: float = 0.1

    # ASS karaoke styling
    font_name: str = "Arial"
    font_size: int = 60
    secondary_color: str = "&H00FFFFFF&"
    outline_color: str = "&H00000000&"
    back_color: str = "&H80000000&"
    outline_width: int = 3
    shadow_offset: int = 2
    margin_left: int = 50
    margin_right: int = 50
    margin_vertical: int = 150

    # Karaoke timing (centiseconds)
    line_lead_in_cs: int = 80
    line_lead_out_cs: int = 20

    ensemble_color: str = "&H0000D7FF&"

    speaker_colors: list = field(
        default_factory=lambda: [
            "&H00A8A800&",
            "&H003232B4&",
            "&H0028D28C&",
            "&H00A03264&",
            "&H006E82A0&",
            "&H006464C8&",
        ]
    )
