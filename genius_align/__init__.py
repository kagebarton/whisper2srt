"""genius_align — pruned alignment prototype.

A leaner fork of ``genius_diarize`` that:
  * owns its own ``WhisperModelConfig`` (decoupled from ``pipeline.config``)
  * keeps only the ``walk`` matcher with gap interpolation

Use ``python -m genius_align <vocal_audio> <lyrics_file>``.
"""
