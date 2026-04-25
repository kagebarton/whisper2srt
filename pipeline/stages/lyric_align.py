"""Lyric alignment/transcription stage: generate ASS (and SRT) from vocal stem.

Two modes:
- Alignment (lyrics_path provided): aligns the given .txt or .srt lyrics to
  the vocal stem via stable-ts model.align(), then refines timestamps.
- Transcription (no lyrics_path): runs model.transcribe() directly; stable-ts
  determines segment/word boundaries from the audio alone.

In both modes the same ASS and SRT generators are used. The difference is
how line objects are built: alignment pairs words to predefined lyric lines
by count; transcription uses stable-ts segments directly as lines.

Each model call (align, transcribe, refine) is wrapped in its own
cancellation activity scope (Phase.ALIGN / Phase.TRANSCRIBE / Phase.REFINE).
ASS/SRT are written to ctx.tmp_dir first and moved to the final output
directory only after both writes succeed — preventing orphan files on
cancellation.

ASS generation ported from snippets/stable_align.py, but reads all
styling/timing parameters from PipelineConfig instead of module-level
constants.
"""

import datetime
import logging
import shutil
from pathlib import Path

import srt

from pipeline.config import PipelineConfig
from pipeline.context import Phase, PipelineCancelled, SetEvent, StageContext
from pipeline.stages.base import BaseStage
from pipeline.workers.whisper_worker import AlignmentCancelledError, WhisperWorker

logger = logging.getLogger(__name__)


class LyricAlignStage(BaseStage):
    """Align lyrics to vocal stem and generate karaoke ASS + optional SRT."""

    name = "lyric_align"

    def __init__(self, whisper_worker: WhisperWorker, config: PipelineConfig) -> None:
        self._worker = whisper_worker
        self._config = config

    def run(self, ctx: StageContext) -> None:
        lyrics_path = ctx.artifacts.get("lyrics_path")
        vocal_wav = ctx.artifacts.get("vocal_wav")

        if vocal_wav is None:
            raise RuntimeError(f"[{self.name}] No vocal_wav in artifacts")

        if lyrics_path is not None:
            # --- Alignment mode: ALIGN → REFINE ---
            lyrics_text, lyrics_format = self._load_lyrics(lyrics_path)
            ctx.artifacts["lyrics_text"] = lyrics_text
            ctx.artifacts["lyrics_format"] = lyrics_format

            logger.info(f"[{self.name}] Aligning lyrics to vocal stem: {Path(vocal_wav).name}")
            result = _model_call(
                ctx, Phase.ALIGN,
                lambda: self._worker.align(
                    vocal_path=vocal_wav,
                    lyrics_text=lyrics_text,
                    cancel_event=ctx.cancel.event if ctx.cancel else None,
                ),
            )

            words = self._extract_words(result)
            lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
            line_objects = self._match_words_to_lines(words, lines)

            # SRT only when input was .txt (no timestamps to preserve from .srt input)
            write_srt = lyrics_format == "txt"

            result = _model_call(
                ctx, Phase.REFINE,
                lambda: self._worker.refine(
                    vocal_path=vocal_wav,
                    result=result,
                    cancel_event=ctx.cancel.event if ctx.cancel else None,
                ),
            )
        else:
            # --- Transcription mode: TRANSCRIBE → (regroup) → REFINE ---
            logger.info(f"[{self.name}] Transcribing vocal stem: {Path(vocal_wav).name}")
            result = _model_call(
                ctx, Phase.TRANSCRIBE,
                lambda: self._worker.transcribe(
                    vocal_path=vocal_wav,
                    cancel_event=ctx.cancel.event if ctx.cancel else None,
                ),
            )

            # regroup is fast (CPU-bound, milliseconds) and runs OUTSIDE any
            # activity scope. It's not cancellable. A cancel arriving during
            # regroup is caught by REFINE's activity() on entry via the sticky
            # `cancelled` flag.
            if self._config.whisper_regroup:
                result.regroup(self._config.whisper_regroup)

            line_objects = self._segments_to_line_objects(result)
            write_srt = True

            result = _model_call(
                ctx, Phase.REFINE,
                lambda: self._worker.refine(
                    vocal_path=vocal_wav,
                    result=result,
                    cancel_event=ctx.cancel.event if ctx.cancel else None,
                ),
            )

        # Generate ASS and SRT strings (CPU-bound, not cancellable)
        # -- but we still need to re-extract line_objects from the *refined*
        # result.  The refine step may have adjusted word timestamps, so
        # rebuild line_objects from the refined result.
        if lyrics_path is not None:
            words = self._extract_words(result)
            lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
            line_objects = self._match_words_to_lines(words, lines)
        else:
            line_objects = self._segments_to_line_objects(result)

        ass_content = self._generate_ass(line_objects)
        srt_content = self._generate_srt(line_objects) if write_srt else None

        # Write to tmp_dir first, then move to final destinations — prevents
        # orphan files on cancellation (tmp_dir is cleaned by orchestrator).
        tmp_ass = ctx.tmp_dir / f"{ctx.song_path.stem}.ass"
        tmp_ass.write_text(ass_content, encoding="utf-8")

        tmp_srt = None
        if write_srt:
            tmp_srt = ctx.tmp_dir / f"{ctx.song_path.stem}.srt"
            tmp_srt.write_text(srt_content, encoding="utf-8")

        # Promote to final locations
        karaoke_dir = ctx.song_path.parent / "karaoke"
        karaoke_dir.mkdir(exist_ok=True)
        final_ass = karaoke_dir / tmp_ass.name
        shutil.move(str(tmp_ass), str(final_ass))
        ctx.artifacts["ass_file"] = final_ass
        logger.info(f"[{self.name}] ASS written: {final_ass}")

        if write_srt and tmp_srt is not None:
            subtitles_dir = ctx.song_path.parent / "subtitles"
            subtitles_dir.mkdir(exist_ok=True)
            final_srt = subtitles_dir / tmp_srt.name
            shutil.move(str(tmp_srt), str(final_srt))
            ctx.artifacts["srt_file"] = final_srt
            logger.info(f"[{self.name}] SRT written: {final_srt}")

    # --- Helpers ---

    def _load_lyrics(self, lyrics_path: Path) -> tuple[str, str]:
        """Return (lyrics_text, lyrics_format) where format is 'txt' or 'srt'.

        If lyrics_path is .srt: parse with srt library, concatenate text,
        discard original timestamps. If .txt: read raw text.
        """
        suffix = Path(lyrics_path).suffix.lower()
        if suffix == ".srt":
            raw = lyrics_path.read_text(encoding="utf-8")
            subs = list(srt.parse(raw))
            lyrics_text = "\n".join(sub.content for sub in subs)
            return lyrics_text, "srt"
        else:
            lyrics_text = lyrics_path.read_text(encoding="utf-8")
            return lyrics_text, "txt"

    def _extract_words(self, result) -> list[dict]:
        """Flatten WhisperResult into [{word, start, end, is_segment_first}, ...]."""
        all_words = []
        for segment in result.segments:
            for i, word in enumerate(segment.words):
                all_words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "is_segment_first": i == 0,
                })
        return all_words

    def _match_words_to_lines(self, words: list[dict], lines: list[str]) -> list[dict]:
        """Assign aligned words to lyrics lines by count.

        Count-based pairing: assumes the lyrics file has the same word
        count and order as what stable-ts aligned.
        """
        line_objects = []
        word_index = 0

        for line in lines:
            line_word_count = len(line.split())
            line_words = words[word_index:word_index + line_word_count]
            word_index += line_word_count

            if not line_words:
                continue

            line_obj = {
                "text": line,
                "words": line_words,
                "start": line_words[0]["start"],
                "end": line_words[-1]["end"],
            }
            line_objects.append(line_obj)

        return line_objects

    def _segments_to_line_objects(self, result) -> list[dict]:
        """Build line objects directly from stable-ts segments (transcription mode).

        Each segment becomes one subtitle line; its words are used for karaoke
        timing. Segments with no words are skipped.
        """
        line_objects = []
        for segment in result.segments:
            if not segment.words:
                continue
            words = [
                {
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "is_segment_first": i == 0,
                }
                for i, w in enumerate(segment.words)
            ]
            line_objects.append({
                "text": segment.text.strip(),
                "words": words,
                "start": words[0]["start"],
                "end": words[-1]["end"],
            })
        return line_objects

    def _generate_ass(self, line_objects: list[dict]) -> str:
        """Build .ass content from line objects using config styling/timing.

        Ported from snippets/stable_align.py:generate_enhanced_karaoke_ass.
        All styling/timing values come from self._config instead of
        module-level constants.
        """
        cfg = self._config
        header = (
            f"[Script Info]\n"
            f"Title: Karaoke Subtitles\n"
            f"ScriptType: v4.00+\n"
            f"PlayResX: 1920\n"
            f"PlayResY: 1080\n"
            f"Timer: 100.0000\n"
            f"\n"
            f"[V4+ Styles]\n"
            f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            f"OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            f"ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            f"Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Karaoke,{cfg.font_name},{cfg.font_size},"
            f"{cfg.primary_color},{cfg.secondary_color},"
            f"{cfg.outline_color},{cfg.back_color},"
            f"0,0,0,0,100,100,0,0,1,"
            f"{cfg.outline_width},{cfg.shadow_offset},2,"
            f"{cfg.margin_left},{cfg.margin_right},{cfg.margin_vertical},1\n"
            f"\n"
            f"[Events]\n"
            f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        events = []
        for line_obj in line_objects:
            words = line_obj["words"]
            if not words:
                continue

            # Pad the event window around the sung word boundaries
            event_start = max(0.0, words[0]["start"] - cfg.line_lead_in_cs / 100.0)
            event_end = words[-1]["end"] + cfg.line_lead_out_cs / 100.0

            # Karaoke cursor starts at the event start time.
            # Tracking prev_end from here means the gap to the first word
            # automatically becomes the silent lead-in tag.
            prev_end = event_start
            parts = []

            for i, word_data in enumerate(words):
                word = word_data["word"]

                word_start = word_data["start"]
                # Apply first_word_nudge_cs: push back the first word of a
                # segment if it starts almost immediately after the previous
                # segment (within ~50ms of the expected lead-in gap),
                # preventing word clipping.
                if (
                    word_data.get("is_segment_first")
                    and abs(word_start - prev_end - cfg.line_lead_in_cs / 100.0) < 0.05
                ):
                    word_start += cfg.first_word_nudge_cs / 100.0

                word_end = word_data["end"]
                word_dur_cs = max(10, round((word_end - word_start) * 100))

                # Silent cursor advance through any gap before this word.
                # For the first word this gap = LINE_LEAD_IN_CS (the lead-in).
                # For subsequent words it covers natural pauses between words.
                gap_cs = max(0, round((word_start - prev_end) * 100))
                if gap_cs > 0:
                    parts.append(f"{{\\k{gap_cs}}}")

                # \kf = left-to-right fill sweep over word_dur_cs centiseconds
                parts.append(f"{{\\kf{word_dur_cs}}}{word}")
                prev_end = word_end

                if i < len(words) - 1:
                    parts.append(" ")

            karaoke_text = "".join(parts)
            events.append(
                f"Dialogue: 0,{_seconds_to_ass_time(event_start)},"
                f"{_seconds_to_ass_time(event_end)},Karaoke,,0,0,0,,{karaoke_text}"
            )

        return header + "\n".join(events) + "\n"

    def _generate_srt(self, line_objects: list[dict]) -> str:
        """Build .srt from line objects so SRT inherits the same segmentation as ASS.

        In alignment mode this preserves the lyric file's curated line breaks;
        in transcription mode line_objects mirror stable-ts segments.
        """
        subtitles = [
            srt.Subtitle(
                index=i,
                start=datetime.timedelta(seconds=line_obj["start"]),
                end=datetime.timedelta(seconds=line_obj["end"]),
                content=line_obj["text"],
            )
            for i, line_obj in enumerate(line_objects, start=1)
        ]
        return srt.compose(subtitles)


def _model_call(ctx, phase: Phase, fn):
    """Wrap a single model call in an activity scope (or run it bare
    when ctx.cancel is None).  Translates AlignmentCancelledError to
    PipelineCancelled so the orchestrator only needs to catch one
    exception type.
    """
    if ctx.cancel is None:
        return fn()
    cancel_event = ctx.cancel.event
    try:
        with ctx.cancel.activity(phase, SetEvent(cancel_event)):
            return fn()
    except AlignmentCancelledError:
        raise PipelineCancelled(phase)


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
