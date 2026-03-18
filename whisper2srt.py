import os
import re
import config
import stable_whisper

GAP_THRESHOLD = 0.4  # seconds; tune this if needed


def load_model():
	return stable_whisper.load_faster_whisper(
		config.WHISPER_MODELS_DIR,
		device=config.DEVICE,
		compute_type=config.COMPUTE_TYPE
	)


def transcribe_audio(model, audio_file_path: str):
	"""
	Transcribes audio using faster-whisper via stable-ts.
	Used when no lyrics file is found.
	"""
	print("No lyrics file found — running in transcription mode.")
	return model.transcribe(
		audio_file_path,
		word_timestamps=True,
		language="en",
		beam_size=10,
		best_of=10,
		temperature=0.0,
		condition_on_previous_text=True,
		initial_prompt="Lyrics:"
	)


def align_audio(model, audio_file_path: str, lyrics_file_path: str):
	"""
	Aligns audio to verified lyrics using stable-ts.
	Used when a .txt file with the same filename prefix is found.
	"""
	print(f"Lyrics file found — running in alignment mode using {lyrics_file_path}")
	with open(lyrics_file_path, "r", encoding="utf-8") as f:
		lyrics = f.read()
	return model.align(audio_file_path, lyrics, language="en")


def seconds_to_srt(seconds: float) -> str:
	"""Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
	h = int(seconds // 3600)
	m = int((seconds % 3600) // 60)
	s = int(seconds % 60)
	ms = int((seconds % 1) * 1000)
	return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt_from_transcription(result, output_path: str):
	"""
	Saves SRT from a transcription result.
	Splits lines on capital letters using word-level timestamps.
	"I" is treated specially since it is always capitalized in English —
	it only triggers a new line if preceded by a pause or punctuation.
	"""
	blocks = []

	for segment in result.segments:
		words = segment.words

		current_line = []
		for word in words:
			text = word.word.strip()
			is_capital = bool(re.match(r'[A-Z]', text))
			is_new_line = False

			if current_line and is_capital:
				if text == "I":
					prev_word = current_line[-1]
					gap = word.start - prev_word.end
					ends_with_punctuation = bool(re.search(r'[.,!?]$', prev_word.word.strip()))
					is_new_line = gap > GAP_THRESHOLD or ends_with_punctuation
				else:
					is_new_line = True

			if is_new_line:
				blocks.append({
					"start": seconds_to_srt(current_line[0].start),
					"end": seconds_to_srt(current_line[-1].end),
					"text": " ".join(w.word.strip() for w in current_line)
				})
				current_line = []

			current_line.append(word)

		# Don't forget the last line in the segment
		if current_line:
			blocks.append({
				"start": seconds_to_srt(current_line[0].start),
				"end": seconds_to_srt(current_line[-1].end),
				"text": " ".join(w.word.strip() for w in current_line)
			})

	with open(output_path, "w", encoding="utf-8") as f:
		for i, block in enumerate(blocks, start=1):
			f.write(f"{i}\n{block['start']} --> {block['end']}\n{block['text']}\n\n")


def save_srt_from_alignment(result, lyrics_file_path: str, output_path: str):
	"""
	Saves SRT from an alignment result, using the lyrics file line breaks
	to determine subtitle blocks. Words from the alignment are matched
	back to the original lyrics lines in order.
	"""
	# Flatten all words from all segments into a single list
	all_words = []
	for segment in result.segments:
		for word in segment.words:
			all_words.append(word)

	# Read lyrics lines, skipping empty ones
	with open(lyrics_file_path, "r", encoding="utf-8") as f:
		lines = [l.strip() for l in f.readlines() if l.strip()]

	blocks = []
	word_index = 0

	for line in lines:
		# Count how many words are in this lyric line
		line_word_count = len(line.split())

		# Grab that many words from the aligned word list
		line_words = all_words[word_index: word_index + line_word_count]
		word_index += line_word_count

		if not line_words:
			continue

		blocks.append({
			"start": seconds_to_srt(line_words[0].start),
			"end": seconds_to_srt(line_words[-1].end),
			"text": line  # use the verified lyrics text, not the transcribed text
		})

	with open(output_path, "w", encoding="utf-8") as f:
		for i, block in enumerate(blocks, start=1):
			f.write(f"{i}\n{block['start']} --> {block['end']}\n{block['text']}\n\n")


if __name__ == "__main__":
	file_path = "/home/ken/Downloads/KPop Demon Hunters - What It Sounds Like.webm"
	output_path = "/home/ken/Downloads/KPop Demon Hunters - What It Sounds Like.srt"

	# Check for a .txt lyrics file with the same filename prefix
	prefix = os.path.splitext(file_path)[0]
	lyrics_path = prefix + ".txt"

	model = load_model()

	if os.path.exists(lyrics_path):
		result = align_audio(model, file_path, lyrics_path)
		save_srt_from_alignment(result, lyrics_path, output_path)
	else:
		result = transcribe_audio(model, file_path)
		save_srt_from_transcription(result, output_path)

	print(f"SRT saved to {output_path}")
