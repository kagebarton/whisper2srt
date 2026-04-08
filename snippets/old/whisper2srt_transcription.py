import os
import glob
import argparse
import config
import stable_whisper

# Priority languages listed first for convenience
PRIORITY_LANGUAGES = {
    "1": ("en", "English"),
    "2": ("zh", "Mandarin"),
    "3": ("yue", "Cantonese"),
    "4": ("ja", "Japanese"),
    "5": ("ko", "Korean"),
}

# All Whisper large-v3 supported languages
ALL_LANGUAGES = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    "zh": "Mandarin",
}


def get_language_choice():
    """Prompt user to select the predominant language of the song."""
    print("\nSelect the predominant language:")
    print("-" * 50)
    print("Common:")
    for key, (code, name) in PRIORITY_LANGUAGES.items():
        print(f"  {key}. {name} ({code})")
    print("-" * 50)
    print("All languages (enter code directly, e.g., 'fr', 'de', 'es'):")
    print("  af=Afrikaans, ar=Arabic, bn=Bengali, bs=Bosnian, ca=Catalan,")
    print("  cs=Czech, da=Danish, de=German, el=Greek, es=Spanish, et=Estonian,")
    print("  fa=Persian, fi=Finnish, fr=French, gu=Gujarati, he=Hebrew, hi=Hindi,")
    print("  hr=Croatian, hu=Hungarian, hy=Armenian, id=Indonesian, is=Icelandic,")
    print("  it=Italian, ka=Georgian, kk=Kazakh, km=Khmer, kn=Kannada, la=Latin,")
    print("  lt=Lithuanian, lv=Latvian, mk=Macedonian, ml=Malayalam, mn=Mongolian,")
    print("  mr=Marathi, ms=Malay, mt=Maltese, my=Burmese, ne=Nepali, nl=Dutch,")
    print("  no=Norwegian, pa=Punjabi, pl=Polish, ps=Pashto, pt=Portuguese, ro=Romanian,")
    print("  ru=Russian, si=Sinhala, sk=Slovak, sl=Slovenian, sq=Albanian, sr=Serbian,")
    print("  sv=Swedish, sw=Swahili, ta=Tamil, te=Telugu, th=Thai, tl=Tagalog,")
    print("  tr=Turkish, tt=Tatar, uk=Ukrainian, ur=Urdu, uz=Uzbek, vi=Vietnamese,")
    print("  and more...")
    print("-" * 50)

    while True:
        choice = input("Enter choice (1-5 or language code, default=1): ").strip().lower()
        if not choice:
            choice = "1"
        if choice in PRIORITY_LANGUAGES:
            code, name = PRIORITY_LANGUAGES[choice]
            print(f"Language: {name} ({code})")
            return code, name
        if choice in ALL_LANGUAGES:
            name = ALL_LANGUAGES[choice]
            print(f"Language: {name} ({choice})")
            return choice, name
        print("Invalid choice. Please enter 1-5 or a valid language code.")


def load_model():
    return stable_whisper.load_faster_whisper(
        config.WHISPER_MODEL,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE
    )


def transcribe_audio(model, audio_file_path: str, language: str = "en"):
    """Transcribes audio using faster-whisper via stable-ts."""
    print(f"Transcribing audio (language: {language})...")
    return model.transcribe(
        audio_file_path,
        word_timestamps=True,
        language=language,
        beam_size=10,
        best_of=10,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt="Lyrics",
    )


def seconds_to_srt(seconds: float) -> str:
    """Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(result, output_path: str):
    """Saves transcription result to SRT format, one segment per subtitle."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result.segments, start=1):
            start = seconds_to_srt(segment.start)
            end = seconds_to_srt(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def save_raw_text(result, output_path: str):
    """Saves raw Whisper transcription to plain text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in result.segments:
            f.write(segment.text.strip() + "\n")


def process_file(model, file_path: str, language: str = "en"):
    prefix = os.path.splitext(file_path)[0]
    srt_path = prefix + ".srt"
    txt_path = prefix + ".txt"

    print(f"\nProcessing: {file_path}")

    result = transcribe_audio(model, file_path, language)
    save_srt(result, srt_path)
    save_raw_text(result, txt_path)

    print(f"SRT saved to {srt_path}")
    print(f"Raw text saved to {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT subtitles.")
    parser.add_argument("files", nargs="+", help="Audio file path(s); supports wildcards e.g. *.mp3")
    args = parser.parse_args()

    # Expand wildcards manually
    file_paths = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            file_paths.extend(sorted(matches))
        else:
            print(f"Warning: no files matched '{pattern}'")

    if not file_paths:
        print("No files to process.")
    else:
        model = load_model()
        language_code, language_name = get_language_choice()
        for file_path in file_paths:
            process_file(model, file_path, language_code)
