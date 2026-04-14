#!/usr/bin/env python3
"""Convert a .srt file to plain text."""

import sys
import srt


def srt_to_txt(srt_path: str, txt_path: str | None = None) -> str:
    """Read an SRT file and return/write plain text.

    If *txt_path* is given the text is also written to that file.
    Returns the extracted text.
    """
    with open(srt_path, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    # Join all subtitle lines, separated by newlines
    text = "\n".join(sub.content for sub in subs)

    if txt_path:
        with open(txt_path, "w", encoding="utf-8") as out:
            out.write(text + "\n")
        print(f"Written → {txt_path}")

    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.srt> [output.txt]")
        sys.exit(1)

    srt_file = sys.argv[1]
    txt_file = sys.argv[2] if len(sys.argv) > 2 else None

    srt_to_txt(srt_file, txt_file)
