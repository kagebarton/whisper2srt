"""Minimal LRC parser for LRCLIB `synced_lyrics`.

Format: each line is one or more `[mm:ss.cc]` stamps followed by text.

Examples handled:
    [00:22.36] I've been spinning now for time
    [00:01.00][00:30.00] So tell me when it kicks in
    [04:42.23]                                  ← trailing blank-text stamp

Multi-stamp lines split into one entry per stamp. Blank-text stamps are
*kept* — they're useful end-of-section anchors. Returned entries are in
file order (not sorted), but with multi-stamp expansion the resulting
list is effectively time-ascending for sane inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STAMP_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


@dataclass
class LrcLine:
    start: float
    text: str


def parse_lrc(synced_lyrics: str) -> list[LrcLine]:
    out: list[LrcLine] = []
    for raw in synced_lyrics.splitlines():
        stamps = list(_STAMP_RE.finditer(raw))
        if not stamps:
            continue
        text = raw[stamps[-1].end():].strip()
        for m in stamps:
            mm = int(m.group(1))
            ss = float(m.group(2))
            out.append(LrcLine(start=mm * 60.0 + ss, text=text))
    return out
