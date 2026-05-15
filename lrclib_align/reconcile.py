"""Line-level Genius ↔ LRCLIB reconciliation.

LRCLIB describes *this mix* with per-line stamps; Genius describes the
*canonical* song with section headers and possibly extra/missing lines.
We need to know, for each Genius line, when (if ever) it occurs in this
mix.

Algorithm: Needleman–Wunsch on lines, similarity via
``rapidfuzz.fuzz.token_set_ratio`` on normalized text. After NW, walk the
alignment and label each Genius line:

- **match**: aligned to an LRCLIB line with score ≥ ``MATCH_THRESHOLD``.
- **weak**: aligned with score in ``[WEAK_THRESHOLD, MATCH_THRESHOLD)``.
  Accepted only if both immediate neighbors are matches; else demoted
  to ``unmatched``.
- **unmatched**: NW gave it a gap, or the matched score < weak.

Post-process unmatched runs:
- A run of ≥ ``CUT_RUN_MIN`` consecutive unmatched lines → ``status=cut``
  (likely a deliberately-cut section in this mix; will be dropped from
  the output by the caller).
- Shorter runs → ``status=interpolated`` with linear-interp times from
  the nearest matched neighbors on each side. Runs at the very start /
  end with only one matched neighbor get that neighbor's time (no
  smearing past song bounds).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from lrclib_align.lrc import LrcLine

logger = logging.getLogger(__name__)

# Tunables (constants — adjust here, not at call sites).
MATCH_THRESHOLD = 75.0
WEAK_THRESHOLD = 60.0
CUT_RUN_MIN = 3
GAP_PENALTY = -10.0  # NW gap cost (similarity is 0..100)


_PAREN_RE = re.compile(r"\([^)]*\)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def _normalize(text: str) -> str:
    t = text.lower()
    t = _PAREN_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    return " ".join(t.split())


@dataclass
class ReconciledLine:
    genius_idx: int
    lrclib_idx: Optional[int]
    lrclib_time: Optional[float]
    score: float  # 0..100 (0 if unmatched/gap)
    status: str  # "match" | "weak" | "interpolated" | "cut"


def _needleman_wunsch(
    genius_norm: list[str],
    lrclib_norm: list[str],
) -> list[tuple[Optional[int], Optional[int], float]]:
    """Return list of (genius_idx|None, lrclib_idx|None, score) in order."""
    n, m = len(genius_norm), len(lrclib_norm)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP_PENALTY
    for j in range(1, m + 1):
        dp[0][j] = j * GAP_PENALTY

    sim = [[0.0] * m for _ in range(n)]
    for i in range(n):
        gi = genius_norm[i]
        if not gi:
            continue
        for j in range(m):
            lj = lrclib_norm[j]
            if not lj:
                continue
            sim[i][j] = fuzz.token_set_ratio(gi, lj)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + sim[i - 1][j - 1]
            up = dp[i - 1][j] + GAP_PENALTY
            left = dp[i][j - 1] + GAP_PENALTY
            dp[i][j] = max(diag, up, left)

    # Traceback.
    out: list[tuple[Optional[int], Optional[int], float]] = []
    i, j = n, m
    while i > 0 and j > 0:
        diag = dp[i - 1][j - 1] + sim[i - 1][j - 1]
        up = dp[i - 1][j] + GAP_PENALTY
        if dp[i][j] == diag:
            out.append((i - 1, j - 1, sim[i - 1][j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == up:
            out.append((i - 1, None, 0.0))
            i -= 1
        else:
            out.append((None, j - 1, 0.0))
            j -= 1
    while i > 0:
        out.append((i - 1, None, 0.0))
        i -= 1
    while j > 0:
        out.append((None, j - 1, 0.0))
        j -= 1
    out.reverse()
    return out


def reconcile(
    genius_lines: list[dict],
    lrc_lines: list[LrcLine],
) -> list[ReconciledLine]:
    genius_norm = [_normalize(gl["align_text"]) for gl in genius_lines]
    lrclib_norm = [_normalize(ll.text) for ll in lrc_lines]

    alignment = _needleman_wunsch(genius_norm, lrclib_norm)

    # Build per-Genius result skeleton.
    by_g: dict[int, ReconciledLine] = {}
    for gi, lj, score in alignment:
        if gi is None:
            continue
        if lj is None or score < WEAK_THRESHOLD:
            by_g[gi] = ReconciledLine(
                genius_idx=gi,
                lrclib_idx=None,
                lrclib_time=None,
                score=score if lj is not None else 0.0,
                status="unmatched",
            )
        elif score >= MATCH_THRESHOLD:
            by_g[gi] = ReconciledLine(
                genius_idx=gi,
                lrclib_idx=lj,
                lrclib_time=lrc_lines[lj].start,
                score=score,
                status="match",
            )
        else:
            by_g[gi] = ReconciledLine(
                genius_idx=gi,
                lrclib_idx=lj,
                lrclib_time=lrc_lines[lj].start,
                score=score,
                status="weak",
            )

    # Demote weak matches that aren't bordered by strong matches.
    n = len(genius_lines)
    for i in range(n):
        rl = by_g[i]
        if rl.status != "weak":
            continue
        left_ok = i > 0 and by_g[i - 1].status == "match"
        right_ok = i < n - 1 and by_g[i + 1].status == "match"
        if left_ok and right_ok:
            rl.status = "match"
        else:
            rl.lrclib_idx = None
            rl.lrclib_time = None
            rl.status = "unmatched"

    # Mark cut-section runs (≥ CUT_RUN_MIN unmatched in a row).
    i = 0
    while i < n:
        if by_g[i].status == "unmatched":
            j = i
            while j < n and by_g[j].status == "unmatched":
                j += 1
            run_len = j - i
            if run_len >= CUT_RUN_MIN:
                for k in range(i, j):
                    by_g[k].status = "cut"
            i = j
        else:
            i += 1

    # Interpolate the remaining short unmatched runs.
    for i in range(n):
        if by_g[i].status != "unmatched":
            continue
        # Find nearest matched neighbors on each side.
        left_t: Optional[float] = None
        left_idx: Optional[int] = None
        for k in range(i - 1, -1, -1):
            if by_g[k].status == "match" and by_g[k].lrclib_time is not None:
                left_t = by_g[k].lrclib_time
                left_idx = k
                break
        right_t: Optional[float] = None
        right_idx: Optional[int] = None
        for k in range(i + 1, n):
            if by_g[k].status == "match" and by_g[k].lrclib_time is not None:
                right_t = by_g[k].lrclib_time
                right_idx = k
                break

        if left_t is not None and right_t is not None:
            assert left_idx is not None and right_idx is not None
            span = right_idx - left_idx
            frac = (i - left_idx) / span if span else 0.0
            t = left_t + (right_t - left_t) * frac
        elif left_t is not None:
            t = left_t
        elif right_t is not None:
            t = right_t
        else:
            t = None

        by_g[i].lrclib_time = t
        by_g[i].status = "interpolated"

    result = [by_g[i] for i in range(n)]

    matched = sum(1 for r in result if r.status == "match")
    interp = sum(1 for r in result if r.status == "interpolated")
    cut = sum(1 for r in result if r.status == "cut")
    logger.info(
        "Reconcile: %d match / %d interp / %d cut (of %d Genius lines)",
        matched, interp, cut, n,
    )
    return result
