"""Speaker assignment: appearance-order remap, cluster→name mapping,
word→speaker assignment, line splitting.

Functions:
- remap_speakers_by_appearance: raw pyannote labels → A/B/C by first appearance
- map_clusters_to_names: pyannote labels → named singers via Genius majority vote
- assign_speakers_to_words: attach a speaker label (letter or name) to every word
- reset_segment_first_flags: fix is_segment_first after word regrouping
- split_lines_at_speaker_boundaries: break lines at speaker hand-offs
"""

import logging

logger = logging.getLogger(__name__)


def remap_speakers_by_appearance(turns, cfg):
    """Remap raw pyannote speaker labels to A/B/C by appearance order.

    Walks turns sorted by start time; the first-seen pyannote label is
    mapped to cfg.speaker_letters[0] (= "A"), second to "B", etc.

    Returns:
        (turns, label_map) — turns are mutated in-place (each gets a
        "letter" key), label_map maps raw label → letter.

    Raises:
        ValueError: if more speakers are detected than there are
        configured letters.
    """
    label_map = {}
    for t in sorted(turns, key=lambda t: t["start"]):
        raw = t["speaker"]
        if raw not in label_map:
            idx = len(label_map)
            if idx >= len(cfg.speaker_letters):
                raise ValueError(
                    f"More speakers than configured letters "
                    f"({len(cfg.speaker_letters)}). "
                    f"Bump speaker_letters / speaker_colors in config, "
                    f"or set max_speakers."
                )
            label_map[raw] = cfg.speaker_letters[idx]
    for t in turns:
        t["letter"] = label_map[t["speaker"]]
    return turns, label_map


def map_clusters_to_names(
    turns: list[dict],
    line_objects: list[dict],
    genius_lines: list[dict],
) -> dict[str, str]:
    """Build {pyannote_label -> singer_name} via majority vote.

    Algorithm:
    1. Pair line_objects with genius_lines by index (they're aligned
       1:1 — match_words_to_lines already preserved order).
    2. For each pair where genius_line is single-named:
       For each word in line_obj["words"]:
         find the pyannote turn covering that word's midpoint;
         credit (pyannote_label, singer_name) += word_duration.
    3. For each pyannote_label, pick the singer_name with max credit.
    4. Resolve collisions (two clusters mapping to the same name): keep
       the one with the higher credit; the other gets the runner-up
       name. If runner-up is also taken, fall back to letter (A/B/C).
    5. Ambiguity threshold: if the winning name's credit is less than
       2× the runner-up's, drop that cluster from the mapping.

    Returns {} if no single-named lines are available.
    """
    if not line_objects or not genius_lines:
        return {}

    # Build credit matrix: (pyannote_label, singer_name) -> seconds
    votes: dict[tuple[str, str], float] = {}

    for line_obj, genius_line in zip(line_objects, genius_lines):
        singers = genius_line.get("singers")
        # Only consider single-named lines (len==1 and not "All")
        if singers is None or len(singers) != 1 or singers[0] == "All":
            continue
        singer_name = singers[0]

        for w in line_obj.get("words", []):
            mid = 0.5 * (w["start"] + w["end"])
            duration = w["end"] - w["start"]
            label = _find_turn_label_at_midpoint(mid, turns)
            if label is None:
                continue
            key = (label, singer_name)
            votes[key] = votes.get(key, 0.0) + duration

    if not votes:
        return {}

    # Group votes by pyannote_label
    label_votes: dict[str, list[tuple[str, float]]] = {}
    for (label, name), credit in votes.items():
        label_votes.setdefault(label, []).append((name, credit))

    # Sort each label's candidates by credit descending
    for label in label_votes:
        label_votes[label].sort(key=lambda x: x[1], reverse=True)

    # First pass: pick the max-credit name per label
    raw_mapping: dict[str, tuple[str, float]] = {}  # label -> (name, credit)
    for label, candidates in label_votes.items():
        winner_name, winner_credit = candidates[0]
        # Ambiguity threshold: winner must have >= 2× runner-up credit
        if len(candidates) > 1:
            runner_up_credit = candidates[1][1]
            if winner_credit < 2.0 * runner_up_credit:
                logger.warning(
                    f"Ambiguous cluster {label}: '{winner_name}' credit "
                    f"{winner_credit:.1f}s vs runner-up {candidates[1][0]} "
                    f"{runner_up_credit:.1f}s — dropping from name mapping"
                )
                continue
        raw_mapping[label] = (winner_name, winner_credit)

    # Resolve name collisions: if two clusters claim the same name,
    # the one with higher credit keeps it; the other takes runner-up.
    # If runner-up is also taken, fall back to letter (not included
    # in the returned dict — caller uses label_map for letter fallback).
    name_to_label: dict[str, str] = {}  # name -> best label
    final_mapping: dict[str, str] = {}  # label -> name

    # Process labels sorted by credit descending so the strongest claim wins
    sorted_labels = sorted(
        raw_mapping.keys(), key=lambda l: raw_mapping[l][1], reverse=True
    )
    for label in sorted_labels:
        name, credit = raw_mapping[label]
        if name not in name_to_label:
            # First claim on this name
            name_to_label[name] = label
            final_mapping[label] = name
        else:
            # Name already taken — try runner-up names from this label's candidates
            assigned = False
            for runner_name, runner_credit in label_votes[label][1:]:
                if runner_name not in name_to_label:
                    name_to_label[runner_name] = label
                    final_mapping[label] = runner_name
                    logger.info(
                        f"Cluster {label}: '{name}' already claimed by "
                        f"{name_to_label[name]}, falling back to '{runner_name}'"
                    )
                    assigned = True
                    break
            if not assigned:
                logger.warning(
                    f"Cluster {label}: all candidate names already taken, "
                    f"falling back to letter"
                )

    return final_mapping


def _find_turn_label_at_midpoint(mid: float, turns: list[dict]) -> str | None:
    """Return the pyannote speaker label for the turn covering *mid*.

    Falls back to the nearest turn if the midpoint falls in a gap.
    Returns None if turns is empty.
    """
    if not turns:
        return None
    for t in turns:
        if t["start"] <= mid <= t["end"]:
            return t["speaker"]
    # Gap fallback: nearest turn by midpoint distance
    nearest = min(
        turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"]))
    )
    return nearest["speaker"]


def assign_speakers_to_words(
    line_objects,
    turns,
    overlap_intervals=None,
    genius_lines=None,
    cluster_to_name=None,
):
    """Attach a speaker label (name, letter, or None) to every word.

    Per-word resolution order (§7):

    1. **Genius ensemble:** if the parent line's genius_lines[idx] has
       is_ensemble=True → speaker = None.
    2. **Genius single-name:** if the parent line is single-named and
       that name is in cluster_to_name.values() → speaker = name.
       (We trust Genius even if pyannote's nearest cluster disagrees —
       Genius is ground truth at line granularity.)
    3. **Pyannote + overlap:** existing logic — _word_speaker(...) with
       overlap intervals → either cluster_to_name[label] or None.
    4. **Letter fallback:** if a label isn't in cluster_to_name (e.g.
       dropped by ambiguity threshold), use the appearance-order letter
       (existing remap_speakers_by_appearance output).

    Steps 1–2 require pairing line_objects[i] with genius_lines[i],
    which the alignment-mode caller already enforces (count-based
    pairing in match_words_to_lines).

    When genius_lines is None, steps 1–2 are skipped entirely and the
    function behaves exactly as before (letter-based assignment).

    Raises:
        RuntimeError: if turns is empty and no Genius attribution is
            available (cannot label any words).
    """
    use_genius = genius_lines is not None and len(genius_lines) == len(line_objects)
    known_names = set(cluster_to_name.values()) if cluster_to_name else set()

    if not turns and not use_genius:
        raise RuntimeError("Diarization produced zero turns — cannot label words")

    for idx, line in enumerate(line_objects):
        # Determine line-level Genius attribution once per line
        genius_line = genius_lines[idx] if use_genius else None
        genius_ensemble = False
        genius_single_name = None

        if genius_line is not None:
            singers = genius_line.get("singers")
            if singers is not None:
                if len(singers) == 1 and singers[0] != "All":
                    # Single-named line — this name is the candidate
                    genius_single_name = singers[0]
                else:
                    # Multi-name or ["All"] → ensemble
                    genius_ensemble = True

        for w in line["words"]:
            if genius_ensemble:
                # Step 1: Genius ensemble
                w["speaker"] = None
            elif genius_single_name and genius_single_name in known_names:
                # Step 2: Genius single-name (trusted over pyannote)
                w["speaker"] = genius_single_name
            else:
                # Steps 3–4: pyannote + overlap, then letter fallback
                w["speaker"] = _word_speaker(
                    w, turns, overlap_intervals, cluster_to_name
                )

    return line_objects


def _word_speaker(word, turns, overlap_intervals=None, cluster_to_name=None):
    """Return the speaker label for a word, or None if it's in an overlap zone.

    Strategy: check overlap intervals first; if the word's midpoint is in a
    multi-speaker region, return None. Otherwise pick the turn with the
    largest temporal overlap; gap fallback to nearest turn by midpoint.

    If cluster_to_name is provided and the turn's raw speaker label is in
    it, return the mapped name. Otherwise fall back to the turn's letter
    (from remap_speakers_by_appearance).
    """
    if overlap_intervals and _midpoint_in_overlap(word, overlap_intervals):
        return None

    best_turn, best_overlap = None, 0.0
    for t in turns:
        overlap = max(0.0, min(word["end"], t["end"]) - max(word["start"], t["start"]))
        if overlap > best_overlap:
            best_turn, best_overlap = t, overlap
    if best_turn is not None:
        raw_label = best_turn["speaker"]
        if cluster_to_name and raw_label in cluster_to_name:
            return cluster_to_name[raw_label]
        return best_turn["letter"]
    # gap fallback: nearest turn by midpoint (turns guaranteed non-empty)
    mid = 0.5 * (word["start"] + word["end"])
    nearest = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
    raw_label = nearest["speaker"]
    if cluster_to_name and raw_label in cluster_to_name:
        return cluster_to_name[raw_label]
    return nearest["letter"]


def _midpoint_in_overlap(word, overlap_intervals):
    """Return True if the word's midpoint falls inside any overlap interval."""
    mid = 0.5 * (word["start"] + word["end"])
    return any(start <= mid <= end for start, end in overlap_intervals)


def reset_segment_first_flags(line_objects):
    """Reset is_segment_first so only the first word of each line is True.

    Called from run.py after match_words_to_lines (alignment path only).
    The whisper alignment marks is_segment_first=True on the first word
    of each *whisper* segment, but those boundaries don't match the
    lyric-line boundaries we just regrouped to. Stray True flags in
    the middle of a line would cause first_word_nudge_cs to fire at
    the wrong position.

    Not needed in the transcription path — segments_to_line_objects
    builds each line from one whisper segment, so the flags already
    line up.
    """
    for line in line_objects:
        for i, w in enumerate(line["words"]):
            w["is_segment_first"] = i == 0


def split_lines_at_speaker_boundaries(line_objects):
    """Split lines wherever the speaker changes between consecutive words.

    Option (b): walk each line's words; whenever the speaker letter
    changes between consecutive words, close the current sub-line and
    start a new one. The first word of every emitted sub-line gets
    is_segment_first=True so the existing first_word_nudge_cs logic
    still applies inside the karaoke generator.
    """
    out = []
    for line in line_objects:
        words = line["words"]
        if not words:
            continue
        cur_words = [dict(words[0], is_segment_first=True)]
        cur_speaker = words[0]["speaker"]
        for w in words[1:]:
            if w["speaker"] != cur_speaker:
                out.append(_make_sub(cur_words, cur_speaker))
                cur_words = [dict(w, is_segment_first=True)]
                cur_speaker = w["speaker"]
            else:
                cur_words.append(w)
        out.append(_make_sub(cur_words, cur_speaker))
    return out


def _make_sub(words, speaker):
    """Build a line-object dict from a contiguous run of same-speaker words."""
    return {
        "text": " ".join(w["word"] for w in words),
        "words": words,
        "speaker": speaker,
        "start": words[0]["start"],
        "end": words[-1]["end"],
    }
