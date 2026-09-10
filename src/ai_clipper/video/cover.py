"""
Pick one frame per clip to stand in for it.

Every platform takes a cover image and every platform picks a bad one if you let
it: the frame at a fixed offset, which lands mid-blink, mid-gesture, or on the
one moment the speaker is looking at their notes. Choosing it by hand is thirty
seconds per clip and nobody does it.

Two things decide the moment, and the second one arrived after looking at a real
run.

**The picture has to be settled.** The frame analysis that already runs for
framing knows which sampled frames had a face the detector was confident about,
how much the picture changed between samples, and where the face sat. "Settled"
is two measurements that are not the same: how much the frame changed, which
catches a gesture or a cut, and how far the tracked face moved, which catches
the middle of a pan. A cover taken mid-pan is soft even when the frame it came
from was sharp.

**Something has to be on screen worth reading.** Stillness alone will happily
choose a second where nobody is talking, and a thumbnail with no words on it
throws away the line that would have made someone stop. So a visible caption is
a requirement rather than a preference, and among the moments that have one, a
line whose words carry something is preferred and the calmest of those wins. The
requirement drops only for a clip where no moment has both a caption and a
confident face.

Which lines carry is not measured here. It is asked of `scoring/signals.py`, the
same banks the clip scorer uses, so the vocabulary that decides which moment is
worth clipping also decides which frame represents it.

The frame is then taken out of the rendered clip rather than out of the source.
That is not an optimisation, it is the whole reason the cover matches: the clip
has already been reframed for the platform and has its captions burned in, so a
still from it is by construction exactly what a viewer would see if they paused
there. The first version cropped the source itself and had to repeat the crop
arithmetic to stay in sync with the video, which is two implementations of one
idea and only one of them was ever tested.
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional, Sequence, Tuple

# Nobody looks their best on the first or last frame of a cut.
EDGE_MARGIN = 1.0

# How far either side of a candidate to look when judging whether the moment is
# settled. Two samples is a fifth of a second at the usual sampling rate.
NEIGHBOURS = 2

# Past this, a caption is a paragraph at thumbnail size whatever it says.
MAX_CHARS = 44

# What a marker word or a figure is worth on top of the content fraction. It is
# added rather than clamped in, so that "MARGIN CALL" still beats "KEBIJAKAN
# PEMERINTAH" even though both are two words that carry.
LOADED = 0.35


def _in_range(keyframes: Sequence, start: float, end: float) -> List:
    return [k for k in keyframes if start <= k.t <= end]


def _stillness(keyframes: Sequence, index: int) -> float:
    """How settled the picture is around one keyframe. Higher is calmer."""
    window = keyframes[max(0, index - NEIGHBOURS): index + NEIGHBOURS + 1]
    if not window:
        return 0.0

    measured = [k.motion for k in window if k.motion >= 0]
    motion = sum(measured) / len(measured) if measured else 0.0

    centres = [k.center_x for k in window]
    drift = max(centres) - min(centres)

    return -(motion + drift * 4.0)


def line_value(text: str) -> float:
    """
    How well a caption line works as the words on a thumbnail.

    1.0 means every word on screen carries something; a figure with a unit adds
    on top, so a loaded line can score above 1. The scale only has to be
    consistent, because it is weighed against stillness and nothing else.

    Two versions of this were wrong before the third, and both were wrong in the
    same way: they measured something that correlated with a good line instead
    of the line itself.

    The first measured length. The caption style puts two words on screen at a
    time, so every line came out between 15 and 21 characters and the rule never
    chose between anything. "TRANSAKSI HARIAN" and "MANAJEMEN DALAM" scored
    identically, and only one of those reads as a phrase.

    The second counted content words, tokenising the line first. Tokenising
    drops numerals, so "Hampir 90" was scored as the single word "hampir",
    came out perfect, and pushed number fragments to the top of eleven of
    fifteen clips. The lines got measurably better and visibly worse.

    So the denominator is now the words a viewer actually sees, and a bare
    numeral only counts when the line gives it a unit: "16 RIBU" is a thumbnail,
    "PALING 3" is the middle of a sentence.
    """
    from ..scoring import signals

    stripped = text.strip()
    shown = [w for w in stripped.split() if any(ch.isalnum() for ch in w)]
    if not shown:
        return 0.0

    pack = signals.active()
    weak = pack.stopwords | pack.filler | pack.weak_words
    plain = [w.strip(".,!?:;\"'-").lower() for w in shown]
    united = _has_unit(plain, stripped, pack)

    carrying = 0
    for word in plain:
        if any(ch.isdigit() for ch in word):
            carrying += 1 if united else 0
        elif word not in weak and len(word) >= 4:
            carrying += 1

    value = carrying / len(shown)

    # A chunk can end up holding a single word, at a sentence end or where a
    # pause split it early, and one word scoring full marks put "BEDA." on a
    # cover. A thumbnail needs a phrase, so a lone word is worth half of one.
    if len(shown) < 2:
        value *= 0.5

    # Whisper repeats itself on a stumble, and "TERNYATA TERNYATA" is not a
    # thumbnail however well its words score.
    if len(set(plain)) < len(plain):
        value *= 0.5

    if united or _is_loaded(plain, stripped, pack):
        value += LOADED
    if len(stripped) > MAX_CHARS:
        value *= 0.5
    return value


def _has_unit(plain: Sequence, text: str, pack) -> bool:
    """
    Whether a figure in the line is attached to something that gives it scale.

    Without one a numeral is the middle of a sentence rather than a claim.
    """
    if not any(any(ch.isdigit() for ch in w) for w in plain):
        return False
    return "%" in text or any(w in pack.quantity_words for w in plain)


def _is_loaded(plain: Sequence, text: str, pack) -> bool:
    """Whether the line carries weight rather than only naming things."""
    banks = (pack.stakes_markers, pack.curiosity_markers,
             pack.contrarian_markers, pack.intensity_markers)
    lowered = text.lower()
    for bank in banks:
        for phrase in bank:
            if " " in phrase:
                if phrase in lowered:
                    return True
            elif phrase in plain:
                return True
    return False


def pick_time(crop_path, start: float, end: float,
              words: Optional[Sequence] = None, per_chunk: int = 3,
              edge: float = EDGE_MARGIN) -> Optional[float]:
    """
    The best moment to freeze, or None if the analysis offers nothing better
    than a guess.

    Returning None rather than the midpoint is deliberate. The midpoint is what
    the caller would have done anyway, and a function that quietly returns it
    cannot be told apart from one that found a good frame.
    """
    if crop_path is None or not getattr(crop_path, "keyframes", None):
        return None

    inner = _in_range(crop_path.keyframes, start + edge, end - edge)
    if not inner:
        inner = _in_range(crop_path.keyframes, start, end)
    if not inner:
        return None

    confident = [(i, k) for i, k in enumerate(inner) if k.confident]
    if not confident:
        return None

    spoken = _spoken_windows(words, start, end, per_chunk) if words else []

    # A caption on the cover is a requirement, not a preference, and the
    # fallback is only for a clip where no moment has both a caption and a face.
    # Making it a weight instead was the first attempt: a very calm moment in
    # the gap between two words outscored every moment with words on screen, and
    # the cover came out wordless on a run where the report said otherwise.
    speaking = [(i, k) for i, k in confident if _line_at(spoken, k.t) is not None]
    candidates = speaking or confident

    # Among them, the calmest wins, and ties go to the earlier one: a cover near
    # the hook is more likely to show what the clip is about than one from the
    # tail, where the subject has usually moved on.
    scored = []
    for i, k in candidates:
        bonus = (_line_at(spoken, k.t) or 0.0) * LINE_WEIGHT
        scored.append((_stillness(inner, i) + bonus - (k.t - inner[0].t) * 0.02, k.t))
    return max(scored)[1]


def _spoken_windows(words: Sequence, start: float, end: float,
                    per_chunk: int) -> List[Tuple[float, float, float]]:
    """
    (from, to, value) for every span where a caption is actually visible.

    Spans are per word, not per chunk, and that distinction is the whole point.
    A chunk looks continuous in the transcript, but the karaoke effect emits one
    event per word, running from that word's start to its end, so the short gaps
    between words are gaps where nothing is on screen at all. Scoring by chunk
    put a cover in one of those holes on the first real run: the picture was
    good, the report said a line was showing, and the thumbnail was wordless.

    The last word of a chunk is held until the chunk ends, exactly as the
    subtitle builder holds it, so this has to mirror that too. Readability is
    still judged on the whole chunk, because that is what a viewer reads.
    """
    from .subtitles import chunk_words

    inside = [w for w in words if w.end > start and w.start < end]
    if not inside:
        return []

    windows = []
    for chunk in chunk_words(inside, per_chunk):
        score = line_value(" ".join(w.text for w in chunk.words))
        for i, word in enumerate(chunk.words):
            last = i == len(chunk.words) - 1
            windows.append((word.start, chunk.end if last else word.end, score))
    return windows


def _line_at(windows: Sequence, t: float) -> Optional[float]:
    """
    The value of whatever caption is on screen at t, or None if none is.

    None and 0.0 are different answers and the caller depends on the
    difference: nothing showing at all, versus a line showing whose words
    happen to carry nothing. A cover still wants the second one.
    """
    for a, b, score in windows:
        if a <= t <= b:
            return score
    return None


# How much a well-sized line is worth against holding still, once both moments
# already have a caption on them.
LINE_WEIGHT = 8.0


def grab(clip_file: str, at: float, out_path: str, quality: int = 3) -> str:
    """
    Take one frame out of a rendered clip.

    `at` is a time inside the clip, not inside the source episode. Same atomic
    write as the renderer: a half-written JPEG is a valid-looking file, and a
    resumed run would take it for finished work.
    """
    final = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    target = final + ".part.jpg"

    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-accurate_seek", "-ss", f"{max(0.0, at):.3f}", "-i", clip_file,
        "-frames:v", "1", "-q:v", str(quality), target,
    ], capture_output=True, text=True)

    if proc.returncode != 0:
        if os.path.exists(target):
            os.remove(target)
        raise RuntimeError(f"cover frame failed for {final}:\n{proc.stderr[-1000:]}")

    os.replace(target, final)
    return final
