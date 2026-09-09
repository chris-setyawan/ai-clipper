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

**Something has to be being said.** Stillness alone will happily choose a second
where nobody is talking, and a thumbnail with no words on it throws away the
line that would have made someone stop. So candidate moments are drawn from
inside the caption chunks, and a chunk long enough to read but short enough to
take in at a glance is preferred.

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

# A caption this long is a paragraph at thumbnail size; this short is a fragment.
IDEAL_CHARS = (12, 44)


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


def readability(text: str) -> float:
    """
    How well a caption line works as the words on a thumbnail.

    Zero outside the readable range rather than negative, so a clip whose lines
    are all too long still gets a cover; it just stops preferring one line over
    another and lets stillness decide.
    """
    n = len(text.strip())
    low, high = IDEAL_CHARS
    if n < low or n > high:
        return 0.0
    return 1.0 - abs(n - (low + high) / 2.0) / ((high - low) / 2.0)


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

    # Among settled moments, the earlier one wins. A cover taken near the hook
    # is more likely to show what the clip is actually about than one from the
    # tail, where the subject has usually moved on.
    scored = []
    for i, k in confident:
        bonus = _line_bonus(spoken, k.t)
        scored.append((_stillness(inner, i) + bonus - (k.t - inner[0].t) * 0.02, k.t))
    return max(scored)[1]


def _spoken_windows(words: Sequence, start: float, end: float,
                    per_chunk: int) -> List[Tuple[float, float, float]]:
    """
    (from, to, readability) for each caption chunk inside the clip.

    Built with the same chunking the subtitles use, so a moment that scores well
    here is a moment where that exact line is on screen, not an approximation of
    one.
    """
    from .subtitles import chunk_words

    inside = [w for w in words if w.end > start and w.start < end]
    if not inside:
        return []

    windows = []
    for chunk in chunk_words(inside, per_chunk):
        text = " ".join(w.text for w in chunk.words)
        windows.append((chunk.start, chunk.end, readability(text)))
    return windows


def _line_bonus(windows: Sequence, t: float, weight: float = 8.0) -> float:
    """
    How much preferring a readable line is worth against holding still.

    The weight is what decides which of the two rules wins when they disagree,
    and it is set so that a well-sized line beats a moderately calmer frame but
    loses to an obviously bad one. A cover that is sharp and wordless is still a
    usable cover; a blurred one with a good line on it is not.
    """
    for a, b, score in windows:
        if a <= t <= b:
            return score * weight
    return 0.0


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
