"""
Pick one frame per clip to stand in for it.

Every platform takes a cover image and every platform picks a bad one if you let
it: the frame at a fixed offset, which lands mid-blink, mid-gesture, or on the
one moment the speaker is looking at their notes. Choosing it by hand is thirty
seconds per clip and nobody does it.

The frame analysis this project already runs for framing knows more than enough
to choose better. It knows which sampled frames had a face the detector was
confident about, how much the picture changed between samples, and where the
face sat. A good cover is a confident detection during a settled moment, which
is a two-line rule over data that has already been computed and cached.

What it deliberately does not do is add text. A cover with a headline burned
into it is a design decision that depends on the channel, and the clip already
ships with a title in its caption pack for whoever wants to make that call.
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional, Sequence

# Nobody looks their best on the first or last frame of a cut.
EDGE_MARGIN = 1.0

# How far either side of a candidate to look when judging whether the moment is
# settled. Two samples is a fifth of a second at the usual sampling rate.
NEIGHBOURS = 2


def _in_range(keyframes: Sequence, start: float, end: float) -> List:
    return [k for k in keyframes if start <= k.t <= end]


def _stillness(keyframes: Sequence, index: int) -> float:
    """
    How settled the picture is around one keyframe. Higher is calmer.

    Two things count and they are different: how much the frame changed, which
    catches a gesture or a cut, and how far the tracked face moved, which
    catches the middle of a pan. A cover taken mid-pan is soft even when the
    frame it came from was sharp.
    """
    window = keyframes[max(0, index - NEIGHBOURS): index + NEIGHBOURS + 1]
    if not window:
        return 0.0

    measured = [k.motion for k in window if k.motion >= 0]
    motion = sum(measured) / len(measured) if measured else 0.0

    centres = [k.center_x for k in window]
    drift = max(centres) - min(centres)

    return -(motion + drift * 4.0)


def pick_time(crop_path, start: float, end: float,
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

    # Among settled moments, the earlier one wins. A cover taken near the hook
    # is more likely to show what the clip is actually about than one from the
    # tail, where the subject has usually moved on.
    scored = [
        (_stillness(inner, i) - (k.t - inner[0].t) * 0.02, k.t)
        for i, k in confident
    ]
    return max(scored)[1]


def crop_x(crop_path, t: float, scaled_width: int, out_width: int) -> int:
    """
    Where the crop window sits at t, in the scaled frame.

    The same arithmetic the sendcmd script uses, so the cover is framed exactly
    the way the video is at that instant rather than approximately.
    """
    center = crop_path.center_at(t) if crop_path is not None else 0.5
    x = int(round(center * scaled_width - out_width / 2))
    return min(max(x, 0), max(0, scaled_width - out_width))


def _filter(mode: str, out_w: int, out_h: int,
            scaled_w: Optional[int], x: Optional[int]) -> str:
    if mode in ("face", "per_shot") and scaled_w and x is not None:
        return f"scale={scaled_w}:{out_h},crop={out_w}:{out_h}:{x}:0"
    return (f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}")


def write(source: str, t: float, out_path: str, out_w: int, out_h: int,
          mode: str = "crop", crop_path=None, scaled_w: Optional[int] = None,
          quality: int = 3) -> str:
    """
    Write one frame, framed and sized the way the clip is.

    Same atomic-write discipline as the renderer: a half-written JPEG is a
    valid-looking file, and a resumed run would take it for finished work.
    """
    final = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    target = final + ".part.jpg"

    x = crop_x(crop_path, t, scaled_w, out_w) if scaled_w else None
    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-accurate_seek", "-ss", f"{t:.3f}", "-i", source,
        "-frames:v", "1", "-vf", _filter(mode, out_w, out_h, scaled_w, x),
        "-q:v", str(quality), target,
    ], capture_output=True, text=True)

    if proc.returncode != 0:
        if os.path.exists(target):
            os.remove(target)
        raise RuntimeError(f"cover frame failed for {final}:\n{proc.stderr[-1000:]}")

    os.replace(target, final)
    return final
