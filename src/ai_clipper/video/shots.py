"""
Shot-change detection.

A multi-camera podcast cuts between angles every few seconds. The framing stage
has to know where those cuts are, because a crop window that glides smoothly
across a hard cut looks broken - on a cut it should jump instantly, and only
between cuts should it drift.

Detection compares colour histograms of consecutive sampled frames. Histograms
rather than raw pixel difference, because a person waving an arm changes a lot
of pixels without changing the shot, while a camera change moves the whole
distribution at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .decode import iter_frames, probe


@dataclass
class Shot:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


def _histogram(frame) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def detect_shots(
    video_path: str,
    threshold: float = 0.45,
    sample_every: int = 3,
    min_shot_duration: float = 0.5,
    analysis_width: int = 320,
) -> List[Shot]:
    """
    Return the shots making up the video.

    threshold is a histogram correlation distance: higher means fewer, larger
    shots. min_shot_duration merges anything shorter, which suppresses the
    double-trigger you get on a cross-dissolve.
    """
    info = probe(video_path)
    fps = info.fps or 30.0
    duration = info.duration

    cut_times: List[float] = []
    prev_hist = None
    last_time = 0.0

    for _, t, frame in iter_frames(video_path, step=sample_every, width=analysis_width):
        hist = _histogram(frame)
        if prev_hist is not None:
            # correlation is 1.0 for identical frames
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if (1.0 - corr) > threshold:
                cut_times.append(t)
        prev_hist = hist
        last_time = t

    if duration <= 0:
        duration = last_time + sample_every / fps

    boundaries = [0.0] + cut_times + [duration]

    shots: List[Shot] = []
    for a, b in zip(boundaries, boundaries[1:]):
        if shots and (b - a) < min_shot_duration:
            # too short to be a real shot - fold it into the previous one
            shots[-1] = Shot(shots[-1].start, b)
        else:
            shots.append(Shot(a, b))

    return shots


def shot_at(shots: List[Shot], t: float) -> int:
    """Index of the shot containing time t, or the last shot if past the end."""
    for i, s in enumerate(shots):
        if s.contains(t):
            return i
    return len(shots) - 1
