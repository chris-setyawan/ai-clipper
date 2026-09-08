"""
Frame decoding, done the same way on every machine.

OpenCV's VideoCapture hands off to whatever FFmpeg build the wheel was compiled
against, and the result is not reproducible: the same HEVC file, the same code,
and two machines disagreed about how many camera cuts it contains - 43 shots on
one, 4 on the other, with byte-identical input. Face detection agreed to within
a percent, so frames were arriving; they were simply not the same frames.

PyAV carries its own FFmpeg and is already a dependency (faster-whisper installs
it), so decoding through it costs nothing and removes the variable. OpenCV is
still used for everything downstream - histograms, cascades, colour work - where
it behaves consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import av
import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    frames: int


def probe(path: str) -> VideoInfo:
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if not duration and container.duration:
            duration = container.duration / av.time_base
        frames = stream.frames or int(duration * fps)
        return VideoInfo(
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=fps,
            duration=duration,
            frames=frames,
        )


def iter_frames(path: str, step: int = 1, width: Optional[int] = None
                ) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Yield (frame index, timestamp, BGR image) every `step` frames.

    Decoding is sequential - every frame passes through the decoder, and `step`
    only controls which ones are converted and handed back. Skipping the decode
    itself is what makes seek-based sampling unreliable on long-GOP codecs.
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 30.0

        for index, frame in enumerate(container.decode(video=0)):
            if index % step:
                continue
            img = frame.to_ndarray(format="bgr24")
            if width and img.shape[1] > width:
                scale = width / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale)
            yield index, index / fps, img


def frames_at(path: str, times: List[float], width: Optional[int] = None
              ) -> Iterator[Tuple[float, np.ndarray]]:
    """
    Yield (requested time, BGR image) for each time, in order.

    One sequential pass that hands back a frame whenever the decoder reaches the
    next requested timestamp. Seeking to each one separately is faster in
    principle, but on long-GOP video it lands on the nearest keyframe and the
    frame returned can be well away from the time asked for.
    """
    wanted = sorted(times)
    if not wanted:
        return

    cursor = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 30.0

        for index, frame in enumerate(container.decode(video=0)):
            if cursor >= len(wanted):
                break
            t = index / fps
            if t + 1e-9 < wanted[cursor]:
                continue

            img = frame.to_ndarray(format="bgr24")
            if width and img.shape[1] > width:
                scale = width / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale)

            # one frame can satisfy several closely-spaced requests
            while cursor < len(wanted) and wanted[cursor] <= t + 1e-9:
                yield wanted[cursor], img
                cursor += 1
