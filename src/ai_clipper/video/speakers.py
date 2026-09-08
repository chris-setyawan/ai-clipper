"""
Which of the visible people is talking right now.

Audio diarization was tried first and failed (see asr/diarize.py). It was also
answering the wrong question: framing does not need to know *who* someone is,
only which face is currently speaking. Mouths move when people talk, and that is
visible in the video, so measure it there.

The method, per sampled frame:

  1. Find the faces and assign each to a persistent seat. On a static camera
     people stay where they sat down, so a seat is just a stable x position.
  2. Crop the mouth region of each seat and measure how much it changed since
     the previous sample.
  3. Whoever's mouth is moving most is speaking - with hysteresis, so the
     answer does not flicker between two people every half second.

STATUS OF THE ACTIVE-SPEAKER PART: MEASURED AND REJECTED.

Seat finding works and is used - find_seats reliably locates where people sit
on a static camera. The active-speaker half does not work and is not wired in.

Measured against the audio on the static sample: mouth motion during the loudest
seconds was 0.78x its level during the quietest, and per-second correlation with
the audio envelope was -0.15. Both point the wrong way. Retried at full
resolution (73x52 px mouths rather than 38x29) and with head motion subtracted
to isolate the jaw; the best any variant reached was r=+0.11, which is not a
usable signal. What the measure actually captures is general movement, and a
listener nodding along moves more than a talker sitting still, which is why the
correlation comes out negative.

Frame differencing is simply the wrong tool. Real active-speaker detection
(TalkNet, SyncNet, Light-ASD) learns audio-visual synchrony rather than motion
magnitude, and needs a trained model. Stereo panning would have been a shortcut,
but this recording is mono duplicated across both channels (L/R correlation
0.9993), so there is no direction to recover either.

The split-view feature does not depend on any of this - it needs two faces
located in one frame, which works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .decode import frames_at, iter_frames, probe
from .framing import FaceTracker


@dataclass
class Seat:
    """A place someone sits, in fractions of frame width and height."""
    index: int
    center_x: float
    center_y: float
    width: float
    height: float
    samples: int = 0

    def mouth_box(self, frame_w: int, frame_h: int) -> tuple:
        """
        The lower-middle of the face box.

        Mouth landmarks would be better, but every landmark model is a download.
        The lower third of a detected face box contains the mouth reliably
        enough for a motion measurement, and it excludes the eyes, which blink
        and would otherwise register as speech.
        """
        fw = self.width * frame_w
        fh = self.height * frame_h
        cx = self.center_x * frame_w
        cy = self.center_y * frame_h

        x0 = int(cx - fw * 0.30)
        x1 = int(cx + fw * 0.30)
        y0 = int(cy + fh * 0.05)
        y1 = int(cy + fh * 0.45)
        return (max(0, x0), max(0, y0), min(frame_w, x1), min(frame_h, y1))


@dataclass
class ActivityTrack:
    """Mouth-motion energy per seat, sampled on a fixed grid."""
    times: List[float]
    energy: dict            # seat index -> list of floats, same length as times
    seats: List[Seat]
    fps: float

    def smoothed(self, window: int = 7) -> dict:
        out = {}
        for idx, values in self.energy.items():
            arr = np.array(values, dtype=float)
            if len(arr) >= window:
                kernel = np.ones(window) / window
                arr = np.convolve(np.pad(arr, window // 2, mode="edge"), kernel, mode="valid")[:len(values)]
            out[idx] = arr
        return out


def find_seats(video_path: str, sample_times: Optional[List[float]] = None,
               analysis_width: int = 640, tolerance: float = 0.10,
               min_share: float = 0.35) -> List[Seat]:
    """
    Work out where people are sitting by sampling across the whole video.

    Detections are grouped by horizontal position: anything within `tolerance`
    of an existing group joins it. A group then has to appear in `min_share` of
    the samples to count as a seat.

    A share rather than a fixed count, because the same absolute threshold means
    very different things at 30 samples and at 12: a fixed "seen at least three
    times" let a false positive on a table edge through as a third person on a
    twelve-sample clip, which would have handed split-view two panels of the
    same man.
    """
    info = probe(video_path)
    duration = info.duration or 0.0

    if sample_times is None:
        sample_times = list(np.linspace(1.0, max(2.0, duration - 1.0), 30))

    tracker = FaceTracker()
    groups: List[dict] = []

    for _, small in frames_at(video_path, sample_times, width=analysis_width):
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        for det in tracker.detect(small, gray):
            if det.source != "cascade":
                continue                     # seats should come from real detections
            x, y, bw, bh = det.box
            sh, sw = small.shape[:2]
            entry = {
                "cx": (x + bw / 2) / sw,
                "cy": (y + bh / 2) / sh,
                "w": bw / sw,
                "h": bh / sh,
            }
            for g in groups:
                if abs(float(np.median(g["cx"])) - entry["cx"]) < tolerance:
                    for k in ("cx", "cy", "w", "h"):
                        g[k].append(entry[k])
                    break
            else:
                groups.append({k: [v] for k, v in entry.items()})

    needed = max(3, int(round(len(sample_times) * min_share)))
    groups = [g for g in groups if len(g["cx"]) >= needed]
    groups.sort(key=lambda g: float(np.median(g["cx"])))

    # median, not mean: people lean forward and gesture, and a single detection
    # that lands on a raised hand or a shifted posture drags an average enough
    # to visibly mis-frame the panel. The measured symptom was one person
    # sitting 12 points lower in his panel than the other.
    return [
        Seat(index=i,
             center_x=float(np.median(g["cx"])),
             center_y=float(np.median(g["cy"])),
             width=float(np.median(g["w"])),
             height=float(np.median(g["h"])),
             samples=len(g["cx"]))
        for i, g in enumerate(groups)
    ]


def seats_for_range(video_path: str, start: float, end: float,
                    samples: int = 12) -> List[Seat]:
    """
    Seat positions measured inside one clip's own time range.

    Seats found across a whole episode describe where people sit on average,
    which is not where they are sitting during any particular twelve seconds.
    Framing a clip from the episode-wide median left one panel's face 12 points
    lower than the other's; measuring within the clip cut that to 5, and put the
    lower panel exactly on its target. The crop is still fixed for the duration
    of the clip - this changes which fixed position, not whether it moves.
    """
    times = list(np.linspace(start, max(start + 0.5, end), samples))
    return find_seats(video_path, sample_times=times)


def mouth_activity(video_path: str, seats: List[Seat], sample_fps: float = 10.0,
                   analysis_width: int = 640) -> ActivityTrack:
    """
    Measure how much each seat's mouth region changes over time.

    Frames are read sequentially rather than sought, because this needs every
    sample and seeking per frame is far slower. The measure is the mean absolute
    difference from the previous sample, divided by the region's own mean
    brightness so that a brightly lit face does not score higher than a dim one
    just for being brighter.
    """
    info = probe(video_path)
    fps = info.fps or 30.0
    step = max(1, int(round(fps / sample_fps)))

    times: List[float] = []
    energy = {s.index: [] for s in seats}
    previous = {s.index: None for s in seats}

    for _, t, small in iter_frames(video_path, step=step, width=analysis_width):
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sh, sw = gray.shape[:2]

        times.append(t)
        for seat in seats:
            x0, y0, x1, y1 = seat.mouth_box(sw, sh)
            patch = gray[y0:y1, x0:x1].astype(np.float32)
            prev = previous[seat.index]
            if patch.size == 0 or prev is None or prev.shape != patch.shape:
                energy[seat.index].append(0.0)
            else:
                diff = np.abs(patch - prev).mean()
                energy[seat.index].append(float(diff / (patch.mean() + 1e-6)))
            previous[seat.index] = patch

    return ActivityTrack(times=times, energy=energy, seats=seats, fps=sample_fps)


def active_speaker_series(track: ActivityTrack, margin: float = 1.25,
                          silence_percentile: float = 35.0) -> List[Optional[int]]:
    """
    Turn the per-seat energy into one answer per sample: seat index, or None.

    Two guards keep it stable. A seat only takes over if its energy exceeds the
    runner-up by `margin`, so near-ties leave the previous answer standing
    rather than flickering. And if every seat is below its own quiet baseline,
    the answer is None - during a pause nobody is speaking, and claiming
    otherwise would make the framing chase noise.
    """
    smoothed = track.smoothed()
    if not smoothed:
        return []

    # each seat is scored against its own history, since face size and lighting
    # differ between seats and raw energies are not comparable
    normalised = {}
    floors = {}
    for idx, arr in smoothed.items():
        median = np.median(arr) or 1e-6
        normalised[idx] = arr / median
        floors[idx] = np.percentile(arr, silence_percentile) / median

    n = len(next(iter(normalised.values())))
    order = sorted(normalised)
    out: List[Optional[int]] = []
    current: Optional[int] = None

    for i in range(n):
        values = [(idx, normalised[idx][i]) for idx in order]
        values.sort(key=lambda kv: kv[1], reverse=True)

        top_idx, top_val = values[0]
        runner_up = values[1][1] if len(values) > 1 else 0.0

        if top_val < floors[top_idx] * 1.05:
            current = None
        elif runner_up <= 0 or top_val > runner_up * margin:
            current = top_idx
        # otherwise: too close to call, keep whatever was already showing

        out.append(current)

    return out


def split_view_crops(seats: List[Seat], src_w: int, src_h: int,
                     panels: int = 2, zoom: float = 1.9,
                     face_target: float = 0.36) -> List[tuple]:
    """
    Crop rectangles for a stacked split-view, one per seat, as (x, y, w, h).

    This is the feature paid clippers only manage when the source happens to
    contain an angle with everyone in it. Here both crops come out of the same
    static wide shot, which is how most one-camera podcasts are actually
    recorded.

    Each panel is 9:8 - half of a 9:16 frame - so two stacked panels fill the
    vertical output exactly. `zoom` sets how tightly each crop sits around its
    face; the rectangle is clamped inside the frame, and shifted rather than
    shrunk when it would fall outside, so both panels keep the same scale and
    the two people end up the same size on screen.
    """
    # when more seats were found than there are panels, keep the ones seen most
    # often - a spurious seat is by definition the one detected least
    if len(seats) > panels:
        seats = sorted(sorted(seats, key=lambda s: -s.samples)[:panels],
                       key=lambda s: s.center_x)

    panel_aspect = 9.0 / (16.0 / panels)

    # size the crop from the face, then make every panel identical so nobody
    # ends up looking closer to the camera than they are
    face_heights = [s.height * src_h for s in seats[:panels]]
    crop_h = min(src_h, max(face_heights) * zoom * 2.2)
    crop_w = crop_h * panel_aspect
    if crop_w > src_w:
        crop_w = src_w
        crop_h = crop_w / panel_aspect

    crops = []
    for seat in seats[:panels]:
        cx = seat.center_x * src_w
        cy = seat.center_y * src_h
        x = cx - crop_w / 2
        y = cy - crop_h * face_target
        x = max(0, min(x, src_w - crop_w))
        y = max(0, min(y, src_h - crop_h))
        crops.append((int(x), int(y), int(crop_w) & ~1, int(crop_h) & ~1))

    return crops


def dominant_seat(track: ActivityTrack, series: List[Optional[int]],
                  start: float, end: float) -> Optional[int]:
    """Which seat held the floor for most of a time range."""
    counts = {}
    for t, seat in zip(track.times, series):
        if seat is not None and start <= t < end:
            counts[seat] = counts.get(seat, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)
