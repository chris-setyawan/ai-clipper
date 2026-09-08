"""
Work out where the crop window should sit, frame by frame.

The job: turn a landscape source into a vertical clip without cutting the
speaker's head off. Three things have to be true at once.

  1. The crop follows whoever is on screen.
  2. It moves smoothly, not in the jittery way raw face detection does.
  3. It jumps instantly on a shot change, because gliding across a hard cut
     reads as a mistake.

Detection runs on downscaled frames a few times a second - faces do not move
fast enough to justify running it on every frame, and this keeps a three-minute
episode's analysis to a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .decode import iter_frames, probe
from .shots import Shot, detect_shots


@dataclass
class Keyframe:
    t: float
    center_x: float   # 0..1, fraction of source width
    confident: bool   # False = no face found, holding the last known position
    # mean pixel change since the previous sample. -1 means not measured, which
    # is not the same as "did not move" - a path built without motion data must
    # not have every shot mistaken for a held graphic.
    motion: float = -1.0


@dataclass
class CropPath:
    keyframes: List[Keyframe]
    shots: List[Shot]
    detection_rate: float   # fraction of sampled frames with a face

    def center_at(self, t: float) -> float:
        if not self.keyframes:
            return 0.5
        if t <= self.keyframes[0].t:
            return self.keyframes[0].center_x
        for a, b in zip(self.keyframes, self.keyframes[1:]):
            if a.t <= t <= b.t:
                span = b.t - a.t
                if span <= 0:
                    return a.center_x
                f = (t - a.t) / span
                return a.center_x + (b.center_x - a.center_x) * f
        return self.keyframes[-1].center_x


def _cascade_api():
    """
    (CascadeClassifier, path to the bundled cascade XMLs).

    OpenCV 5 reorganised the Python bindings and no longer exposes
    CascadeClassifier or cv2.data at the top level, so importing it and calling
    cv2.CascadeClassifier fails with a bare AttributeError several frames deep.
    Look in both places, and if neither works say what is actually wrong.
    """
    classifier = getattr(cv2, "CascadeClassifier", None)
    if classifier is None:
        classifier = getattr(getattr(cv2, "objdetect", None), "CascadeClassifier", None)

    data = getattr(cv2, "data", None)
    base = getattr(data, "haarcascades", None) if data else None

    if classifier is None or base is None:
        raise RuntimeError(
            f"this OpenCV build ({getattr(cv2, '__version__', 'unknown')}) does not "
            f"expose the Haar cascades this project uses. Install the 4.x line:\n"
            f"    pip install 'opencv-python-headless<5'"
        )
    return classifier, base


def skin_fraction(bgr, box) -> float:
    """
    Fraction of pixels inside a box that look like skin, in YCrCb.

    This is what separates a face from a microphone. Haar cascades key on
    light/dark gradient patterns, and a dark mic capsule against a bright
    jacket produces exactly the pattern they look for. Skin tone does not care
    about gradients, so it rejects that case cheaply. The Cr/Cb range used here
    is the standard one and holds across skin tones, because chroma varies far
    less between people than luminance does.
    """
    x, y, w, h = box
    patch = bgr[max(0, y):y + h, max(0, x):x + w]
    if patch.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float(np.count_nonzero(mask)) / mask.size


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


@dataclass
class Detection:
    box: tuple
    skin: float
    score: float
    source: str = "cascade"   # "cascade" (confident) or "skin" (fallback)

    @property
    def center_x_px(self) -> float:
        return self.box[0] + self.box[2] / 2


class FaceTracker:
    """
    Haar cascades, which ship inside opencv-python - no model download, which
    keeps the project installable from PyPI alone.

    Cascades alone are not enough. Two rules make them usable here:

      - Run every detector and rank the results, rather than stopping at the
        first that returns something. Running frontal first and only falling
        back to profile meant a false positive from the frontal pass hid the
        real, side-on face entirely.
      - Score candidates on plausibility - skin content, vertical position,
        size - instead of trusting the cascade's word for it.
    """

    MIN_SKIN = 0.25          # a real face is mostly skin inside the box
    MAX_CENTER_Y = 0.72      # faces sit in the upper part of a podcast frame
    MIN_REL_SIZE = 0.02      # relative to frame width
    MAX_REL_SIZE = 0.60

    def __init__(self):
        classifier, base = _cascade_api()
        self.frontal = classifier(base + "haarcascade_frontalface_default.xml")
        self.alt = classifier(base + "haarcascade_frontalface_alt2.xml")
        self.profile = classifier(base + "haarcascade_profileface.xml")

    def _raw_candidates(self, gray) -> List[tuple]:
        w = gray.shape[1]
        found = []
        for cascade in (self.frontal, self.alt):
            for f in cascade.detectMultiScale(gray, 1.15, 5, minSize=(28, 28)):
                found.append(tuple(int(v) for v in f))

        for f in self.profile.detectMultiScale(gray, 1.15, 5, minSize=(28, 28)):
            found.append(tuple(int(v) for v in f))

        # the profile cascade is trained facing one way only; a mirrored pass
        # picks up faces turned the other direction
        flipped = cv2.flip(gray, 1)
        for (x, y, fw, fh) in self.profile.detectMultiScale(flipped, 1.15, 5, minSize=(28, 28)):
            found.append((int(w - x - fw), int(y), int(fw), int(fh)))

        # merge boxes several cascades agree on; agreement is itself evidence
        merged: List[tuple] = []
        votes: List[int] = []
        for box in found:
            for i, kept in enumerate(merged):
                if _iou(box, kept) > 0.4:
                    votes[i] += 1
                    break
            else:
                merged.append(box)
                votes.append(1)

        return list(zip(merged, votes))

    def _skin_blobs(self, bgr) -> List[Detection]:
        """
        Fallback for when the cascades find nothing.

        Haar cascades are trained on frontal and one profile orientation; a head
        turned three-quarters away matches neither, which on a two-person podcast
        is a large share of the footage. Skin colour does not depend on
        orientation, so thresholding for it and taking the largest blob in the
        upper frame recovers those frames. Less precise than a cascade hit -
        hence the separate source label - but far better than holding a stale
        position for seconds at a time.
        """
        fh, fw = bgr.shape[:2]
        ycrcb = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        min_area = 0.0015 * fw * fh

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if (y + h / 2) / fh > self.MAX_CENTER_Y:
                continue
            out.append(Detection(box=(x, y, w, h), skin=1.0, score=area, source="skin"))

        out.sort(key=lambda d: d.score, reverse=True)
        return out

    def detect(self, bgr, gray) -> List[Detection]:
        fh, fw = gray.shape[:2]
        out: List[Detection] = []

        for box, vote in self._raw_candidates(gray):
            x, y, w, h = box

            rel_size = w / fw
            if not (self.MIN_REL_SIZE <= rel_size <= self.MAX_REL_SIZE):
                continue
            if (y + h / 2) / fh > self.MAX_CENTER_Y:
                continue

            skin = skin_fraction(bgr, box)
            if skin < self.MIN_SKIN:
                continue

            # bigger, skin-rich, multiply-confirmed boxes win
            score = (w * h) * (0.5 + skin) * (1.0 + 0.25 * (vote - 1))
            out.append(Detection(box=box, skin=skin, score=score))

        out.sort(key=lambda d: d.score, reverse=True)
        if out:
            return out

        return self._skin_blobs(bgr)


def _smooth(values: List[float], window: int = 9) -> List[float]:
    """Median filter to kill single-frame outliers, then a mean pass to soften."""
    if len(values) < 3:
        return values

    arr = np.array(values, dtype=float)
    k = min(window, len(arr) if len(arr) % 2 else len(arr) - 1)
    if k >= 3:
        padded = np.pad(arr, k // 2, mode="edge")
        arr = np.array([np.median(padded[i:i + k]) for i in range(len(arr))])

    k2 = min(5, len(arr) if len(arr) % 2 else len(arr) - 1)
    if k2 >= 3:
        padded = np.pad(arr, k2 // 2, mode="edge")
        kernel = np.ones(k2) / k2
        arr = np.convolve(padded, kernel, mode="valid")

    return arr.tolist()


def _ease_toward(targets: List[float], alpha: float = 0.22, deadzone: float = 0.010) -> List[float]:
    """
    Move the frame toward the target the way a camera operator would: stay put
    while the subject is basically still, and glide when they actually move.

    An earlier version held position and then jumped straight to the new target
    once the deadzone was exceeded. That is what made the motion look stepped -
    the frame was either frozen or teleporting, never travelling. Here the
    deadzone gates whether to move at all, and when it does move the position
    approaches the target exponentially, so every step is a fraction of the
    remaining distance instead of all of it.

    alpha is the fraction of the remaining gap closed per sample. At 5 samples a
    second, 0.22 settles in about a second - fast enough to keep up with someone
    leaning across a table, slow enough that it never snaps.
    """
    if not targets:
        return targets

    out = [targets[0]]
    pos = targets[0]
    for t in targets[1:]:
        if abs(t - pos) > deadzone:
            pos += (t - pos) * alpha
        out.append(pos)
    return out


def jerkiness(values: List[float]) -> float:
    """
    Mean absolute second difference of a path - how much the motion changes
    direction or speed. Useful as a plain number for comparing smoothing
    settings instead of squinting at renders.
    """
    if len(values) < 3:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.mean(np.abs(np.diff(arr, n=2))))


STATIC_SHOT_MOTION = 1.2      # mean 0-255 pixel change per sample


def is_static_shot(path: "CropPath", shot) -> bool:
    """
    True when a shot is a held graphic rather than a camera on a person.

    Measured as median frame-to-frame pixel change: a poster or title card sits
    near zero.

    A face in the shot overrules the measurement, and has to. "Even a
    still-sitting speaker produces several times more motion" was wrong: on a
    locked-off camera with flat lighting, a person listening quietly measures
    the same as a poster. Of the three shots motion alone called static on the
    sample episode, two had a confident face in every single sampled frame -
    and those are the ones that came back rendered as a narrow strip between two
    blurred bands, because a graphic gets letterboxed whole instead of cropped.
    A held graphic with a detectable face in it is rare; a talking head
    mistaken for a poster ruins the clip.
    """
    in_shot = [k for k in path.keyframes if shot.start <= k.t < shot.end]
    if any(k.confident for k in in_shot):
        return False
    motions = [k.motion for k in in_shot if k.motion >= 0]
    if len(motions) < 3:
        return False
    return float(np.median(motions)) < STATIC_SHOT_MOTION


def segment_modes(path: "CropPath", start: float, end: float,
                  base_mode: str, graphic_mode: str = "blur") -> List[tuple]:
    """
    Split a clip into (start, end, mode) runs.

    A held graphic cannot be face-cropped and cannot be centre-cropped either -
    a 9:16 window shows about a third of a 16:9 frame, so a film poster loses
    its edges whatever you do with the position. It has to be fitted whole and
    letterboxed, which is a different filter chain, which means the clip has to
    be rendered in pieces and joined.

    Runs shorter than MIN_RUN are folded into their neighbour: switching
    framing briefly looks like a glitch, not an intention. It was half a second,
    which still let a 1.1-second letterboxed flash through in the middle of an
    interview; a mode change has to last long enough to read as deliberate.
    """
    MIN_RUN = 1.5
    runs: List[tuple] = []
    for shot in path.shots:
        a, b = max(shot.start, start), min(shot.end, end)
        if b <= a:
            continue
        mode = graphic_mode if is_static_shot(path, shot) else base_mode
        if runs and runs[-1][2] == mode:
            runs[-1] = (runs[-1][0], b, mode)
        else:
            runs.append((a, b, mode))

    merged: List[tuple] = []
    for a, b, mode in runs:
        if merged and (b - a) < MIN_RUN:
            merged[-1] = (merged[-1][0], b, merged[-1][2])
        elif merged and merged[-1][2] == mode:
            merged[-1] = (merged[-1][0], b, mode)
        else:
            merged.append((a, b, mode))

    if not merged:
        merged = [(start, end, base_mode)]
    else:
        merged[0] = (start, merged[0][1], merged[0][2])
        merged[-1] = (merged[-1][0], end, merged[-1][2])
    return merged


def flatten_per_shot(path: "CropPath") -> "CropPath":
    """
    Collapse the path to one fixed position per shot.

    On multi-camera footage this is often better than tracking: within a shot
    the camera never moves at all, and the frame still lands correctly on
    whoever is in it, because the position comes from the detections in that
    shot rather than from the centre of the frame. All the repositioning happens
    on cuts, where the viewer expects a jump anyway.
    """
    out: List[Keyframe] = []
    for shot in path.shots:
        group = [k for k in path.keyframes if shot.start <= k.t < shot.end]
        if not group:
            continue
        if is_static_shot(path, shot):
            # a graphic gets the middle of the frame. It still loses the sides
            # in a 9:16 crop, but centred reads as a deliberate framing choice
            # rather than a crop through someone's printed face.
            fixed = 0.5
        else:
            confident = [k.center_x for k in group if k.confident] or [k.center_x for k in group]
            fixed = float(np.median(confident))
        for k in group:
            out.append(Keyframe(k.t, fixed, k.confident))

    return CropPath(keyframes=out, shots=path.shots, detection_rate=path.detection_rate)


def build_crop_path(
    video_path: str,
    target_aspect: float = 9 / 16,
    sample_fps: float = 5.0,
    shots: Optional[List[Shot]] = None,
    analysis_width: int = 480,
) -> CropPath:
    """
    Analyse the whole video once and return a crop centre for every sample time.

    Smoothing is applied per shot rather than across the whole video, so a cut
    never blends the positions of two different camera angles.
    """
    if shots is None:
        shots = detect_shots(video_path)

    info = probe(video_path)
    fps = info.fps or 30.0
    src_w, src_h = info.width, info.height
    step = max(1, int(round(fps / sample_fps)))

    tracker = FaceTracker()

    # half the crop width, as a fraction of source width - used to keep the
    # window inside the frame
    crop_w_frac = min(1.0, (src_h * target_aspect) / src_w)
    half = crop_w_frac / 2

    samples = []      # (time, center_x or None, motion)
    last_center = None
    previous_gray = None
    hits = 0
    total = 0

    for _, t, small in iter_frames(video_path, step=step, width=analysis_width):
        total += 1
        plain = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(plain)

        # how much the picture itself is moving. A poster or title card held on
        # screen barely changes; a person talking never stops changing. The
        # difference is what tells a graphic apart from a camera shot, and a
        # graphic should not be cropped to a face - even though a film poster
        # genuinely has faces printed on it, which is exactly how the opening
        # shot of the sample ended up cropped through the middle of the artwork.
        if previous_gray is not None and previous_gray.shape == plain.shape:
            motion = float(np.abs(plain.astype(np.float32)
                                  - previous_gray.astype(np.float32)).mean())
        else:
            motion = 0.0
        previous_gray = plain
        faces = tracker.detect(small, gray)

        center = None
        if faces:
            best = faces[0]
            # prefer whoever we were already following: if a previous target is
            # still detected nearby, stay on them instead of hopping to a
            # marginally better-scoring face
            if last_center is not None:
                for d in faces:
                    cx = d.center_x_px / small.shape[1]
                    if abs(cx - last_center) < 0.12 and d.score > best.score * 0.5:
                        best = d
                        break
            center = best.center_x_px / small.shape[1]
            last_center = center
            hits += 1

        samples.append((t, center, motion))

    # fill gaps: hold the last confident position, or fall back to centre
    filled = []
    last = 0.5
    for t, c, motion in samples:
        if c is None:
            filled.append((t, last, False, motion))
        else:
            last = c
            filled.append((t, c, True, motion))

    # smooth inside each shot independently
    keyframes: List[Keyframe] = []
    for shot in shots:
        group = [(t, c, ok, m) for (t, c, ok, m) in filled if shot.start <= t < shot.end]
        if not group:
            continue
        centers = _smooth([c for (_, c, _, _) in group])
        centers = _ease_toward(centers)
        for (t, _, ok, motion), c in zip(group, centers):
            clamped = min(max(c, half), 1.0 - half)
            keyframes.append(Keyframe(t, clamped, ok, motion))

    keyframes.sort(key=lambda k: k.t)
    rate = hits / total if total else 0.0
    return CropPath(keyframes=keyframes, shots=shots, detection_rate=rate)


def recommend_mode(path: "CropPath", min_detection: float = 0.40,
                   cut_heavy_seconds: float = 8.0) -> tuple:
    """
    Pick a framing mode from what the analysis actually found.

    (mode, reason). The user can always override, but most people should not
    have to think about this - the right answer is deducible from the footage:

      few detections    -> blur. Nothing reliable to frame on, so don't pretend.
      frequent cuts     -> per_shot. The editor already framed each shot; hold
                           position and let the cuts do the repositioning.
      long takes        -> face. One camera running for minutes means the
                           subject will move, and only tracking keeps up.
    """
    if path.detection_rate < min_detection:
        return "blur", (
            f"a face was found in only {path.detection_rate * 100:.0f}% of frames, "
            f"too few to frame on reliably"
        )

    durations = [s.duration for s in path.shots] or [999.0]
    median_shot = float(np.median(durations))

    if median_shot < cut_heavy_seconds:
        return "per_shot", (
            f"shots change every {median_shot:.1f}s on average, so the edit is "
            f"already doing the reframing - holding still inside each shot is smoother"
        )

    return "face", (
        f"shots run {median_shot:.1f}s on average, long enough that the subject "
        f"will move within them"
    )


def to_sendcmd(
    path: CropPath,
    clip_start: float,
    clip_end: float,
    scaled_width: int,
    out_width: int,
    rate: float = 20.0,
) -> str:
    """
    Render the crop path as an ffmpeg sendcmd script.

    The crop filter's x is re-sent at `rate` times a second. Because the path is
    already smoothed, each step is sub-pixel to a few pixels, so the movement
    reads as continuous rather than stepped.
    """
    max_x = max(0, scaled_width - out_width)
    lines = []
    t = clip_start
    step = 1.0 / rate

    while t < clip_end:
        center = path.center_at(t)
        x = int(round(center * scaled_width - out_width / 2))
        x = min(max(x, 0), max_x)
        lines.append(f"{t - clip_start:.3f} crop x {x};")
        t += step

    return "\n".join(lines) + "\n"
