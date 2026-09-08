import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video.framing import CropPath, Keyframe, _ease_toward, _smooth, to_sendcmd
from ai_clipper.video.render import RenderSpec, scaled_width_for, target_size
from ai_clipper.video.shots import Shot, shot_at


def path_with(centers):
    kfs = [Keyframe(i * 0.5, c, True) for i, c in enumerate(centers)]
    return CropPath(keyframes=kfs, shots=[Shot(0.0, 100.0)], detection_rate=1.0)


def test_center_interpolates_between_keyframes():
    p = path_with([0.2, 0.8])
    assert p.center_at(0.0) == 0.2
    assert p.center_at(0.5) == 0.8
    assert abs(p.center_at(0.25) - 0.5) < 1e-6


def test_center_before_and_after_the_path_is_clamped():
    p = path_with([0.3, 0.7])
    assert p.center_at(-5.0) == 0.3
    assert p.center_at(99.0) == 0.7


def test_deadzone_holds_position_through_small_moves():
    out = _ease_toward([0.5, 0.505, 0.51, 0.9], alpha=0.22, deadzone=0.05)
    assert out[:3] == [0.5, 0.5, 0.5], "small drifts should not move the frame"
    assert out[3] > 0.5, "a large move should start being followed"
    assert out[3] < 0.9, "but not jumped to in a single step"


def test_smoothing_removes_a_single_frame_spike():
    values = [0.5] * 6 + [0.95] + [0.5] * 6
    out = _smooth(values)
    assert max(out) < 0.7, "a one-sample outlier should be filtered out"


def test_sendcmd_stays_inside_the_frame():
    # centre pinned hard left, then hard right
    p = path_with([0.0, 1.0])
    script = to_sendcmd(p, clip_start=0.0, clip_end=1.0, scaled_width=2000, out_width=720, rate=10)
    xs = [int(line.split()[-1].rstrip(";")) for line in script.strip().splitlines()]
    assert min(xs) >= 0
    assert max(xs) <= 2000 - 720


def test_sendcmd_times_are_relative_to_the_clip():
    p = path_with([0.5, 0.5])
    script = to_sendcmd(p, clip_start=30.0, clip_end=30.5, scaled_width=2000, out_width=720, rate=10)
    first = script.strip().splitlines()[0]
    assert first.startswith("0.000"), "a clip cut from mid-episode must start its commands at zero"


def test_scaled_width_is_even_and_preserves_aspect():
    w = scaled_width_for(1048, 592, 1280)
    assert w % 2 == 0
    assert abs(w / 1280 - 1048 / 592) < 0.01


def test_target_size_for_each_ratio():
    assert target_size(RenderSpec(ratio="9:16", resolution="720p")) == (720, 1280)
    w, h = target_size(RenderSpec(ratio="1:1", resolution="720p"))
    assert w == h


def test_shot_lookup():
    shots = [Shot(0, 5), Shot(5, 10)]
    assert shot_at(shots, 2.0) == 0
    assert shot_at(shots, 7.0) == 1
    assert shot_at(shots, 999.0) == 1
