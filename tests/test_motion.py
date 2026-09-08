import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video.framing import (
    STATIC_SHOT_MOTION,
    CropPath,
    Keyframe,
    _ease_toward,
    flatten_per_shot,
    jerkiness,
    recommend_mode,
)
from ai_clipper.video.shots import Shot


def test_easing_holds_still_inside_the_deadzone():
    out = _ease_toward([0.5, 0.502, 0.498, 0.501], deadzone=0.01)
    assert all(abs(v - 0.5) < 1e-9 for v in out), "tiny jitter must not move the frame"


def test_easing_approaches_a_moved_target_gradually():
    out = _ease_toward([0.2] + [0.8] * 20, alpha=0.22, deadzone=0.01)
    steps = [abs(b - a) for a, b in zip(out, out[1:])]
    assert steps[0] < 0.6 * 0.5, "must not teleport to the new target in one step"
    assert steps[0] > steps[5], "each step should close less ground than the last"
    assert abs(out[-1] - 0.8) < 0.02, "should still arrive"


def test_easing_is_smoother_than_jumping():
    targets = [0.2] * 5 + [0.8] * 5 + [0.3] * 5
    jumped = targets
    eased = _ease_toward(targets)
    assert jerkiness(eased) < jerkiness(jumped)


def test_per_shot_is_constant_within_a_shot():
    kfs = [Keyframe(t / 10, 0.3 + t / 100, True) for t in range(20)]
    path = CropPath(keyframes=kfs, shots=[Shot(0.0, 1.0), Shot(1.0, 2.0)], detection_rate=1.0)
    flat = flatten_per_shot(path)

    first = [k.center_x for k in flat.keyframes if k.t < 1.0]
    second = [k.center_x for k in flat.keyframes if k.t >= 1.0]
    assert len(set(first)) == 1, "no movement allowed inside a shot"
    assert len(set(second)) == 1
    assert first[0] != second[0], "each shot gets its own position"


def test_per_shot_uses_confident_detections_only():
    kfs = [
        Keyframe(0.0, 0.4, True),
        Keyframe(0.1, 0.4, True),
        Keyframe(0.2, 0.9, False),   # held value, should be ignored
    ]
    path = CropPath(keyframes=kfs, shots=[Shot(0.0, 1.0)], detection_rate=1.0)
    flat = flatten_per_shot(path)
    assert abs(flat.keyframes[0].center_x - 0.4) < 1e-9


def test_per_shot_falls_back_when_nothing_was_confident():
    kfs = [Keyframe(0.0, 0.6, False), Keyframe(0.1, 0.6, False)]
    path = CropPath(keyframes=kfs, shots=[Shot(0.0, 1.0)], detection_rate=0.0)
    flat = flatten_per_shot(path)
    assert abs(flat.keyframes[0].center_x - 0.6) < 1e-9


def test_jerkiness_is_zero_for_a_straight_line():
    assert jerkiness([0.1, 0.2, 0.3, 0.4]) < 1e-9


def _path(shot_lengths, detection_rate=0.9):
    shots, kfs, t = [], [], 0.0
    for length in shot_lengths:
        shots.append(Shot(t, t + length))
        kfs.append(Keyframe(t + 0.1, 0.5, True))
        t += length
    return CropPath(keyframes=kfs, shots=shots, detection_rate=detection_rate)


def test_recommend_blur_when_detection_is_poor():
    mode, reason = recommend_mode(_path([30, 30], detection_rate=0.1))
    assert mode == "blur"
    assert "10%" in reason


def test_recommend_per_shot_on_cut_heavy_footage():
    mode, _ = recommend_mode(_path([3, 4, 2, 5, 3]))
    assert mode == "per_shot"


def test_recommend_face_on_long_takes():
    mode, _ = recommend_mode(_path([120, 90, 150]))
    assert mode == "face"


def test_recommendation_always_explains_itself():
    for lengths in ([3, 4, 3], [120, 90]):
        _, reason = recommend_mode(_path(lengths))
        assert len(reason) > 20


def _path_with_motion(shot_specs):
    """
    shot_specs: (duration, motion) or (duration, motion, face_detected).

    Confidence defaults to "no face on a still shot", because that is what a
    held graphic looks like. A still shot *with* a face is a locked-off camera
    on a person, and there is a test below for exactly that.
    """
    shots, kfs, t = [], [], 0.0
    for spec in shot_specs:
        length, motion = spec[0], spec[1]
        confident = spec[2] if len(spec) > 2 else motion >= STATIC_SHOT_MOTION
        shots.append(Shot(t, t + length))
        for i in range(5):
            kfs.append(Keyframe(t + i * length / 5, 0.5, confident, motion))
        t += length
    return CropPath(keyframes=kfs, shots=shots, detection_rate=0.9)


def test_a_held_graphic_is_recognised_as_static():
    from ai_clipper.video.framing import is_static_shot

    path = _path_with_motion([(4.0, 0.1), (4.0, 4.5)])
    assert is_static_shot(path, path.shots[0]), "a still frame should read as static"
    assert not is_static_shot(path, path.shots[1]), "a talking head should not"


def test_a_graphic_shot_is_centred_not_face_cropped():
    from ai_clipper.video.framing import flatten_per_shot

    path = _path_with_motion([(4.0, 0.1)])
    for k in path.keyframes:
        k.center_x = 0.2                     # off-centre artwork, no face found
    assert abs(flatten_per_shot(path).keyframes[0].center_x - 0.5) < 1e-9


def test_a_still_camera_on_a_person_is_not_a_graphic():
    """
    The bug behind a clip that came back as a narrow strip between two blurred
    bands. A locked-off camera on someone sitting quietly measures as little
    frame-to-frame change as a poster does, so motion alone called it a graphic
    and the whole 16:9 frame was letterboxed into a 9:16 box. Two of the three
    shots motion called static on the sample episode had a confident face in
    every sampled frame.
    """
    from ai_clipper.video.framing import is_static_shot, segment_modes

    path = _path_with_motion([(4.0, 0.1, True)])
    assert not is_static_shot(path, path.shots[0])
    assert [m for _, _, m in segment_modes(path, 0.0, 4.0, "per_shot")] == ["per_shot"]


def test_segment_modes_isolates_the_graphic_run():
    from ai_clipper.video.framing import segment_modes

    path = _path_with_motion([(5.0, 4.0), (3.0, 0.1), (5.0, 4.0)])
    runs = segment_modes(path, 0.0, 13.0, "per_shot")
    assert [m for _, _, m in runs] == ["per_shot", "blur", "per_shot"]
    assert runs[0][0] == 0.0 and runs[-1][1] == 13.0, "runs must cover the clip"


def test_uniform_footage_produces_a_single_run():
    from ai_clipper.video.framing import segment_modes

    path = _path_with_motion([(5.0, 4.0), (5.0, 3.5)])
    runs = segment_modes(path, 0.0, 10.0, "per_shot")
    assert len(runs) == 1, "no need to split and rejoin when nothing changes"


def test_a_very_short_graphic_flash_is_not_split_out():
    from ai_clipper.video.framing import segment_modes

    path = _path_with_motion([(5.0, 4.0), (1.0, 0.1), (5.0, 4.0)])
    runs = segment_modes(path, 0.0, 10.3, "per_shot")
    assert len(runs) == 1, "switching framing for a few frames looks like a glitch"
