import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video import cover
from ai_clipper.video.framing import CropPath, Keyframe


def path_of(*rows):
    """rows: (t, center_x, confident, motion)"""
    return CropPath(
        keyframes=[Keyframe(t, cx, ok, motion) for t, cx, ok, motion in rows],
        shots=[],
        detection_rate=sum(1 for r in rows if r[2]) / max(1, len(rows)),
    )


def steady(t0, n, step=0.5, center=0.5, confident=True, motion=1.0):
    return [(t0 + i * step, center, confident, motion) for i in range(n)]


def test_nothing_to_choose_from():
    assert cover.pick_time(None, 0, 10) is None
    assert cover.pick_time(path_of(), 0, 10) is None


def test_a_clip_with_no_confident_frame_admits_it():
    """
    Falling back to the midpoint here would be indistinguishable from having
    found a good frame, and the run report would claim something it did not do.
    """
    frames = steady(0, 20, confident=False)
    assert cover.pick_time(path_of(*frames), 0, 10) is None


def test_the_first_and_last_second_are_avoided():
    frames = steady(0, 24, step=0.5)
    at = cover.pick_time(path_of(*frames), 0, 12, edge=1.0)
    assert at is not None
    assert 1.0 <= at <= 11.0


def test_a_settled_moment_beats_a_busy_one():
    frames = (
        steady(0, 6, center=0.5, motion=40.0)      # someone gesturing
        + steady(3.0, 6, center=0.5, motion=0.5)   # settled
        + steady(6.0, 6, center=0.5, motion=40.0)
    )
    at = cover.pick_time(path_of(*frames), 0, 9, edge=0.5)
    assert 3.0 <= at <= 5.5


def test_the_middle_of_a_pan_is_not_chosen():
    panning = [(i * 0.5, 0.2 + i * 0.05, True, 1.0) for i in range(10)]
    settled = [(5.0 + i * 0.5, 0.7, True, 1.0) for i in range(10)]
    at = cover.pick_time(path_of(*(panning + settled)), 0, 10, edge=0.5)
    assert at >= 5.0


def test_an_earlier_settled_moment_wins_a_tie():
    frames = steady(0, 40, step=0.25, motion=1.0)
    at = cover.pick_time(path_of(*frames), 0, 10, edge=0.5)
    assert at < 5.0


def test_a_clip_shorter_than_the_edge_margin_still_gets_a_frame():
    frames = steady(0, 4, step=0.2)
    at = cover.pick_time(path_of(*frames), 0, 0.8, edge=1.0)
    assert at is not None


def test_the_crop_matches_what_the_video_does_at_that_moment():
    frames = [(0.0, 0.5, True, 1.0), (10.0, 0.5, True, 1.0)]
    p = path_of(*frames)
    # centred face, 1280-wide scaled frame, 720-wide output
    assert cover.crop_x(p, 5.0, 1280, 720) == 280


def test_the_crop_window_cannot_leave_the_frame():
    p = path_of((0.0, 0.02, True, 1.0), (10.0, 0.98, True, 1.0))
    assert cover.crop_x(p, 0.0, 1280, 720) == 0
    assert cover.crop_x(p, 10.0, 1280, 720) == 560


def test_no_crop_path_means_a_centred_crop():
    assert cover.crop_x(None, 3.0, 1280, 720) == 280
