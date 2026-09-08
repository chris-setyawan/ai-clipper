import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video.render import RenderSpec, build_filter
from ai_clipper.video.speakers import Seat, split_view_crops


def seats():
    return [
        Seat(index=0, center_x=0.303, center_y=0.316, width=0.100, height=0.175, samples=29),
        Seat(index=1, center_x=0.784, center_y=0.328, width=0.093, height=0.163, samples=30),
    ]


def test_crops_stay_inside_the_frame():
    for x, y, w, h in split_view_crops(seats(), 1260, 720):
        assert x >= 0 and y >= 0
        assert x + w <= 1260
        assert y + h <= 720


def test_both_panels_are_the_same_size():
    crops = split_view_crops(seats(), 1260, 720)
    assert crops[0][2] == crops[1][2], "unequal widths would make one person look closer"
    assert crops[0][3] == crops[1][3]


def test_crop_dimensions_are_even():
    for _, _, w, h in split_view_crops(seats(), 1260, 720):
        assert w % 2 == 0 and h % 2 == 0, "yuv420p requires even dimensions"


def test_panels_match_the_target_panel_aspect():
    # two stacked panels of a 9:16 frame are 9:8 each
    for _, _, w, h in split_view_crops(seats(), 1260, 720):
        assert abs(w / h - 9 / 8) < 0.02


def test_each_panel_contains_its_own_face():
    crops = split_view_crops(seats(), 1260, 720)
    for seat, (x, y, w, h) in zip(seats(), crops):
        cx = seat.center_x * 1260
        cy = seat.center_y * 720
        assert x <= cx <= x + w, "the face must be inside its panel"
        assert y <= cy <= y + h


def test_a_panel_never_contains_the_other_persons_face():
    """
    A few pixels of shared background between panels is harmless; the same face
    appearing in both is not, because the viewer cannot tell the panels apart.
    """
    crops = split_view_crops(seats(), 1260, 720)
    for i, (x, _, w, _) in enumerate(crops):
        for j, seat in enumerate(seats()):
            if i == j:
                continue
            other_cx = seat.center_x * 1260
            assert not (x <= other_cx <= x + w), "a panel is showing the wrong person"


def test_filter_stacks_one_panel_per_crop():
    crops = split_view_crops(seats(), 1260, 720)
    chain = build_filter(RenderSpec(mode="split"), 720, 1280, None, split_crops=crops)
    assert chain.count("crop=") == 2
    assert "vstack=inputs=2" in chain
    assert "scale=720:640" in chain, "each panel is half the output height"


def test_split_mode_refuses_to_guess_with_one_person():
    from ai_clipper.video.render import render_clip

    with pytest.raises(ValueError) as err:
        render_clip("nonexistent.mp4", 0, 1, "/tmp/out.mp4",
                    RenderSpec(mode="split"), None, None, seats()[:1])
    assert "two seats" in str(err.value)


def test_face_target_controls_vertical_placement():
    """Each panel should put its face at the requested fraction of panel height."""
    src_w, src_h = 1260, 720
    for target in (0.30, 0.36, 0.45):
        crops = split_view_crops(seats(), src_w, src_h, face_target=target)
        for seat, (_, y, _, h) in zip(seats(), crops):
            face_y = seat.center_y * src_h
            placed = (face_y - y) / h
            assert abs(placed - target) < 0.02, f"{target}: got {placed:.3f}"


def test_two_people_at_different_heights_still_line_up():
    """
    The whole point of per-seat positioning: someone sitting lower in the frame
    must not end up sitting lower in their panel too.
    """
    low = [
        Seat(index=0, center_x=0.30, center_y=0.25, width=0.10, height=0.175),
        Seat(index=1, center_x=0.78, center_y=0.40, width=0.10, height=0.175),
    ]
    crops = split_view_crops(low, 1260, 720)
    placed = [((s.center_y * 720) - y) / h for s, (_, y, _, h) in zip(low, crops)]
    assert abs(placed[0] - placed[1]) < 0.02, "panels disagree on where the face sits"


def test_a_spurious_seat_is_dropped_before_splitting():
    """
    Three seats and two panels must not produce two panels of the same person.
    The extra seat is the one detected least often, so that is the one to drop.
    """
    three = [
        Seat(index=0, center_x=0.30, center_y=0.25, width=0.10, height=0.175, samples=12),
        Seat(index=1, center_x=0.39, center_y=0.65, width=0.05, height=0.080, samples=3),
        Seat(index=2, center_x=0.78, center_y=0.33, width=0.10, height=0.163, samples=12),
    ]
    crops = split_view_crops(three, 1260, 720)
    assert len(crops) == 2
    xs = [x for x, _, _, _ in crops]
    assert xs == sorted(xs), "panels must stay in left-to-right order"
    # the two kept panels must be the well-attested pair, not the neighbours
    assert xs[1] - xs[0] > 400, "both panels landed on the same person"
