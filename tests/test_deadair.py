import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video import deadair
from ai_clipper.video.render import coalesce, plan_runs
from ai_clipper.video.render import RenderSpec


@dataclass
class W:
    start: float
    end: float
    text: str = "x"


def speech(*pairs):
    return [W(a, b) for a, b in pairs]


def test_a_clip_with_no_long_pause_is_left_whole():
    words = speech((0, 1), (1.1, 2), (2.2, 3))
    line = deadair.plan(words, 0, 3)
    assert line.cuts == 0
    assert line.ranges() == [(0, 3)]
    assert line.removed == 0


def test_a_long_pause_is_shortened_not_closed():
    words = speech((0, 10), (14, 30))
    line = deadair.plan(words, 0, 30)
    assert line.cuts == 1
    # 4s of silence, 0.4s of it kept
    assert round(line.removed, 2) == 3.6
    assert round(line.duration, 2) == 26.4


def test_a_short_pause_is_part_of_the_delivery():
    words = speech((0, 10), (11.2, 30))
    assert deadair.plan(words, 0, 30, min_gap=1.5).cuts == 0


def test_the_threshold_is_adjustable():
    words = speech((0, 10), (11.2, 30))
    assert deadair.plan(words, 0, 30, min_gap=1.0).cuts == 1


def test_the_room_at_either_end_is_never_taken():
    """
    A clip that opens on a consonant sounds clipped, and the tail is what keeps
    a loop from cutting off the last syllable. Only gaps between words count.
    """
    words = speech((5, 10), (11, 20))
    line = deadair.plan(words, 0, 30, min_gap=1.5)
    assert line.cuts == 0


def test_total_removal_is_capped():
    words = speech(*[(i * 10.0, i * 10.0 + 2.0) for i in range(10)])
    line = deadair.plan(words, 0, 100, max_fraction=0.1)
    assert line.removed <= 100 * 0.1 + 1e-6
    assert line.cuts >= 1


def test_the_biggest_silence_is_shortened_even_when_it_exceeds_the_budget():
    """
    The version that skipped an oversized gap left the one silence a viewer
    would actually leave over completely intact.
    """
    words = speech((0, 5), (35, 40))
    line = deadair.plan(words, 0, 40, max_fraction=0.1)
    assert line.cuts == 1
    assert round(line.removed, 2) == 4.0


def test_cuts_come_back_in_time_order():
    words = speech((0, 5), (10, 15), (25, 30), (40, 45))
    ranges = deadair.plan(words, 0, 45).ranges()
    assert ranges == sorted(ranges)
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert start > end


def test_remap_moves_later_words_earlier_by_what_was_cut():
    words = speech((0, 10), (14, 30))
    line = deadair.plan(words, 0, 30)
    assert line.remap(0) == 0
    assert round(line.remap(10), 2) == 10.0
    assert round(line.remap(14), 2) == round(14 - 3.6, 2)
    assert round(line.remap(30), 2) == round(line.duration, 2)


def test_captions_are_moved_onto_the_shortened_timeline():
    words = speech((0, 10), (14, 30))
    line = deadair.plan(words, 0, 30)
    moved = deadair.shift_words(words, line, W)
    assert round(moved[1].start, 2) == round(14 - 3.6, 2)
    assert [w.text for w in moved] == [w.text for w in words]


def test_describe_is_empty_when_nothing_was_cut():
    words = speech((0, 1), (1.1, 2))
    assert deadair.describe(deadair.plan(words, 0, 2)) == ""
    assert deadair.describe(None) == ""


# --- how the cuts reach ffmpeg ----------------------------------------------

def test_touching_runs_of_the_same_mode_are_merged():
    runs = [(0.0, 5.0, "face"), (5.0, 9.0, "face"), (9.0, 12.0, "blur")]
    assert coalesce(runs) == [(0.0, 9.0, "face"), (9.0, 12.0, "blur")]


def test_a_removed_gap_keeps_two_runs_apart():
    """
    The failure this guards against is silent: merge across the cut and the
    dead air is back in the output with every other part of the pipeline still
    reporting that it was removed.
    """
    runs = [(0.0, 5.0, "face"), (8.6, 20.0, "face")]
    assert coalesce(runs) == runs


def test_each_kept_span_becomes_its_own_piece():
    spec = RenderSpec(mode="crop")
    runs = plan_runs(None, spec, [(0.0, 5.0), (8.6, 20.0)])
    assert runs == [(0.0, 5.0, "crop"), (8.6, 20.0, "crop")]
