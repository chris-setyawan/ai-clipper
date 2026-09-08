import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.scoring.selection import Selection, overlaps


@dataclass
class FakeClip:
    start: float
    end: float
    score: float


def pool():
    return [
        FakeClip(0, 20, 9.0),
        FakeClip(0, 25, 8.0),      # same region as the first, re-cut
        FakeClip(30, 50, 8.5),
        FakeClip(60, 80, 8.2),
        FakeClip(90, 110, 7.0),
        FakeClip(120, 140, 6.5),
    ]


def test_build_picks_highest_scoring_non_overlapping_clips():
    sel = Selection.build(pool(), 3)
    assert [c.start for c in sel.chosen] == [0, 30, 60]
    assert all(not overlaps(a, b) for a in sel.chosen for b in sel.chosen if a is not b)


def test_chosen_clips_are_in_time_order():
    sel = Selection.build(pool(), 4)
    starts = [c.start for c in sel.chosen]
    assert starts == sorted(starts)


def test_regenerate_different_avoids_the_rejected_region():
    sel = Selection.build(pool(), 3)
    sel.regenerate(0, mode="different")
    assert all(c.start != 0 for c in sel.chosen), "should not re-offer the same moment"
    assert len(sel.chosen) == 3


def test_regenerate_retime_offers_the_same_moment_recut():
    sel = Selection.build(pool(), 3)
    replacement = sel.regenerate(0, mode="retime")
    assert replacement is not None
    assert replacement.start == 0 and replacement.end == 25


def test_regenerate_never_collides_with_kept_clips():
    sel = Selection.build(pool(), 3)
    kept = [c for c in sel.chosen if c.start != 0]
    replacement = sel.regenerate(0)
    if replacement is not None:
        assert all(not overlaps(replacement, k) for k in kept)


def test_a_rejected_clip_is_never_offered_again():
    sel = Selection.build(pool(), 3)
    first = sel.chosen[0]
    sel.regenerate(0)
    alts = [c for i in range(len(sel.chosen)) for c in sel.alternatives_for(i)]
    assert all(not (c.start == first.start and c.end == first.end) for c in alts)


def test_regenerate_returns_none_and_drops_the_clip_when_nothing_fits():
    small = [FakeClip(0, 20, 9.0)]
    sel = Selection.build(small, 1)
    assert sel.regenerate(0) is None
    assert sel.chosen == []


def test_regenerate_all_replaces_the_whole_set():
    sel = Selection.build(pool(), 2)
    before = {(c.start, c.end) for c in sel.chosen}
    sel.regenerate_all()
    after = {(c.start, c.end) for c in sel.chosen}
    assert before.isdisjoint(after)


def test_remaining_counts_unrejected_candidates():
    sel = Selection.build(pool(), 2)
    before = sel.remaining()
    sel.regenerate(0)
    assert sel.remaining() == before - 1


def test_out_of_range_index_is_handled():
    sel = Selection.build(pool(), 2)
    assert sel.regenerate(99) is None
    assert sel.alternatives_for(-1) == []


def test_reserve_excludes_what_is_already_chosen():
    sel = Selection.build(pool(), 3)
    assert sel.remaining() == len(sel.pool) - len(sel.chosen)


def test_reserve_shrinks_as_clips_are_rejected():
    sel = Selection.build(pool(), 2)
    before = sel.remaining()
    sel.regenerate(0)
    assert sel.remaining() < before
