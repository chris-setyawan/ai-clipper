import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video import cover
from ai_clipper.video.framing import CropPath, Keyframe
from ai_clipper.video.subtitles import Word


def path_of(*rows):
    """rows: (t, center_x, confident, motion)"""
    return CropPath(
        keyframes=[Keyframe(t, cx, ok, motion) for t, cx, ok, motion in rows],
        shots=[],
        detection_rate=sum(1 for r in rows if r[2]) / max(1, len(rows)),
    )


def steady(t0, n, step=0.5, center=0.5, confident=True, motion=1.0):
    return [(t0 + i * step, center, confident, motion) for i in range(n)]


def speech(t0, n, text="kata", step=0.4):
    return [Word(t0 + i * step, t0 + i * step + 0.3, text) for i in range(n)]


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


# --- the words on the thumbnail ---------------------------------------------

def test_a_line_is_judged_by_what_its_words_carry():
    """
    Both of these are two words and about the same length. Measuring length,
    which the first version did, scored them identically.
    """
    assert cover.line_value("transaksi harian") > cover.line_value("manajemen dalam")


def test_glue_and_filler_score_nothing():
    assert cover.line_value("mungkin, pokoknya") == 0
    assert cover.line_value("ya") == 0
    assert cover.line_value("") == 0


def test_a_stakes_word_lifts_a_line():
    assert cover.line_value("margin call") > cover.line_value("kebetulan ditawarin")


def test_a_figure_counts_even_though_it_is_not_a_word():
    """
    "1,8T" is short and has no long tokens, so the content rule alone scores it
    at nothing, and it is a better thumbnail than most full phrases.
    """
    assert cover.line_value("1,8 triliun") > 0


def test_a_paragraph_is_discounted_however_good_its_words_are():
    long_line = "margin call " * 8
    assert cover.line_value(long_line) < cover.line_value("margin call")


def test_a_moment_with_a_readable_line_beats_a_silent_one():
    """
    Stillness alone will happily pick a second where nobody is talking, and a
    thumbnail with no words on it throws away the line that makes someone stop.
    """
    frames = steady(0, 40, step=0.25, motion=1.0)
    # nothing said until 6s, then a line worth reading
    words = speech(6.0, 5, "gontokan")
    at = cover.pick_time(path_of(*frames), 0, 10, words=words, per_chunk=3, edge=0.5)
    assert 6.0 <= at <= 8.0


def test_a_caption_is_required_not_merely_preferred():
    """
    A wordless cover throws away the line that makes someone stop, so a calmer
    silent moment does not win. Weighting this instead of requiring it is what
    produced a wordless cover on the first real run.
    """
    frames = (
        steady(0, 12, step=0.25, motion=0.2)                  # very calm, silent
        + steady(3.0, 12, step=0.25, motion=8.0)              # busier, but spoken
    )
    words = speech(3.0, 6, "gontokan")
    at = cover.pick_time(path_of(*frames), 0, 6, words=words, per_chunk=3, edge=0.5)
    assert at >= 3.0


def test_the_gap_between_two_words_is_not_a_caption():
    """
    The bug this exists for: a chunk looks continuous in the transcript, but
    the karaoke effect draws one event per word, so the moment between two
    words has nothing on screen. Scoring by chunk put a cover in one of those
    holes while the report claimed a line was showing.
    """
    frames = [(1.0, 0.5, True, 40.0), (1.45, 0.5, True, 0.1), (2.0, 0.5, True, 40.0)]
    # words at 0.9-1.1 and 1.9-2.1, so 1.45 is a hole even though the chunk
    # spans it
    words = [Word(0.9, 1.1, "gontokan"), Word(1.9, 2.1, "banget")]
    at = cover.pick_time(path_of(*frames), 0, 3, words=words, per_chunk=3, edge=0.0)
    assert at != 1.45


def test_the_last_word_is_held_until_the_chunk_ends():
    """Mirrors how the subtitle builder holds the final word of a chunk."""
    words = [Word(0.0, 0.4, "sampe"), Word(0.5, 0.7, "gontokan")]
    spans = cover._spoken_windows(words, 0, 5, per_chunk=2)
    assert spans[-1][1] >= 0.7


def test_a_visible_but_weak_line_still_counts_as_visible():
    """
    A value of 0 means "these words carry nothing", not "nothing is on screen".
    A cover with a weak line still beats one with no line at all.
    """
    frames = steady(0, 12, step=0.25, motion=0.2) + steady(3.0, 12, step=0.25, motion=8.0)
    words = [Word(3.0 + i * 0.4, 3.0 + i * 0.4 + 0.35, "ya") for i in range(6)]
    at = cover.pick_time(path_of(*frames), 0, 6, words=words, per_chunk=1, edge=0.5)
    assert at >= 3.0


def test_words_are_optional():
    frames = steady(0, 20, step=0.5)
    assert cover.pick_time(path_of(*frames), 0, 10, words=[], edge=0.5) is not None


def test_a_clip_where_every_line_is_weak_still_gets_a_cover():
    frames = steady(0, 20, step=0.5)
    words = [Word(i * 0.5, i * 0.5 + 0.4, "mungkin") for i in range(20)]
    assert cover.pick_time(path_of(*frames), 0, 10, words=words, edge=0.5) is not None
