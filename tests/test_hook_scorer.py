import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.scoring.hook_scorer import HookScorer, TranscriptSegment


def make_segments():
    return [
        TranscriptSegment(0.0, 5.0, "Jadi hari ini kita ngobrol soal cuaca ya."),
        TranscriptSegment(5.0, 10.0, "Cuacanya lumayan panas belakangan ini."),
        TranscriptSegment(10.0, 16.0, "Kalian tau gak sih kenapa 90 persen startup gagal di tahun pertama?"),
        TranscriptSegment(16.0, 22.0, "Ternyata bukan karena kehabisan uang, padahal banyak yang ngira gitu."),
        TranscriptSegment(22.0, 27.0, "Oke lanjut ke topik berikutnya."),
        TranscriptSegment(27.0, 32.0, "Kita bahas soal makanan favorit deh."),
    ]


def test_top_candidate_has_highest_score():
    scorer = HookScorer()
    candidates = scorer.rank_candidates(make_segments(), min_duration=5, max_duration=20, top_n=5)
    assert candidates, "expected at least one candidate"
    scores = [c.score for c in candidates]
    best = max(candidates, key=lambda c: c.score)
    # the hook-heavy window (question + curiosity + contrarian) should win
    assert "90 persen" in best.text
    assert best.score == max(scores)


def test_candidates_do_not_overlap():
    scorer = HookScorer()
    candidates = scorer.rank_candidates(make_segments(), min_duration=5, max_duration=20, top_n=5)
    candidates.sort(key=lambda c: c.start)
    for a, b in zip(candidates, candidates[1:]):
        assert a.end <= b.start, f"overlap between {a.start}-{a.end} and {b.start}-{b.end}"


def test_explain_lists_only_fired_signals():
    scorer = HookScorer()
    candidates = scorer.rank_candidates(make_segments(), min_duration=5, max_duration=20, top_n=5)
    best = max(candidates, key=lambda c: c.score)
    explanation = best.explain()
    assert "opening_question" in explanation or "curiosity_marker" in explanation


def test_calibration_spreads_the_scale_across_the_pool():
    """
    A fixed exponential curve saturated: on a one-hour episode the median
    window scored 5.9 and the very best 9.4, so every clip worth considering
    sat inside a fraction of a point. Fitting the scale to the episode makes
    the best window a 10 and the median a 5.
    """
    scorer = HookScorer()
    segments = make_segments()
    pool = scorer.candidate_pool(segments, min_duration=5, max_duration=20)
    assert pool
    assert max(c.score for c in pool) == 10.0
    assert min(c.score for c in pool) < 10.0


def test_raw_score_survives_calibration():
    scorer = HookScorer()
    pool = scorer.candidate_pool(make_segments(), min_duration=5, max_duration=20)
    best = max(pool, key=lambda c: c.score)
    assert best.raw_score != 0.0, "the absolute total is kept for cross-episode use"
    assert abs(best.raw_score - sum(best.signals.values())) < 1e-9


def test_percentile_is_reported():
    scorer = HookScorer()
    pool = scorer.candidate_pool(make_segments(), min_duration=5, max_duration=20)
    best = max(pool, key=lambda c: c.score)
    assert 0.0 <= best.percentile <= 100.0
