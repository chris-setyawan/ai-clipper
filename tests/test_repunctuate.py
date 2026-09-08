import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.scoring.hook_scorer import HookScorer, TranscriptSegment
from ai_clipper.scoring.repunctuate import (
    needs_repunctuation,
    punctuated_fraction,
    restore_sentence_ends,
)


def raw(*spans):
    """spans: (start, end, text)"""
    return [TranscriptSegment(a, b, t) for a, b, t in spans]


def test_a_punctuated_transcript_is_left_alone():
    segs = raw(
        (0.0, 2.0, "Waktu itu gue hampir kena margin call."),
        (2.1, 4.0, "Akhirnya gue nego sama sekuritasnya."),
    )
    assert not needs_repunctuation(segs)
    before = [s.text for s in segs]
    restore_sentence_ends(segs)
    assert [s.text for s in segs] == before


def test_a_raw_transcript_is_detected():
    segs = raw(
        (0.0, 2.0, "Waktu itu gue hampir kena margin call"),
        (2.1, 4.0, "Akhirnya gue nego sama sekuritasnya"),
    )
    assert needs_repunctuation(segs)
    assert punctuated_fraction(segs) == 0.0


def test_a_pause_becomes_a_full_stop_and_a_gapless_join_does_not():
    segs = raw(
        (0.0, 2.0, "Waktu itu gue hampir kena margin call"),   # 1.0s pause after
        (3.0, 5.0, "Terus gue"),                               # runs straight on
        (5.0, 7.0, "nego sama sekuritasnya"),                  # last, always closed
    )
    restore_sentence_ends(segs, min_pause=0.35)
    assert segs[0].text.endswith(".")
    assert not segs[1].text.endswith("."), "no pause, so the sentence continues"
    assert segs[2].text.endswith("."), "a transcript should not end mid-sentence"


def test_restoring_makes_the_completeness_penalty_discriminate_again():
    """
    The failure this module exists for. Without punctuation, ends_incomplete
    fires on every candidate, and a penalty that fires on everything ranks
    nothing: measured across two one-hour episodes, 6% of candidates on the
    punctuated one and 100% on the raw one.
    """
    spans, t = [], 0.0
    lines = [
        "Waktu itu gue hampir kena margin call di harga tujuh ribu lima ratus",
        "Akhirnya gue nego sama sekuritasnya biar gak kena likuidasi semua",
        "Terus dia kasih keringanan sampai gue bisa selamat waktu itu",
        "Nah dari situ gue belajar jangan pernah pakai repo segede itu",
    ]
    for line in lines:
        spans.append((t, t + 4.0, line))
        t += 5.0                                  # a one-second pause each time
    segs = raw(*spans)

    scorer = HookScorer()
    before = scorer.candidate_pool(segs, 5.0, 60.0)
    assert all(c.signals["ends_incomplete"] < 0 for c in before), "all look broken"

    restore_sentence_ends(segs)
    after = scorer.candidate_pool(segs, 5.0, 60.0)
    assert any(c.signals["ends_incomplete"] == 0.0 for c in after)
