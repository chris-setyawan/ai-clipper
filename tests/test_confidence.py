import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.asr.transcriber import UNSURE_BELOW, Word, segments_to_json, uncertain
from ai_clipper.scoring.hook_scorer import TranscriptSegment


def test_a_word_is_confident_unless_told_otherwise():
    """
    A transcript written before confidence was saved must load as one where
    nothing is flagged, not one where every word looks suspect.
    """
    assert Word(0.0, 1.0, "halo").confidence == 1.0
    assert uncertain([Word(0.0, 1.0, "halo")]) == []


def test_confidence_reaches_the_transcript_file():
    payload = segments_to_json([TranscriptSegment(0, 2, "halo dunia")],
                               [Word(0, 1, "halo", 0.42), Word(1, 2, "dunia")])
    assert payload["words"][0]["confidence"] == 0.42
    assert payload["words"][1]["confidence"] == 1.0


def test_only_the_unsure_words_come_back():
    words = [Word(0, 1, "halo", 0.99), Word(1, 2, "santung", 0.31),
             Word(2, 3, "dunia", 0.85)]
    assert [w.text for w in uncertain(words, below=0.5)] == ["santung"]


def test_the_least_certain_word_comes_first():
    words = [Word(0, 1, "satu", 0.55), Word(1, 2, "dua", 0.12),
             Word(2, 3, "tiga", 0.40)]
    assert [w.text for w in uncertain(words)] == ["dua", "tiga", "satu"]


def test_the_threshold_is_adjustable():
    words = [Word(0, 1, "mungkin", 0.7)]
    assert uncertain(words, below=UNSURE_BELOW) == []
    assert len(uncertain(words, below=0.8)) == 1


def test_words_from_an_older_transcript_are_never_flagged():
    class Plain:
        def __init__(self):
            self.start, self.end, self.text = 0.0, 1.0, "halo"

    assert uncertain([Plain()], below=0.99) == []
