import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.scoring import signals
from ai_clipper.scoring.hook_scorer import (
    HookScorer,
    TranscriptSegment,
    ends_dangling,
    opening_stands_alone,
)


@pytest.fixture(autouse=True)
def restore_pack():
    """Every test leaves the built-in pack installed."""
    before = signals.active()
    yield
    signals.use(before)


def test_the_built_in_pack_is_active_by_default():
    assert signals.active().name == "indonesian"
    assert "margin call" in signals.active().stakes_markers


def test_a_bank_can_be_replaced_outright():
    pack = signals.merge(signals.INDONESIAN, {"markers": {"intensity": ["wild"]}})
    assert pack.intensity_markers == {"wild"}


def test_a_bank_can_be_added_to_without_retyping_it():
    pack = signals.merge(signals.INDONESIAN, {"markers": {"+stakes": ["komplikasi"]}})
    assert "komplikasi" in pack.stakes_markers
    assert "margin call" in pack.stakes_markers, "the existing bank must survive"


def test_weights_and_thresholds_are_patched_not_replaced():
    pack = signals.merge(signals.INDONESIAN, {
        "weights": {"stakes": 5.0},
        "thresholds": {"closing_window": 20.0},
    })
    assert pack.weights["stakes"] == 5.0
    assert pack.weights["ends_incomplete"] == -3.0, "untouched weights stay"
    assert pack.closing_window == 20.0


def test_an_unknown_bank_is_ignored_rather_than_fatal():
    """A file written for a later version should still load in an older one."""
    pack = signals.merge(signals.INDONESIAN, {"markers": {"sarcasm": ["yeah right"]}})
    assert pack.stakes_markers == signals.INDONESIAN.stakes_markers


def test_loading_with_no_file_returns_the_built_in_pack(tmp_path):
    assert signals.load(str(tmp_path / "nothing.json")) is signals.INDONESIAN


def test_a_file_on_disk_is_read_and_applied(tmp_path):
    path = tmp_path / "signals.json"
    path.write_text(json.dumps({
        "name": "kesehatan",
        "extends": "indonesian",
        "markers": {"+stakes": ["komplikasi"]},
    }), encoding="utf-8")

    pack = signals.load(str(path))
    assert pack.name == "kesehatan"
    assert "komplikasi" in pack.stakes_markers


def test_swapping_the_pack_changes_what_the_scorer_sees():
    """
    The point of the whole split: adding a language is adding a file, not
    editing Python.
    """
    seg = [TranscriptSegment(0, 10, "The surgery nearly went wrong that night.")]
    assert HookScorer().score_window(seg, {}, 1).signals["stakes"] == 0.0

    english = signals.merge(signals.INDONESIAN, {
        "name": "english",
        "markers": {"stakes": ["nearly", "went wrong"], "stopwords": ["the", "that"]},
    })
    signals.use(english)
    assert HookScorer().score_window(seg, {}, 1).signals["stakes"] > 0


def test_the_structural_rules_follow_the_pack_too():
    """
    ends_dangling and opening_stands_alone are structure, but the words they
    look for are vocabulary, so they read the active pack as well.
    """
    assert ends_dangling("Gue bilang sama Maybank.")

    signals.use(signals.merge(signals.INDONESIAN, {
        "markers": {"speech_verbs": [], "speech_act_verbs": ["said", "told"]}}))
    assert not ends_dangling("Gue bilang sama Maybank.")
    assert ends_dangling("And then I told Maybank.")


def test_a_pack_with_empty_quantity_words_does_not_crash_the_number_signal():
    signals.use(signals.merge(signals.INDONESIAN, {"markers": {"quantity_words": []}}))
    seg = [TranscriptSegment(0, 10, "Harganya naik banyak sekali waktu itu.")]
    assert HookScorer().score_window(seg, {}, 1).signals["has_number"] == 0.0
