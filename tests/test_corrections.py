import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.asr.corrections import (
    Lexicon,
    apply,
    correct_text,
    correct_word_sequence,
    diff_summary,
    snap_word,
)


def lex():
    return Lexicon(
        exact={"sudah dares": "sutradara", "ngedirect": "nge-direct"},
        vocabulary=["sutradara", "produser", "syuting"],
    )


def words(*pairs):
    out, t = [], 0.0
    for text in pairs:
        out.append({"start": round(t, 3), "end": round(t + 0.3, 3), "text": text})
        t += 0.3
    return out


def test_exact_phrase_replaced_in_text():
    assert correct_text("Gue sudah dares juga", lex()) == "Gue sutradara juga"


def test_replacement_keeps_original_capitalisation():
    assert correct_text("Sudah dares itu", lex()).startswith("Sutradara")


def test_multi_word_phrase_collapses_in_the_word_list():
    w = correct_word_sequence(words("Gue", "sudah", "dares", "juga"), lex())
    assert [x["text"] for x in w] == ["Gue", "sutradara", "juga"]


def test_collapsed_span_keeps_start_and_end():
    original = words("Gue", "sudah", "dares", "juga")
    w = correct_word_sequence(original, lex())
    fixed = w[1]
    assert fixed["start"] == original[1]["start"]
    assert fixed["end"] == original[2]["end"]


def test_timings_stay_monotonic_after_correction():
    w = correct_word_sequence(words("a", "sudah", "dares", "b", "ngedirect"), lex())
    for a, b in zip(w, w[1:]):
        assert a["end"] <= b["start"] + 1e-6


def test_trailing_punctuation_survives():
    w = correct_word_sequence(words("Gue", "sudah", "dares,"), lex())
    assert w[-1]["text"] == "sutradara,"


def test_fuzzy_snap_pulls_near_miss_to_vocabulary():
    assert snap_word("sutradarah", lex()) == "sutradara"
    assert snap_word("produsen", lex()) in ("produser", "produsen")


def test_short_words_are_not_fuzzy_snapped():
    assert snap_word("gue", lex()) == "gue"


def test_empty_lexicon_is_a_no_op():
    t = {"segments": [{"text": "apa adanya"}], "words": words("apa", "adanya")}
    assert apply(t, Lexicon.empty()) == t


def test_apply_fixes_both_segments_and_words():
    t = {
        "segments": [{"start": 0, "end": 1, "text": "Gue sudah dares"}],
        "words": words("Gue", "sudah", "dares"),
    }
    out = apply(t, lex())
    assert out["segments"][0]["text"] == "Gue sutradara"
    assert [w["text"] for w in out["words"]] == ["Gue", "sutradara"]
    # the input is not mutated
    assert t["segments"][0]["text"] == "Gue sudah dares"


def test_whisper_prompt_lists_terms_without_duplicates():
    prompt = lex().as_whisper_prompt()
    assert "sutradara" in prompt
    assert prompt.count("sutradara") == 1


def test_diff_summary_reports_changes():
    before = {"words": words("sudah", "dares")}
    after = apply({"segments": [], **before}, lex())
    assert diff_summary(before, after)


def test_word_list_correction_keeps_capitalisation():
    lx = Lexicon(exact={"dangang": "dangak"}, vocabulary=[])
    w = correct_word_sequence(words("Dangang."), lx)
    assert w[0]["text"] == "Dangak.", "a sentence-initial fix must stay capitalised"


def test_lowercase_source_stays_lowercase():
    lx = Lexicon(exact={"dangang": "dangak"}, vocabulary=[])
    w = correct_word_sequence(words("dangang"), lx)
    assert w[0]["text"] == "dangak"


def test_multi_word_replacement_capitalises_only_the_first():
    lx = Lexicon(exact={"pasang bandang": "pasang badan"}, vocabulary=[])
    w = correct_word_sequence(words("Pasang", "bandang"), lx)
    assert [x["text"] for x in w] == ["Pasang", "badan"]


def test_whisper_prompt_is_a_punctuated_sentence():
    prompt = lex().as_whisper_prompt()
    assert prompt[0].isupper(), "a lowercase prompt makes Whisper drop capitalisation"
    assert prompt.endswith("."), "an unpunctuated prompt makes Whisper drop punctuation"


def test_whisper_prompt_prefix_is_overridable():
    lx = Lexicon(exact={}, vocabulary=["Arap"], prompt_prefix="This conversation mentions")
    assert lx.as_whisper_prompt() == "This conversation mentions Arap."


def test_empty_lexicon_produces_no_prompt():
    assert Lexicon.empty().as_whisper_prompt() == ""
