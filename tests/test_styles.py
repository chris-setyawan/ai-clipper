import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video import fonts
from ai_clipper.video.subtitles import CaptionStyle, Word, build_ass, chunk_words, load_styles


def test_bundled_fonts_present():
    assert fonts.missing_bundled() == [], "a bundled font file is missing from assets/fonts"
    assert fonts.DEFAULT_FONT in fonts.bundled_fonts()


def test_style_round_trips_through_dict():
    style = CaptionStyle(name="x", font="Anton", font_size=70, uppercase=True)
    assert CaptionStyle.from_dict(style.to_dict()) == style


def test_unknown_field_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError) as err:
        CaptionStyle.from_dict({"font_colour": "#fff"})
    assert "font_colour" in str(err.value)


def test_user_file_patches_a_preset_without_clobbering_it(tmp_path):
    p = tmp_path / "styles.json"
    p.write_text(json.dumps({"punch": {"highlight": "#FF0000"}}), encoding="utf-8")

    styles = load_styles(str(p))
    assert styles["punch"].highlight == "#FF0000"
    # untouched fields survive the patch
    assert styles["punch"].uppercase is True
    assert styles["punch"].words_per_chunk == 2


def test_user_file_can_add_a_new_style(tmp_path):
    p = tmp_path / "styles.json"
    p.write_text(json.dumps({"mine": {"font_size": 99}}), encoding="utf-8")
    styles = load_styles(str(p))
    assert styles["mine"].font_size == 99
    assert styles["mine"].name == "mine"
    assert {"clean", "punch", "calm"} <= set(styles)


def test_comment_keys_are_ignored(tmp_path):
    p = tmp_path / "styles.json"
    p.write_text(json.dumps({"_comment": "hello", "mine": {"font_size": 40}}), encoding="utf-8")
    styles = load_styles(str(p))
    assert "_comment" not in styles


def test_pause_forces_a_chunk_break():
    words = [Word(0.0, 0.4, "a"), Word(0.5, 0.9, "b"), Word(3.0, 3.4, "c")]
    chunks = chunk_words(words, per_chunk=5, max_gap=0.7)
    assert len(chunks) == 2, "a long silence should split the caption"


def test_ass_output_shifts_times_and_names_the_font():
    words = [Word(10.0, 10.5, "satu"), Word(10.6, 11.0, "dua")]
    out = build_ass(words, 720, 1280, CaptionStyle(font="Anton"), clip_start=10.0)
    assert "Anton" in out
    assert "Dialogue: 0,0:00:00.00" in out, "first word should start at zero"


# --- word tokens that were never separate words -----------------------------

from ai_clipper.video.subtitles import merge_split_tokens


def test_a_number_split_at_its_separator_is_rejoined():
    """
    Whisper returns "19.000" as "19" + ".000", and the renderer, joining words
    with spaces, burned "19 .000" onto the screen. It was the first thing anyone
    noticed in the first frame anyone looked at closely.
    """
    words = [Word(0.0, 0.3, "ke"), Word(0.3, 0.6, "19"), Word(0.6, 1.1, ".000")]
    out = merge_split_tokens(words)
    assert [w.text for w in out] == ["ke", "19.000"]


def test_the_merged_word_keeps_the_whole_span():
    """Karaoke highlighting has to cover the time the speaker actually took."""
    out = merge_split_tokens([Word(0.3, 0.6, "19"), Word(0.6, 1.1, ".000")])
    assert (out[0].start, out[0].end) == (0.3, 1.1)


def test_a_percent_sign_joins_the_number_before_it():
    out = merge_split_tokens([Word(0, 1, "naik"), Word(1, 2, "50"), Word(2, 2.4, "%")])
    assert [w.text for w in out] == ["naik", "50%"]


def test_a_token_that_is_only_punctuation_joins_the_word_before_it():
    out = merge_split_tokens([Word(0, 1, "bilang"), Word(1, 1.2, ".")])
    assert [w.text for w in out] == ["bilang."]


def test_ordinary_words_are_left_alone():
    words = [Word(0, 1, "Nah"), Word(1, 2, "gue"), Word(2, 3, "bilang.")]
    assert [w.text for w in merge_split_tokens(words)] == ["Nah", "gue", "bilang."]


def test_a_leading_separator_without_a_digit_is_not_merged():
    """A dash starting a real word is a word, not a fragment."""
    words = [Word(0, 1, "beli"), Word(1, 2, "-mungkin")]
    assert [w.text for w in merge_split_tokens(words)] == ["beli", "-mungkin"]
