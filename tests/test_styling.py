import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video import deadair
from ai_clipper.video.subtitles import (
    ANIMATIONS, PRESETS, Word, build_ass, entrance, glow_tag,
    merge_split_tokens, typed, word_tags,
)

PUNCH = PRESETS["punch"].scaled(1280)


def events(text):
    return [l for l in text.splitlines() if l.startswith("Dialogue: 0,")]


# --- per-word styling --------------------------------------------------------

def test_a_word_with_no_style_gets_no_tags():
    assert word_tags(None, PUNCH) == ("", "")
    assert word_tags({}, PUNCH) == ("", "")


def test_every_override_is_closed_again():
    """
    A caption event holds the whole chunk, so a tag left open leaks onto the
    words after it.
    """
    opening, closing = word_tags({"italic": True, "size": 90}, PUNCH)
    assert opening.startswith("{") and opening.endswith("}")
    assert r"\i1" in opening and r"\i0" in closing
    assert r"\fs90" in opening
    assert rf"\fs{PUNCH.font_size}" in closing


def test_closing_returns_to_the_style_not_to_nothing():
    _, closing = word_tags({"bold": False, "font": "Bebas Neue"}, PUNCH)
    assert (r"\b1" if PUNCH.bold else r"\b0") in closing
    assert r"\fn" + PUNCH.font in closing


def test_an_unknown_override_is_ignored_rather_than_fatal():
    """A file written by a newer caption editor still has to render here."""
    assert word_tags({"wobble": 3}, PUNCH) == ("", "")


def test_styling_survives_the_number_merge():
    """
    Whisper splits "19.000" into two tokens and the renderer rejoins them. The
    styling has to come along, or every edited caption loses its look the moment
    a number appears in it.
    """
    merged = merge_split_tokens([
        Word(0.0, 0.4, "19", {"italic": True}), Word(0.4, 0.6, ".000"),
    ])
    assert len(merged) == 1
    assert merged[0].text == "19.000"
    assert merged[0].style == {"italic": True}


def test_styling_survives_a_dead_air_cut():
    words = [Word(0.0, 1.0, "satu", {"color": "#FF0000"}), Word(9.0, 10.0, "dua")]
    line = deadair.plan(words, 0, 10)
    moved = deadair.shift_words(words, line)
    assert moved[0].style == {"color": "#FF0000"}
    assert moved[0].text == "satu"


def test_a_styled_word_reaches_the_file(tmp_path=None):
    out = build_ass([Word(0.0, 1.0, "setiap"),
                     Word(1.0, 2.0, "lakilaki", {"italic": True})],
                    720, 1280, PRESETS["punch"])
    assert r"\i1" in out


# --- animations --------------------------------------------------------------

def test_no_animation_adds_nothing():
    assert entrance(dataclasses.replace(PUNCH, animation="none")) == ""


def test_every_named_animation_produces_something():
    for name in ANIMATIONS:
        if name in ("none", "type"):
            continue
        style = dataclasses.replace(PUNCH, animation=name)
        assert entrance(style, (360, 986)), name


def test_the_moving_animations_need_a_point_and_say_so():
    """
    rise and drop are the only two that need explicit coordinates, and asking
    for one without them should do nothing rather than render at the corner.
    """
    style = dataclasses.replace(PUNCH, animation="rise")
    assert entrance(style, None) == ""
    assert r"\move(" in entrance(style, (360, 986))


def test_rise_comes_from_below_and_drop_from_above():
    rise = entrance(dataclasses.replace(PUNCH, animation="rise"), (360, 900))
    drop = entrance(dataclasses.replace(PUNCH, animation="drop"), (360, 900))
    from_rise = int(rise.split("(")[1].split(",")[1])
    from_drop = int(drop.split("(")[1].split(",")[1])
    assert from_rise > 900 > from_drop


def test_only_the_moving_animations_pin_the_caption():
    """
    Forcing a position changes how a long caption wraps, so the ones that do not
    need it must not use it.
    """
    for name in ("pop", "fade", "zoom", "blur"):
        out = build_ass([Word(0.0, 1.0, "satu")], 720, 1280,
                        dataclasses.replace(PRESETS["punch"], animation=name))
        assert r"\move" not in out and r"\pos" not in out, name


def test_a_typewriter_is_one_event_per_character():
    lines = typed("HALO", 0.0, 1.0, 200, "")
    assert len(lines) == 4
    assert lines[0].endswith("H")
    assert lines[-1].endswith("HALO")


def test_a_typewriter_does_not_outrun_the_words_being_spoken():
    """
    The reveal is capped, so a long line types faster rather than still
    appearing after the speaker has moved on.
    """
    lines = typed("SATU DUA TIGA EMPAT", 0.0, 10.0, 300, "")
    assert lines
    last_start = lines[-1].split(",")[1]
    assert last_start < "0:00:00.40"


def test_typing_an_empty_line_produces_nothing():
    assert typed("   ", 0.0, 1.0, 200, "") == []


def test_the_whole_chunk_stays_on_screen_after_it_has_typed():
    style = dataclasses.replace(PRESETS["punch"], animation="type",
                                words_per_chunk=2)
    out = build_ass([Word(0.0, 1.0, "satu"), Word(1.0, 2.0, "dua")],
                    720, 1280, style)
    last = events(out)[-1]
    # the karaoke highlight is still on the active word, so check the words are
    # both there rather than checking the line ends with them
    assert "SATU" in last and "DUA" in last


# --- glow --------------------------------------------------------------------

def test_glow_is_off_by_default():
    assert glow_tag(PUNCH) == ""


def test_glow_is_a_blurred_border_not_a_second_copy():
    tag = glow_tag(dataclasses.replace(PUNCH, glow=True, glow_size=7.0))
    assert r"\blur7.0" in tag and r"\bord" in tag
    out = build_ass([Word(0.0, 1.0, "satu")], 720, 1280,
                    dataclasses.replace(PRESETS["punch"], glow=True))
    assert len(events(out)) == 1


def test_glow_defaults_to_the_outline_colour():
    """
    A glow is drawn behind the fill and bleeds over its edges, so one the same
    colour as a word turns that word into a blob. The first version defaulted to
    the same green `punch` highlights with, and every highlighted word vanished.
    """
    style = dataclasses.replace(PUNCH, glow=True)
    assert style.glow_color is None
    tag = glow_tag(style)
    from ai_clipper.video.subtitles import ass_color
    assert ass_color(PUNCH.outline_color) in tag
    assert ass_color(PUNCH.highlight) not in tag


def test_glow_scales_with_the_frame():
    big = dataclasses.replace(PRESETS["punch"], glow=True, glow_size=8.0)
    assert big.scaled(1280).glow_size < big.scaled(1920).glow_size
