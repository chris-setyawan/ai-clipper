import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.export import platforms as plat
from ai_clipper.video.subtitles import (
    CaptionStyle, PRESETS, Word, build_ass, chars_per_line, title_event,
    wrap_title,
)


def test_a_short_title_stays_on_one_line():
    assert wrap_title("MARGIN CALL", 20) == ["MARGIN CALL"]


def test_lines_are_balanced_rather_than_greedily_filled():
    """
    Greedy filling puts as much as possible on each line and leaves the
    remainder alone on the last one, which is what makes a title card look
    unfinished. No line should be much shorter than the others.
    """
    lines = wrap_title("HAMPIR KENA MARGIN CALL WAKTU ITU", 20)
    assert len(lines) > 1
    lengths = [len(line) for line in lines]
    assert max(lengths) - min(lengths) <= 6


def test_a_word_longer_than_the_line_is_not_dropped():
    lines = wrap_title("PERTANGGUNGJAWABANNYA OKE", 8)
    assert "PERTANGGUNGJAWABANNYA" in " ".join(lines)


def test_an_overlong_title_is_cut_at_the_line_limit():
    text = " ".join(["KATA"] * 40)
    lines = wrap_title(text, 20, max_lines=3)
    assert len(lines) == 3
    assert lines[-1].endswith("...")


def test_a_title_that_fits_exactly_is_not_marked_as_cut():
    lines = wrap_title("HAMPIR KENA MARGIN CALL WAKTU ITU", 20, max_lines=3)
    assert not any(line.endswith("...") for line in lines)


def test_empty_title_produces_nothing():
    assert wrap_title("   ", 20) == []
    assert title_event("", PRESETS["punch"], 720, 1280) == ""


def test_line_width_follows_the_frame_and_the_font():
    wide = chars_per_line(1080, 60, 60, 60)
    narrow = chars_per_line(720, 60, 60, 60)
    assert wide > narrow
    assert chars_per_line(1080, 60, 60, 120) < wide


# --- how it lands in the file ------------------------------------------------

WORDS = [Word(0.0, 0.5, "gue"), Word(0.5, 1.0, "hampir"), Word(1.0, 1.5, "kena")]


def test_the_title_becomes_one_event_above_the_captions():
    out = build_ass(WORDS, 720, 1280, PRESETS["punch"], title="Margin Call")
    title_lines = [l for l in out.splitlines() if l.startswith("Dialogue: 1,")]
    assert len(title_lines) == 1
    assert "MARGIN CALL" in title_lines[0]
    assert "Title" in title_lines[0]


def test_the_title_fades_rather_than_cutting():
    out = build_ass(WORDS, 720, 1280, PRESETS["punch"], title="Margin Call")
    assert r"\fad(" in out


def test_no_title_means_no_title_event():
    out = build_ass(WORDS, 720, 1280, PRESETS["punch"])
    assert not [l for l in out.splitlines() if l.startswith("Dialogue: 1,")]
    # the style can still be declared; what matters is that nothing draws it
    assert "MARGIN" not in out


def test_a_style_with_the_title_switched_off_draws_nothing():
    style = CaptionStyle.from_dict({**PRESETS["punch"].to_dict(), "title_seconds": 0.0})
    out = build_ass(WORDS, 720, 1280, style, title="Margin Call")
    assert not [l for l in out.splitlines() if l.startswith("Dialogue: 1,")]


def test_the_title_hangs_from_the_top():
    out = build_ass(WORDS, 720, 1280, PRESETS["punch"], title="Margin Call")
    style_line = next(l for l in out.splitlines() if l.startswith("Style: Title,"))
    # alignment 8 is top centre, and the vertical margin is measured from there
    assert style_line.split(",")[18] == "8"


def test_scaling_carries_every_field_including_new_ones():
    """
    scaled() used to name each field, and adding one meant remembering to add
    it there too. It is a copy now, so this checks nothing is lost.
    """
    style = PRESETS["punch"]
    small = style.scaled(1280)
    assert small.title_size < style.title_size
    assert small.title_uppercase == style.title_uppercase
    assert set(small.to_dict()) == set(style.to_dict())


def test_a_platform_pushes_the_title_clear_of_its_own_interface():
    base = PRESETS["punch"]
    tiktok = plat.style_for(base, "tiktok")
    square = plat.style_for(base, "square")
    # TikTok draws over the top of the frame, a square post does not
    assert tiktok.title_top_margin_frac > square.title_top_margin_frac
