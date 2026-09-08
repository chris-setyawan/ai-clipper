import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.export import platforms
from ai_clipper.export.caption_pack import (
    build_hashtags,
    build_pack,
    extract_hook,
    keywords,
)
from ai_clipper.video.subtitles import PRESETS


@dataclass
class FakeClip:
    start: float
    end: float
    score: float
    text: str

    def explain(self):
        return "opening_question (+2.5)"


# --- platforms -------------------------------------------------------------

def test_every_platform_keeps_captions_clear_of_its_ui():
    for key, platform in platforms.PLATFORMS.items():
        margins = platform.caption_margins()
        assert margins["bottom_margin_frac"] >= platform.bottom_ui, key
        assert margins["side_margin_frac"] >= max(platform.left_ui, platform.right_ui), key


def test_margins_never_swallow_the_frame():
    for platform in platforms.PLATFORMS.values():
        m = platform.caption_margins()
        assert m["bottom_margin_frac"] < 0.5
        assert m["side_margin_frac"] < 0.35


def test_tiktok_needs_more_side_clearance_than_the_feed():
    assert (platforms.get("tiktok").caption_margins()["side_margin_frac"]
            > platforms.get("feed").caption_margins()["side_margin_frac"])


def test_style_for_moves_margins_and_leaves_the_look_alone():
    base = PRESETS["punch"]
    styled = platforms.style_for(base, "tiktok")
    assert styled.font == base.font
    assert styled.highlight == base.highlight
    assert styled.words_per_chunk == base.words_per_chunk
    assert styled.bottom_margin_frac > base.bottom_margin_frac


def test_unknown_platform_names_the_valid_ones():
    with pytest.raises(ValueError) as err:
        platforms.get("myspace")
    assert "tiktok" in str(err.value)


def test_a_long_clip_is_excluded_from_platforms_that_cap_duration():
    long_clip = FakeClip(0, 200, 8.0, "x")
    targets = [p.key for p in platforms.targets_for(200.0, ["tiktok", "reels", "shorts"])]
    assert "tiktok" in targets          # 10 minute cap
    assert "reels" not in targets       # 3 minute cap
    assert "shorts" not in targets


def test_too_short_clips_are_excluded_everywhere():
    assert platforms.targets_for(2.0) == []


def test_plan_covers_every_clip_and_platform_pair():
    clips = [FakeClip(0, 30, 8.0, "x"), FakeClip(40, 70, 7.0, "y")]
    plans = platforms.plan_exports(clips, ["tiktok", "shorts"])
    assert len(plans) == 4
    assert len({p.filename for p in plans}) == 4, "filenames must be unique"


def test_rejected_explains_why():
    clips = [FakeClip(0, 200, 8.0, "x")]
    out = platforms.rejected(clips, ["reels"])
    assert out and "200s" in out[0][1]


# --- caption packs ---------------------------------------------------------

def test_hook_prefers_a_substantial_question():
    text = "Betul ya? Kenapa 90 persen pelamar magang ditolak di tahap CV?"
    assert "90 persen" in extract_hook(text)


def test_hook_ignores_throwaway_questions():
    text = "Gitu ya? Iya. Scene ini adalah scene termahal di film tersebut."
    hook = extract_hook(text)
    assert "termahal" in hook
    assert hook != "Gitu ya?"


def test_hook_is_trimmed_to_a_readable_length():
    text = " ".join(["katakata"] * 40) + "."
    assert len(extract_hook(text).split()) <= 15


def test_hook_of_empty_text_is_empty():
    assert extract_hook("") == ""


def test_keywords_skip_filler_and_stopwords():
    words = keywords("Iya gitu banget emang terus kayak sutradara sutradara produser")
    assert "sutradara" in words
    assert "banget" not in words
    assert "gitu" not in words


def test_hashtags_start_with_the_channel_tags():
    tags = build_hashtags("sutradara produser syuting", base=["#podcast"])
    assert tags[0] == "#podcast"
    assert any("sutradara" in t for t in tags)


def test_hashtags_are_unique_and_capped():
    tags = build_hashtags("scene scene scene film film", limit=5)
    assert len(tags) == len(set(tags))
    assert len(tags) <= 5


def test_pack_carries_the_scoring_explanation():
    clip = FakeClip(10, 40, 7.9, "Scene ini adalah scene termahal di film tersebut.")
    pack = build_pack(clip, 1)
    assert pack.why
    assert pack.score == 7.9
    assert "termahal" in pack.title or "termahal" in pack.hook


def test_pack_text_is_labelled_as_a_draft():
    clip = FakeClip(0, 30, 7.0, "Scene ini adalah scene termahal di film tersebut.")
    assert "Drafts" in build_pack(clip, 1).to_text()


def test_captions_stay_centred_on_every_platform():
    """
    Off-centre subtitles read as a mistake in every frame. Clearing the button
    column is worth narrowing the text block for, not worth shifting it for.
    """
    for key in platforms.PLATFORMS:
        left, right = platforms.style_for(PRESETS["punch"], key).margins()
        assert abs(left - right) < 1e-9, key


def test_tiktok_narrows_the_caption_more_than_the_feed():
    tiktok, _ = platforms.style_for(PRESETS["punch"], "tiktok").margins()
    feed, _ = platforms.style_for(PRESETS["punch"], "feed").margins()
    assert tiktok > feed, "a wider interface column should narrow the text block"


def test_feed_keeps_captions_centred():
    style = platforms.style_for(PRESETS["punch"], "feed")
    left, right = style.margins()
    assert abs(left - right) < 1e-9, "nothing overlaps a feed post, so stay centred"


def test_margins_default_to_the_symmetric_value():
    base = PRESETS["clean"]
    left, right = base.margins()
    assert left == right == base.side_margin_frac


def test_ass_can_still_write_asymmetric_margins_when_asked():
    """The capability stays; no platform uses it by default."""
    from ai_clipper.video.subtitles import CaptionStyle, Word, build_ass

    style = CaptionStyle(left_margin_frac=0.03, right_margin_frac=0.20)
    ass = build_ass([Word(0, 1, "halo")], 720, 1280, style)
    line = next(l for l in ass.splitlines() if l.startswith("Style:"))
    fields = line.split(",")
    margin_l, margin_r = int(fields[-4]), int(fields[-3])
    assert margin_r > margin_l
