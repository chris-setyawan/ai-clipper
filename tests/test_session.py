import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.scoring.selection import Selection
from ai_clipper.scoring.session import Session, key_of


@dataclass
class FakeClip:
    start: float
    end: float
    score: float


def pool():
    return [
        FakeClip(0, 20, 9.0),
        FakeClip(30, 50, 8.5),
        FakeClip(60, 80, 8.2),
        FakeClip(90, 110, 7.0),
        FakeClip(120, 140, 6.5),
    ]


SETTINGS = {"transcript": "t.json", "clips": 3, "min_duration": 15.0}


def fresh_session(chosen, rejected=()):
    session = Session(settings=dict(SETTINGS))
    numbers = session.numbers_for(chosen)
    entries = list(zip(numbers, [key_of(c) for c in chosen], chosen))
    session.record(entries, rejected)
    return session, numbers


def test_a_session_round_trips_through_disk(tmp_path):
    sel = Selection.build(pool(), 3)
    session, _ = fresh_session(sel.chosen, [(0.0, 20.0)])
    session.save(tmp_path)

    back = Session.load(tmp_path)
    assert back is not None
    assert back.settings == SETTINGS
    assert [s.index for s in back.slots] == [1, 2, 3]
    assert back.rejected == [(0.0, 20.0)]


def test_no_session_file_is_not_an_error(tmp_path):
    assert Session.load(tmp_path) is None


def test_a_changed_setting_invalidates_the_saved_selection():
    session, _ = fresh_session(Selection.build(pool(), 3).chosen)
    assert session.usable_with(SETTINGS)
    assert not session.usable_with({**SETTINGS, "clips": 5})


def test_restoring_brings_back_the_rejections(tmp_path):
    sel = Selection.build(pool(), 3)
    sel.regenerate(0)
    session, _ = fresh_session(sel.chosen, sel.rejected)
    session.save(tmp_path)

    # a second run rebuilds the pool from scratch and must not re-offer the
    # window that was turned down
    again = Selection.build(pool(), 3)
    assert Session.load(tmp_path).restore(again, pool())
    assert again.rejected == sel.rejected
    assert [key_of(c) for c in again.chosen] == [key_of(c) for c in sel.chosen]


def test_restore_refuses_when_a_saved_window_is_gone():
    session, _ = fresh_session(Selection.build(pool(), 3).chosen)
    smaller = [c for c in pool() if c.start != 30]
    assert not session.restore(Selection.build(smaller, 3), smaller)


def test_a_replacement_takes_the_number_of_the_clip_it_replaced():
    """
    Reject clip 2 and the new clip is clip 2, wherever in the episode it came
    from. Renumbering by time would move clips the user never complained about,
    and every moved clip is a file that has to be encoded again.
    """
    sel = Selection.build(pool(), 3)
    session, numbers = fresh_session(sel.chosen)
    before = dict(zip([key_of(c) for c in sel.chosen], numbers))

    replaced = sel.chosen[1]
    sel.regenerate(1)
    after = dict(zip([key_of(c) for c in sel.chosen], session.numbers_for(sel.chosen)))

    for clip in sel.chosen:
        k = key_of(clip)
        if k in before and k != key_of(replaced):
            assert after[k] == before[k], "kept clips must not move"
    new = [c for c in sel.chosen if key_of(c) not in before]
    assert len(new) == 1
    assert after[key_of(new[0])] == before[key_of(replaced)]


def test_a_file_is_current_only_when_it_was_encoded_from_this_clip():
    sel = Selection.build(pool(), 3)
    session, _ = fresh_session(sel.chosen)
    clip = sel.chosen[0]

    assert not session.is_current("clip_01_tiktok.mp4", clip), "never rendered"
    session.mark_rendered("clip_01_tiktok.mp4", clip)
    assert session.is_current("clip_01_tiktok.mp4", clip)

    clip.end += 12.0                    # an extension moved the boundary
    assert not session.is_current("clip_01_tiktok.mp4", clip)


def test_a_dry_run_does_not_make_a_stale_file_look_current():
    """
    The bug this replaced. Comparing "have the boundaries changed since the last
    run" breaks the moment a --dry-run updates the selection without encoding:
    the saved boundaries describe the new clip, the file on disk is still the
    old one, and the comparison finds nothing to do.
    """
    sel = Selection.build(pool(), 3)
    session, _ = fresh_session(sel.chosen)
    session.mark_rendered("clip_02_tiktok.mp4", sel.chosen[1])

    sel.regenerate(1)                   # previewed with --dry-run
    numbers = session.numbers_for(sel.chosen)
    session.record(list(zip(numbers, [key_of(c) for c in sel.chosen], sel.chosen)),
                   sel.rejected)

    replacement = next(c for n, c in zip(numbers, sel.chosen) if n == 2)
    assert not session.is_current("clip_02_tiktok.mp4", replacement)


def test_what_was_rendered_survives_a_round_trip(tmp_path):
    sel = Selection.build(pool(), 3)
    session, _ = fresh_session(sel.chosen)
    session.mark_rendered("clip_01_tiktok.mp4", sel.chosen[0])
    session.save(tmp_path)

    back = Session.load(tmp_path)
    assert back.is_current("clip_01_tiktok.mp4", sel.chosen[0])


def test_records_for_files_no_longer_produced_are_dropped():
    session = Session()
    session.rendered = {"clip_01_tiktok.mp4": (0.0, 20.0),
                        "clip_09_reels.mp4": (0.0, 20.0)}
    session.forget_rendered(["clip_01_tiktok.mp4"])
    assert list(session.rendered) == ["clip_01_tiktok.mp4"]


RECIPE = {"style": "punch", "resolution": "720p", "framing_mode": "face",
          "safe_area": True, "audio_filter": None}


def test_the_first_recipe_is_not_a_change():
    session = Session()
    assert session.adopt_recipe(RECIPE) is False


def test_an_upgraded_session_keeps_its_finished_files():
    """
    A session.json written before recipes existed has none. Reading that as
    "the recipe changed" would re-render an entire episode the first time the
    user updated the project, for nothing.
    """
    session = Session()
    session.rendered = {"clip_01_tiktok.mp4": (0.0, 20.0)}
    assert session.adopt_recipe(RECIPE) is False
    assert session.rendered


def test_same_recipe_twice_renders_nothing_again():
    session = Session()
    session.adopt_recipe(RECIPE)
    session.rendered = {"clip_01_tiktok.mp4": (0.0, 20.0)}
    assert session.adopt_recipe(dict(RECIPE)) is False
    assert session.rendered


def test_changing_the_caption_style_makes_every_file_stale():
    session = Session()
    session.adopt_recipe(RECIPE)
    session.rendered = {"clip_01_tiktok.mp4": (0.0, 20.0)}
    assert session.adopt_recipe({**RECIPE, "style": "clean"}) is True
    assert session.rendered == {}


def test_changing_the_audio_level_makes_every_file_stale():
    session = Session()
    session.adopt_recipe(RECIPE)
    session.rendered = {"clip_01_tiktok.mp4": (0.0, 20.0)}
    assert session.adopt_recipe({**RECIPE, "audio_filter": "volume=6.00dB"}) is True
    assert session.rendered == {}


def test_the_recipe_survives_a_round_trip(tmp_path):
    session = Session(settings=dict(SETTINGS))
    session.adopt_recipe(RECIPE)
    session.save(tmp_path)
    assert Session.load(tmp_path).adopt_recipe(dict(RECIPE)) is False
