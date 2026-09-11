import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.cli.pipeline import build_parser
from ai_clipper.core import Event, PipelineError, Settings, run


LINES = [
    "Kenapa banyak orang gagal di trading saham?",
    "Ternyata masalahnya bukan di analisa teknikal sama sekali.",
    "Gue hampir kena margin call waktu itu dan hampir bangkrut total.",
    "Broker nya nelpon jam dua pagi minta gue nambah dana.",
    "Padahal gue udah taruh delapan puluh persen portofolio di satu saham.",
    "Akhirnya gue nego sama Maybank supaya dikasih waktu tiga hari.",
    "Waktu itu gue nyesel banget nggak pasang stop loss dari awal.",
    "Sekarang gue selalu bagi modal jadi lima bagian yang terpisah.",
]


def transcript(tmp_path, repeats=16):
    """A transcript long enough to produce a real candidate pool."""
    segments, words, t = [], [], 0.0
    for i in range(repeats):
        text = LINES[i % len(LINES)]
        t += 0.3
        start = t
        for token in text.split():
            length = 0.22 + len(token) * 0.028
            words.append({"start": round(t, 3), "end": round(t + length, 3),
                          "text": token})
            t += length + 0.05
        segments.append({"start": round(start, 3), "end": round(t, 3), "text": text})

    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({"segments": segments, "words": words}),
                    encoding="utf-8")
    return str(path)


def dry(tmp_path, **overrides):
    kwargs = dict(
        video="never-opened.mp4",
        transcript=transcript(tmp_path),
        out=str(tmp_path / "out"),
        clips=2,
        min_duration=15.0,
        max_duration=40.0,
        dry_run=True,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


# --- settings ----------------------------------------------------------------

def test_the_defaults_match_the_command_line():
    """
    Two places declare a default and they must not drift. A UI calling
    Settings() and a person running the command with no flags should get the
    same run.
    """
    parsed = Settings.from_args(build_parser().parse_args(["video.mp4", "t.json"]))
    assert parsed == Settings(video="video.mp4", transcript="t.json")


def test_from_args_ignores_anything_it_does_not_know():
    """The parser may grow options that are the command's business alone."""
    args = build_parser().parse_args(["v.mp4", "t.json"])
    args.some_terminal_only_flag = True
    assert Settings.from_args(args).video == "v.mp4"


# --- running -----------------------------------------------------------------

def test_a_dry_run_needs_no_video_and_returns_its_preview(tmp_path):
    result = run(dry(tmp_path))
    assert result.dry_run
    assert not result.cancelled
    assert result.preview and len(result.preview) == 2
    assert result.report is None
    assert set(result.numbers) == {1, 2}

    entry = result.preview[0]
    for key in ("clip", "start", "end", "duration", "score", "opens", "ends",
                "text", "why"):
        assert key in entry
    assert entry["start"] < entry["end"]


def test_a_dry_run_writes_what_the_command_writes(tmp_path):
    out = Path(dry(tmp_path).out)
    run(dry(tmp_path, out=str(out)))
    assert json.loads((out / "dry_run.json").read_text(encoding="utf-8"))
    assert (out / "session.json").exists()
    assert (out / "caption_packs.json").exists()


def test_running_prints_nothing(tmp_path, capsys):
    """
    The whole reason this module exists. A stray print here is invisible in a
    UI and impossible to turn off.
    """
    run(dry(tmp_path))
    assert capsys.readouterr().out == ""


def test_progress_arrives_as_events_with_data_not_only_text(tmp_path):
    seen = []
    run(dry(tmp_path), on_progress=seen.append)

    assert all(isinstance(e, Event) for e in seen)
    kinds = [e.kind for e in seen]
    assert "transcript" in kinds
    assert "selection" in kinds
    assert kinds.count("preview_clip") == 2
    assert kinds[-1] == "done"

    transcript_event = next(e for e in seen if e.kind == "transcript")
    assert transcript_event.data["segments"] > 0
    assert transcript_event.data["words"] > 0

    selection = next(e for e in seen if e.kind == "selection")
    assert selection.data["chosen"] == 2
    assert selection.data["scored"] > 2


def test_every_event_carries_a_line_for_the_terminal(tmp_path):
    seen = []
    run(dry(tmp_path), on_progress=seen.append)
    assert all(e.message.strip() for e in seen)


def test_the_run_is_reported_in_the_order_it_happened(tmp_path):
    seen = []
    run(dry(tmp_path), on_progress=seen.append)
    kinds = [e.kind for e in seen]
    assert kinds.index("transcript") < kinds.index("selection")
    assert kinds.index("selection") < kinds.index("preview_clip")


# --- failing -----------------------------------------------------------------

def test_regenerate_without_a_previous_run_is_an_error_not_an_exit(tmp_path):
    """
    It used to raise SystemExit, which ends a command tidily and kills a UI.
    """
    with pytest.raises(PipelineError) as caught:
        run(dry(tmp_path, regenerate="2"))
    assert "--regenerate" in str(caught.value)


def test_regenerating_a_clip_that_was_never_offered_is_reported_not_fatal(tmp_path):
    settings = dry(tmp_path)
    run(settings)

    seen = []
    run(dry(tmp_path, transcript=settings.transcript, out=settings.out,
            regenerate="9"), on_progress=seen.append)
    skipped = [e for e in seen if e.kind == "regenerate_skipped"]
    assert skipped and skipped[0].data["clip"] == 9


def test_a_second_dry_run_keeps_the_clip_numbers(tmp_path):
    settings = dry(tmp_path)
    first = run(settings)
    second = run(dry(tmp_path, transcript=settings.transcript, out=settings.out))
    assert first.numbers == second.numbers
    assert [c["start"] for c in first.preview] == [c["start"] for c in second.preview]


# --- edits -------------------------------------------------------------------

def test_edits_arrive_with_string_keys_and_are_coerced():
    from ai_clipper.core import ClipEdit, as_edits
    edits = as_edits({"3": {"gain_db": -2.0}})
    assert edits[3] == ClipEdit(gain_db=-2.0)


def test_an_empty_edit_has_no_fingerprint():
    from ai_clipper.core import ClipEdit
    assert ClipEdit().fingerprint() == ""
    assert ClipEdit(gain_db=-2.0).fingerprint()


def test_the_fingerprint_is_stable_and_tells_edits_apart():
    from ai_clipper.core import ClipEdit
    one = ClipEdit(keep_spans=[[0, 5]], gain_db=-2.0)
    same = ClipEdit(keep_spans=[[0, 5]], gain_db=-2.0)
    other = ClipEdit(keep_spans=[[0, 6]], gain_db=-2.0)
    assert one.fingerprint() == same.fingerprint()
    assert one.fingerprint() != other.fingerprint()


def test_a_hand_set_boundary_wins(tmp_path):
    settings = dry(tmp_path)
    before = run(settings)
    first = before.preview[0]

    after = run(dry(tmp_path, transcript=settings.transcript, out=settings.out),
                edits={first["clip"]: {"start": first["start"] + 2.0}})
    moved = next(c for c in after.preview if c["clip"] == first["clip"])
    assert moved["start"] == round(first["start"] + 2.0, 2)


def test_a_trim_that_leaves_nothing_is_refused(tmp_path):
    settings = dry(tmp_path)
    first = run(settings).preview[0]
    with pytest.raises(PipelineError):
        run(dry(tmp_path, transcript=settings.transcript, out=settings.out),
            edits={first["clip"]: {"start": first["start"],
                                   "end": first["start"] + 0.2}})


def test_trimming_is_reported(tmp_path):
    settings = dry(tmp_path)
    first = run(settings).preview[0]
    seen = []
    run(dry(tmp_path, transcript=settings.transcript, out=settings.out),
        edits={first["clip"]: {"end": first["end"] - 2.0}},
        on_progress=seen.append)
    trimmed = [e for e in seen if e.kind == "trimmed"]
    assert trimmed and first["clip"] in trimmed[0].data["clips"]


def test_clips_nobody_edited_are_untouched(tmp_path):
    settings = dry(tmp_path)
    before = run(settings)
    target = before.preview[0]["clip"]
    after = run(dry(tmp_path, transcript=settings.transcript, out=settings.out),
                edits={target: {"end": before.preview[0]["end"] - 1.0}})
    for was, now in zip(before.preview, after.preview):
        if now["clip"] == target:
            continue
        assert (was["start"], was["end"]) == (now["start"], now["end"])


# --- one run at a time -------------------------------------------------------

def test_a_second_run_on_the_same_folder_is_refused(tmp_path):
    """
    Two runs against one folder both write session.json, and the loser does not
    fail, it corrupts: clip numbers from one and rendered files from the other.
    Nothing catches that afterwards, which is why it is a lock and not a
    convention.
    """
    from ai_clipper.core import Busy, LOCK

    settings = dry(tmp_path)
    Path(settings.out).mkdir(parents=True, exist_ok=True)
    (Path(settings.out) / LOCK).write_text("{}")

    with pytest.raises(Busy):
        run(settings)


def test_busy_is_something_a_ui_can_tell_apart(tmp_path):
    """Nothing is wrong here, so it should not read like a broken run."""
    from ai_clipper.core import Busy
    assert issubclass(Busy, PipelineError)


def test_a_lock_left_by_a_run_that_never_came_back_is_taken_over(tmp_path):
    import os
    from ai_clipper.core import LOCK

    settings = dry(tmp_path)
    Path(settings.out).mkdir(parents=True, exist_ok=True)
    stale = Path(settings.out) / LOCK
    stale.write_text("{}")
    os.utime(stale, (0, 0))

    assert run(settings).preview


def test_the_lock_is_released_even_when_the_run_fails(tmp_path):
    from ai_clipper.core import LOCK

    settings = dry(tmp_path, regenerate="2")
    with pytest.raises(PipelineError):
        run(settings)
    assert not (Path(settings.out) / LOCK).exists()


def test_the_lock_does_not_outlive_a_finished_run(tmp_path):
    from ai_clipper.core import LOCK

    settings = dry(tmp_path)
    run(settings)
    assert not (Path(settings.out) / LOCK).exists()
