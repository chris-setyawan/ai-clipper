"""
The pipeline as a function, so something other than a terminal can drive it.

`cli/pipeline.py` used to be the pipeline: it parsed arguments, did the work,
and printed as it went. That is fine for a command and useless for anything
else. A UI that wants a progress bar has to read the printed lines back, and a
stop button has to kill the process, and neither of those is a thing you can
test.

So the work lives here and takes three arguments instead of a command line:

    run(settings, on_progress=..., should_stop=...)

`settings` is a dataclass with one field per command-line option and the same
defaults, so `Settings()` and running with no flags are the same run.

`on_progress` is called with an `Event` at every point the command used to
print. Each event carries both the line the terminal wants and the numbers a
program wants, which is what lets the CLI stay byte-for-byte what it was while a
UI ignores `message` entirely and reads `data`.

`should_stop` is asked before each file. Stopping between files is safe rather
than merely tolerable: renders are atomic and the session is written as each
file lands, so a stopped run is a resumable one.

Nothing here prints. That is the whole point, and it is worth keeping true.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .export import platforms as plat
from .export.caption_pack import build_pack, write_packs
from .input.fetch import resolve
from .scoring import signals as signal_packs
from .scoring.hook_scorer import HookScorer, TranscriptSegment
from .scoring.repunctuate import (
    needs_repunctuation, punctuated_fraction, restore_sentence_ends,
)
from .scoring.selection import Selection
from .scoring.session import Session, key_of
from .scoring.topics import TopicModel, shift_to_a_real_opening
from .video.subtitles import Word, build_ass, load_styles

# OpenCV and PyAV are imported where the video work starts, not here. A dry run
# never opens the video, and it is the loop a user runs most often, so it should
# work on a machine that has nothing installed but Python and numpy.


class PipelineError(Exception):
    """
    A run that cannot proceed, for a reason the caller can show a user.

    The two cases are an unknown caption style and `--regenerate` with no
    previous run to reject from. Both used to raise SystemExit, which is correct
    for a command and wrong for anything holding a window open.
    """


@dataclass
class Settings:
    """One field per command-line option, same names, same defaults."""

    video: str
    transcript: str
    out: str = "out"
    clips: int = 5
    min_duration: float = 15.0
    max_duration: float = 75.0
    hard_max: float = 150.0
    no_extend: bool = False
    regenerate: str = ""
    regenerate_mode: str = "different"
    fresh: bool = False
    dry_run: bool = False
    platforms: str = "tiktok,reels,shorts"
    style: str = "punch"
    styles_file: str = "styles.json"
    signals_file: str = "signals.json"
    resolution: str = "720p"
    downloads: str = "downloads"
    no_cover: bool = False
    no_trim_silence: bool = False
    min_gap: Optional[float] = None
    title: bool = False
    no_loudness: bool = False
    no_safe_area: bool = False
    mode: str = "auto"

    @classmethod
    def from_args(cls, args) -> "Settings":
        """Build from an argparse namespace, taking only the fields we know."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in vars(args).items() if k in known})


@dataclass(frozen=True)
class Event:
    """
    One thing that happened.

    `kind` is stable and is what a program should switch on. `message` is the
    line the command prints, indentation included, because the indentation is
    the terminal's way of saying "this is a detail of the step above". `data`
    holds the same information as values.
    """

    kind: str
    message: str
    data: Dict = field(default_factory=dict)


@dataclass
class Result:
    """What a finished run produced."""

    out_dir: str
    seconds: float
    dry_run: bool = False
    cancelled: bool = False
    rendered: List[str] = field(default_factory=list)
    numbers: List[int] = field(default_factory=list)
    # the contents of run_report.json, or None on a dry run
    report: Optional[Dict] = None
    # the contents of dry_run.json, or None on a real run
    preview: Optional[List[Dict]] = None


class _Stopped(Exception):
    """Raised inside the run when should_stop() says so. Never escapes."""


def load_transcript(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments = [TranscriptSegment(s["start"], s["end"], s["text"]) for s in data["segments"]]
    words = [Word(w["start"], w["end"], w["text"]) for w in data["words"]]
    return segments, words


def eta(started: float, done: int, total: int) -> str:
    """
    Time left, once there is enough evidence to guess with.

    Empty for the first file, because one sample is not a rate and a wrong
    estimate is worse than none.
    """
    if done < 1:
        return ""
    per_file = (time.time() - started) / done
    left = per_file * (total - done)
    if left < 90:
        return f"  about {left:.0f}s left"
    return f"  about {left / 60:.0f}m left"


def run(settings: Settings,
        on_progress: Optional[Callable[[Event], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None) -> Result:
    """
    Pick clips out of an episode and render them.

    Raises PipelineError for a run that cannot start. Everything else is
    reported through `on_progress` and returned in the Result.
    """
    emit = on_progress or (lambda event: None)

    def say(kind: str, message: str, **data) -> None:
        emit(Event(kind, message, data))

    def stopping() -> bool:
        return bool(should_stop and should_stop())

    out_dir = Path(settings.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    video = settings.video
    # a link is downloaded once and reused; a path passes straight through.
    # A dry run never opens the video, so it must not download one either.
    if not settings.dry_run:
        video = resolve(video, dest_dir=settings.downloads)

    has_signals_file = Path(settings.signals_file).exists()
    pack = signal_packs.load(settings.signals_file if has_signals_file else None)
    signal_packs.use(pack)
    if pack.name != signal_packs.INDONESIAN.name or has_signals_file:
        say("signals", f"Signals: {pack.name} (from {settings.signals_file})",
            name=pack.name, path=settings.signals_file)

    segments, words = load_transcript(settings.transcript)
    say("transcript", f"Transcript: {len(segments)} segments, {len(words)} words",
        segments=len(segments), words=len(words))

    # Every completeness rule reads punctuation. A transcript without any makes
    # all of them fire on every candidate, which ranks nothing - see
    # scoring/repunctuate.py for the measurement that found this.
    if needs_repunctuation(segments):
        before = punctuated_fraction(segments)
        segments = restore_sentence_ends(segments)
        after = punctuated_fraction(segments)
        say("repunctuated",
            f"  only {before * 100:.0f}% of segments were punctuated - sentence "
            f"ends inferred from pauses ({after * 100:.0f}% now)",
            before=round(before, 4), after=round(after, 4))

    # --- score ---------------------------------------------------------------
    scorer = HookScorer()
    pool = scorer.candidate_pool(segments, settings.min_duration, settings.max_duration)

    # The pool is a pure function of the transcript and the duration limits, so
    # it is rebuilt rather than stored. What has to be remembered is which
    # windows the user already turned down - otherwise a second run cheerfully
    # hands back the clip they just rejected.
    pool_settings = {
        "transcript": str(settings.transcript),
        "clips": settings.clips,
        "min_duration": settings.min_duration,
        "max_duration": settings.max_duration,
        "hard_max": settings.hard_max,
        "no_extend": bool(settings.no_extend),
    }
    session = Session.load(out_dir) if not settings.fresh else None
    if session and not session.usable_with(pool_settings):
        say("session_reset",
            "Settings changed since the last run, so the saved selection no "
            "longer applies - starting fresh", reason="settings")
        session = None

    selection = Selection.build(pool, settings.clips)
    restored = bool(session and session.restore(selection, pool))
    if session and not restored:
        say("session_reset",
            "The saved selection no longer matches this pool - starting fresh",
            reason="pool")
        session = None

    wanted = [int(n) for n in settings.regenerate.replace(" ", "").split(",") if n]
    if wanted and not restored:
        raise PipelineError(
            "--regenerate needs a previous run to reject from. Run once without "
            "it first, look at the clips, then come back and name the ones to "
            f"replace. (No usable {out_dir}/session.json here.)"
        )

    for number in wanted:
        held = {slot.index: slot.key for slot in session.slots}
        if number not in held:
            say("regenerate_skipped",
                f"  no clip {number} in the last run - skipping",
                clip=number, reason="not in the last run")
            continue
        position = next((i for i, c in enumerate(selection.chosen)
                         if key_of(c) == held[number]), None)
        if position is None:
            say("regenerate_skipped",
                f"  clip {number} is not in the current selection - skipping",
                clip=number, reason="not in the current selection")
            continue
        old_clip = selection.chosen[position]
        replacement = selection.regenerate(position, mode=settings.regenerate_mode)
        if replacement is None:
            say("regenerate_exhausted",
                f"  clip {number} rejected, and nothing else fits between the "
                f"clips you kept", clip=number)
        else:
            say("regenerated",
                f"  clip {number}: {old_clip.start:.0f}-{old_clip.end:.0f}s "
                f"replaced by {replacement.start:.0f}-{replacement.end:.0f}s "
                f"(score {replacement.score})",
                clip=number,
                was=[round(old_clip.start, 2), round(old_clip.end, 2)],
                now=[round(replacement.start, 2), round(replacement.end, 2)],
                score=replacement.score)

    session = session or Session(settings=pool_settings)
    session.settings = pool_settings
    # Both captured before any boundary adjustment: a window's identity is its
    # original pool coordinates, and extension is about to move the ones on the
    # object.
    numbers = session.numbers_for(selection.chosen)
    pool_keys = [key_of(c) for c in selection.chosen]
    entries = list(zip(numbers, pool_keys, selection.chosen))
    say("selection",
        f"Scored {len(pool)} distinct candidates, chose {len(selection.chosen)}, "
        f"{selection.remaining()} held in reserve for regeneration",
        scored=len(pool), chosen=len(selection.chosen),
        reserve=selection.remaining())

    # --- let a clip finish its subject ---------------------------------------
    # Scoring picks where a good moment starts. Where it should stop is a
    # different question, and a window capped at max_duration cannot answer it
    # when the speaker is still explaining. Extension happens after selection so
    # it never changes which moments were chosen, only how much of each is kept.
    extensions: Dict[int, float] = {}
    openings: Dict[int, float] = {}
    if not settings.no_extend:
        topics = TopicModel(segments)
        occupied = [(c.start, c.end) for c in selection.chosen]

        for i, clip in enumerate(selection.chosen):
            ext = topics.extend(clip, hard_max=settings.hard_max)
            if not ext:
                continue
            _, new_end, added = ext
            # never run into the next chosen clip
            nxt = [s for s, _ in occupied if s > clip.end]
            if nxt and new_end > nxt[0]:
                continue
            clip.end = new_end
            clip.segments = [s for s in segments
                             if s.start >= clip.start and s.end <= new_end]
            extensions[numbers[i]] = round(added, 2)
        if extensions:
            say("extended",
                f"Extended {len(extensions)} clips to the end of their subject: "
                + ", ".join(f"#{k} +{v:.0f}s" for k, v in sorted(extensions.items())),
                clips=dict(sorted(extensions.items())))

        # and the same for the other end: enter the moment on a line that works
        # for someone who has seen nothing before it
        for i, clip in enumerate(selection.chosen):
            moved = shift_to_a_real_opening(clip, segments)
            if not moved:
                continue
            new_start, _, dropped = moved
            clip.start = new_start
            clip.segments = [s for s in segments
                             if s.start >= new_start and s.end <= clip.end]
            openings[numbers[i]] = round(dropped, 2)
        if openings:
            say("openings",
                f"Moved {len(openings)} clips to a self-contained opening: "
                + ", ".join(f"#{k} +{v:.0f}s in" for k, v in sorted(openings.items())),
                clips=dict(sorted(openings.items())))

    # --- boundaries only, no video -------------------------------------------
    # Rendering 15 clips across three platforms takes about fourteen minutes,
    # and every question about a clip's boundaries ("does it stop too early?")
    # is answerable from the transcript alone. This makes that loop seconds.
    if settings.dry_run:
        preview = _preview(selection, numbers, extensions, openings, say)
        session.record(entries, selection.rejected)
        session.save(out_dir)
        write_packs(selection.chosen, str(out_dir), numbers=numbers)
        (out_dir / "dry_run.json").write_text(
            json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
        seconds = time.time() - started
        say("done",
            f"\nDone in {seconds:.0f}s (nothing rendered). "
            f"Boundaries in {out_dir}/dry_run.json",
            seconds=round(seconds, 1), dry_run=True)
        return Result(out_dir=str(out_dir), seconds=seconds, dry_run=True,
                      numbers=numbers, preview=preview)

    # --- analyse the video once ----------------------------------------------
    from .video.framing import build_crop_path, recommend_mode
    from .video.render import RenderSpec, render_clip, target_size
    from .video.speakers import find_seats, seats_for_range

    cache = out_dir / "analysis.pkl"
    if cache.exists():
        crop_path, seats = pickle.loads(cache.read_bytes())
        say("analysis_cached", "Reusing cached video analysis", cached=True)
    else:
        say("analysing", "Analysing shots and faces ...", cached=False)
        crop_path = build_crop_path(video)
        seats = find_seats(video)
        cache.write_bytes(pickle.dumps((crop_path, seats)))

    # One measurement for the episode, reused by every clip cut from it, so the
    # quiet moments stay quieter than the loud ones. Cached separately from the
    # frame analysis because it is cheap to redo and the two have no reason to
    # be invalidated together.
    audio_filter = None
    loudness = None
    if not settings.no_loudness:
        from .video import audio as audio_level

        level_cache = out_dir / "loudness.json"
        if level_cache.exists():
            loudness = audio_level.Loudness(**json.loads(level_cache.read_text()))
        else:
            say("measuring_audio", "Measuring audio level ...")
            loudness = audio_level.measure(video)
            if loudness:
                level_cache.write_text(json.dumps(loudness.__dict__), encoding="utf-8")
        audio_filter = audio_level.filter_chain(loudness)
        say("loudness", f"  {audio_level.describe(loudness)}",
            integrated=loudness.integrated if loudness else None,
            true_peak=loudness.true_peak if loudness else None,
            gain=round(audio_level.gain_for(loudness), 2) if loudness else 0.0,
            filter=audio_filter)

    # Planned before rendering so the count can be reported up front, and so the
    # subtitle builder and the renderer read the same plan rather than each
    # working it out.
    from .video import deadair

    min_gap = deadair.MIN_GAP if settings.min_gap is None else settings.min_gap
    timelines = {}
    if not settings.no_trim_silence:
        for i, clip in zip(numbers, selection.chosen):
            timelines[i] = deadair.plan(words, clip.start, clip.end, min_gap=min_gap)
        cut = {i: t for i, t in timelines.items() if t.cuts}
        if cut:
            total = sum(t.removed for t in cut.values())
            say("dead_air",
                f"Trimming dead air from {len(cut)} of {len(timelines)} clips, "
                f"{total:.0f}s in total",
                clips={i: round(t.removed, 2) for i, t in sorted(cut.items())},
                total=round(total, 2))

    mode, reason = recommend_mode(crop_path)
    if settings.mode != "auto":
        mode, reason = settings.mode, "chosen on the command line"
    elif len(seats) >= 2 and len(crop_path.shots) <= 3:
        # a static camera holding two people is exactly the split-view case
        mode, reason = "split", (
            f"{len(seats)} people in a single unbroken shot - split-view shows both"
        )
    say("shots",
        f"  {len(crop_path.shots)} shots, {len(seats)} seats, "
        f"faces in {crop_path.detection_rate * 100:.0f}% of frames",
        shots=len(crop_path.shots), seats=len(seats),
        detection_rate=round(crop_path.detection_rate, 3))
    say("framing", f"  framing mode: {mode} ({reason})", mode=mode, reason=reason)

    # --- render --------------------------------------------------------------
    styles = load_styles(settings.styles_file if Path(settings.styles_file).exists() else None)
    if settings.style not in styles:
        raise PipelineError(
            f"unknown style '{settings.style}'. Known: {', '.join(sorted(styles))}")

    keys = [k.strip() for k in settings.platforms.split(",") if k.strip()]
    plans = plat.plan_exports(selection.chosen, keys, settings.style,
                              settings.resolution, numbers=numbers)

    for index, why in plat.rejected(selection.chosen, keys, numbers=numbers):
        say("clip_skipped", f"  skipping clip {index}: {why}", clip=index, why=why)

    # Two reasons to skip a file, and they are different. A clip whose
    # boundaries have not moved would re-encode to the same bytes. A file that
    # is simply already on disk is finished work from a run that was
    # interrupted. Recording the selection *before* rendering is what makes the
    # second case recoverable: a crash in the middle of a fourteen-minute render
    # used to mean starting over.
    all_plans = plans
    by_number = {n: c for n, c in zip(numbers, selection.chosen)}

    # anything left over from a killed run: the file it belonged to is either
    # about to be rendered again or was never wanted
    for leftover in out_dir.glob("*.part.mp4"):
        leftover.unlink()

    # Everything that changes the bytes without changing the boundaries. The
    # platform list is deliberately not in here: adding a fourth platform should
    # render the fourth platform, not redo the three that were already fine.
    recipe = {
        "style": settings.style,
        "resolution": settings.resolution,
        "framing_mode": mode,
        "safe_area": not settings.no_safe_area,
        "audio_filter": audio_filter,
        "title": bool(settings.title),
        "trim_silence": None if settings.no_trim_silence else min_gap,
    }

    session.record(entries, selection.rejected)
    session.forget_rendered([p.filename for p in all_plans])
    if session.adopt_recipe(recipe):
        say("recipe_changed",
            "  render settings changed since the last run, so every file is "
            "being made again")
    session.save(out_dir)

    # A file is done when it exists *and* was encoded from exactly this clip.
    # Both halves matter: it can be missing after an interrupted run, and it can
    # be present but stale after a regeneration.
    plans = [p for p in all_plans
             if not (out_dir / p.filename).exists()
             or not session.is_current(p.filename, by_number[p.clip_index])]
    already = len(all_plans) - len(plans)
    if already:
        say("resuming",
            f"  {already} files already done, {len(plans)} to render",
            done=already, remaining=len(plans))

    say("render_start", f"\nRendering {len(plans)} files ...", total=len(plans))

    # split mode reframes per clip, so each one gets seats measured inside its
    # own time range rather than the episode-wide average
    clip_seats = {}
    if mode == "split":
        for i, clip in zip(numbers, selection.chosen):
            local = seats_for_range(video, clip.start, clip.end)
            clip_seats[i] = local if len(local) >= 2 else seats

    rendered: List[str] = []
    cancelled = False
    render_started = time.time()

    try:
        for done, plan in enumerate(plans):
            if stopping():
                raise _Stopped
            clip = by_number[plan.clip_index]
            spec = RenderSpec(ratio=plan.ratio, resolution=plan.resolution,
                              mode=mode, audio_filter=audio_filter)
            out_w, out_h = target_size(spec)

            style = (styles[settings.style] if settings.no_safe_area
                     else plat.style_for(styles[settings.style], plan.platform.key))
            clip_words = [w for w in words if w.end > clip.start and w.start < clip.end]
            # the same line the caption pack files under TITLE, so what is on the
            # screen and what is in the copy cannot drift apart
            title = build_pack(clip, plan.clip_index).title if settings.title else None

            # With cuts, the word times are moved onto the shortened timeline and
            # the file starts at zero. Without them, nothing changes and the
            # subtitle builder does the shifting itself, as it always has.
            timeline = timelines.get(plan.clip_index)
            cutting = bool(timeline and timeline.cuts)
            ass_words = (deadair.shift_words(clip_words, timeline, Word)
                         if cutting else clip_words)
            ass_path = out_dir / f"clip_{plan.clip_index:02d}_{plan.platform.key}.ass"
            ass_path.write_text(
                build_ass(ass_words, out_w, out_h, style,
                          clip_start=0.0 if cutting else clip.start, title=title),
                encoding="utf-8",
            )

            target = out_dir / plan.filename
            length = timeline.duration if cutting else clip.duration
            say("rendering",
                f"  [{done + 1}/{len(plans)}] {plan.filename} "
                f"({length:.0f}s){eta(render_started, done, len(plans))}",
                index=done + 1, total=len(plans), file=plan.filename,
                clip=plan.clip_index, platform=plan.platform.key,
                duration=round(length, 2))
            render_clip(video, clip.start, clip.end, str(target), spec,
                        str(ass_path), crop_path,
                        clip_seats.get(plan.clip_index, seats),
                        keep_spans=timeline.ranges() if cutting else None)
            rendered.append(plan.filename)
            # recorded as each file lands, so an interrupted run resumes from the
            # last finished file rather than the last finished run
            session.mark_rendered(plan.filename, clip)
            session.save(out_dir)
    except _Stopped:
        cancelled = True
        say("cancelled",
            f"\nStopped after {len(rendered)} files. Run again to carry on.",
            rendered=len(rendered))

    # --- one cover frame per clip --------------------------------------------
    # Per clip rather than per file: the same moment is the right one whatever
    # ratio it is cropped to, and fifteen clips across three platforms would
    # otherwise mean forty-five near-identical images.
    covers: Dict[int, Optional[float]] = {}
    if not settings.no_cover and not cancelled:
        from .video import cover

        first_plan = {}
        for plan in all_plans:
            first_plan.setdefault(plan.clip_index, plan)

        for index, plan in sorted(first_plan.items()):
            if stopping():
                cancelled = True
                break
            clip = by_number[index]
            source_clip = out_dir / plan.filename
            if not source_clip.exists():
                continue

            # The frame comes out of the finished clip, so it carries the
            # platform's framing and the burned-in caption without this having
            # to reproduce either.
            at = cover.pick_time(crop_path, clip.start, clip.end, words,
                                 per_chunk=styles[settings.style].words_per_chunk)
            source_time = at if at is not None else (clip.start + clip.end) / 2

            timeline = timelines.get(index)
            in_clip = (timeline.remap(source_time)
                       if timeline and timeline.cuts else source_time - clip.start)

            path = out_dir / f"clip_{index:02d}_cover.jpg"
            try:
                cover.grab(str(source_clip), in_clip, str(path))
            except RuntimeError as exc:
                say("cover_failed", f"  no cover for clip {index}: {exc}",
                    clip=index, error=str(exc))
                continue
            covers[index] = round(source_time, 2) if at is not None else None

        confident = sum(1 for v in covers.values() if v is not None)
        say("covers",
            f"{len(covers)} cover frames, {confident} of them on a moment "
            f"the tracker was sure about",
            written=len(covers), confident=confident)

    # The report describes the whole current set, not only what was re-encoded.
    # A run that replaced one clip should still leave a report you can read on
    # its own.
    outputs = []
    for plan in all_plans:
        clip = by_number[plan.clip_index]
        style = (styles[settings.style] if settings.no_safe_area
                 else plat.style_for(styles[settings.style], plan.platform.key))
        outputs.append({
            "file": plan.filename,
            "clip": plan.clip_index,
            "platform": plan.platform.key,
            "ratio": plan.ratio,
            "framing_mode": mode,
            "start": round(clip.start, 2),
            "end": round(clip.end, 2),
            "duration": round(clip.duration, 2),
            "dead_air_removed": round(timelines[plan.clip_index].removed, 2)
                                if plan.clip_index in timelines else 0,
            "cover_at": covers.get(plan.clip_index),
            "score": clip.score,
            "raw_score": round(clip.raw_score, 2),
            "percentile": clip.percentile,
            "extended_by": extensions.get(plan.clip_index, 0),
            "opening_moved_by": openings.get(plan.clip_index, 0),
            "rendered_this_run": plan.filename in rendered,
            "signals": {k: round(v, 2) for k, v in clip.signals.items() if v > 0},
            "why": clip.explain(),
            "caption_style": style.name,
            "caption_bottom_margin": style.bottom_margin_frac,
        })

    report = {
        "source": video,
        "framing_mode": mode,
        "framing_reason": reason,
        "shots": len(crop_path.shots),
        "seats": len(seats),
        "face_detection_rate": round(crop_path.detection_rate, 3),
        "source_loudness_lufs": round(loudness.integrated, 2) if loudness else None,
        "audio_filter": audio_filter,
        "candidates_scored": len(pool),
        "candidates_in_reserve": selection.remaining(),
        "outputs": outputs,
    }

    # --- the writing and the paperwork ---------------------------------------
    write_packs(selection.chosen, str(out_dir), numbers=numbers)
    (out_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    seconds = time.time() - started
    if not cancelled:
        say("done", f"\nDone in {seconds:.0f}s", seconds=round(seconds, 1))
        say("summary",
            f"{len(rendered)} files written, {len(selection.chosen)} caption packs, "
            f"and run_report.json in {out_dir}/",
            written=len(rendered), clips=len(selection.chosen), out=str(out_dir))

    return Result(out_dir=str(out_dir), seconds=seconds, cancelled=cancelled,
                  rendered=rendered, numbers=numbers, report=report)


def _preview(selection, numbers, extensions, openings, say) -> List[Dict]:
    """
    The dry run's report, and the events that describe it.

    In clip-number order, not time order: after a rejection the replacement can
    sit anywhere in the episode, and a list that jumps from clip 4 to clip 3
    reads as a bug.
    """
    preview = []
    for i, clip in sorted(zip(numbers, selection.chosen), key=lambda p: p[0]):
        headline = (
            f"\nclip {i:02d}  {clip.start:.1f}-{clip.end:.1f}  "
            f"{clip.duration:.0f}s  score {clip.score}"
            + (f"  (+{extensions[i]:.0f}s to finish the subject)"
               if i in extensions else "")
            + (f"  (started {openings[i]:.0f}s later, on a line that "
               f"stands alone)" if i in openings else "")
        )
        entry = {
            "clip": i,
            "start": round(clip.start, 2),
            "end": round(clip.end, 2),
            "duration": round(clip.duration, 2),
            "score": clip.score,
            "extended_by": extensions.get(i, 0),
            "opening_moved_by": openings.get(i, 0),
            "opens": clip.segments[0].text.strip(),
            "ends": clip.segments[-1].text.strip(),
            # the whole clip, because opening and closing lines are not enough
            # to judge whether the middle is worth watching
            "text": clip.text,
            "why": clip.explain(),
        }
        say("preview_clip",
            headline + f"\n  opens: {entry['opens']}"
                       f"\n  ends : {entry['ends']}"
                       f"\n  why  : {entry['why']}",
            **entry)
        preview.append(entry)
    return preview
