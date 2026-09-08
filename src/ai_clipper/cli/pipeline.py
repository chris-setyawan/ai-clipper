"""
The whole thing, end to end.

    ai-clipper <video> <transcript.json> [-o out] [--clips 5]
               [--platforms tiktok,reels,shorts] [--style punch]

Takes a video and its transcript, and writes ready-to-post clips: scored,
reframed, captioned inside each platform's safe area, plus the caption copy and
a report of how every decision was made.

Transcription is a separate step (transcribe_local.py) because it is the only
part that needs a model download, and because caching it makes every re-run of
this script fast.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

from ..export import platforms as plat
from ..input.fetch import resolve
from ..export.caption_pack import write_packs
from ..scoring.hook_scorer import HookScorer, TranscriptSegment
from ..scoring.repunctuate import needs_repunctuation, punctuated_fraction, restore_sentence_ends
from ..scoring import signals as signal_packs
from ..scoring.selection import Selection
from ..scoring.session import Session, key_of
from ..scoring.topics import TopicModel, shift_to_a_real_opening
from ..video.subtitles import Word, build_ass, load_styles

# OpenCV and PyAV are imported where the video work starts, not here. A
# --dry-run never opens the video, and it is the loop a user runs most often, so
# it should work on a machine that has nothing installed but Python and numpy.


def _eta(started, done, total):
    """
    Time left, once there is enough evidence to guess with.

    Nothing is printed for the first file, because one sample is not a rate and
    a wrong estimate is worse than none.
    """
    if done < 1:
        return ""
    per_file = (time.time() - started) / done
    left = per_file * (total - done)
    if left < 90:
        return f"  about {left:.0f}s left"
    return f"  about {left / 60:.0f}m left"


def load_transcript(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments = [TranscriptSegment(s["start"], s["end"], s["text"]) for s in data["segments"]]
    words = [Word(w["start"], w["end"], w["text"]) for w in data["words"]]
    return segments, words


def main():
    ap = argparse.ArgumentParser(description="Turn a video into ready-to-post clips.")
    ap.add_argument("video", help="path to a video file, or a link to download")
    ap.add_argument("transcript")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--min-duration", type=float, default=15.0)
    ap.add_argument("--max-duration", type=float, default=75.0,
                    help="longest window the scorer will consider (default 75s). "
                         "It was 60, which is shorter than some stories take to "
                         "tell - a clip that has to stop before the point lands "
                         "is worse than a long one.")
    ap.add_argument("--hard-max", type=float, default=150.0,
                    help="a clip may run past --max-duration up to this, but "
                         "only to reach the end of the subject (default 150s)")
    ap.add_argument("--no-extend", action="store_true",
                    help="do not extend clips to the end of their subject")
    ap.add_argument("--regenerate", default="",
                    help="clip numbers to replace, e.g. --regenerate 4 or 4,7. "
                         "The rejected windows are remembered in session.json, "
                         "so they are never offered again, and every other clip "
                         "keeps its number and its existing files.")
    ap.add_argument("--regenerate-mode", default="different",
                    choices=["different", "retime"],
                    help="'different' looks elsewhere in the episode; 'retime' "
                         "keeps the same moment and re-cuts it (default: different)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore session.json and start the selection over")
    ap.add_argument("--dry-run", action="store_true",
                    help="pick the clips and print where each one starts and "
                         "ends, without touching the video. Seconds instead of "
                         "minutes, for judging boundaries before committing to "
                         "a render.")
    ap.add_argument("--platforms", default="tiktok,reels,shorts")
    ap.add_argument("--style", default="punch")
    ap.add_argument("--styles-file", default="styles.json")
    ap.add_argument("--signals-file", default="signals.json",
                    help="word lists and weights for the scorer. Picked up "
                         "automatically if it sits next to the project; see "
                         "signals.example.json")
    ap.add_argument("--resolution", default="720p", choices=["720p", "1080p"])
    ap.add_argument("--downloads", default="downloads",
                    help="where downloaded videos are kept (default: downloads/)")
    ap.add_argument("--no-safe-area", action="store_true",
                    help="keep captions centred instead of shifting them clear "
                         "of each platform's interface")
    ap.add_argument("--mode", default="auto",
                    help="auto | face | per_shot | split | crop | blur")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # a link is downloaded once and reused; a path passes straight through.
    # A dry run never opens the video, so it must not download one either.
    if not args.dry_run:
        args.video = resolve(args.video, dest_dir=args.downloads)

    pack = signal_packs.load(args.signals_file if Path(args.signals_file).exists() else None)
    signal_packs.use(pack)
    if pack.name != signal_packs.INDONESIAN.name or Path(args.signals_file).exists():
        print(f"Signals: {pack.name} (from {args.signals_file})")

    segments, words = load_transcript(args.transcript)
    print(f"Transcript: {len(segments)} segments, {len(words)} words")

    # Every completeness rule reads punctuation. A transcript without any makes
    # all of them fire on every candidate, which ranks nothing - see
    # scoring/repunctuate.py for the measurement that found this.
    if needs_repunctuation(segments):
        before = punctuated_fraction(segments)
        segments = restore_sentence_ends(segments)
        print(f"  only {before * 100:.0f}% of segments were punctuated - sentence "
              f"ends inferred from pauses ({punctuated_fraction(segments) * 100:.0f}% now)")

    # --- score -------------------------------------------------------------
    scorer = HookScorer()
    pool = scorer.candidate_pool(segments, args.min_duration, args.max_duration)

    # The pool is a pure function of the transcript and the duration limits, so
    # it is rebuilt rather than stored. What has to be remembered is which
    # windows the user already turned down - otherwise a second run cheerfully
    # hands back the clip they just rejected.
    settings = {
        "transcript": str(args.transcript),
        "clips": args.clips,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "hard_max": args.hard_max,
        "no_extend": bool(args.no_extend),
    }
    session = Session.load(out_dir) if not args.fresh else None
    if session and not session.usable_with(settings):
        print("Settings changed since the last run, so the saved selection no "
              "longer applies - starting fresh")
        session = None

    selection = Selection.build(pool, args.clips)
    restored = bool(session and session.restore(selection, pool))
    if session and not restored:
        print("The saved selection no longer matches this pool - starting fresh")
        session = None

    wanted = [int(n) for n in args.regenerate.replace(" ", "").split(",") if n]
    if wanted and not restored:
        raise SystemExit(
            "--regenerate needs a previous run to reject from. Run once without "
            "it first, look at the clips, then come back and name the ones to "
            f"replace. (No usable {out_dir}/session.json here.)"
        )

    for number in wanted:
        held = {slot.index: slot.key for slot in session.slots}
        if number not in held:
            print(f"  no clip {number} in the last run - skipping")
            continue
        position = next((i for i, c in enumerate(selection.chosen)
                         if key_of(c) == held[number]), None)
        if position is None:
            print(f"  clip {number} is not in the current selection - skipping")
            continue
        old_clip = selection.chosen[position]
        replacement = selection.regenerate(position, mode=args.regenerate_mode)
        if replacement is None:
            print(f"  clip {number} rejected, and nothing else fits between the "
                  f"clips you kept")
        else:
            print(f"  clip {number}: {old_clip.start:.0f}-{old_clip.end:.0f}s "
                  f"replaced by {replacement.start:.0f}-{replacement.end:.0f}s "
                  f"(score {replacement.score})")

    session = session or Session(settings=settings)
    session.settings = settings
    # Both captured before any boundary adjustment: a window's identity is its
    # original pool coordinates, and extension is about to move the ones on the
    # object.
    numbers = session.numbers_for(selection.chosen)
    pool_keys = [key_of(c) for c in selection.chosen]
    entries = list(zip(numbers, pool_keys, selection.chosen))
    print(f"Scored {len(pool)} distinct candidates, chose {len(selection.chosen)}, "
          f"{selection.remaining()} held in reserve for regeneration")

    # --- let a clip finish its subject --------------------------------------
    # Scoring picks where a good moment starts. Where it should stop is a
    # different question, and a window capped at --max-duration cannot answer it
    # when the speaker is still explaining. Extension happens after selection so
    # it never changes which moments were chosen, only how much of each is kept.
    extensions = {}
    openings = {}
    if not args.no_extend:
        topics = TopicModel(segments)
        occupied = [(c.start, c.end) for c in selection.chosen]

        for i, clip in enumerate(selection.chosen):
            ext = topics.extend(clip, hard_max=args.hard_max)
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
            print(f"Extended {len(extensions)} clips to the end of their subject: "
                  + ", ".join(f"#{k} +{v:.0f}s" for k, v in sorted(extensions.items())))

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
            print(f"Moved {len(openings)} clips to a self-contained opening: "
                  + ", ".join(f"#{k} +{v:.0f}s in" for k, v in sorted(openings.items())))

    # --- boundaries only, no video ------------------------------------------
    # Rendering 15 clips across three platforms takes about fourteen minutes,
    # and every question about a clip's boundaries ("does it stop too early?")
    # is answerable from the transcript alone. This makes that loop seconds.
    if args.dry_run:
        preview = []
        # in clip-number order, not time order: after a rejection the
        # replacement can sit anywhere in the episode, and a list that jumps
        # from clip 4 to clip 3 reads as a bug
        for i, clip in sorted(zip(numbers, selection.chosen), key=lambda p: p[0]):
            print(f"\nclip {i:02d}  {clip.start:.1f}-{clip.end:.1f}  "
                  f"{clip.duration:.0f}s  score {clip.score}"
                  + (f"  (+{extensions[i]:.0f}s to finish the subject)"
                     if i in extensions else "")
                  + (f"  (started {openings[i]:.0f}s later, on a line that "
                     f"stands alone)" if i in openings else ""))
            print(f"  opens: {clip.segments[0].text.strip()}")
            print(f"  ends : {clip.segments[-1].text.strip()}")
            print(f"  why  : {clip.explain()}")
            preview.append({
                "clip": i,
                "start": round(clip.start, 2),
                "end": round(clip.end, 2),
                "duration": round(clip.duration, 2),
                "score": clip.score,
                "extended_by": extensions.get(i, 0),
                "opening_moved_by": openings.get(i, 0),
                "opens": clip.segments[0].text.strip(),
                "ends": clip.segments[-1].text.strip(),
                # the whole clip, because opening and closing lines are not
                # enough to judge whether the middle is worth watching
                "text": clip.text,
                "why": clip.explain(),
            })
        session.record(entries, selection.rejected)
        session.save(out_dir)
        write_packs(selection.chosen, str(out_dir), numbers=numbers)
        (out_dir / "dry_run.json").write_text(
            json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nDone in {time.time() - started:.0f}s (nothing rendered). "
              f"Boundaries in {out_dir}/dry_run.json")
        return

    # --- analyse the video once -------------------------------------------
    from ..video.framing import build_crop_path, recommend_mode
    from ..video.render import RenderSpec, render_clip, target_size
    from ..video.speakers import find_seats, seats_for_range

    cache = out_dir / "analysis.pkl"
    if cache.exists():
        crop_path, seats = pickle.loads(cache.read_bytes())
        print("Reusing cached video analysis")
    else:
        print("Analysing shots and faces ...")
        crop_path = build_crop_path(args.video)
        seats = find_seats(args.video)
        cache.write_bytes(pickle.dumps((crop_path, seats)))

    mode, reason = recommend_mode(crop_path)
    if args.mode != "auto":
        mode, reason = args.mode, "chosen on the command line"
    elif len(seats) >= 2 and len(crop_path.shots) <= 3:
        # a static camera holding two people is exactly the split-view case
        mode, reason = "split", (
            f"{len(seats)} people in a single unbroken shot - split-view shows both"
        )
    print(f"  {len(crop_path.shots)} shots, {len(seats)} seats, "
          f"faces in {crop_path.detection_rate * 100:.0f}% of frames")
    print(f"  framing mode: {mode} ({reason})")

    # --- render -------------------------------------------------------------
    styles = load_styles(args.styles_file if Path(args.styles_file).exists() else None)
    if args.style not in styles:
        raise SystemExit(f"unknown style '{args.style}'. Known: {', '.join(sorted(styles))}")

    keys = [k.strip() for k in args.platforms.split(",") if k.strip()]
    plans = plat.plan_exports(selection.chosen, keys, args.style, args.resolution,
                              numbers=numbers)

    for index, why in plat.rejected(selection.chosen, keys, numbers=numbers):
        print(f"  skipping clip {index}: {why}")

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

    session.record(entries, selection.rejected)
    session.forget_rendered([p.filename for p in all_plans])
    session.save(out_dir)

    # A file is done when it exists *and* was encoded from exactly this clip.
    # Both halves matter: it can be missing after an interrupted run, and it can
    # be present but stale after a regeneration.
    plans = [p for p in all_plans
             if not (out_dir / p.filename).exists()
             or not session.is_current(p.filename, by_number[p.clip_index])]
    done = len(all_plans) - len(plans)
    if done:
        print(f"  {done} files already done, {len(plans)} to render")

    print(f"\nRendering {len(plans)} files ...")
    # split mode reframes per clip, so each one gets seats measured inside its
    # own time range rather than the episode-wide average
    clip_seats = {}
    if mode == "split":
        for i, clip in zip(numbers, selection.chosen):
            local = seats_for_range(args.video, clip.start, clip.end)
            clip_seats[i] = local if len(local) >= 2 else seats

    rendered = set()
    render_started = time.time()
    for done, plan in enumerate(plans):
        clip = by_number[plan.clip_index]
        spec = RenderSpec(ratio=plan.ratio, resolution=plan.resolution, mode=mode)
        out_w, out_h = target_size(spec)

        style = (styles[args.style] if args.no_safe_area
                 else plat.style_for(styles[args.style], plan.platform.key))
        clip_words = [w for w in words if w.end > clip.start and w.start < clip.end]
        ass_path = out_dir / f"clip_{plan.clip_index:02d}_{plan.platform.key}.ass"
        ass_path.write_text(
            build_ass(clip_words, out_w, out_h, style, clip_start=clip.start),
            encoding="utf-8",
        )

        target = out_dir / plan.filename
        print(f"  [{done + 1}/{len(plans)}] {plan.filename} "
              f"({clip.duration:.0f}s){_eta(render_started, done, len(plans))}",
              flush=True)
        render_clip(args.video, clip.start, clip.end, str(target), spec,
                    str(ass_path), crop_path, clip_seats.get(plan.clip_index, seats))
        rendered.add(plan.filename)
        # recorded as each file lands, so an interrupted run resumes from the
        # last finished file rather than the last finished run
        session.mark_rendered(plan.filename, clip)
        session.save(out_dir)

    # The report describes the whole current set, not only what was re-encoded.
    # A run that replaced one clip should still leave a report you can read on
    # its own.
    report = []
    for plan in all_plans:
        clip = by_number[plan.clip_index]
        style = (styles[args.style] if args.no_safe_area
                 else plat.style_for(styles[args.style], plan.platform.key))
        report.append({
            "file": plan.filename,
            "clip": plan.clip_index,
            "platform": plan.platform.key,
            "ratio": plan.ratio,
            "framing_mode": mode,
            "start": round(clip.start, 2),
            "end": round(clip.end, 2),
            "duration": round(clip.duration, 2),
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

    # --- the writing and the paperwork -------------------------------------
    write_packs(selection.chosen, str(out_dir), numbers=numbers)

    (out_dir / "run_report.json").write_text(
        json.dumps({
            "source": args.video,
            "framing_mode": mode,
            "framing_reason": reason,
            "shots": len(crop_path.shots),
            "seats": len(seats),
            "face_detection_rate": round(crop_path.detection_rate, 3),
            "candidates_scored": len(pool),
            "candidates_in_reserve": selection.remaining(),
            "outputs": report,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone in {time.time() - started:.0f}s")
    print(f"{len(rendered)} files written, {len(selection.chosen)} caption packs, "
          f"and run_report.json in {out_dir}/")
    print(f"Not happy with one? python run_pipeline.py ... --regenerate <number>")
