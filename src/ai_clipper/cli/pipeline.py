"""
The command line around the pipeline.

    ai-clipper <video> <transcript.json> [-o out] [--clips 5]
               [--platforms tiktok,reels,shorts] [--style punch]

Takes a video and its transcript, and writes ready-to-post clips: scored,
reframed, captioned inside each platform's safe area, plus the caption copy and
a report of how every decision was made.

This file does three things and nothing else: read the arguments, print what
comes back, and turn a failure into an exit code. The work is in
`ai_clipper.core`, so a UI can call it without a terminal in the way.

Transcription is a separate step (transcribe_local.py) because it is the only
part that needs a model download, and because caching it makes every re-run of
this script fast.
"""

import argparse

from ..core import Event, PipelineError, Settings, load_transcript, run

# re-exported: the tests and a few scripts import it from here
__all__ = ["main", "build_parser", "load_transcript"]


def show(event: Event) -> None:
    """
    Print an event the way the command has always printed it.

    Every message carries its own indentation, and the ones that used to start
    with a blank line still do, so this stays a one-liner rather than a place
    where formatting decisions accumulate.
    """
    print(event.message, flush=True)


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--no-cover", action="store_true",
                    help="do not write a cover frame beside each clip")
    ap.add_argument("--no-trim-silence", action="store_true",
                    help="keep every pause at its original length")
    ap.add_argument("--min-gap", type=float, default=None,
                    help="shortest pause that counts as dead air, in seconds "
                         "(default 1.5). Raise it if the cuts feel rushed.")
    ap.add_argument("--title", action="store_true",
                    help="burn the hook across the opening seconds of each "
                         "clip. Off by default: the hook is the first sentence "
                         "of the transcript, so it repeats the subtitle that "
                         "arrives underneath it a moment later. See the README.")
    ap.add_argument("--no-loudness", action="store_true",
                    help="export the audio at whatever level it sits at in the "
                         "source, instead of matching what the platforms "
                         "normalise towards")
    ap.add_argument("--no-safe-area", action="store_true",
                    help="keep captions centred instead of shifting them clear "
                         "of each platform's interface")
    ap.add_argument("--mode", default="auto",
                    help="auto | face | per_shot | split | crop | blur")
    return ap


def main():
    args = build_parser().parse_args()

    try:
        result = run(Settings.from_args(args), on_progress=show)
    except PipelineError as exc:
        raise SystemExit(str(exc))
    except KeyboardInterrupt:
        # Renders are atomic and the session is written as each file lands, so
        # there is nothing to clean up and nothing to warn about.
        raise SystemExit("\nStopped. Run the same command again to carry on.")

    if not result.dry_run and not result.cancelled:
        print("Not happy with one? python run_pipeline.py ... --regenerate <number>")
