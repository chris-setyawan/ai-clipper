# Interface reference

What the pipeline takes, what it writes, and what a program on the other side of
it can rely on. Written for building a UI against this project without reading
the source first.

Everything below was taken from a real fifteen-clip run, not from the code.

## Two commands

Installed, they are `ai-clipper-transcribe` and `ai-clipper`. From a clone they
are `python transcribe_local.py` and `python run_pipeline.py`, same arguments.

### Transcription

```
ai-clipper-transcribe <media> [options]
```

| option | default | what it does |
|---|---|---|
| `--model` | `small` | Whisper size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--language` | `id` | language code |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--lexicon` | none | path to a `lexicon.json` of known transcription fixes |
| `--out` | beside the media | where to write the transcript |

`media` accepts a link as well as a path. This is the only step that downloads a
model, and it is separate from the rest so that its result can be cached and
every later run is fast.

The transcript it writes is the second argument to the pipeline:

```json
{
  "segments": [{"start": 31.04, "end": 34.9, "text": "Makanya agak unik..."}],
  "words":    [{"start": 31.04, "end": 31.3, "text": "Makanya"}]
}
```

Both lists are required. Word timings drive the karaoke captions, the dead air
plan and the cover choice, so a transcript without them will not work.

### The pipeline

```
ai-clipper <video> <transcript.json> [options]
```

`video` accepts a link. A downloaded file is kept in `--downloads` and reused.

**Choosing clips**

| option | default | what it does |
|---|---|---|
| `--clips` | 5 | how many to pick |
| `--min-duration` | 15 | shortest window the scorer considers, seconds |
| `--max-duration` | 75 | longest window the scorer considers |
| `--hard-max` | 150 | how far a clip may run past `--max-duration` to finish its subject |
| `--no-extend` | off | do not extend a clip to the end of its subject |
| `--signals-file` | `signals.json` | word banks and weights for the scorer |

**Reviewing and replacing**

| option | default | what it does |
|---|---|---|
| `--dry-run` | off | pick the clips and write `dry_run.json`, touch no video |
| `--regenerate` | none | clip numbers to replace, `4` or `4,7` |
| `--regenerate-mode` | `different` | `different` looks elsewhere, `retime` re-cuts the same moment |
| `--fresh` | off | ignore `session.json` and start the selection over |

**Output**

| option | default | what it does |
|---|---|---|
| `-o`, `--out` | `out` | output folder |
| `--platforms` | `tiktok,reels,shorts` | also `feed`, `square`, `youtube` |
| `--resolution` | `720p` | `720p` or `1080p` |
| `--style` | `punch` | caption style: `clean`, `punch`, `calm`, or one from `styles.json` |
| `--styles-file` | `styles.json` | user styles layered over the presets |
| `--mode` | `auto` | framing: `face`, `per_shot`, `split`, `crop`, `blur` |
| `--no-safe-area` | off | keep captions centred instead of clear of the platform interface |

**Finishing**

| option | default | what it does |
|---|---|---|
| `--no-trim-silence` | off | keep every pause at its original length |
| `--min-gap` | 1.5 | shortest pause counted as dead air, seconds |
| `--no-loudness` | off | export at the source level instead of matching the platforms |
| `--no-cover` | off | do not write a cover frame per clip |
| `--title` | off | burn the hook over the opening seconds. See the README for why this is off |

Exit code is 0 on success. A bad `--style`, or `--regenerate` with no previous
run to reject from, exits non-zero with the reason on stderr.

## What lands in the output folder

For fifteen clips across three platforms, 45 videos and their sidecars:

| file | one per | what it is |
|---|---|---|
| `clip_01_tiktok.mp4` | clip and platform | the rendered clip |
| `clip_01_tiktok.ass` | clip and platform | the subtitles behind it, editable, re-burnable |
| `clip_01_tiktok.cmd` | clip and platform | the crop path, only in `face` and `per_shot` mode |
| `clip_01_cover.jpg` | clip | the cover frame |
| `clip_01_caption.txt` | clip | title, description and hashtags, for a person to read |
| `caption_packs.json` | run | the same copy, for a program to read |
| `run_report.json` | run | every decision the run made |
| `dry_run.json` | run | boundaries only, written by `--dry-run` |
| `session.json` | run | what was chosen, what was rejected, what was rendered |
| `analysis.pkl` | run | cached shot and face analysis, pickled |
| `loudness.json` | run | cached loudness measurement of the source |

`analysis.pkl` and `loudness.json` are caches. Deleting them costs time on the
next run and nothing else. A `.part.mp4` is an unfinished render and is cleaned
up automatically at the start of the next run.

Copying `analysis.pkl` and `session.json` into a new output folder is how you
re-render an episode with different settings without redoing the analysis or
losing the clips you already rejected.

## run_report.json

Written after every render. Describes the whole current set, not only the files
this run re-encoded.

```json
{
  "source": "downloads\\1ziIpehWMiI.mp4",
  "framing_mode": "per_shot",
  "framing_reason": "shots change every 6.0s on average, so the edit is already doing the reframing",
  "shots": 291,
  "seats": 1,
  "face_detection_rate": 1.0,
  "source_loudness_lufs": -16.12,
  "audio_filter": "volume=2.12dB,alimiter=limit=0.8913:attack=5:release=50:level=disabled",
  "candidates_scored": 400,
  "candidates_in_reserve": 384,
  "outputs": [ ... ]
}
```

`candidates_in_reserve` is how many replacements `--regenerate` still has. When
it reaches zero, rejecting another clip has nothing to offer.

One entry in `outputs` per rendered file:

```json
{
  "file": "clip_01_tiktok.mp4",
  "clip": 1,
  "platform": "tiktok",
  "ratio": "9:16",
  "framing_mode": "per_shot",
  "start": 31.04,
  "end": 97.22,
  "duration": 66.18,
  "dead_air_removed": 4.42,
  "cover_at": 55.42,
  "score": 9.7,
  "raw_score": 19.21,
  "percentile": 100.0,
  "extended_by": 0,
  "opening_moved_by": 0,
  "rendered_this_run": false,
  "signals": {"opening_question": 2.5, "stakes": 1.2, "...": 0},
  "why": "topic_distinctiveness (+4.5); opening_question (+2.5); ...",
  "caption_style": "punch@tiktok",
  "caption_bottom_margin": 0.23
}
```

Notes on fields that are easy to misread:

- `start` and `end` are positions in the source episode. `duration` is
  `end - start`, so on a clip with `dead_air_removed` above zero the file on
  disk is shorter than `duration` by that amount.
- `cover_at` is also a source position, not an offset into the clip. It is
  `null` when no moment in the clip had a confident face, in which case the
  midpoint was used and nothing claims otherwise.
- `score` is on a 0 to 10 scale fitted to this episode. `raw_score` is the sum
  of the signals before fitting, and is not comparable between episodes.
  `percentile` is the position within this run's candidate pool.
- `rendered_this_run` is false for a file that was already current and skipped.
- `signals` holds only the signals that fired. `why` is the same information as
  a sentence, already sorted, ready to show a user.

## caption_packs.json

One entry per clip, the same content as the `_caption.txt` files.

```json
{
  "clip_index": 1,
  "hook": "Makanya agak unik, lu dari dulu dia tuh sukanya nongkrong di pik, padahal waktu...",
  "title": "Makanya agak unik, lu dari dulu dia tuh sukanya nongkrong...",
  "description": "Makanya agak unik, lu dari dulu ...",
  "hashtags": ["#podcast", "#shorts", "#fyp", "#bisnis"],
  "start": 31.04,
  "end": 97.22,
  "score": 9.7,
  "why": "topic_distinctiveness (+4.5); ..."
}
```

These are drafts taken from the transcript, not written copy. A UI should
present them as editable rather than final.

## dry_run.json

Written by `--dry-run`, which never opens the video. This is the file to build a
review screen on: seconds instead of minutes per iteration.

```json
{
  "clip": 1,
  "start": 31.04,
  "end": 97.22,
  "duration": 66.18,
  "score": 9.7,
  "extended_by": 0,
  "opening_moved_by": 0,
  "opens": "Makanya agak unik, lu dari dulu dia tuh sukanya nongkrong di pik...",
  "ends": "DP-nya itu gue partneran sama temen gue.",
  "text": "the whole clip, so the middle can be judged too",
  "why": "topic_distinctiveness (+4.5); opening_question (+2.5); ..."
}
```

`extended_by` is how many seconds the clip was allowed to run past its scored
window to finish its subject. `opening_moved_by` is how many seconds were
dropped from the front to start on a line that stands alone. Both are zero when
nothing moved.

Clips are listed in clip-number order, not time order, because after a rejection
a replacement can come from anywhere in the episode.

## session.json

The memory between runs. A UI that offers a reject button is really editing this
file, through `--regenerate`.

```json
{
  "settings": {
    "transcript": "downloads\\transcript.json",
    "clips": 15, "min_duration": 15.0, "max_duration": 75.0,
    "hard_max": 150.0, "no_extend": false
  },
  "slots": [{"index": 1, "key": [31.04, 97.22], "start": 31.04, "end": 97.22}],
  "rejected": [[257.48, 330.52]],
  "rendered": {"clip_01_reels.mp4": [31.04, 97.22]},
  "recipe": {
    "style": "punch", "resolution": "720p", "framing_mode": "per_shot",
    "safe_area": true, "title": false, "trim_silence": 1.5,
    "audio_filter": "volume=2.12dB,alimiter=..."
  }
}
```

- `settings` decides whether the saved selection still applies. Change any of
  it and the run starts over, because the candidate pool is a different pool.
- `slots` maps a clip number to the pool window it came from. `key` is the
  window's original coordinates and is its identity; `start` and `end` are where
  it ended up after extension and opening adjustment, and can differ.
- `rejected` is every window turned down so far. They are never offered again.
- `rendered` records what each file was actually encoded from, so an interrupted
  run resumes from the last finished file.
- `recipe` is everything that changes the output bytes without moving the
  boundaries. Change any of it and every file is re-rendered.

## Reading progress from stdout

The pipeline prints as it goes. The lines worth parsing:

```
Transcript: 1841 segments, 24350 words
Scored 400 distinct candidates, chose 15, 384 held in reserve for regeneration
Extended 3 clips to the end of their subject: #2 +12s, #7 +8s
Moved 2 clips to a self-contained opening: #4 +3s in
Reusing cached video analysis
  source measures -16.1 LUFS, peak -0.4 dBTP - turned up 2.1 dB
Trimming dead air from 4 of 15 clips, 8s in total
  291 shots, 1 seats, faces in 100% of frames
  framing mode: per_shot (shots change every 6.0s on average)
  45 files already done, 0 to render

Rendering 45 files ...
  [1/45] clip_01_tiktok.mp4 (62s)  about 14m left
15 cover frames, 15 of them on a moment the tracker was sure about

Done in 824s
45 files written, 15 caption packs, and run_report.json in out/
```

For a progress bar, `Rendering N files ...` gives the total and each
`  [i/N] name (Ds)` line gives the position. The trailing estimate appears from
the second file onward, because one sample is not a rate.

Lines beginning with two spaces are details of the step above them. Everything
else is a step boundary.

## Calling it from Python

The pipeline is a function. `cli/pipeline.py` reads the arguments and prints;
everything it does is in `ai_clipper.core`, which prints nothing and has a test
that says so.

```python
from ai_clipper.core import Settings, run

def on_progress(event):
    print(event.kind, event.data)

result = run(Settings(video="podcast.mov", transcript="transcript.json",
                      out="out", clips=15),
             on_progress=on_progress,
             should_stop=lambda: user_pressed_stop)
```

`Settings` has one field per command-line option, with the same names and the
same defaults, and a test checks the two do not drift. `Settings.from_args()`
builds one from an argparse namespace and ignores fields it does not recognise.

`run()` returns a `Result`:

| field | what it is |
|---|---|
| `out_dir` | where everything was written |
| `seconds` | how long the run took |
| `dry_run` | whether the video was ever opened |
| `cancelled` | whether `should_stop` ended it early |
| `rendered` | filenames this run encoded, in order |
| `numbers` | the clip numbers in play |
| `report` | the contents of `run_report.json`, or None on a dry run |
| `preview` | the contents of `dry_run.json`, or None on a real run |

A run that cannot start raises `PipelineError` with a message meant for a user:
an unknown caption style, or `--regenerate` with no previous run to reject from.
Those used to be `SystemExit`, which ends a command tidily and kills anything
holding a window open.

### Progress events

`on_progress` is called with an `Event` at every point the command prints.

```python
Event(kind="rendering",
      message="  [3/45] clip_02_tiktok.mp4 (62s)  about 13m left",
      data={"index": 3, "total": 45, "file": "clip_02_tiktok.mp4",
            "clip": 2, "platform": "tiktok", "duration": 62.4})
```

`message` is the line the terminal wants, indentation and all. `data` is the
same information as values. The CLI is one function that prints `message`; a UI
should switch on `kind` and read `data` and ignore `message` entirely.

The kinds, in the order a full run emits them:

| kind | when | useful in `data` |
|---|---|---|
| `signals` | a `signals.json` was loaded | `name`, `path` |
| `transcript` | the transcript was read | `segments`, `words` |
| `repunctuated` | punctuation was inferred from pauses | `before`, `after` |
| `session_reset` | the saved selection no longer applies | `reason` |
| `regenerated` | a clip was replaced | `clip`, `was`, `now`, `score` |
| `regenerate_skipped` | a named clip could not be replaced | `clip`, `reason` |
| `regenerate_exhausted` | nothing else fits | `clip` |
| `selection` | clips were chosen | `scored`, `chosen`, `reserve` |
| `extended` | clips were run on to finish their subject | `clips` |
| `openings` | clips were moved to a self-contained opening | `clips` |
| `preview_clip` | one clip, on a dry run | the whole `dry_run.json` entry |
| `analysing` / `analysis_cached` | shot and face analysis | `cached` |
| `measuring_audio` / `loudness` | the level measurement | `integrated`, `true_peak`, `gain`, `filter` |
| `dead_air` | pauses to be shortened | `clips`, `total` |
| `shots` | what the analysis found | `shots`, `seats`, `detection_rate` |
| `framing` | the framing mode and why | `mode`, `reason` |
| `clip_skipped` | too long for a platform | `clip`, `why` |
| `recipe_changed` | render settings moved, so everything is stale | |
| `resuming` | files already done | `done`, `remaining` |
| `render_start` | the render loop begins | `total` |
| `rendering` | one file, before it is encoded | `index`, `total`, `file`, `clip`, `platform`, `duration` |
| `cancelled` | `should_stop` said so | `rendered` |
| `cover_failed` | one cover could not be written | `clip`, `error` |
| `covers` | cover frames written | `written`, `confident` |
| `done` / `summary` | the end | `seconds`, `written`, `clips` |

For a progress bar, `render_start` gives the total and `rendering` gives the
position. The estimate in `message` appears from the second file onward, because
one sample is not a rate.

### Stopping

`should_stop` is called before each file and each cover frame. Returning true
ends the run with `cancelled=True` rather than an exception.

Stopping between files is safe rather than merely tolerable: renders are atomic,
and the session is written as each file lands, so running again picks up where
it stopped. Measured on a three-clip render: stopped after two files, no
`.part.mp4` left behind, and the next run encoded only the third.

### The pieces underneath

`run()` is one way of wiring these together. A UI backend assembling its own is
a supported use rather than a workaround.

| what you want | where |
|---|---|
| score a transcript | `scoring.hook_scorer.HookScorer().candidate_pool(segments, min_d, max_d)` |
| pick a set, reject one | `scoring.selection.Selection.build(pool, n)`, `.regenerate(i, mode)` |
| remember across runs | `scoring.session.Session` |
| find where a subject ends | `scoring.topics.TopicModel(segments).extend(clip, hard_max)` |
| plan the dead air cuts | `video.deadair.plan(words, start, end)` |
| measure and match loudness | `video.audio.measure(src)`, `.filter_chain(measured)` |
| choose a cover moment | `video.cover.pick_time(crop_path, start, end, words, per_chunk)` |
| build subtitles | `video.subtitles.build_ass(words, w, h, style)` |
| render one clip | `video.render.render_clip(src, start, end, out, spec, ass, crop_path)` |
| cut pieces out of a clip | `render_clip(..., keep_spans=[(a, b), (c, d)])` |
| per-platform plans and margins | `export.platforms.plan_exports(...)`, `.style_for(style, key)` |
| titles and hashtags | `export.caption_pack.build_pack(clip, index)` |

None of these print, none of them read `sys.argv`, and all of them are covered
by the test suite.

## What a UI will want that does not exist yet

Listed so the first hour is not spent discovering them.

- **Analysis and rendering are one call.** There is no way to ask for the shot
  and face analysis on its own and show it before committing to a render, short
  of running a dry run first, which skips analysis entirely.
- **One episode at a time.** No queue, no job ids. Two runs against the same
  output folder will fight over `session.json`.
- **No preview render.** `render_clip` with a short range and
  `preset="ultrafast"` is a few seconds, but nothing exposes it.
- **No per-clip overrides.** `render_clip` already accepts `keep_spans` and a
  per-clip audio filter, which is everything a manual trim-and-volume editor
  needs, but `run()` decides both itself and takes no instruction from outside.
- **`--regenerate` needs a previous run.** A reject button has to have run the
  pipeline at least once against that output folder first, or it raises
  `PipelineError` rather than picking something.
- **An unknown style is caught late.** The check happens after the shot and face
  analysis, so a typo costs minutes before it reports. Validate against
  `subtitles.load_styles()` before starting the run.
