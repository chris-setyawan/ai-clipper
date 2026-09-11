"""
The one test that actually runs ffmpeg.

Every other test in this project works on data. That is on purpose and it is
what makes the suite run in two seconds. But it leaves a whole class of bug
uncovered, and that class is not theoretical: a glow that erased the word it
was drawn behind, a caption landing in a hole between two words, a filter chain
that ffmpeg rejects outright. Every one of those was caught by a person looking
at a frame, which does not scale and does not run in CI.

So this renders. It builds a two-second video with ffmpeg's own test source, cuts
a clip out of it with captions burned in, and checks the result is a real,
playable file with both streams and roughly the right length. It will not tell
you a caption is ugly. It will tell you the chain still works, which is the part
a refactor breaks.

Skipped when ffmpeg is not on the PATH, so a machine that only ever runs the
scoring half is not held back by it.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video.subtitles import PRESETS, Word, build_ass
from ai_clipper.video.render import RenderSpec, render_clip, target_size

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or
                                shutil.which("ffprobe") is None,
                                reason="needs ffmpeg and ffprobe on the PATH")

WORDS = [Word(0.0, 0.6, "setiap"),
         Word(0.6, 1.2, "lakilaki", {"italic": True, "color": "#39FF6A"}),
         Word(1.2, 1.8, "harus"),
         Word(1.8, 2.4, "tau")]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """Three seconds of moving picture and a tone, made by ffmpeg itself."""
    path = tmp_path_factory.mktemp("smoke") / "source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return str(path)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return (float(data["format"]["duration"]),
            {s["codec_type"] for s in data["streams"]})


def render(source, out_dir, name="clip.mp4", style=None, **kwargs):
    spec = RenderSpec(ratio="9:16", resolution="720p", mode="crop",
                      preset="ultrafast", crf=32)
    width, height = target_size(spec)
    ass = Path(out_dir) / "clip.ass"
    ass.write_text(build_ass(WORDS, width, height, style or PRESETS["punch"]),
                   encoding="utf-8")
    target = str(Path(out_dir) / name)
    return render_clip(source, 0.0, 2.5, target, spec, str(ass), **kwargs)


def test_a_clip_comes_out_playable_with_both_streams(source, tmp_path):
    out = render(source, tmp_path)
    assert os.path.getsize(out) > 0
    seconds, streams = probe(out)
    assert streams == {"video", "audio"}
    assert 2.3 < seconds < 2.8


def test_the_output_is_the_shape_that_was_asked_for(source, tmp_path):
    out = render(source, tmp_path)
    size = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", out],
        capture_output=True, text=True, check=True).stdout.strip()
    assert size == "720x1280"


def test_nothing_is_left_behind_when_a_render_succeeds(source, tmp_path):
    render(source, tmp_path)
    assert not list(Path(tmp_path).glob("*.part.mp4"))


def test_a_clip_cut_in_two_pieces_is_joined(source, tmp_path):
    """
    The path dead air removal and hand edits both take: encode the surviving
    stretches on their own, join them, burn the captions over the join.
    """
    out = render(source, tmp_path, name="cut.mp4",
                 keep_spans=[(0.0, 0.8), (1.7, 2.5)])
    seconds, streams = probe(out)
    assert streams == {"video", "audio"}
    assert 1.4 < seconds < 1.9


def test_every_animation_survives_ffmpeg(source, tmp_path):
    """
    Cheap insurance against a tag libass rejects. It says nothing about how any
    of them look; it says the file still renders.
    """
    import dataclasses

    from ai_clipper.video.subtitles import ANIMATIONS

    for name in ANIMATIONS:
        style = dataclasses.replace(PRESETS["punch"], animation=name)
        out = render(source, tmp_path, name=f"{name}.mp4", style=style)
        seconds, _ = probe(out)
        assert seconds > 1.0, name


def test_glow_survives_ffmpeg(source, tmp_path):
    import dataclasses

    style = dataclasses.replace(PRESETS["punch"], glow=True, glow_size=7.0)
    out = render(source, tmp_path, name="glow.mp4", style=style)
    assert probe(out)[0] > 1.0


def test_a_loudness_filter_survives_ffmpeg(source, tmp_path):
    """
    The gain and the limiter go in as one -af string, and a typo in it is an
    ffmpeg error rather than a quiet no-op.
    """
    from ai_clipper.video import audio

    measured = audio.Loudness(integrated=-24.0, true_peak=-0.4, range=9.0)
    chain = audio.filter_chain(measured)
    assert chain

    spec = RenderSpec(ratio="9:16", resolution="720p", mode="crop",
                      preset="ultrafast", crf=32, audio_filter=chain)
    out = render_clip(source, 0.0, 2.0, str(Path(tmp_path) / "loud.mp4"), spec)
    assert probe(out)[1] == {"video", "audio"}


def test_a_cover_frame_comes_out_of_a_rendered_clip(source, tmp_path):
    from ai_clipper.video import cover

    clip = render(source, tmp_path, name="for_cover.mp4")
    out = cover.grab(clip, 1.0, str(Path(tmp_path) / "cover.jpg"))
    assert os.path.getsize(out) > 0
    assert not list(Path(tmp_path).glob("*.part.jpg"))
