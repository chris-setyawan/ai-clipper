import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video import render as R


def test_a_failed_render_leaves_no_file_behind(tmp_path, monkeypatch):
    """
    An interrupted render used to leave a playable-looking fragment at the final
    path, and a resumed run would take it for finished work and skip it. ffmpeg
    now writes to a .part.mp4 that is moved into place only on success.
    """
    out = tmp_path / "clip_01_tiktok.mp4"

    def fail(cmd, **kwargs):
        target = cmd[-1]
        assert target.endswith(".part.mp4"), "ffmpeg must not write the final name"
        Path(target).write_bytes(b"half a video")
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(R.subprocess, "run", fail)
    with pytest.raises(RuntimeError):
        R.render_clip("src.mp4", 0.0, 5.0, str(out), R.RenderSpec(mode="crop"))

    assert not out.exists(), "a failed render must not leave the final file"
    assert not list(tmp_path.glob("*.part.mp4")), "and must clean up after itself"


def test_a_successful_render_moves_the_file_into_place(tmp_path, monkeypatch):
    out = tmp_path / "clip_01_tiktok.mp4"

    def succeed(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"a whole video")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(R.subprocess, "run", succeed)
    result = R.render_clip("src.mp4", 0.0, 5.0, str(out), R.RenderSpec(mode="crop"))

    assert Path(result) == out.resolve()
    assert out.read_bytes() == b"a whole video"
    assert not list(tmp_path.glob("*.part.mp4"))


def test_a_stale_fragment_is_replaced_not_appended_to(tmp_path, monkeypatch):
    out = tmp_path / "clip_01_tiktok.mp4"
    (tmp_path / "clip_01_tiktok.mp4.part.mp4").write_bytes(b"junk from a killed run")

    def succeed(cmd, **kwargs):
        assert not Path(cmd[-1]).exists(), "the old fragment must be gone first"
        Path(cmd[-1]).write_bytes(b"fresh")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(R.subprocess, "run", succeed)
    R.render_clip("src.mp4", 0.0, 5.0, str(out), R.RenderSpec(mode="crop"))
    assert out.read_bytes() == b"fresh"
