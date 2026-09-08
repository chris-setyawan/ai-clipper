import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.input.fetch import build_options, format_selector, is_url, resolve


def test_links_are_recognised():
    for link in ("https://youtu.be/abc", "http://example.com/v.mp4",
                 "HTTPS://WWW.YOUTUBE.COM/watch?v=x", "  https://youtu.be/x  "):
        assert is_url(link), link


def test_paths_are_not_mistaken_for_links():
    for path in (r"C:\videos\podcast.mov", "podcast sample.mov",
                 "./out/clip.mp4", "/home/user/a.mkv", "ftp://host/x.mp4"):
        assert not is_url(path), path


def test_format_selector_caps_height_and_keeps_fallbacks():
    selector = format_selector(720)
    assert "height<=720" in selector
    assert selector.count("/") >= 2, "a source without separate audio must still work"
    assert selector.endswith("best")


def test_a_higher_cap_is_honoured():
    assert "height<=1080" in format_selector(1080)


def test_options_refuse_to_pull_a_whole_playlist():
    assert build_options("/tmp/dl")["noplaylist"] is True


def test_options_name_files_by_id_so_reruns_can_reuse_them():
    outtmpl = build_options("/tmp/dl")["outtmpl"]
    assert "%(id)s" in outtmpl
    assert outtmpl.startswith("/tmp/dl")


def test_a_local_path_passes_straight_through(tmp_path):
    f = tmp_path / "video.mp4"
    f.write_bytes(b"x")
    assert resolve(str(f)) == str(f)


def test_a_missing_local_file_fails_before_anything_else(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        resolve(str(tmp_path / "nope.mp4"))
    assert "nope.mp4" in str(err.value)
