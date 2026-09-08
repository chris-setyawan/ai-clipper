"""
Get a video from a link.

"Paste a link, get clips" is the flow every paid clipper leads with, and it is
the difference between a tool you try once on a file you happen to have and one
you actually use. yt-dlp does the work; this decides what to ask it for and
where to put the result.

Two choices worth explaining. Downloads are capped at 720p by default, because
the pipeline analyses at 480-640px and renders to 720p or 1080p - fetching 4K of
a three-hour episode costs gigabytes and buys nothing. And a file already
downloaded is reused rather than fetched again, since re-running the pipeline
with different settings is the normal case, not the exception.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def is_url(value: str) -> bool:
    return bool(URL_PATTERN.match(value.strip()))


@dataclass
class Fetched:
    path: str
    title: str
    duration: float
    source_url: str
    was_cached: bool


def format_selector(max_height: int = 720) -> str:
    """
    Best video up to max_height plus best audio, falling back to whatever
    single file exists. The fallback matters: some sources have no separate
    audio stream, and without it yt-dlp fails on videos it could have handled.
    """
    return (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={max_height}]+bestaudio/"
        f"best[height<={max_height}]/best"
    )


def build_options(dest_dir: str, max_height: int = 720, quiet: bool = False) -> dict:
    return {
        "format": format_selector(max_height),
        "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,
        # a playlist link should not quietly pull down forty episodes
        "noplaylist": True,
        "retries": 3,
    }


def fetch(url: str, dest_dir: str = "downloads", max_height: int = 720,
          quiet: bool = False) -> Fetched:
    """
    Download `url` into dest_dir and return where it landed.

    Raises RuntimeError with the underlying message on failure - a link that
    cannot be fetched should say why (age gate, region block, private video)
    rather than surfacing as a missing file three steps later.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "downloading from a link needs yt-dlp:\n    pip install yt-dlp"
        )

    os.makedirs(dest_dir, exist_ok=True)
    options = build_options(dest_dir, max_height, quiet)

    with yt_dlp.YoutubeDL(options) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as err:
            raise RuntimeError(f"could not read {url}: {err}") from err

        if info.get("entries"):
            info = info["entries"][0]

        expected = ydl.prepare_filename(info)
        merged = os.path.splitext(expected)[0] + ".mp4"

        existing = next((p for p in (merged, expected) if os.path.exists(p)), None)
        if existing:
            return Fetched(
                path=existing,
                title=info.get("title", ""),
                duration=float(info.get("duration") or 0.0),
                source_url=url,
                was_cached=True,
            )

        try:
            ydl.download([url])
        except Exception as err:
            raise RuntimeError(f"download failed for {url}: {err}") from err

    path = next((p for p in (merged, expected) if os.path.exists(p)), None)
    if path is None:
        raise RuntimeError(
            f"yt-dlp reported success but no file appeared for {url} "
            f"(expected {merged})"
        )

    return Fetched(
        path=path,
        title=info.get("title", ""),
        duration=float(info.get("duration") or 0.0),
        source_url=url,
        was_cached=False,
    )


def resolve(video: str, dest_dir: str = "downloads",
            max_height: int = 720, quiet: bool = False) -> str:
    """
    Take whatever the user passed - a path or a link - and return a local path.

    This is the one function the rest of the pipeline needs: every entry point
    can accept a link without knowing anything about downloading.
    """
    if not is_url(video):
        if not os.path.exists(video):
            raise FileNotFoundError(f"file not found: {video}")
        return video

    result = fetch(video, dest_dir, max_height, quiet)
    label = "using cached download" if result.was_cached else "downloaded"
    print(f"  {label}: {result.title or result.path}")
    if result.duration:
        print(f"  {result.duration / 60:.0f} minutes -> {result.path}")
    return result.path
