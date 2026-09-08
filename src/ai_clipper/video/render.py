"""
Cut a clip out of the source video, reframe it, and burn captions in.

One ffmpeg invocation per clip: seek, crop or pad to the target ratio, overlay
the ASS subtitles, encode. Doing it in a single pass avoids writing an
intermediate file for every clip, which matters when a three-hour episode
produces thirty of them.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from typing import Optional

from .fonts import fonts_dir


def _escape_filter_path(path: str) -> str:
    """
    Make a filesystem path safe inside an ffmpeg filter argument.

    Windows paths are the reason this exists: backslashes and the drive-letter
    colon both mean something to the filter parser.
    """
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


RATIOS = {
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:5": (4, 5),
    "16:9": (16, 9),
}

# target heights per named resolution, for vertical output
HEIGHTS = {"720p": 1280, "1080p": 1920}


@dataclass
class RenderSpec:
    ratio: str = "9:16"
    resolution: str = "1080p"
    # "face"     - crop window follows the speaker (needs a crop path)
    # "per_shot" - one fixed, correctly-framed position per shot; no motion
    #              within a shot, repositioning only on cuts
    # "crop"     - fixed centre crop, ignores where anyone is
    # "split"    - two people cropped from the same wide frame, stacked 9:16
    # "blur"     - whole frame centred over a blurred backdrop
    mode: str = "face"
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "160k"
    # -af argument from video.audio, or None to pass the audio through. The same
    # string is used for every clip in an episode, which is the point: it is
    # derived from one measurement of the source, not of this clip.
    audio_filter: Optional[str] = None


def target_size(spec: RenderSpec) -> tuple:
    rw, rh = RATIOS[spec.ratio]
    base_h = HEIGHTS.get(spec.resolution, 1920)

    if rh >= rw:
        h = base_h
        w = round(h * rw / rh)
    else:
        # landscape: interpret the resolution as the height of the short side
        h = round(base_h * 9 / 16)
        w = round(h * rw / rh)

    # even dimensions, required by yuv420p
    return (w - w % 2, h - h % 2)


def probe_size(path: str) -> tuple:
    """(width, height) of the first video stream, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def scaled_width_for(src_w: int, src_h: int, out_h: int) -> int:
    """Width the source becomes once scaled to the output height, kept even."""
    w = round(src_w * out_h / src_h)
    return w - (w % 2)


def build_filter(
    spec: RenderSpec,
    out_w: int,
    out_h: int,
    ass_path: Optional[str],
    sendcmd_path: Optional[str] = None,
    scaled_w: Optional[int] = None,
    split_crops: Optional[list] = None,
) -> str:
    """
    Compose the video filter chain.

    face mode: scale to the output height, then crop a window whose x is driven
    by a sendcmd script generated from the tracked crop path.

    crop mode: scale so the frame covers the target box, then centre-crop the
    overflow. The fallback when tracking is unavailable or not wanted.

    blur mode: a scaled, blurred copy fills the target box while the untouched
    frame sits centred on top. Safe when there is no reliable face to follow.
    """
    if spec.mode in ("face", "per_shot") and sendcmd_path and scaled_w:
        cmd = _escape_filter_path(sendcmd_path)
        chain = (
            f"[0:v]scale={scaled_w}:{out_h},"
            f"sendcmd=f='{cmd}',"
            f"crop={out_w}:{out_h}:0:0[v]"
        )
    elif spec.mode == "split" and split_crops:
        # one crop per person out of the same wide frame, stacked to fill 9:16
        panel_h = out_h // len(split_crops)
        parts, labels = [], []
        for i, (x, y, w, h) in enumerate(split_crops):
            parts.append(
                f"[0:v]crop={w}:{h}:{x}:{y},scale={out_w}:{panel_h}[p{i}]"
            )
            labels.append(f"[p{i}]")
        chain = ";".join(parts)
        chain += f";{''.join(labels)}vstack=inputs={len(split_crops)}[v]"
    elif spec.mode == "blur":
        chain = (
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=luma_radius=min(h\\,w)/16:luma_power=2[bg];"
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )
    else:
        chain = (
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}[v]"
        )

    if ass_path:
        escaped = _escape_filter_path(ass_path)
        fonts = _escape_filter_path(fonts_dir())
        # fontsdir points libass at the fonts shipped with the project, so a
        # style using one of them renders the same on any machine
        chain += f";[v]ass='{escaped}':fontsdir='{fonts}'[v]"

    return chain


def _burn_subtitles(video_in: str, ass_path: str, out_path: str, spec) -> str:
    """Second pass: draw the captions onto an already-reframed clip."""
    escaped = _escape_filter_path(ass_path)
    fonts = _escape_filter_path(fonts_dir())
    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_in,
        "-vf", f"ass='{escaped}':fontsdir='{fonts}'",
        "-c:v", "libx264", "-preset", spec.preset, "-crf", str(spec.crf),
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
        out_path,
    ], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"subtitle burn failed:\n{proc.stderr[-1500:]}")
    return out_path


def coalesce(runs: list) -> list:
    """
    Merge neighbouring runs that share a mode and touch in time.

    Two reasons this matters. Shot analysis often reports the same framing mode
    for several consecutive shots, and encoding each of them on its own costs a
    re-encode and a join for no visible difference. And when a clip is being cut
    for dead air, the split must survive: two runs that share a mode but sit on
    either side of a removed gap are not contiguous and must not be merged, or
    the gap comes back.
    """
    merged = []
    for a, b, mode in runs:
        if merged and merged[-1][2] == mode and abs(merged[-1][1] - a) < 1e-6:
            merged[-1] = (merged[-1][0], b, mode)
        else:
            merged.append((a, b, mode))
    return merged


def plan_runs(crop_path, spec, spans: list) -> list:
    """
    Turn the ranges a clip keeps into the pieces ffmpeg will encode.

    Two independent reasons to split a clip meet here: its framing mode can
    change partway through, and dead air can have been removed from the middle.
    Composing them in one place is what stops the second feature from quietly
    undoing the first.
    """
    runs = []
    for a, b in spans:
        if crop_path is not None and spec.mode in ("face", "per_shot"):
            from .framing import segment_modes

            runs.extend(segment_modes(crop_path, a, b, spec.mode))
        else:
            runs.append((a, b, spec.mode))
    return coalesce(runs)


def render_segmented(
    source: str,
    runs: list,
    out_path: str,
    spec,
    ass_path=None,
    crop_path=None,
    seats=None,
) -> str:
    """
    Render a clip whose framing mode changes partway through.

    Each run is encoded on its own, the pieces are joined, and the captions go
    on afterwards in a single pass over the joined file - burning them per piece
    would mean shifting every timestamp and would still risk a caption landing
    across a join.
    """
    import tempfile

    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (a, b, mode) in enumerate(runs):
            part = os.path.join(tmp, f"part{i:03d}.mp4")
            piece_spec = replace(spec, mode=mode)
            render_clip(source, a, b, part, piece_spec, None, crop_path, seats,
                        allow_segmentation=False, sidecar_stem=part)
            parts.append(part)

        listing = os.path.join(tmp, "parts.txt")
        with open(listing, "w", encoding="utf-8") as f:
            for part in parts:
                f.write(f"file '{part}'\n")

        joined = os.path.join(tmp, "joined.mp4") if ass_path else out_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        proc = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", listing,
            "-c", "copy", "-movflags", "+faststart", joined,
        ], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"joining clip parts failed:\n{proc.stderr[-1500:]}")

        if ass_path:
            _burn_subtitles(joined, ass_path, out_path, spec)

    return out_path


def render_clip(
    source: str,
    start: float,
    end: float,
    out_path: str,
    spec: Optional[RenderSpec] = None,
    ass_path: Optional[str] = None,
    crop_path=None,
    seats=None,
    allow_segmentation: bool = True,
    sidecar_stem: Optional[str] = None,
    keep_spans: Optional[list] = None,
) -> str:
    """
    keep_spans: the ranges of the source this clip keeps, from video.deadair.
    More than one means the middle has been cut, and the clip is encoded in
    pieces and joined. None or a single span renders straight through.

    crop_path: a CropPath from framing.build_crop_path. Required for face mode;
    without one, face mode falls back to a fixed centre crop rather than failing,
    so a caller that skipped analysis still gets a usable clip.

    If the clip spans both talking footage and a held graphic, it is rendered in
    pieces and joined - a poster cannot be cropped, only fitted. Pass
    allow_segmentation=False to force a single pass; render_segmented uses that
    when rendering the pieces themselves.

    The file appears at out_path only once it is complete. ffmpeg writes to a
    .part.mp4 beside it and the finished file is moved into place in one step,
    because an interrupted render leaves a playable-looking fragment behind and
    a resumed run would take it for finished work and skip it.
    """
    spec = spec or RenderSpec()
    out_w, out_h = target_size(spec)
    duration = max(0.1, end - start)

    final = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    target = final + ".part.mp4"
    if os.path.exists(target):
        os.remove(target)

    cutting = bool(keep_spans) and len(keep_spans) > 1
    if allow_segmentation and (cutting or
                               (crop_path is not None and spec.mode in ("face", "per_shot"))):
        runs = plan_runs(crop_path, spec, list(keep_spans) if cutting else [(start, end)])
        if cutting or len({mode for _, _, mode in runs}) > 1:
            render_segmented(source, runs, target, spec, ass_path, crop_path, seats)
            os.replace(target, final)
            return final

    sendcmd_path = None
    scaled_w = None
    split_crops = None

    if spec.mode == "split":
        if not seats or len(seats) < 2:
            raise ValueError(
                "split mode needs at least two seats - run speakers.find_seats "
                "on the source first, and fall back to another mode if it finds "
                "fewer than two people"
            )
        from .speakers import split_view_crops

        src_w, src_h = probe_size(source)
        split_crops = split_view_crops(seats, src_w, src_h)

    if spec.mode in ("face", "per_shot") and crop_path is not None:
        from .framing import flatten_per_shot, to_sendcmd

        if spec.mode == "per_shot":
            crop_path = flatten_per_shot(crop_path)

        src_w, src_h = probe_size(source)
        scaled_w = scaled_width_for(src_w, src_h, out_h)
        if scaled_w > out_w:
            sendcmd_path = os.path.splitext(sidecar_stem or final)[0] + ".cmd"
            with open(sendcmd_path, "w", encoding="utf-8") as f:
                f.write(to_sendcmd(crop_path, start, end, scaled_w, out_w))
        else:
            # source is already at or narrower than the target ratio - nothing
            # to pan across
            scaled_w = None

    filter_complex = build_filter(spec, out_w, out_h, ass_path, sendcmd_path,
                                  scaled_w, split_crops)

    audio_args = ["-af", spec.audio_filter] if spec.audio_filter else []

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        # -ss before -i seeks quickly; -accurate_seek keeps the cut frame-exact
        "-accurate_seek", "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", source,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", spec.preset, "-crf", str(spec.crf),
        "-pix_fmt", "yuv420p",
        *audio_args,
        "-c:a", "aac", "-b:a", spec.audio_bitrate,
        "-movflags", "+faststart",
        target,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if os.path.exists(target):
            os.remove(target)
        raise RuntimeError(f"ffmpeg failed for {final}:\n{proc.stderr[-1500:]}")

    os.replace(target, final)
    return final
