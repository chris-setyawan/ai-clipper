"""
What each platform expects, and where its interface covers the frame.

Every short-form app draws its own controls over the video: a caption block and
a button column on TikTok, a different caption block on Reels, a progress bar
and title on Shorts. A subtitle placed without knowing that ends up behind a
follow button, which is a small idea that fixes a problem clippers hit on every
single clip.

The safe areas below are fractions of the frame, measured from published design
guidance and from where the controls actually sit in each app. They are
deliberately a little conservative: a caption sitting slightly higher than
necessary costs nothing, and one hidden behind a button costs the punchline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Platform:
    key: str
    name: str
    ratio: str
    resolution: str = "1080p"

    # fractions of the frame covered by the app's own interface
    top_ui: float = 0.0
    bottom_ui: float = 0.0
    right_ui: float = 0.0
    left_ui: float = 0.0

    max_duration: Optional[float] = None    # seconds, None = no practical cap
    min_duration: float = 5.0
    notes: str = ""

    def caption_margins(self, headroom: float = 0.03) -> dict:
        """
        Where captions may sit, as fractions, with a little clearance added.

        Side margins are symmetric, and deliberately so. An earlier version set
        them per side, which on TikTok pushed captions left of frame centre to
        clear the button column. It is defensible on paper and it looks wrong:
        off-centre subtitles read as a mistake, and the viewer sees that in
        every frame while the button column only sometimes overlaps anything.

        What survives is the useful half. The wider of the two interface columns
        still narrows the caption's usable width on both sides, so a long line
        wraps instead of running underneath the buttons - and the block stays
        centred. Vertical clearance, which costs nothing visually, stays per
        platform.

        CaptionStyle still supports per-side margins for anyone who wants them;
        nothing here sets them.
        """
        return {
            "bottom_margin_frac": min(0.45, self.bottom_ui + headroom),
            "side_margin_frac": min(0.30, max(self.left_ui, self.right_ui) + headroom),
        }

    def fits(self, duration: float) -> bool:
        if duration < self.min_duration:
            return False
        return self.max_duration is None or duration <= self.max_duration


PLATFORMS = {
    "tiktok": Platform(
        key="tiktok", name="TikTok", ratio="9:16",
        # the caption and username block sits low; the action column is on the right
        top_ui=0.10, bottom_ui=0.20, right_ui=0.14,
        max_duration=600.0,
        notes="Action column on the right, caption block bottom-left.",
    ),
    "reels": Platform(
        key="reels", name="Instagram Reels", ratio="9:16",
        top_ui=0.09, bottom_ui=0.22, right_ui=0.13,
        max_duration=180.0,
        notes="Caption and audio strip sit higher than TikTok's.",
    ),
    "shorts": Platform(
        key="shorts", name="YouTube Shorts", ratio="9:16",
        top_ui=0.08, bottom_ui=0.17, right_ui=0.13,
        max_duration=180.0,
        notes="Progress bar and title along the bottom.",
    ),
    "feed": Platform(
        key="feed", name="Instagram feed", ratio="4:5",
        top_ui=0.0, bottom_ui=0.06,
        max_duration=90.0,
        notes="Nothing overlaps the video itself; margins are aesthetic only.",
    ),
    "square": Platform(
        key="square", name="Square post", ratio="1:1",
        top_ui=0.0, bottom_ui=0.06,
        max_duration=None,
        notes="Safe everywhere, weakest reach.",
    ),
    "youtube": Platform(
        key="youtube", name="YouTube landscape", ratio="16:9",
        bottom_ui=0.10,
        max_duration=None,
        notes="Player controls appear over the lower strip on hover.",
    ),
}


def get(key: str) -> Platform:
    try:
        return PLATFORMS[key]
    except KeyError:
        raise ValueError(
            f"unknown platform '{key}'. Known: {', '.join(sorted(PLATFORMS))}"
        )


def style_for(style, platform_key: str):
    """
    Return a copy of a caption style repositioned for a platform.

    Only the margins change - font, colour and rhythm are the user's choices and
    are left exactly as they are. The point is that one style can be authored
    once and still land correctly on three apps that cover different parts of
    the frame.
    """
    from ..video.subtitles import CaptionStyle

    platform = get(platform_key)
    margins = platform.caption_margins()

    data = style.to_dict()
    data.update(margins)
    data["name"] = f"{style.name}@{platform.key}"
    return CaptionStyle.from_dict(data)


def targets_for(duration: float, keys: Optional[List[str]] = None) -> List[Platform]:
    """Platforms a clip of this length can actually be posted to."""
    keys = keys or list(PLATFORMS)
    return [get(k) for k in keys if get(k).fits(duration)]


@dataclass
class ExportPlan:
    """One clip, one platform, one output file."""
    clip_index: int
    platform: Platform
    filename: str
    ratio: str
    resolution: str
    caption_style_name: str


def plan_exports(clips, platform_keys: List[str], style_name: str = "punch",
                 resolution: str = "1080p", numbers: List[int] = None) -> List[ExportPlan]:
    """
    Work out every file a run should produce, before rendering any of them.

    Planning first means the caller can show the user what is about to happen,
    count the work, and skip clips that no requested platform will accept -
    rather than discovering a 4-minute clip is too long for Reels only after
    spending a minute encoding it.

    `numbers` gives each clip its filename number. It exists because a clip's
    number has to survive a rejection: reject clip 4 and its replacement is
    still clip 4, so clips 1-3 and 5 onwards keep the files they already have.
    Left out, clips are numbered by position as before.
    """
    plans: List[ExportPlan] = []
    for i, clip in _numbered(clips, numbers):
        duration = clip.end - clip.start
        for platform in targets_for(duration, platform_keys):
            plans.append(ExportPlan(
                clip_index=i,
                platform=platform,
                filename=f"clip_{i:02d}_{platform.key}.mp4",
                ratio=platform.ratio,
                resolution=resolution,
                caption_style_name=f"{style_name}@{platform.key}",
            ))
    return plans


def _numbered(clips, numbers):
    if numbers is None:
        return list(enumerate(clips, start=1))
    return list(zip(numbers, clips))


def rejected(clips, platform_keys: List[str], numbers: List[int] = None) -> List[tuple]:
    """(clip number, reason) for clips no requested platform will take."""
    out = []
    for i, clip in _numbered(clips, numbers):
        duration = clip.end - clip.start
        if not targets_for(duration, platform_keys):
            limits = ", ".join(
                f"{get(k).name} {get(k).min_duration:.0f}-"
                f"{get(k).max_duration if get(k).max_duration else 'unlimited'}s"
                for k in platform_keys
            )
            out.append((i, f"{duration:.0f}s fits none of: {limits}"))
    return out
