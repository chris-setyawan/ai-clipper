"""
Build ASS subtitle files with word-level highlighting.

ASS is used rather than SRT because it carries styling - font, colour, outline,
position - inside the file itself. That matters for the "re-render without
re-analysing" goal: the .ass file is a real editable artifact, so fixing a typo
or changing a look means re-running only the burn-in step.

The karaoke effect is done by emitting one dialogue event per word. Each event
shows the whole chunk but paints the currently-spoken word in the highlight
colour, so the caption reads as a stable block with a moving emphasis rather
than words appearing one at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ASS colours are &HAABBGGRR - alpha first, then blue, green, red.
def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


@dataclass
class CaptionStyle:
    """
    Everything about how burned-in captions look.

    Every field is a plain value, so a style round-trips to JSON without any
    special handling. That is what makes user-defined styles possible: the
    presets below are just the first three entries in the same format a user's
    own styles.json uses.
    """
    name: str = "default"
    font: str = "Montserrat ExtraBold"
    font_size: int = 64
    primary: str = "#FFFFFF"        # resting word colour
    highlight: str = "#F5D90A"      # active word colour
    outline_color: str = "#000000"
    outline: float = 3.5
    shadow: float = 1.0
    bold: bool = True
    uppercase: bool = False
    words_per_chunk: int = 3
    # vertical placement as a fraction of frame height, measured from the
    # bottom. 0.18 keeps captions clear of the platform UI in most cases.
    bottom_margin_frac: float = 0.18
    side_margin_frac: float = 0.06
    # per-side overrides. None means "use side_margin_frac". They exist because
    # TikTok and Reels put a button column down the right edge only: keeping the
    # caption centred in the *usable* width, rather than in the whole frame,
    # means shifting it left, which symmetric margins cannot express.
    left_margin_frac: Optional[float] = None
    right_margin_frac: Optional[float] = None

    # --- the hook, held on screen while the viewer decides whether to stay ----
    # A short-form viewer gives a clip about a second. The captions tell them
    # what is being said; the title tells them what it is about, which is a
    # different job and needs to be readable before the first word lands.
    title_font: Optional[str] = None        # None: same face as the captions
    title_size: int = 68
    title_color: str = "#FFFFFF"
    title_box_color: str = "#000000"
    title_box_alpha: int = 25                # 0 solid, 255 invisible
    title_top_margin_frac: float = 0.14
    title_seconds: float = 3.0
    title_uppercase: bool = True
    title_max_lines: int = 3
    # None means work it out from the frame width and the font size, which is
    # what makes one style hold up at 720p and 1080p and in 9:16 and 1:1.
    title_chars_per_line: Optional[int] = None

    def scaled(self, height: int) -> "CaptionStyle":
        """
        Font sizes are authored against a 1920px-tall frame and scaled to the
        real output, so one style looks the same in 720p and 1080p.

        Built by copying the whole style and adjusting the size fields, rather
        than by listing every field. The listed version was one line per field
        and silently dropped anything added after it was written.
        """
        factor = height / 1920.0
        data = self.to_dict()
        data["font_size"] = max(14, round(self.font_size * factor))
        data["title_size"] = max(12, round(self.title_size * factor))
        data["outline"] = max(0.5, self.outline * factor)
        data["shadow"] = self.shadow * factor
        return CaptionStyle.from_dict(data)

    def margins(self) -> tuple:
        """(left, right) as fractions, resolving the per-side overrides."""
        left = self.side_margin_frac if self.left_margin_frac is None else self.left_margin_frac
        right = self.side_margin_frac if self.right_margin_frac is None else self.right_margin_frac
        return left, right


    # --- serialization, so styles can live in a JSON file the user edits ---

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "CaptionStyle":
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown caption style fields: {', '.join(sorted(unknown))}. "
                f"Valid fields: {', '.join(sorted(known))}"
            )
        return cls(**data)


PRESETS = {
    "clean": CaptionStyle(name="clean"),
    "punch": CaptionStyle(
        name="punch", font="Anton", font_size=76, highlight="#39FF6A",
        outline=5.0, uppercase=True, words_per_chunk=2,
    ),
    "calm": CaptionStyle(
        name="calm", font="Bebas Neue", font_size=54, highlight="#7FC7FF",
        outline=2.5, bold=False, words_per_chunk=5,
    ),
}


def load_styles(path: Optional[str] = None) -> dict:
    """
    Built-in presets, with the user's own styles.json layered on top.

    A user style that reuses a preset's name overrides it, so someone who just
    wants "punch, but red" writes four lines instead of a whole style. Anything
    the file leaves out keeps the dataclass default.
    """
    import json
    import os

    styles = {name: style for name, style in PRESETS.items()}

    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for name, fields in data.items():
            # keys starting with an underscore are notes to the human editing
            # the file, not styles
            if name.startswith("_"):
                continue
            if not isinstance(fields, dict):
                raise ValueError(
                    f"style '{name}' must be an object of field/value pairs, "
                    f"got {type(fields).__name__}"
                )
            base = styles.get(name, CaptionStyle())
            merged = base.to_dict()
            merged.update(fields)
            merged["name"] = name
            styles[name] = CaptionStyle.from_dict(merged)

    return styles


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    words: List[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


# Whisper splits a number at its separator: "19.000" arrives as the two words
# "19" and ".000", and the renderer, joining words with spaces, put "19 .000"
# on screen. It showed up in the first frame anyone looked at closely.
_GLUED = re.compile(r"^[.,:%\-/]\d")


def merge_split_tokens(words: List[Word]) -> List[Word]:
    """
    Rejoin word tokens that were never separate words.

    Only two cases, both safe: a fragment that starts with a separator followed
    by a digit ("19" + ".000", "3" + ",5"), and a token that is nothing but
    punctuation. Anything else is left alone - a caption renderer should not be
    in the business of guessing where words belong.

    The merged word keeps the first start and the last end, so karaoke
    highlighting still covers exactly the time the speaker took to say it.
    """
    out: List[Word] = []
    for w in words:
        text = w.text.strip()
        if out and text and (
            (_GLUED.match(text) and out[-1].text.strip()[-1:].isdigit())
            or not any(ch.isalnum() for ch in text)
        ):
            previous = out[-1]
            out[-1] = Word(previous.start, w.end, previous.text.strip() + text)
            continue
        out.append(w)
    return out


def chunk_words(words: List[Word], per_chunk: int, max_gap: float = 0.7) -> List[Chunk]:
    """
    Group words into caption chunks. A pause longer than max_gap forces a break
    even mid-chunk, so captions follow how the person actually speaks instead of
    stapling together phrases from either side of a silence.
    """
    chunks: List[Chunk] = []
    current: List[Word] = []

    for w in words:
        if current:
            gap = w.start - current[-1].end
            if gap > max_gap or len(current) >= per_chunk:
                chunks.append(Chunk(current))
                current = []
        current.append(w)

    if current:
        chunks.append(Chunk(current))
    return chunks


def chars_per_line(width: int, margin_l: int, margin_r: int, font_size: int) -> int:
    """
    Roughly how many characters fit across the usable width.

    0.5 is the average advance width of a character as a fraction of the font
    size for the condensed display faces this project ships. It is an estimate,
    and it only has to be close: the cost of being wrong by a character or two
    is a line break in a slightly different place, not a title off the edge of
    the frame, because libass still wraps whatever overflows.
    """
    usable = max(1, width - margin_l - margin_r)
    return max(8, int(usable / (font_size * 0.5)))


def wrap_title(text: str, per_line: int, max_lines: int = 3) -> List[str]:
    """
    Break a title into balanced lines.

    Greedy filling is the obvious approach and it produces the thing that makes
    a title card look amateur: three full lines and a fourth holding one word.
    So the target width is the total length divided by the number of lines it
    will need, which spreads the words evenly, and the lines are filled to that
    instead of to the maximum.

    Anything past max_lines is dropped with an ellipsis. A title that long was
    going to cover the speaker's face either way.
    """
    words = text.split()
    if not words:
        return []

    needed = min(max_lines, max(1, -(-len(text) // per_line)))
    target = max(1, len(text) // needed)

    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max(target, len(word)):
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
        else:
            current = f"{current} {word}".strip()

    if current and len(lines) < max_lines:
        lines.append(current)

    used = sum(len(line.split()) for line in lines)
    if used < len(words) and lines:
        lines[-1] = lines[-1].rstrip(",;:") + "..."
    return lines


def title_event(text: str, style: CaptionStyle, width: int, height: int) -> str:
    """
    The one dialogue line that puts the hook on screen.

    It fades rather than cutting, because a title that appears on frame one
    reads as a hard-coded watermark while one that fades in over a fifth of a
    second reads as part of the edit.
    """
    left_frac, right_frac = style.margins()
    margin_l = int(width * left_frac)
    margin_r = int(width * right_frac)
    per_line = style.title_chars_per_line or chars_per_line(
        width, margin_l, margin_r, style.title_size
    )

    body = text.upper() if style.title_uppercase else text
    lines = wrap_title(body, per_line, style.title_max_lines)
    if not lines:
        return ""

    return (
        f"Dialogue: 1,{ass_time(0.0)},{ass_time(style.title_seconds)},Title,,"
        f"0,0,0,,{{\\fad(200,300)}}" + r"\N".join(lines)
    )


def build_ass(
    words: List[Word],
    width: int,
    height: int,
    style: Optional[CaptionStyle] = None,
    clip_start: float = 0.0,
    title: Optional[str] = None,
) -> str:
    """
    Render a full .ass file. Word times are shifted by clip_start so a clip cut
    out of the middle of an episode still starts its subtitles at zero.

    `title` is the hook, held over the opening seconds. It goes in this file
    rather than through a drawtext filter so that it inherits the fonts the
    project ships, survives the two-pass render that segmented clips use, and
    stays editable afterwards along with everything else on screen.
    """
    style = (style or PRESETS["clean"]).scaled(height)

    joined = merge_split_tokens(words)
    shifted = [Word(w.start - clip_start, w.end - clip_start, w.text) for w in joined]
    chunks = chunk_words(shifted, style.words_per_chunk)

    margin_v = int(height * style.bottom_margin_frac)
    title_margin_v = int(height * style.title_top_margin_frac)
    left_frac, right_frac = style.margins()
    margin_l = int(width * left_frac)
    margin_r = int(width * right_frac)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{style.font},{style.font_size},{ass_color(style.primary)},{ass_color(style.primary)},{ass_color(style.outline_color)},{ass_color("#000000", 160)},{-1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline:.1f},{style.shadow:.1f},2,{margin_l},{margin_r},{margin_v},1
Style: Title,{style.title_font or style.font},{style.title_size},{ass_color(style.title_color)},{ass_color(style.title_color)},{ass_color(style.title_box_color, style.title_box_alpha)},{ass_color(style.title_box_color, style.title_box_alpha)},{-1 if style.bold else 0},0,0,0,100,100,0,0,3,{max(6, round(style.title_size * 0.22))},0,8,{margin_l},{margin_r},{title_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    highlight = ass_color(style.highlight)
    primary = ass_color(style.primary)

    lines = []
    # Layer 1, and first in the file: the title is drawn over the captions in
    # the rare case where a long one reaches down into them.
    if title and style.title_seconds > 0:
        event = title_event(title, style, width, height)
        if event:
            lines.append(event)

    for chunk in chunks:
        for i, active in enumerate(chunk.words):
            parts = []
            for j, w in enumerate(chunk.words):
                text = w.text.upper() if style.uppercase else w.text
                if j == i:
                    parts.append(r"{\c" + highlight + "}" + text + r"{\c" + primary + "}")
                else:
                    parts.append(text)
            body = " ".join(parts)

            # hold the last word of a chunk until the chunk ends, so the caption
            # does not blink out early on a trailing pause
            end = active.end if i < len(chunk.words) - 1 else chunk.end
            lines.append(
                f"Dialogue: 0,{ass_time(active.start)},{ass_time(end)},Cap,,0,0,0,,{body}"
            )

    return header + "\n".join(lines) + "\n"
