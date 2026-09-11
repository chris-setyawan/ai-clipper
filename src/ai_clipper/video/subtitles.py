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

    # --- how a caption arrives -----------------------------------------------
    # none, pop, fade, rise, drop, zoom, blur or type. Applied per caption
    # event, which is per word, so the effect reads as the caption keeping time
    # with the speaker rather than as a single entrance at the top of the clip.
    animation: str = "none"
    animation_ms: int = 150
    # How far rise and drop travel, as a fraction of the font size. Expressed
    # against the font rather than the frame so it stays proportionate when the
    # same style is used at 720p and 1080p.
    animation_travel: float = 0.6

    # Glow is a blurred outline in its own colour, which is one event rather
    # than a second copy of the text drawn underneath.
    #
    # None means the outline colour, and that is the default on purpose. A glow
    # is drawn behind the fill and bleeds over its edges, so a glow the same
    # colour as a word turns that word into a solid blob. The first version
    # defaulted to the same green `punch` highlights with, and every highlighted
    # word disappeared. Pick a colour that differs from both `primary` and
    # `highlight`, or leave it alone for a soft dark halo that works under any
    # fill.
    glow: bool = False
    glow_color: Optional[str] = None
    glow_size: float = 6.0

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
        data["glow_size"] = self.glow_size * factor
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
    """
    One spoken word, and optionally how this one word should look.

    `style` is what a caption editor writes: a dict of overrides that apply to
    this word alone. It lives on the word rather than in a separate table keyed
    by position, because positions move. Words get merged when Whisper splits a
    number, and they get shifted when dead air is cut, and an index into the
    original list stops meaning anything after either. Carried on the word, the
    styling survives both.

        Word(3.2, 3.5, "setiap", {"italic": True, "color": "#39FF6A"})

    Known keys: bold, italic, font, size, color. Anything else is ignored, so a
    file written by a newer editor still renders in an older build.
    """

    start: float
    end: float
    text: str
    style: Optional[dict] = None


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
            out[-1] = Word(previous.start, w.end, previous.text.strip() + text,
                           previous.style)
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



ANIMATIONS = ("none", "pop", "fade", "rise", "drop", "zoom", "blur", "type")

# rise and drop are the two that move, and moving in ASS needs a point to move
# between, which means the caption has to be positioned explicitly rather than
# laid out from its margins.
MOVING = ("rise", "drop")


def word_tags(word_style: Optional[dict], style: CaptionStyle) -> tuple:
    """
    (opening, closing) override tags for one word, or ("", "") for a plain one.

    Every override is closed again immediately, because a caption event holds
    the whole chunk and an unclosed tag would leak onto the words after it.
    Unknown keys are ignored so a file written by a newer caption editor still
    renders here, minus whatever this build has never heard of.
    """
    if not word_style:
        return "", ""

    opening, closing = [], []
    if word_style.get("bold") is not None:
        opening.append(r"\b1" if word_style["bold"] else r"\b0")
        closing.append(r"\b1" if style.bold else r"\b0")
    if word_style.get("italic"):
        opening.append(r"\i1")
        closing.append(r"\i0")
    if word_style.get("font"):
        opening.append(r"\fn" + str(word_style["font"]))
        closing.append(r"\fn" + style.font)
    if word_style.get("size"):
        opening.append(r"\fs" + str(int(word_style["size"])))
        closing.append(r"\fs" + str(style.font_size))
    if word_style.get("color"):
        opening.append(r"\c" + ass_color(str(word_style["color"])))
        closing.append(r"\c" + ass_color(style.primary))

    if not opening:
        return "", ""
    return "{" + "".join(opening) + "}", "{" + "".join(closing) + "}"


def entrance(style: CaptionStyle, anchor: Optional[tuple] = None) -> str:
    """
    The tag that gives a caption its way of arriving.

    `anchor` is (x, y) and is only needed by the animations that move. Anything
    that merely scales, fades or blurs is a transform on the text where it
    already sits, and forcing a position on those would quietly change how
    multi-line captions wrap.
    """
    ms = max(1, int(style.animation_ms))
    kind = style.animation

    if kind == "pop":
        return r"{\fscx60\fscy60\t(0," + str(ms) + r",\fscx100\fscy100)}"
    if kind == "zoom":
        return r"{\fscx130\fscy130\t(0," + str(ms) + r",\fscx100\fscy100)}"
    if kind == "fade":
        return r"{\fad(" + str(ms) + r",0)}"
    if kind == "blur":
        return (r"{\blur" + f"{style.font_size * 0.12:.1f}"
                + r"\t(0," + str(ms) + r",\blur0)}")
    if kind in MOVING and anchor:
        x, y = anchor
        travel = max(2, int(style.font_size * style.animation_travel))
        from_y = y + travel if kind == "rise" else y - travel
        return (r"{\move(" + f"{x},{from_y},{x},{y},0,{ms}"
                + r")\fad(" + str(ms) + r",0)}")
    return ""


def glow_tag(style: CaptionStyle) -> str:
    """
    A blurred outline in the glow colour.

    One event, not two. The obvious way to glow is to draw the text twice, once
    fat and blurred underneath and once sharp on top, and it doubles the event
    count for a result libass already gives for free: blur applies to the
    border, so a wide border plus blur is a halo.
    """
    if not style.glow:
        return ""
    return (r"{\bord" + f"{max(style.outline, style.glow_size):.1f}"
            + r"\blur" + f"{style.glow_size:.1f}"
            + r"\3c" + ass_color(style.glow_color or style.outline_color) + "}")


def typed(text: str, start: float, end: float, ms: int, prefix: str,
          style_name: str = "Cap") -> List[str]:
    """
    A caption that types itself in, one character at a time.

    Six of the seven animations are a tag on an event. This one is not: ASS has
    no reveal transform, so a typewriter is a run of events each showing one
    character more than the last. The reveal is capped at `ms` no matter how
    long the line is, so a long chunk types faster rather than running past the
    words being spoken.
    """
    letters = [i for i, ch in enumerate(text) if not ch.isspace()]
    if not letters:
        return []

    span = min(ms / 1000.0, max(0.0, end - start))
    step = span / len(letters) if letters else 0.0

    events = []
    for n, cut in enumerate(letters):
        at = start + step * n
        until = start + step * (n + 1) if n < len(letters) - 1 else end
        if until <= at:
            continue
        events.append(
            f"Dialogue: 0,{ass_time(at)},{ass_time(until)},{style_name},,0,0,0,,"
            + prefix + text[: cut + 1]
        )
    return events


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
    shifted = [Word(w.start - clip_start, w.end - clip_start, w.text, w.style)
               for w in joined]
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

    # Where the caption block sits, needed only by the animations that move.
    # Alignment 2 anchors at the bottom centre, so this is the same point the
    # margins would have put it at.
    anchor = (width // 2, height - margin_v) if style.animation in MOVING else None
    prefix = glow_tag(style) + entrance(style, anchor)

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
                opening, closing = word_tags(w.style, style)
                if j == i:
                    text = r"{\c" + highlight + "}" + text + r"{\c" + primary + "}"
                parts.append(opening + text + closing)
            body = " ".join(parts)

            # hold the last word of a chunk until the chunk ends, so the caption
            # does not blink out early on a trailing pause
            end = active.end if i < len(chunk.words) - 1 else chunk.end

            if style.animation == "type":
                # One event per character, so this one cannot share the path
                # the other six take. Only the first word of a chunk types; the
                # rest of the chunk is already on screen by then.
                if i == 0:
                    lines.extend(typed(body, active.start, end,
                                       style.animation_ms, prefix))
                else:
                    lines.append(
                        f"Dialogue: 0,{ass_time(active.start)},{ass_time(end)},"
                        f"Cap,,0,0,0,,{prefix}{body}"
                    )
                continue

            lines.append(
                f"Dialogue: 0,{ass_time(active.start)},{ass_time(end)},Cap,,0,0,0,,"
                f"{prefix}{body}"
            )

    return header + "\n".join(lines) + "\n"
