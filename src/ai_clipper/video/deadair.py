"""
Cut the pauses nobody wants to sit through, and keep the ones that are acting.

A podcast is allowed to breathe. Short-form is not: a second and a half of
someone thinking is a second and a half of a viewer's thumb moving. But a pause
is not automatically dead air, and a tool that removes every gap it finds turns
conversation into an auctioneer's read and cuts the beat before a punchline,
which is the one pause that was doing work.

So the rule here is deliberately timid, and it is three rules:

1. Only gaps at or past `MIN_GAP` are candidates. Below that the pause is part
   of how the sentence is delivered.
2. A cut never closes a gap completely. `KEEP` seconds of it stay, so the edit
   sounds like a pause that was shortened rather than a splice.
3. No more than `MAX_FRACTION` of the clip comes out, largest gaps first. This
   is the one that stops a rambling clip from being compressed into something
   breathless: past a point, the answer is a different clip, not a tighter one.

Nothing here touches video or audio. It produces a `Timeline` - which ranges of
the source survive, and where any given moment in the source lands afterwards -
and the renderer and the subtitle builder both read from it. Keeping the plan
separate from the cutting is what makes it testable without a video file, and it
is also what keeps the captions honest: the same object that decides where to
cut is the one that moves the words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Shorter than this and the pause is part of the delivery.
MIN_GAP = 1.5

# What is left of a cut gap. Silence between sentences is not a defect.
KEEP = 0.4

# Ceiling on how much of a clip can disappear.
MAX_FRACTION = 0.12

# A cut shorter than this is inaudible and still costs a re-encode and a join.
# It exists because the budget can run down to a few milliseconds and then hand
# the next gap a cut with no length, which produced two output pieces separated
# by nothing.
MIN_CUT = 0.1


@dataclass(frozen=True)
class Span:
    """A range of the source that survives into the clip."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Timeline:
    """
    What is kept, in source time, and how to translate a moment into the result.

    A timeline with a single span is the no-op case and is what a clip gets when
    there is nothing worth cutting. Callers can check `cuts` rather than
    special-casing it.
    """
    spans: List[Span]
    source_duration: float

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.spans)

    @property
    def removed(self) -> float:
        return max(0.0, self.source_duration - self.duration)

    @property
    def cuts(self) -> int:
        return max(0, len(self.spans) - 1)

    def ranges(self) -> List[Tuple[float, float]]:
        return [(s.start, s.end) for s in self.spans]

    def remap(self, t: float) -> float:
        """
        Where a source timestamp ends up once the cuts are made.

        A time inside a removed gap has no honest answer, so it collapses to the
        edge of the cut. In practice nothing asks: cuts are only ever placed
        between two words, so every word boundary lands inside a span. The
        clamping exists so a caption built from slightly different word
        timings than the plan used cannot produce a negative timestamp.
        """
        elapsed = 0.0
        for span in self.spans:
            if t < span.start:
                return elapsed
            if t <= span.end:
                return elapsed + (t - span.start)
            elapsed += span.duration
        return elapsed


def slice_of(timeline: "Timeline", a: float, b: float) -> List[Tuple[float, float]]:
    """
    The source ranges that make up clip time `a` to `b`.

    `remap` goes one way, from the source into the finished clip. This goes the
    other, and a preview needs it: someone asks for six seconds starting a
    minute into a clip, and what has to be handed to ffmpeg is the one or two
    ranges of the original episode those six seconds were cut from.
    """
    out, elapsed = [], 0.0
    for span in timeline.spans:
        length = span.duration
        first, last = elapsed, elapsed + length
        if last <= a or first >= b:
            elapsed = last
            continue
        from_here = span.start + max(0.0, a - first)
        to_here = span.start + min(length, b - first)
        if to_here > from_here:
            out.append((from_here, to_here))
        elapsed = last
    return out


def _gaps(words: Sequence, start: float, end: float) -> List[Tuple[float, float]]:
    """
    Silent stretches strictly between the first and last word.

    The head and tail of a clip are left alone on purpose. A clip that starts
    exactly on a consonant sounds clipped, and the room at the end is what stops
    a loop from cutting off the last syllable.
    """
    inside = [w for w in words if w.end > start and w.start < end]
    if len(inside) < 2:
        return []

    found = []
    for a, b in zip(inside, inside[1:]):
        gap_start = max(a.end, start)
        gap_end = min(b.start, end)
        if gap_end - gap_start > 0:
            found.append((gap_start, gap_end))
    return found


def plan(words: Sequence, start: float, end: float,
         min_gap: float = MIN_GAP, keep: float = KEEP,
         max_fraction: float = MAX_FRACTION) -> Timeline:
    """
    Decide which stretches of a clip to drop.

    Candidates are taken largest first, because if only some of the dead air can
    come out it should be the part a viewer would notice most. They are applied
    in time order afterwards.
    """
    duration = max(0.0, end - start)
    if duration <= 0:
        return Timeline([Span(start, end)], duration)

    budget = duration * max_fraction
    candidates = [(a, b) for a, b in _gaps(words, start, end) if b - a >= min_gap]
    candidates.sort(key=lambda g: g[1] - g[0], reverse=True)

    chosen = []
    for a, b in candidates:
        cut_from, cut_to = a + keep / 2.0, b - keep / 2.0
        if cut_to - cut_from < MIN_CUT or budget < MIN_CUT:
            continue
        # A gap larger than the whole budget is shortened by as much as the
        # budget allows rather than skipped. Skipping it was the first version,
        # and it meant the longest silence in a clip - the one a viewer would
        # actually leave over - was the one guaranteed to survive.
        cut_to = min(cut_to, cut_from + budget)
        budget -= cut_to - cut_from
        chosen.append((cut_from, cut_to))

    if not chosen:
        return Timeline([Span(start, end)], duration)

    chosen.sort()
    spans, cursor = [], start
    for cut_from, cut_to in chosen:
        if cut_from > cursor:
            spans.append(Span(cursor, cut_from))
        cursor = cut_to
    if cursor < end:
        spans.append(Span(cursor, end))

    return Timeline(spans, duration)


def from_spans(spans: Sequence, start: float, end: float) -> Timeline:
    """
    A timeline someone else decided.

    The gap detector is one way to choose which stretches of a clip survive. A
    person dragging handles in an editor is another, and the renderer does not
    care which produced the list. Spans are in source time, the same coordinates
    `plan` returns, and are clamped to the clip and put in order so a UI cannot
    hand back something that renders as a negative duration.
    """
    kept = []
    for a, b in spans:
        a, b = max(float(a), start), min(float(b), end)
        if b - a > MIN_CUT:
            kept.append(Span(a, b))
    kept.sort(key=lambda s: s.start)
    if not kept:
        kept = [Span(start, end)]
    return Timeline(kept, max(0.0, end - start))


def shift_words(words: Sequence, timeline: Timeline, factory=None) -> List:
    """
    Move captions onto the cut timeline.

    Words keep every field they arrived with, including any per-word styling a
    caption editor attached, because the new word is a copy of the old one with
    two numbers changed rather than a fresh one built from three of its fields.
    An earlier version took a factory and called `factory(start, end, text)`,
    which silently dropped anything else the word was carrying.

    Words that fell entirely inside a cut are dropped rather than stacked on the
    seam, which can only happen when the caption words and the planning words
    disagree.

    `factory` is accepted and ignored, so older callers keep working.
    """
    from dataclasses import replace

    out = []
    for w in words:
        start, end = timeline.remap(w.start), timeline.remap(w.end)
        if end <= start and out:
            continue
        out.append(replace(w, start=start, end=end))
    return out


def describe(timeline: Optional[Timeline]) -> str:
    """One phrase for the run log, or empty when nothing was cut."""
    if timeline is None or not timeline.cuts:
        return ""
    return (f"{timeline.removed:.1f}s of dead air out across "
            f"{timeline.cuts} cut{'s' if timeline.cuts > 1 else ''}")
