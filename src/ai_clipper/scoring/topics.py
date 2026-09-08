"""
Where one subject stops and the next begins.

The scorer already refuses a clip that stops mid-sentence. That was not enough.
Feedback on a real episode:

    "di clip 15 itu masih kepotong penjelasannya. bukan dari segi dia ngomong
     lalu terpotong, tapi setelah itu dia masih menjelaskan pengalamannya
     (topiknya belum tuntas)"

The clip ended on a full stop and still felt cut off, because the speaker kept
telling the same story for another minute. Grammar is the wrong unit; the unit
is the subject.

Two measurements here, both lexical, both explainable, neither needing a model
or a network:

  topic_boundaries - Hearst's TextTiling (1997). Content words are cut into
    fixed blocks; at each gap the vocabulary before is compared with the
    vocabulary after, and a deep dip means the subject turned. On the 60-minute
    episode this was built against: 76 boundaries, one every 48 seconds.

  TopicModel.extend - given a chosen clip, decide whether the speaker is still
    on the same thing past the cut, and if so where they stop.

The honest limit: this reads words, not meaning. A story can continue in
entirely different vocabulary - and on the very clip quoted above it does, when
the subject shifts from "repo / margin call / wipe out" to "Maybank / nego /
reduce". Measured, the cut sits at the 30th percentile of the episode's own
continuation levels: lexically it looks like a clean break. That case is caught
by the dangling-speech rule in hook_scorer instead ("Gue bilang sama Maybank."
announces speech and never delivers it), and full narrative completion is where
an optional LLM re-rank would genuinely earn its cost. The point of doing this
part by hand is that it is free, fast, and every number can be traced.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Sequence

from . import signals
from .hook_scorer import (
    _content_words,
    closing_unfinished,
    ends_dangling,
    is_clean_ending,
    opening_stands_alone,
    _tokenize,
)

BLOCK_WORDS = 12
CONTEXT_BLOCKS = 5
DEPTH_CUTOFF = 0.4     # standard deviations above the mean dip depth

TAIL_SECONDS = 25.0    # how much of a clip counts as "what it is about"
LOOKAHEAD = 20.0       # how far past a candidate end we listen


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    num = sum(a[k] * b[k] for k in shared)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def topic_boundaries(
    segments: Sequence,
    block_words: int = BLOCK_WORDS,
    context: int = CONTEXT_BLOCKS,
    cutoff: float = DEPTH_CUTOFF,
) -> List[float]:
    """
    Times (seconds) where the vocabulary turns over, ascending.

    An episode too short to fill the context window on both sides gets an empty
    list rather than an arbitrary split.
    """
    words = [(w, seg.end) for seg in segments for w in _content_words(seg.text)]
    if len(words) < block_words * (2 * context + 1):
        return []

    blocks = [words[i:i + block_words] for i in range(0, len(words), block_words)]

    gaps = []
    for i in range(context, len(blocks) - context):
        left = Counter(w for b in blocks[i - context:i] for w, _ in b)
        right = Counter(w for b in blocks[i:i + context] for w, _ in b)
        gaps.append((blocks[i][0][1], _cosine(left, right)))
    if not gaps:
        return []

    sims = [s for _, s in gaps]

    # depth: how far a dip sits below the peaks either side. A shallow wobble in
    # a generally low stretch is not a boundary; a sharp drop between two high
    # plateaus is.
    depths = []
    for j in range(len(sims)):
        k = j
        while k > 0 and sims[k - 1] >= sims[k]:
            k -= 1
        left_peak = sims[k]
        k = j
        while k < len(sims) - 1 and sims[k + 1] >= sims[k]:
            k += 1
        right_peak = sims[k]
        depths.append((left_peak - sims[j]) + (right_peak - sims[j]))

    mean = sum(depths) / len(depths)
    var = sum((d - mean) ** 2 for d in depths) / len(depths)
    threshold = mean + cutoff * math.sqrt(var)

    out = []
    for j, d in enumerate(depths):
        if d <= threshold:
            continue
        if j > 0 and depths[j - 1] > d:
            continue
        if j < len(depths) - 1 and depths[j + 1] > d:
            continue
        out.append(gaps[j][0])
    return out


def continuation(clip_segments: Sequence, following: Sequence) -> float:
    """How much of a passage is made of another passage's vocabulary."""
    a = Counter(w for s in clip_segments for w in _content_words(s.text))
    b = Counter(w for s in following for w in _content_words(s.text))
    return _cosine(a, b)


def following_segments(segments: Sequence, end: float, horizon: float = LOOKAHEAD) -> list:
    return [s for s in segments if end <= s.start < end + horizon]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


class TopicModel:
    """
    One episode's subject structure, measured once and reused.

    The thresholds are fitted to the episode, for the same reason the score is:
    a fixed number like "continuation above 0.10 means still talking" turned out
    to sit either side of the median depending on how chatty the speakers are.
    On the episode this was built against the median cut scores 0.13, so 0.10
    would have called almost every cut mid-topic. Percentiles of the episode's
    own levels do not have that problem.
    """

    HIGH_PCT = 65.0    # above this at the cut = the subject is still running
    LOW_PCT = 25.0     # below this ahead = the subject has turned

    def __init__(self, segments: Sequence, sample_every: float = 5.0):
        self.segments = list(segments)
        self.boundaries = topic_boundaries(self.segments)

        levels = []
        if self.segments:
            t = self.segments[0].start + TAIL_SECONDS
            last = self.segments[-1].end - LOOKAHEAD
            while t < last:
                levels.append(self._level_at(t))
                t += sample_every
        self.levels = levels
        self.high = _percentile(levels, self.HIGH_PCT)
        self.low = _percentile(levels, self.LOW_PCT)

    def _tail(self, end: float, start: float = None) -> list:
        floor = end - TAIL_SECONDS
        if start is not None:
            floor = max(floor, start)
        return [s for s in self.segments if s.start >= floor and s.end <= end]

    def _level_at(self, end: float, start: float = None) -> float:
        tail = self._tail(end, start)
        if not tail:
            return 0.0
        return continuation(tail, following_segments(self.segments, end))

    def extend(
        self,
        clip,
        hard_max: float = 150.0,
        max_extension: float = 75.0,
        min_extension: float = 6.0,
    ) -> Optional[tuple]:
        """
        Push a clip's end forward to where the subject actually stops.

        Returns (start, end, added_seconds), or None to leave the clip alone.

        The reference is the clip's own tail, held fixed while the end moves.
        Comparing the whole growing clip does not work: it absorbs the new
        vocabulary as it extends and the measure stops falling.

          1. If continuation at the cut is below this episode's 65th percentile,
             the subject already turned - nothing to do.
          2. Otherwise walk forward to the first point where it has dropped to
             the 25th percentile. That is where the subject ends.
          3. Walk back from there to the last line that ends cleanly, because a
             vocabulary dip is not a full stop.

        `hard_max` caps the clip; `max_extension` stops a missed boundary from
        dragging in half an episode.
        """
        tail = self._tail(clip.end, clip.start)
        if not tail or not self.levels:
            return None

        if self._level_at(clip.end, clip.start) < self.high:
            return None

        # If the clip already ends on the far side of a boundary, anything added
        # belongs to the *next* subject, not this one. This is what turned a
        # 75-second clip into a 99-second one carrying two subjects: the cap
        # below only looks forward, and the boundary it needed was behind.
        if any(clip.start + 20.0 < b < clip.end for b in self.boundaries):
            return None

        limit = min(clip.end + max_extension, clip.start + hard_max)

        # Never extend across a topic boundary. Without this the continuation
        # measure alone pulled a clip 24 seconds past the point where the
        # subject changed, and the result was two subjects in one clip - the
        # reviewer's words: "clip 9 bisa jadi 2 topik berbeda". TextTiling had
        # already put a boundary exactly where he said the second one starts.
        crossing = [b for b in self.boundaries if clip.end + 1.0 < b <= limit]
        if crossing:
            limit = crossing[0]

        turn = None
        for seg in self.segments:
            if not (clip.end + min_extension <= seg.end <= limit):
                continue
            ahead = continuation(tail, following_segments(self.segments, seg.end))
            if ahead <= self.low:
                turn = seg.end
                break
        if turn is None:
            return None

        # Back off to a line that can be stopped on - and check the whole
        # closing exchange, not just that line's grammar, or the extension
        # happily lands on the first half of a question.
        new_end = None
        for seg in self.segments:
            if not (clip.end < seg.end <= turn and is_clean_ending(seg.text)):
                continue
            body = [s for s in self.segments
                    if s.start >= clip.start and s.end <= seg.end]
            if closing_unfinished(body, following_segments(self.segments, seg.end)):
                continue
            new_end = seg.end
        if new_end is None or new_end <= clip.end + 1.0:
            return None

        return (clip.start, new_end, new_end - clip.end)


__all__ = [
    "TopicModel", "topic_boundaries", "continuation", "following_segments",
    "shift_to_a_real_opening", "ends_dangling", "is_clean_ending",
    "opening_stands_alone",
]


def _names_in_line(text: str) -> set:
    """
    Capitalised words in one line that are not first in a sentence.

    Whisper capitalises the first word of every segment, so a line has to be
    read on its own: joining a whole transcript first and splitting on full
    stops makes almost every segment-initial word look like a proper noun. On a
    transcript with barely any punctuation that turned the name guard into a
    blanket refusal, and no clip opening could ever be moved.
    """
    not_names = signals.active().not_names
    found = set()
    for sentence in re.split(r"[.!?]\s*", text.strip()):
        for token in re.findall(r"[A-Za-z][A-Za-z']*", sentence)[1:]:
            if token[:1].isupper() and len(token) >= 4 and token.lower() not in not_names:
                found.add(token.lower())
    return found


def _names_in(segments: Sequence) -> set:
    out = set()
    for seg in segments:
        out |= _names_in_line(seg.text)
    return out


def _drops_a_name(skipped: Sequence, kept: Sequence) -> bool:
    """True when skipping this material removes a name the rest never repeats."""
    kept_text = " ".join(s.text for s in kept)
    return bool(_names_in(skipped) - _names_in(kept) - set(_tokenize(kept_text)))


def shift_to_a_real_opening(
    clip,
    segments: Sequence,
    max_shift: float = 25.0,
    min_duration: float = 20.0,
) -> Optional[tuple]:
    """
    Move a clip's start forward to the first line that works cold.

    Returns (start, end, removed_seconds) or None.

    The mirror of extend(), and deliberately the same shape: it runs after
    selection, so it never changes which moment was chosen - only where that
    moment is entered. A clip whose first line already stands alone is left
    exactly as it was.

    It cannot fix everything, and it is worth saying where it stops. "Nah,
    suatu ketika gue lagi datang ke kantor, gue dipanggil ke ruangan dia" reads
    as self-contained by every lexical test here and still leaves a viewer
    asking who "dia" is. Resolving that reference needs more than word counts.
    """
    if not segments:
        return None
    if opening_stands_alone(clip.segments[0].text if getattr(clip, "segments", None)
                            else segments[0].text):
        return None

    limit = min(clip.start + max_shift, clip.end - min_duration)
    inside = [s for s in segments if clip.start <= s.start and s.end <= clip.end]

    for seg in segments:
        if seg.start <= clip.start:
            continue
        if seg.start > limit:
            break
        if not opening_stands_alone(seg.text):
            continue

        # Do not skip past the only place the subject is named. "Perusahaan
        # kapal kan itu? Kapal Hilong. Kapal dari China." is a weak opening by
        # every test here, and dropping it leaves the clip starting on "waktu
        # itu market cap-nya masih ratusan M" with no way to know of what.
        skipped = [s for s in inside if s.start < seg.start]
        kept = [s for s in inside if s.start >= seg.start]
        if _drops_a_name(skipped, kept):
            # this candidate would lose the name; a later one loses even more
            return None

        return (seg.start, clip.end, seg.start - clip.start)
    return None
