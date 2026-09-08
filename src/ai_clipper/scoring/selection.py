"""
Choosing which candidates become clips, and swapping one out when the user
rejects it.

The scorer produces thousands of overlapping candidate windows. Picking the
final set is a separate job from scoring them, and it has to survive the user
saying "not this one" without recomputing anything or disturbing the clips they
already liked.

The rule: a replacement must not overlap the clips still being kept, but it may
sit anywhere inside the rejected clip's time range - usually the good moment is
there, just with different boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set


def overlaps(a, b) -> bool:
    return not (a.end <= b.start or a.start >= b.end)


def _key(c) -> tuple:
    return (round(c.start, 2), round(c.end, 2))


@dataclass
class Selection:
    """
    A working set of chosen clips plus the pool they came from.

    Rejections are remembered so the same window is never offered twice, which
    is the difference between a regenerate button that feels useful and one that
    keeps handing back what you just turned down.
    """
    pool: List
    chosen: List = field(default_factory=list)
    rejected: Set[tuple] = field(default_factory=set)

    @classmethod
    def build(cls, candidates: List, n: int) -> "Selection":
        sel = cls(pool=sorted(candidates, key=lambda c: c.score, reverse=True))
        sel.chosen = sel._fill([], n)
        return sel

    def _fill(self, keep: List, n: int) -> List:
        out = list(keep)
        for cand in self.pool:
            if len(out) >= n:
                break
            if _key(cand) in self.rejected:
                continue
            if any(overlaps(cand, c) for c in out):
                continue
            out.append(cand)
        out.sort(key=lambda c: c.start)
        return out

    @staticmethod
    def _overlap_fraction(a, b) -> float:
        """How much of the shorter clip's duration the two share."""
        start, end = max(a.start, b.start), min(a.end, b.end)
        if end <= start:
            return 0.0
        shorter = min(a.end - a.start, b.end - b.start) or 1.0
        return (end - start) / shorter

    def alternatives_for(self, index: int, mode: str = "different") -> List:
        """
        Candidates that could replace the clip at `index`, ranked.

        Rejections mean two different things and need different answers:

          "different" - the moment itself is not worth posting. Offer something
                        from elsewhere in the episode, so candidates sharing
                        most of the rejected clip's timeline are filtered out.
          "retime"    - the moment is good but the cut is wrong. Offer other
                        windows over the same stretch, longer or shorter.

        Either way a replacement may never collide with the clips being kept.
        """
        if not (0 <= index < len(self.chosen)):
            return []

        current = self.chosen[index]
        keep = [c for i, c in enumerate(self.chosen) if i != index]

        out = []
        for cand in self.pool:
            if _key(cand) in self.rejected:
                continue
            if _key(cand) == _key(current):
                continue
            if any(overlaps(cand, c) for c in keep):
                continue

            shared = self._overlap_fraction(cand, current)
            if mode == "different" and shared > 0.3:
                continue
            if mode == "retime" and shared < 0.3:
                continue

            out.append(cand)
        return out

    def regenerate(self, index: int, mode: str = "different") -> Optional[object]:
        """
        Reject the clip at `index` and put the best remaining alternative in its
        place. Returns the replacement, or None when nothing else fits - in
        which case the clip is removed rather than silently kept, so the user is
        not told a rejection happened when it did not.
        """
        if not (0 <= index < len(self.chosen)):
            return None

        self.rejected.add(_key(self.chosen[index]))
        alts = self.alternatives_for(index, mode=mode)
        if not alts and mode == "different":
            # nothing elsewhere in the episode fits between the kept clips -
            # a re-cut of the same moment beats returning nothing
            alts = self.alternatives_for(index, mode="retime")

        keep = [c for i, c in enumerate(self.chosen) if i != index]
        if not alts:
            self.chosen = sorted(keep, key=lambda c: c.start)
            return None

        replacement = alts[0]
        keep.append(replacement)
        self.chosen = sorted(keep, key=lambda c: c.start)
        return replacement

    def regenerate_all(self, n: Optional[int] = None) -> List:
        """Reject everything currently chosen and fill from what is left."""
        n = n or len(self.chosen)
        for c in self.chosen:
            self.rejected.add(_key(c))
        self.chosen = self._fill([], n)
        return self.chosen

    def remaining(self) -> int:
        """
        How many alternatives are actually available - the pool minus what was
        rejected and minus what is already on screen.

        The earlier version counted the chosen clips too, so a run of 15 out of
        a 400-candidate pool reported "400 held in reserve", which is both wrong
        and suspiciously round.
        """
        taken = {_key(c) for c in self.chosen} | self.rejected
        return sum(1 for c in self.pool if _key(c) not in taken)
