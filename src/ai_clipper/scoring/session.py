"""
Remembering a run, so "not this one" survives closing the terminal.

Selection has had regenerate() since early on, and until now nothing could call
it. The scorer keeps its whole ranked pool, a rejection is cheap, and none of it
was reachable from the command line: the only caller was the test suite. A
feature that has been paid for and never collected.

The missing piece was not the rejection logic, it was memory. Every run rebuilds
the pool from scratch, so without a record of what was rejected last time, a
second run hands back exactly the clips the user just turned down.

So a run writes `session.json` beside its clips: the settings it ran with, which
pool windows became which numbered clip, and every window rejected so far. The
pool itself is not stored. It is derived from the transcript and the duration
settings, so given the same inputs it comes back identical, and storing four
hundred candidates to disk would be storing a cache of a pure function.

Clip numbers stay put. If clip 4 is rejected, its replacement becomes clip 4
even when it comes from an hour later in the episode, and clips 1 to 3 and 5
onwards keep both their numbers and their files. Renumbering everything would
mean re-rendering everything, and would move clips the user never complained
about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

FILENAME = "session.json"


def key_of(candidate) -> Tuple[float, float]:
    """The pool identity of a window, before any boundary adjustment."""
    return (round(candidate.start, 2), round(candidate.end, 2))


@dataclass
class Slot:
    """One numbered clip: which pool window it is, and where it ended up."""
    index: int
    key: Tuple[float, float]
    start: float
    end: float

    def as_dict(self) -> dict:
        return {"index": self.index, "key": list(self.key),
                "start": round(self.start, 2), "end": round(self.end, 2)}

    @classmethod
    def from_dict(cls, d: dict) -> "Slot":
        return cls(int(d["index"]), tuple(d["key"]), float(d["start"]), float(d["end"]))


@dataclass
class Session:
    settings: Dict = field(default_factory=dict)
    slots: List[Slot] = field(default_factory=list)
    rejected: List[Tuple[float, float]] = field(default_factory=list)
    # filename -> the (start, end) it was actually encoded with
    rendered: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # --- disk ---------------------------------------------------------------

    @classmethod
    def load(cls, out_dir) -> Optional["Session"]:
        path = Path(out_dir) / FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return cls(
            settings=data.get("settings", {}),
            slots=[Slot.from_dict(s) for s in data.get("slots", [])],
            rejected=[tuple(r) for r in data.get("rejected", [])],
            rendered={k: tuple(v) for k, v in data.get("rendered", {}).items()},
        )

    def save(self, out_dir) -> None:
        path = Path(out_dir) / FILENAME
        path.write_text(json.dumps({
            "settings": self.settings,
            "slots": [s.as_dict() for s in self.slots],
            "rejected": [list(r) for r in self.rejected],
            "rendered": {k: list(v) for k, v in sorted(self.rendered.items())},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- reuse --------------------------------------------------------------

    def usable_with(self, settings: Dict) -> bool:
        """
        Whether a saved session still describes the run being asked for.

        Change the transcript or the duration limits and the pool is a different
        pool, so the saved window coordinates mean nothing. Better to start
        clean than to silently reuse numbers that no longer point anywhere.
        """
        return all(self.settings.get(k) == v for k, v in settings.items())

    def restore(self, selection, pool: Sequence) -> bool:
        """
        Put a fresh Selection back into the state this session recorded.

        Returns False when a saved window is no longer in the pool, which means
        something changed that `usable_with` did not catch. The caller should
        fall back to a fresh selection rather than render a half-restored one.
        """
        by_key = {key_of(c): c for c in pool}

        chosen = []
        for slot in sorted(self.slots, key=lambda s: s.index):
            candidate = by_key.get(slot.key)
            if candidate is None:
                return False
            chosen.append(candidate)

        selection.rejected = set(self.rejected)
        selection.chosen = sorted(chosen, key=lambda c: c.start)
        return True

    # --- numbering ----------------------------------------------------------

    def numbers_for(self, chosen: Sequence) -> List[int]:
        """
        A clip number for each chosen window, keeping the ones already in use.

        A window that was already on screen keeps its number. Numbers freed by a
        rejection go to the new windows, smallest first, so a single rejection
        moves exactly one clip.

        Call this *before* any boundary adjustment. A window's identity is its
        original pool coordinates, and extension is about to change the ones on
        the object.
        """
        held = {slot.key: slot.index for slot in self.slots}
        keys = [key_of(c) for c in chosen]
        taken = {held[k] for k in keys if k in held}

        free = (n for n in range(1, len(keys) + len(self.slots) + 2) if n not in taken)
        return [held[k] if k in held else next(free) for k in keys]

    def record(self, entries: Sequence, rejected: Sequence) -> None:
        """
        entries: (number, key, clip) per chosen clip, keys taken before
        adjustment and clips read after it.
        """
        self.slots = [Slot(n, k, c.start, c.end) for n, k, c in entries]
        self.rejected = sorted({tuple(r) for r in rejected})

    def is_current(self, filename: str, clip) -> bool:
        """
        Whether the file on disk was encoded from exactly this clip.

        The first version asked a different question - have this clip's
        boundaries changed since the last run - and it was wrong in a way that
        only showed up in use. A --dry-run updates the selection without
        encoding anything, so after previewing a regenerated clip the saved
        boundaries described the new clip while the file on disk was still the
        old one. The run then compared new against new, found no difference, and
        skipped a file that was genuinely stale.

        What matters is not what changed but what was actually written, so that
        is what gets recorded, per file, at the moment ffmpeg succeeds.
        """
        was = self.rendered.get(filename)
        return was is not None and tuple(was) == (round(clip.start, 2), round(clip.end, 2))

    def mark_rendered(self, filename: str, clip) -> None:
        self.rendered[filename] = (round(clip.start, 2), round(clip.end, 2))

    def forget_rendered(self, keep: Sequence[str]) -> None:
        """Drop records for files this run is no longer producing."""
        self.rendered = {k: v for k, v in self.rendered.items() if k in set(keep)}
