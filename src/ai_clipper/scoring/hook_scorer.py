"""
Rule-based hook / highlight scoring for podcast transcripts.

Given a timestamped transcript, this finds candidate clip windows and scores
how likely each one is to work as a standalone short-form hook - and, unlike
a plain "ask an LLM which part is interesting" approach, it says *why*: which
signal fired and how much it contributed. That's the whole point of building
this ourselves instead of wrapping a prompt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from . import signals


# --- data model ----------------------------------------------------------

@dataclass
class TranscriptSegment:
    """One timestamped chunk of transcript (e.g. one Whisper segment)."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class ClipCandidate:
    start: float
    end: float
    segments: list
    score: float          # 0-10, calibrated against this episode's own pool
    signals: dict         # signal name -> contribution, for explainability
    raw_score: float = 0.0    # sum of signals, comparable across episodes
    percentile: float = 0.0   # where this window sits among all candidates

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    def explain(self) -> str:
        fired = sorted(self.signals.items(), key=lambda kv: -kv[1])
        parts = [f"{name} (+{val:.1f})" for name, val in fired if val > 0]
        return "; ".join(parts) if parts else "no strong signals"


# --- vocabulary -----------------------------------------------------------
#
# The word lists used to live here. They are the one part of this scorer that is
# guaranteed to need changing for a new language or a new kind of podcast, so
# they moved to scoring/signals.py, where a signals.json next to the project can
# override or extend any of them. Everything below reads whichever pack is
# active; nothing below knows a single Indonesian word.
#
# The names are kept as module-level aliases because other modules and the tests
# import them, and because the built-in pack is the answer in the common case.

STOPWORDS = signals.INDONESIAN.stopwords
LEAD_IN_ENDINGS = signals.INDONESIAN.lead_in_endings
SPEECH_ACT_VERBS = signals.INDONESIAN.speech_act_verbs
QUESTION_OPENERS = signals.INDONESIAN.question_openers
PROMISE_MARKERS = signals.INDONESIAN.promise_markers


def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _count_markers(text_lower: str, markers: set) -> int:
    return sum(1 for m in markers if m in text_lower)


def _content_words(text: str) -> list:
    """Words that carry topic, ignoring stopwords and very short tokens."""
    stop = signals.active().stopwords
    return [t for t in _tokenize(text) if t not in stop and len(t) >= 4]


def ends_dangling(text: str) -> bool:
    """
    True when a sentence announces speech or an act and stops before the
    substance of it - "Gue bilang sama Maybank.", "Nah gue hari Senin gue tanya."

    A proper noun after the verb does not count as substance: it names who was
    spoken to, not what was said. Anything else does, so "bokap gue bilang
    gausah lah" is complete.
    """
    pack = signals.active()
    tokens = re.findall(r"[A-Za-z']+", text)
    lower = [t.lower() for t in tokens]
    hits = [i for i, t in enumerate(lower) if t in pack.speech_act_verbs]
    if not hits:
        return False
    after = [(tokens[k], lower[k]) for k in range(hits[-1] + 1, len(tokens))]
    substance = [
        raw for raw, low in after
        if low not in pack.empty_after_verb and len(low) >= 3 and not raw[:1].isupper()
    ]
    return not substance


def _strip_particles(tokens: list) -> list:
    particles = signals.active().discourse_particles
    i = 0
    while i < len(tokens) and tokens[i] in particles:
        i += 1
    return tokens[i:]


def opening_stands_alone(text: str) -> bool:
    """
    True when a first line works for someone who has seen nothing before it.

    Endings were the first thing fixed, and once they were, the openings were
    what stood out. The strongest moment in the test episode - a konglomerat
    telling him "kalau nggak ngerti saham, lu nyangkul sampai bongkok lu nggak
    bakal kaya" - opened with "Kawan lama, temen dia juga.", which is the tail
    of an answer to a question the viewer never heard. Another opened with "Nah
    gue hari Senin gue tanya." - the same dangling speech act already caught at
    the other end of a clip.

    A line stands alone when, with its opening particles stripped, it is either
    a question carrying real content, or a statement carrying more, and does not
    start by pointing at something off-screen.
    """
    pack = signals.active()
    t = text.strip()
    low = t.lower()
    if ends_dangling(low):
        return False

    tokens = _strip_particles(_tokenize(low))
    if not tokens:
        return False
    if tokens[0] in pack.back_references:
        return False

    content = len(_content_words(" ".join(tokens)))
    asks = low.rstrip().endswith("?") or _contains_phrase(low, pack.question_openers)
    return content >= (pack.opening_min_content_question if asks
                       else pack.opening_min_content)


def _contains_phrase(text_lower: str, phrases) -> bool:
    """
    Whole-word phrase match. Substring matching found "berapa" inside
    "beberapa" and called a statement a question, which is how a clip the
    reviewer had passed came back marked unfinished.
    """
    for phrase in phrases:
        pattern = r"(?<![a-z])" + r"\s+".join(re.escape(w) for w in phrase.split())
        if re.search(pattern + r"(?![a-z])", text_lower):
            return True
    return False


def _opening_event(text: str) -> bool:
    """A line that raises something the clip then has to deliver."""
    pack = signals.active()
    low = text.strip().lower()
    if _contains_phrase(low, pack.promise_markers) or ends_dangling(low):
        return True
    asks = low.rstrip().endswith("?") or _contains_phrase(low, pack.question_openers)
    return asks and len(_content_words(low)) >= pack.question_min_content


def closing_unfinished(segments: list, following: list = None) -> bool:
    """
    True when a clip stops inside an exchange that has not landed.

    `ends_incomplete` only ever read the *last* segment, and three real clips
    slipped past it:

      - a host asking a long question across four segments, none of which
        Whisper ended with "?" - the clip stopped with the question still being
        asked and the answer starting one second later;
      - "Terus gue tanya juga. Kan papa udah jual nih." - the question itself
        ("Gue boleh beli ga di pasar?") was on the other side of the cut;
      - "Nah ini gue kasih ilmu yang cuma gue buka di ... Timothy Ronald doang
        ya." - a promise, with the ilmu never delivered.

    So the last CLOSING_WINDOW seconds are read as a unit. If something is
    raised in there - a question, a promise, an announcement of speech - and
    fewer than PAYLOAD_WORDS content words follow it inside the clip, the clip
    ends unfinished.

    `following` is what comes after the cut. It is used for one thing only: a
    question still being asked on the other side of the boundary means the
    question was not resolved on this side either, however many words trailed
    after it.
    """
    if not segments:
        return False

    pack = signals.active()
    end = segments[-1].end
    window = [s for s in segments if s.end >= end - pack.closing_window]
    if not window:
        window = segments[-1:]

    last_opening = None
    for i, seg in enumerate(window):
        if _opening_event(seg.text):
            last_opening = i
    if last_opening is None:
        return False

    payload = sum(len(_content_words(s.text)) for s in window[last_opening + 1:])
    if payload < pack.payload_words:
        return True

    # the question ran past the cut - whatever followed it inside the clip was
    # the asker still asking, not anyone answering
    for seg in (following or []):
        if seg.start >= end + pack.spillover_window:
            break
        if _opening_event(seg.text):
            return True
    return False


def is_clean_ending(text: str) -> bool:
    """A last line a clip can stop on without feeling cut off."""
    t = text.strip()
    if not t.endswith((".", "!")):
        return False
    # Whisper writes a trailing-off sentence as "..." or "…" - the first passed
    # the endswith(".") test and let a clip stop on "Gue pinjam duit, sebagian
    # gue pinjam duit sama..."
    if t.endswith(".."):
        return False
    toks = _tokenize(t)
    if toks and toks[-1] in signals.active().lead_in_endings:
        return False
    if ends_dangling(t):
        return False
    return len(_content_words(t)) >= 2


# --- scoring ---------------------------------------------------------------

class HookScorer:
    """
    Scores candidate clip windows on how likely they are to work as a
    short-form hook. Weights are plain numbers you can see and tune, not a
    black box - that's the differentiator from a pure LLM-prompt approach.
    """

    def __init__(self, weights: dict = None, pack=None):
        self.pack = pack or signals.active()
        self.weights = {**self.pack.weights, **(weights or {})}

    def _term_frequencies(self, segments: Iterable) -> dict:
        freq = {}
        for seg in segments:
            for tok in _tokenize(seg.text):
                if tok in self.pack.stopwords or len(tok) < 3:
                    continue
                freq[tok] = freq.get(tok, 0) + 1
        return freq

    def _distinctiveness(self, window_text: str, global_freq: dict, total_terms: int) -> float:
        """
        How much this window's vocabulary stands out from the rest of the
        transcript. High = the window covers a specific, distinct point
        rather than generic filler ("jadi gini", "menurut saya", ...).
        """
        toks = [t for t in _tokenize(window_text) if t not in self.pack.stopwords and len(t) >= 3]
        if not toks:
            return 0.0
        scores = []
        for tok in set(toks):
            global_count = global_freq.get(tok, 1)
            idf = math.log((total_terms + 1) / global_count)
            scores.append(idf)
        return sum(scores) / len(scores)

    def score_window(self, segments: list, global_freq: dict, total_terms: int,
                     following: list = None) -> ClipCandidate:
        text = " ".join(s.text.strip() for s in segments)
        text_lower = text.lower()
        opening = segments[0].text.strip().lower() if segments else ""

        signals = {}

        opens_with_question = opening.endswith("?") or any(
            f" {w} " in f" {opening} " or opening.startswith(w) for w in self.pack.question_words
        )
        signals["opening_question"] = self.weights["opening_question"] if opens_with_question else 0.0

        curiosity_hits = _count_markers(text_lower, self.pack.curiosity_markers)
        signals["curiosity_marker"] = min(curiosity_hits, 2) * self.weights["curiosity_marker"]

        contrarian_hits = _count_markers(text_lower, self.pack.contrarian_markers)
        signals["contrarian_marker"] = min(contrarian_hits, 2) * self.weights["contrarian_marker"]

        intensity_hits = _count_markers(text_lower, self.pack.intensity_markers)
        signals["intensity_marker"] = min(intensity_hits, 2) * self.weights["intensity_marker"]

        quantity = "|".join(re.escape(w) for w in sorted(self.pack.quantity_words))
        has_number = bool(re.search(r"\d", text)) or (
            bool(quantity) and bool(re.search(rf"\b({quantity})\b", text_lower))
        )
        signals["has_number"] = self.weights["has_number"] if has_number else 0.0

        stakes_hits = _count_markers(text_lower, self.pack.stakes_markers)
        signals["stakes"] = min(stakes_hits, 3) * self.weights["stakes"]

        # a story needs both halves: who it happened to, and that it happened
        told_first_person = any(f" {p} " in f" {text_lower} " for p in self.pack.first_person)
        looks_past = _count_markers(text_lower, self.pack.past_markers)
        signals["personal_story"] = (
            self.weights["personal_story"] * min(looks_past, 2) / 2.0
            if told_first_person else 0.0
        )

        signals["topic_distinctiveness"] = self._distinctiveness(
            text, global_freq, total_terms
        ) * (self.weights["topic_distinctiveness"] / 3.0)

        # --- does the clip finish what it started? -------------------------

        closing = segments[-1].text.strip()
        ends_on_question = closing.endswith("?")
        ends_mid_sentence = not closing.endswith((".", "!", "?", "\u2026"))

        last_words = _tokenize(closing)
        ends_on_lead_in = bool(last_words) and last_words[-1] in LEAD_IN_ENDINGS

        # trailing off on a one-word reaction ("Yes.", "Iya.") is a complete
        # sentence and a bad last frame
        ends_on_reaction = len(_content_words(closing)) < 2

        # "Gue bilang sama Maybank." - the sentence is finished, the thought is
        # not. See ends_dangling.
        ends_on_speech_act = ends_dangling(closing)

        # the last line can look finished while the exchange it belongs to is not
        exchange_open = closing_unfinished(segments, following)

        signals["ends_incomplete"] = (
            self.weights["ends_incomplete"]
            if (ends_on_question or ends_mid_sentence or ends_on_lead_in
                or ends_on_reaction or ends_on_speech_act or exchange_open) else 0.0
        )

        # a window made mostly of two-word reactions carries no information,
        # however many hook words happen to appear in it
        short = sum(1 for seg in segments if len(_content_words(seg.text)) < 3)
        chatter_ratio = short / len(segments)
        signals["chatter"] = (
            self.weights["chatter"] * (chatter_ratio - 0.5) / 0.5
            if chatter_ratio > 0.5 else 0.0
        )

        # question early, substance after it: the shape of a clip that actually
        # tells the viewer something
        answered = 0.0
        for i, seg in enumerate(segments[:max(1, len(segments) // 3)]):
            if seg.text.strip().endswith("?"):
                after = sum(len(_content_words(s.text)) for s in segments[i + 1:])
                if after >= 12:
                    answered = self.weights["answered_question"]
                break
        signals["answered_question"] = answered

        # An opening has to work for someone who has seen nothing before it.
        # The old test - first word not a stopword, two content words - passed
        # "Kawan lama, temen dia juga." and "Nah gue hari Senin gue tanya.",
        # which is how the episode's best moment ended up with an opening that
        # explains nothing. opening_stands_alone is the stricter version, and
        # failing it now costs points rather than merely earning none.
        # Scoring deliberately keeps the *old*, lax test. Making this signal
        # strict and adding a penalty was tried and measured: openings improved
        # on one clip out of fifteen, two got worse, and because the winning
        # window moved, three endings that had been fixed came apart again. The
        # pool's openings are all mediocre, so a penalty only reshuffles equally
        # poor options while disturbing everything downstream of them.
        # opening_stands_alone is applied after selection instead, where it can
        # move a clip's start without changing which moment was chosen.
        first_toks = _tokenize(opening)
        first_tok = first_toks[0] if first_toks else ""
        self_contained = first_tok not in self.pack.stopwords and len(_content_words(opening)) >= 2
        signals["self_contained_opening"] = (
            self.weights["self_contained_opening"] if self_contained else 0.0
        )

        raw_score = sum(signals.values())

        # The displayed score is calibrated later, against the whole episode -
        # see calibrate(). This provisional value only orders windows before
        # that happens, so any monotonic function of the raw total will do.
        provisional = round(max(0.0, 10 * (1 - math.exp(-max(0.0, raw_score) / 6))), 1)

        return ClipCandidate(
            start=segments[0].start,
            end=segments[-1].end,
            segments=segments,
            score=provisional,
            signals=signals,
            raw_score=raw_score,
        )

    def rank_candidates(
        self,
        segments: list,
        min_duration: float = 15.0,
        max_duration: float = 60.0,
        top_n: int = 10,
    ) -> list:
        """
        Slide windows of varying length over the transcript, score each, then
        keep the top non-overlapping candidates (mirrors SOCL's "clip count" /
        "clip duration" controls, but the ranking itself is transparent).
        """
        global_freq = self._term_frequencies(segments)
        total_terms = sum(global_freq.values()) or 1

        candidates = []
        n = len(segments)
        for i in range(n):
            window = []
            for j in range(i, n):
                window.append(segments[j])
                duration = segments[j].end - segments[i].start
                if duration < min_duration:
                    continue
                if duration > max_duration:
                    break
                candidates.append(self.score_window(
                    list(window), global_freq, total_terms, segments[j + 1:j + 8]))

        candidates.sort(key=lambda c: c.score, reverse=True)

        # greedy non-max suppression on overlapping time ranges
        selected = []
        for cand in candidates:
            overlaps = any(
                not (cand.end <= sel.start or cand.start >= sel.end) for sel in selected
            )
            if not overlaps:
                selected.append(cand)
            if len(selected) >= top_n:
                break

        selected.sort(key=lambda c: c.start)
        return selected

    @staticmethod
    def calibrate(candidates: list, all_raw: list) -> list:
        """
        Turn raw signal totals into a 0-10 score that means something.

        The old fixed curve, 10 * (1 - e^(-raw/6)), saturates. Measured on a
        one-hour episode: the median window scored 5.9, the 95th percentile 8.1,
        the 99th 8.7 and the single best 9.4. Every clip anyone would actually
        consider landed inside 0.7 points of every other, which is the opposite
        of what a score is for. On a three-minute sample the same curve pinned
        everything near 7 instead. The number was describing the curve, not the
        content.

        So the scale is fitted to the episode: the median window is 5, the best
        window in the episode is 10, linear between. A 9 now means "clearly
        better than almost anything else here", which is the judgement someone
        picking clips is actually making. raw_score is kept alongside for
        comparing across episodes, where a fitted scale cannot.
        """
        import numpy as np

        if not all_raw:
            return candidates

        arr = np.asarray(all_raw, dtype=float)
        median = float(np.median(arr))
        top = float(arr.max())
        bottom = float(arr.min())

        for c in candidates:
            raw = c.raw_score
            if top > median and raw >= median:
                value = 5.0 + 5.0 * (raw - median) / (top - median)
            elif median > bottom:
                value = 5.0 * (raw - bottom) / (median - bottom)
            else:
                value = 5.0
            c.score = round(min(10.0, max(0.0, value)), 1)
            c.percentile = round(float((arr < raw).mean()) * 100, 1)

        return candidates

    def candidate_pool(
        self,
        segments: list,
        min_duration: float = 15.0,
        max_duration: float = 60.0,
        keep: int = 400,
    ) -> list:
        """
        Every scored window, ranked, before any non-overlap filtering.

        rank_candidates throws this away and returns only the final set, which
        is fine for a one-shot run but leaves nothing to offer when the user
        rejects a clip. Keeping the pool is what makes regeneration possible
        without re-analysing the episode.

        Near-duplicate windows - same region, boundaries a second apart - are
        collapsed, because offering one as the "alternative" to the other is
        not a real choice.
        """
        global_freq = self._term_frequencies(segments)
        total_terms = sum(global_freq.values()) or 1

        scored = []
        n = len(segments)
        for i in range(n):
            window = []
            for j in range(i, n):
                window.append(segments[j])
                duration = segments[j].end - segments[i].start
                if duration < min_duration:
                    continue
                if duration > max_duration:
                    break
                scored.append(self.score_window(
                    list(window), global_freq, total_terms, segments[j + 1:j + 8]))

        scored.sort(key=lambda c: c.raw_score, reverse=True)
        all_raw = [c.raw_score for c in scored]

        deduped = []
        for cand in scored:
            if any(
                abs(cand.start - k.start) < 3.0 and abs(cand.end - k.end) < 3.0
                for k in deduped
            ):
                continue
            deduped.append(cand)
            if len(deduped) >= keep:
                break

        return self.calibrate(deduped, all_raw)
