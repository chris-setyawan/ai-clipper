"""
Putting sentence ends back when the transcript has none.

Every completeness rule in this project reads punctuation: does the last line
end on a full stop, is it a question, does the clip stop mid-sentence. That was
a hidden dependency, and testing a second episode found it hard.

Measured across two one-hour Indonesian podcasts:

| | segments ending in punctuation | ends_incomplete fires on |
|---|---|---|
| episode A | 99% | 6% of candidates |
| episode B | 5% | 100% of candidates |

A penalty that fires on every candidate ranks nothing. On episode B the whole
completeness apparatus was inert - not wrong, just silent.

The cause was not the podcast. `transcribe_local.py` only passes an
`initial_prompt` when a lexicon is supplied, and Whisper imitates the style of
its prompt: given a punctuated sentence it punctuates, given nothing it may not.
The fix at the source is to always prompt with a punctuated sentence, and that
is now the default. But a user can bring a transcript from anywhere, so the
pipeline cannot depend on having been the one to make it.

This module is the safety net. Where the transcript has no sentence ends, they
are inferred from silence: a gap between one segment and the next is where a
speaker stopped. On episode B a 0.35 second gap marks 17% of segments, about one
sentence per twenty words, which is what unhurried speech looks like.

It is an approximation and it is meant to be. A speaker who pauses mid-thought
gets a full stop they did not earn. It is still much better than every clip
looking equally unfinished.
"""

from __future__ import annotations

from typing import List, Sequence

# below this share of punctuated segments, the transcript is treated as raw
PUNCTUATED_ENOUGH = 0.2

# a gap this long between segments reads as the end of a sentence
SENTENCE_PAUSE = 0.35

_ENDINGS = (".", "!", "?", "…", ",")


def punctuated_fraction(segments: Sequence) -> float:
    """Share of segments that end in any punctuation at all."""
    if not segments:
        return 1.0
    ends = sum(1 for s in segments if s.text.strip().endswith(_ENDINGS))
    return ends / len(segments)


def needs_repunctuation(segments: Sequence) -> bool:
    return punctuated_fraction(segments) < PUNCTUATED_ENOUGH


def restore_sentence_ends(segments: Sequence, min_pause: float = SENTENCE_PAUSE) -> List:
    """
    Return the segments with a full stop added wherever the speaker stopped.

    The last segment always gets one: a transcript should not end mid-sentence
    just because nothing follows it.
    """
    out = list(segments)
    for i, seg in enumerate(out):
        text = seg.text.strip()
        if not text or text.endswith(_ENDINGS):
            continue
        last = i == len(out) - 1
        gap = (out[i + 1].start - seg.end) if not last else min_pause
        if last or gap >= min_pause:
            seg.text = text + "."
    return out
