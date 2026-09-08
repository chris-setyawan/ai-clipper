"""
The writing that ships with each clip.

Thirty clips means thirty titles, thirty descriptions and thirty sets of
hashtags. That is the part of clipping that actually takes an afternoon, and no
competitor hands it over with the video. This does, using only what the pipeline
already computed - no API key, no model.

Everything here is a draft for a person to edit, and it says so. Suggesting a
title is useful; pretending it is finished copy is not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

from ..scoring import signals
from ..scoring.hook_scorer import _tokenize


@dataclass
class CaptionPack:
    clip_index: int
    hook: str
    title: str
    description: str
    hashtags: List[str]
    start: float
    end: float
    score: float
    why: str

    def to_text(self) -> str:
        return (
            f"# Clip {self.clip_index:02d}  ({self.start:.1f}s - {self.end:.1f}s, "
            f"hot {self.score}/10)\n\n"
            f"HOOK\n{self.hook}\n\n"
            f"TITLE\n{self.title}\n\n"
            f"DESCRIPTION\n{self.description}\n\n"
            f"HASHTAGS\n{' '.join(self.hashtags)}\n\n"
            f"WHY THIS CLIP\n{self.why}\n\n"
            f"---\nDrafts, not finished copy. Read them before posting.\n"
        )


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _content_words(text: str) -> List[str]:
    pack = signals.active()
    return [
        t for t in _tokenize(text)
        if t not in pack.stopwords and t not in pack.filler and len(t) >= 4
    ]


def extract_hook(text: str, max_words: int = 14, min_content: int = 3) -> str:
    """
    The line a viewer hears in the first three seconds.

    A question is the strongest opening short-form has, but only a real one.
    Conversational backchannel - "Betul ya?", "Gitu?", "Ngerasa ga?" - is
    grammatically a question and carries nothing, and an earlier version of this
    happily titled a clip "Betul ya". So a question has to clear a minimum of
    content words before it wins; otherwise the first sentence that carries
    actual subject matter does.
    """
    sentences = _sentences(text)
    if not sentences:
        return ""

    for sentence in sentences[:3]:
        if sentence.rstrip().endswith("?") and len(_content_words(sentence)) >= min_content:
            return _trim(sentence, max_words)

    for sentence in sentences[:4]:
        if len(_content_words(sentence)) >= min_content:
            return _trim(sentence, max_words)

    return _trim(sentences[0], max_words)


def _trim(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def keywords(text: str, limit: int = 6) -> List[str]:
    """
    The topic words of a clip, most frequent first.

    Frequency within the clip, filtered by stopwords and conversational filler.
    Crude, but the alternative is inventing topics the clip does not contain,
    which is worse than a plain word list.
    """
    pack = signals.active()
    counts = {}
    for token in _tokenize(text):
        if token in pack.stopwords or token in pack.filler or len(token) < 4:
            continue
        counts[token] = counts.get(token, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:limit]]


def build_hashtags(text: str, base: Optional[List[str]] = None, limit: int = 8) -> List[str]:
    """
    Channel-wide tags first, then whatever this clip is actually about.

    Generic tags do the reach and topical tags do the targeting, so both belong;
    the generic ones go first because that is the order most creators use.
    """
    base = base or ["#podcast", "#shorts", "#fyp"]
    tags = list(base)

    for word in keywords(text, limit=limit):
        tag = "#" + re.sub(r"[^a-z0-9]", "", word.lower())
        if len(tag) > 3 and tag not in tags:
            tags.append(tag)

    return tags[:limit]


def build_pack(clip, index: int, base_hashtags: Optional[List[str]] = None) -> CaptionPack:
    text = clip.text
    hook = extract_hook(text)

    body = " ".join(_sentences(text)[:3])
    description = _trim(body, 40)

    return CaptionPack(
        clip_index=index,
        hook=hook,
        title=_trim(hook.rstrip("?.!"), 10),
        description=description,
        hashtags=build_hashtags(text, base_hashtags),
        start=round(clip.start, 2),
        end=round(clip.end, 2),
        score=clip.score,
        why=clip.explain(),
    )


def write_packs(clips, out_dir: str, base_hashtags: Optional[List[str]] = None,
                numbers: Optional[List[int]] = None) -> List[str]:
    """
    One text file per clip, plus a combined JSON for anything programmatic.

    `numbers` keeps a clip's copy filed under the same number as its video after
    a rejection has reshuffled the set. Left out, clips are numbered by position.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    packs, paths = [], []

    for i, clip in (zip(numbers, clips) if numbers else enumerate(clips, start=1)):
        pack = build_pack(clip, i, base_hashtags)
        packs.append(pack)
        path = os.path.join(out_dir, f"clip_{i:02d}_caption.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(pack.to_text())
        paths.append(path)

    combined = os.path.join(out_dir, "caption_packs.json")
    with open(combined, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in packs], f, ensure_ascii=False, indent=2)
    paths.append(combined)

    return paths
