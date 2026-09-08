"""
Fix the words Whisper reliably gets wrong.

Speech models mangle proper nouns, slang and domain terms because those are
rare in training data - "sutradara" comes back as "sudah dares". No amount of
model size fully fixes this, but the mistakes are consistent per channel, so a
small per-project lexicon does.

Two matching modes:

  exact  - a phrase mapping, applied case-insensitively across word boundaries.
           Use for the errors you have actually seen.
  fuzzy  - a list of correct terms; any transcript word close enough to one of
           them is snapped to it. Use for names that come back slightly
           different every time.

Word timings are preserved. When a correction spans a different number of words
than the original, the replaced span keeps the original span's start and end, so
captions stay in sync.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Lexicon:
    exact: Dict[str, str]        # "sudah dares" -> "sutradara"
    vocabulary: List[str]        # terms to snap near-misses to
    fuzzy_threshold: float = 0.82
    # opening of the sentence handed to Whisper as initial_prompt; override it
    # per project or per language
    prompt_prefix: str = "Percakapan ini menyebut"

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            exact={k.lower(): v for k, v in data.get("exact", {}).items()},
            vocabulary=data.get("vocabulary", []),
            fuzzy_threshold=data.get("fuzzy_threshold", 0.82),
            prompt_prefix=data.get("prompt_prefix", "Percakapan ini menyebut"),
        )

    @classmethod
    def empty(cls) -> "Lexicon":
        return cls(exact={}, vocabulary=[])

    def terms(self) -> List[str]:
        """Vocabulary plus the correct side of every exact fix, deduplicated."""
        seen, out = set(), []
        for t in list(self.vocabulary) + list(self.exact.values()):
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    def as_whisper_prompt(self) -> str:
        """
        Terms to feed Whisper as initial_prompt, written as a proper sentence.

        The sentence matters. Whisper copies the *style* of its prompt, not just
        the vocabulary: given a bare lowercase comma list, it returns a
        transcript with no capital letters and no full stops. Wrapping the same
        terms in one correctly punctuated sentence keeps the vocabulary benefit
        without flattening the output, which the burned-in captions would
        otherwise inherit.
        """
        terms = self.terms()
        if not terms:
            return ""
        return f"{self.prompt_prefix} {', '.join(terms)}."


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def correct_text(text: str, lex: Lexicon) -> str:
    """Apply exact phrase mappings, longest phrase first so overlaps resolve."""
    out = text
    for wrong in sorted(lex.exact, key=len, reverse=True):
        right = lex.exact[wrong]
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
        out = pattern.sub(lambda m: _match_case(m.group(0), right), out)
    return out


def snap_word(word: str, lex: Lexicon) -> str:
    """Pull a single word to a vocabulary term if it is close enough."""
    if not lex.vocabulary:
        return word

    bare = word.strip(".,!?;:\"'")
    if not bare or len(bare) < 4:
        return word

    match = difflib.get_close_matches(
        bare.lower(), [v.lower() for v in lex.vocabulary], n=1, cutoff=lex.fuzzy_threshold
    )
    if not match:
        return word

    for v in lex.vocabulary:
        if v.lower() == match[0]:
            return word.replace(bare, _match_case(bare, v))
    return word


def apply(transcript: dict, lex: Optional[Lexicon]) -> dict:
    """
    Correct a transcript dict in the shape transcribe_local.py writes.
    Returns a new dict; the input is left alone.
    """
    if lex is None or (not lex.exact and not lex.vocabulary):
        return transcript

    out = dict(transcript)

    out["segments"] = [
        {**seg, "text": correct_text(seg["text"], lex)}
        for seg in transcript.get("segments", [])
    ]

    words = correct_word_sequence(transcript.get("words", []), lex)
    out["words"] = [{**w, "text": snap_word(w["text"], lex)} for w in words]

    return out


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s-]", "", s).strip().lower()


def correct_word_sequence(words: List[dict], lex: Lexicon) -> List[dict]:
    """
    Apply phrase corrections across the word list, not just within single words.

    This matters because the captions are built from the word list, and the
    common Whisper errors are phrase-level: "sutradara" comes back as two words,
    "sudah dares". Correcting only the segment text would fix the transcript
    while leaving the burned-in captions wrong.

    When a replacement has a different word count than the original, the new
    words share the original span's start and end, split in proportion to their
    length. Timings stay monotonic and the caption stays in sync.
    """
    if not words or not lex.exact:
        return list(words)

    result: List[dict] = []
    i = 0
    keys = sorted(lex.exact, key=lambda k: len(k.split()), reverse=True)

    while i < len(words):
        matched = False

        for key in keys:
            n = len(key.split())
            if n == 0 or i + n > len(words):
                continue

            span = words[i:i + n]
            joined = _normalize(" ".join(w["text"] for w in span))
            if joined != _normalize(key):
                continue

            replacement = lex.exact[key].split()
            # carry the original's capitalisation onto the replacement, so a
            # correction at the start of a sentence does not arrive lowercase
            if replacement and span[0]["text"][:1].isupper():
                replacement[0] = _match_case(span[0]["text"], replacement[0])
            start = float(span[0]["start"])
            end = float(span[-1]["end"])
            total = sum(len(r) for r in replacement) or 1

            # keep any trailing punctuation the original span ended with
            trailing = re.search(r"[.,!?;:]+$", span[-1]["text"])
            cursor = start
            for j, token in enumerate(replacement):
                share = (end - start) * (len(token) / total)
                w_end = end if j == len(replacement) - 1 else cursor + share
                text = token
                if trailing and j == len(replacement) - 1:
                    text += trailing.group(0)
                result.append({"start": round(cursor, 3), "end": round(w_end, 3), "text": text})
                cursor = w_end

            i += n
            matched = True
            break

        if not matched:
            w = words[i]
            result.append({**w, "text": correct_text(w["text"], lex)})
            i += 1

    return result


def diff_summary(before: dict, after: dict) -> List[tuple]:
    """(original, corrected) pairs for every word the lexicon changed."""
    changes = []
    for a, b in zip(before.get("words", []), after.get("words", [])):
        if a["text"] != b["text"]:
            changes.append((a["text"], b["text"]))
    return changes
