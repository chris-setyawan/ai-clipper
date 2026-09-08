"""Load transcripts into TranscriptSegment lists from a few common formats."""

import json

from .hook_scorer import TranscriptSegment


def load_json_transcript(path: str) -> list:
    """
    Load a transcript from a simple JSON format:

        [{"start": 0.0, "end": 4.2, "text": "...", "speaker": "A"}, ...]

    This is also the shape faster-whisper / openai-whisper segments already
    come in (minus "speaker"), so this doubles as the loader once ASR is
    wired up in a later milestone.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        TranscriptSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"],
            speaker=seg.get("speaker"),
        )
        for seg in raw
    ]
