"""
Speech-to-text with word-level timestamps.

Uses faster-whisper (CTranslate2 reimplementation of Whisper) because it runs
comfortably on CPU, which matters for the "cheap to run" angle of this project.
Word-level timestamps are requested on purpose: the subtitle burn-in step later
needs them for karaoke-style highlighting, and the clip cutter needs them so a
clip never starts halfway through a word.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from typing import Optional

from ..scoring.hook_scorer import TranscriptSegment


@dataclass
class Word:
    start: float
    end: float
    text: str


def extract_audio(video_path: str, out_path: Optional[str] = None) -> str:
    """
    Pull a 16 kHz mono WAV out of any video/audio file with ffmpeg.
    Whisper resamples to 16 kHz mono internally anyway, so doing it once here
    keeps the file small and avoids re-decoding the video on every run.
    """
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "id",
    ):
        from faster_whisper import WhisperModel

        self.language = language
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, media_path: str, vad_filter: bool = True):
        """
        Returns (segments, words).

        segments: list[TranscriptSegment] - feeds straight into HookScorer
        words:    list[Word]              - feeds the subtitle renderer later

        vad_filter drops long silences before they reach the model. On podcast
        audio this cuts runtime noticeably and prevents Whisper's habit of
        hallucinating text over dead air.
        """
        needs_cleanup = False
        if not media_path.lower().endswith(".wav"):
            media_path = extract_audio(media_path)
            needs_cleanup = True

        try:
            raw_segments, info = self.model.transcribe(
                media_path,
                language=self.language,
                word_timestamps=True,
                vad_filter=vad_filter,
                beam_size=5,
            )

            segments = []
            words = []
            for seg in raw_segments:
                segments.append(
                    TranscriptSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=seg.text.strip(),
                    )
                )
                for w in (seg.words or []):
                    words.append(Word(start=float(w.start), end=float(w.end), text=w.word.strip()))

            self.detected_language = info.language
            self.duration = info.duration
            return segments, words
        finally:
            if needs_cleanup and os.path.exists(media_path):
                os.unlink(media_path)


def segments_to_json(segments, words=None) -> dict:
    """Serialize a transcription so it can be cached and reused between runs."""
    payload = {
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in segments
        ]
    }
    if words is not None:
        payload["words"] = [asdict(w) for w in words]
    return payload
