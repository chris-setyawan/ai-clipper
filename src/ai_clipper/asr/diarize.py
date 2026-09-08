"""
Work out who is speaking, and when.

The framing stage needs this: "the largest face in frame" is a guess, "the face
belonging to the person currently talking" is an answer. It is also what makes a
split-view meaningful, because it decides which panel gets the emphasis.

STATUS: MEASURED AND REJECTED. Do not wire this into the pipeline.

Proper diarization models exist, but every one of them wants to download
weights, so this attempts the job with MFCC features and clustering out of
scipy - no download, every step inspectable.

It does not work. Evaluated against the sample episode (eval_diarization.py,
which checks the labels against independently clustered video identity), it
scored 52% agreement where chance is 50%, with an adjusted Rand index of -0.006:
no relationship at all. Segment-level MFCC statistics on people sharing a room
and similar microphones cluster on recording conditions, not on who is talking.
The labels read plausibly, which is exactly why the measurement was necessary.

Kept because the evaluation is worth having and a better embedding could be
dropped in behind the same interface. The problem it was meant to solve - which
face is the one currently speaking - is better answered by mouth-region motion
in the video, which does not require knowing who anyone is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.fftpack import dct
from scipy.signal import get_window


SAMPLE_RATE = 16000


# --- MFCC, written out rather than imported ---------------------------------

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """
    Triangular filters spaced evenly on the mel scale.

    The mel scale compresses high frequencies the way hearing does, which is
    what makes the resulting features track vocal tract shape - the thing that
    differs between speakers - rather than raw pitch.
    """
    low_mel, high_mel = _hz_to_mel(50), _hz_to_mel(sample_rate / 2)
    points = _mel_to_hz(np.linspace(low_mel, high_mel, n_filters + 2))
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)

    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        for k in range(left, min(centre, fb.shape[1])):
            fb[i - 1, k] = (k - left) / (centre - left)
        for k in range(centre, min(right, fb.shape[1])):
            fb[i - 1, k] = (right - k) / (right - centre)
    return fb


def mfcc(signal: np.ndarray, sample_rate: int = SAMPLE_RATE, n_coeffs: int = 13,
         frame_ms: float = 25.0, hop_ms: float = 10.0, n_filters: int = 26) -> np.ndarray:
    """Frames x coefficients. Returns an empty array for too-short input."""
    frame_len = int(sample_rate * frame_ms / 1000)
    hop = int(sample_rate * hop_ms / 1000)
    if len(signal) < frame_len:
        return np.zeros((0, n_coeffs))

    # pre-emphasis lifts the high frequencies the vocal tract attenuates
    emphasised = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    n_frames = 1 + (len(emphasised) - frame_len) // hop
    indices = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = emphasised[indices] * get_window("hamming", frame_len, fftbins=False)

    n_fft = 512
    power = (np.abs(np.fft.rfft(frames, n_fft)) ** 2) / n_fft
    energy = _mel_filterbank(n_filters, n_fft, sample_rate) @ power.T
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    coeffs = dct(np.log(energy.T), type=2, axis=1, norm="ortho")[:, :n_coeffs]
    return coeffs


# --- speaker embedding and clustering ---------------------------------------

@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    confidence: float


def segment_embedding(audio: np.ndarray, start: float, end: float,
                      sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    One fixed-length vector describing how a stretch of speech sounds.

    Mean and standard deviation of the MFCCs over the segment. Crude next to a
    trained embedding, but it captures timbre well enough to separate two people
    on different microphones, and it needs nothing but the audio.
    """
    a = int(max(0, start) * sample_rate)
    b = int(min(len(audio) / sample_rate, end) * sample_rate)
    if b - a < sample_rate * 0.4:      # under 0.4s is too little to characterise
        return None

    coeffs = mfcc(audio[a:b], sample_rate)
    if coeffs.shape[0] < 3:
        return None

    # the 0th coefficient is loudness, which says more about mic distance than
    # about who is talking
    coeffs = coeffs[:, 1:]
    return np.concatenate([coeffs.mean(axis=0), coeffs.std(axis=0)])


def diarize(audio: np.ndarray, segments: List[dict], n_speakers: Optional[int] = None,
            sample_rate: int = SAMPLE_RATE, max_speakers: int = 4) -> List[SpeakerTurn]:
    """
    Label each transcript segment with a speaker.

    n_speakers=None picks the count by silhouette score between 2 and
    max_speakers. Segments too short to embed inherit the previous speaker,
    which is right far more often than guessing, because conversation comes in
    runs rather than alternating every line.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    embeddings, indices = [], []
    for i, seg in enumerate(segments):
        vec = segment_embedding(audio, float(seg["start"]), float(seg["end"]), sample_rate)
        if vec is not None:
            embeddings.append(vec)
            indices.append(i)

    if len(embeddings) < 2:
        return [
            SpeakerTurn(float(s["start"]), float(s["end"]), "SPEAKER_00", 0.0)
            for s in segments
        ]

    X = np.array(embeddings)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    best_labels, best_score, best_k = None, -1.0, 1
    candidates = [n_speakers] if n_speakers else range(2, min(max_speakers, len(X) - 1) + 1)
    for k in candidates:
        if k >= len(X):
            continue
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_labels, best_score, best_k = labels, score, k

    if best_labels is None:
        return [
            SpeakerTurn(float(s["start"]), float(s["end"]), "SPEAKER_00", 0.0)
            for s in segments
        ]

    # silhouette runs -1..1; report it as a 0..1 confidence
    confidence = max(0.0, min(1.0, (best_score + 1) / 2))

    assigned = {idx: f"SPEAKER_{label:02d}" for idx, label in zip(indices, best_labels)}

    turns, last = [], "SPEAKER_00"
    for i, seg in enumerate(segments):
        speaker = assigned.get(i)
        if speaker is None:
            speaker = last
        last = speaker
        turns.append(SpeakerTurn(
            start=float(seg["start"]),
            end=float(seg["end"]),
            speaker=speaker,
            confidence=confidence if i in assigned else 0.0,
        ))

    return turns


def apply_to_transcript(transcript: dict, turns: List[SpeakerTurn]) -> dict:
    """Write speaker labels back onto segments and words."""
    out = dict(transcript)
    out["segments"] = [
        {**seg, "speaker": turn.speaker}
        for seg, turn in zip(transcript.get("segments", []), turns)
    ]

    def who(t: float) -> Optional[str]:
        for turn in turns:
            if turn.start <= t < turn.end:
                return turn.speaker
        return None

    out["words"] = [
        {**w, "speaker": who((float(w["start"]) + float(w["end"])) / 2)}
        for w in transcript.get("words", [])
    ]
    return out


def speaker_at(turns: List[SpeakerTurn], t: float) -> Optional[str]:
    for turn in turns:
        if turn.start <= t < turn.end:
            return turn.speaker
    return None


def summarize(turns: List[SpeakerTurn]) -> dict:
    """Speaking time per speaker, and how often the floor changes hands."""
    totals = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + (turn.end - turn.start)

    switches = sum(1 for a, b in zip(turns, turns[1:]) if a.speaker != b.speaker)
    return {
        "speakers": len(totals),
        "speaking_time": {k: round(v, 1) for k, v in sorted(totals.items())},
        "turn_changes": switches,
        "confidence": round(max((t.confidence for t in turns), default=0.0), 3),
    }
