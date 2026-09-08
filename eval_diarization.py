"""
Does the audio-only diarizer actually work? Measure it, don't assume.

The method: cluster transcript segments by voice, cluster the video's shots by
what they look like, and check whether the two agree. On multi-camera footage
the editor cuts to whoever is talking, so visual identity is a usable stand-in
for ground truth - imperfect, but independent of the audio, which is what makes
it worth measuring against.

Two clusters means chance agreement is 50%. Anything near that is a failure, no
matter how plausible the labels look when you read them.

Run: python eval_diarization.py <video> <transcript.json>
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ai_clipper.asr.diarize import mfcc, segment_embedding
from ai_clipper.video.framing import build_crop_path


def shot_signatures(video, crop_path):
    """A colour histogram of the framed subject at the middle of every shot."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sigs, shot_ids = [], []

    for i, shot in enumerate(crop_path.shots):
        t = (shot.start + shot.end) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        cx = crop_path.center_at(t)
        roi = frame[int(h * 0.25):int(h * 0.95),
                    int(max(0, cx - 0.22) * w):int(min(1, cx + 0.22) * w)]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        sigs.append(hist.flatten())
        shot_ids.append(i)

    cap.release()
    return np.array(sigs), shot_ids


def main():
    if len(sys.argv) < 3:
        print("usage: python eval_diarization.py <video> <transcript.json>")
        sys.exit(1)

    video, transcript_path = sys.argv[1], sys.argv[2]
    from faster_whisper.audio import decode_audio

    audio = decode_audio(video, sampling_rate=16000)
    segments = json.load(open(transcript_path, encoding="utf-8"))["segments"]

    print("Analysing shots ...")
    crop_path = build_crop_path(video)

    print("Clustering shots by appearance ...")
    sigs, shot_ids = shot_signatures(video, crop_path)
    visual = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(sigs)
    print(f"  {len(sigs)} shots, silhouette {silhouette_score(sigs, visual):.3f}")
    shot_label = {s: int(l) for s, l in zip(shot_ids, visual)}

    def shot_of(t):
        for i, shot in enumerate(crop_path.shots):
            if shot.start <= t < shot.end:
                return i
        return len(crop_path.shots) - 1

    print("Embedding voices ...")
    X, truth = [], []
    for seg in segments:
        vec = segment_embedding(audio, float(seg["start"]), float(seg["end"]))
        if vec is None:
            continue
        label = shot_label.get(shot_of((float(seg["start"]) + float(seg["end"])) / 2))
        if label is None:
            continue
        X.append(vec)
        truth.append(label)

    X = np.array(X)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    truth = np.array(truth)

    audio_labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)
    # cluster ids are arbitrary, so take the better of the two label matchings
    agreement = max((audio_labels == truth).mean(), (audio_labels != truth).mean())

    print()
    print(f"  segments compared     {len(X)}")
    print(f"  audio silhouette      {silhouette_score(X, audio_labels):.3f}")
    print(f"  agreement with video  {agreement * 100:.1f}%   (50% = chance)")
    print(f"  adjusted Rand index   {adjusted_rand_score(truth, audio_labels):+.3f}   (0 = no relationship)")
    print()
    if agreement < 0.65:
        print("  VERDICT: audio-only diarization is not working on this recording.")
        print("  Do not use it to drive framing decisions.")
    else:
        print("  VERDICT: labels track visual identity well enough to be useful.")


if __name__ == "__main__":
    main()
