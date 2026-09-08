"""
Check that OpenCV is really decoding this video, frame by frame.

Written because the same video produced 43 shots on one machine and 4 on
another, with identical code. Face detection agreed (98% both), so the frames
were arriving - the question is whether they were arriving as *different*
frames. A decoder that hands back a stale buffer looks like a video that never
cuts.

Run: python diagnose_decode.py "podcast sample.mov"
"""

import sys

import cv2
import numpy as np


def main():
    if len(sys.argv) < 2:
        print('usage: python diagnose_decode.py "video.mov"')
        sys.exit(1)

    path = sys.argv[1]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"could not open {path}")
        sys.exit(1)

    print(f"OpenCV      {cv2.__version__}")
    print(f"backend     {cap.getBackendName()}")
    print(f"reported    {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.2f} fps, "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    read = 0
    identical = 0
    distances = []
    previous = None
    prev_hist = None

    while read < 900:                      # first 30 seconds at 30fps
        ok, frame = cap.read()
        if not ok:
            break
        read += 1

        small = cv2.resize(frame, (160, 90))
        if previous is not None and np.array_equal(small, previous):
            identical += 1
        previous = small

        hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        if prev_hist is not None:
            distances.append(1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
        prev_hist = hist

    cap.release()

    print(f"decoded     {read} frames")
    print(f"duplicates  {identical} ({identical / max(1, read) * 100:.0f}% identical to the one before)")

    if distances:
        d = np.array(distances)
        print(f"histogram distance: median {np.median(d):.4f}, "
              f"p95 {np.percentile(d, 95):.4f}, max {d.max():.4f}")
        print(f"frames over the 0.45 cut threshold: {(d > 0.45).sum()}")

    print()
    if identical > read * 0.2:
        print("PROBLEM: many frames are byte-identical to the previous one, so the")
        print("decoder is handing back stale buffers. Shot detection cannot work.")
    elif distances and (np.array(distances) > 0.45).sum() < 3:
        print("PROBLEM: almost no frame-to-frame change is large enough to look like")
        print("a cut, even though the footage has them. Decoding is likely degraded.")
    else:
        print("Decoding looks healthy.")


if __name__ == "__main__":
    main()
