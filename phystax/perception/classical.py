"""Classical baseline: median background subtraction + blob association.

This is the non-foundation-model baseline for SAM. With a static camera and a static
background it can be strong; flicker, noise, occluders and distractors are
where it should break. If SAM does not beat it at some degradation level,
report that: it is a finding, not a failure.
"""
from __future__ import annotations

import cv2
import numpy as np

from .tracks import Track


def track_video(path: str, bodies: list[str], frame_h: int, min_area: int = 40) -> dict[str, Track]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    stack = np.stack(frames).astype(np.float32)
    bg = np.median(stack, axis=0)
    n = len(frames)
    blobs_per_frame = []
    for g in stack:
        diff = cv2.GaussianBlur(np.abs(g - bg), (5, 5), 0)
        _, mask = cv2.threshold(diff.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
        k, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        blobs_per_frame.append([(cents[i][0], cents[i][1], stats[i][4])
                                for i in range(1, k) if stats[i][4] >= min_area])

    # greedy association: each body follows its nearest blob from the last position
    tracks = {b: Track(b, n, np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
                       np.full(n, np.nan), 0) for b in bodies}
    last: dict[str, tuple[float, float] | None] = {b: None for b in bodies}
    for f, blobs in enumerate(blobs_per_frame):
        used = set()
        order = sorted(blobs, key=lambda t: -t[2])
        for b in bodies:
            cands = [(i, bl) for i, bl in enumerate(order) if i not in used]
            if not cands:
                continue
            if last[b] is None:
                i, bl = cands[0]
            else:
                i, bl = min(cands, key=lambda c: (c[1][0] - last[b][0]) ** 2 + (c[1][1] - last[b][1]) ** 2)
            used.add(i)
            tracks[b].x[f], tracks[b].y[f], tracks[b].area[f] = bl[0], frame_h - bl[1], bl[2]
            last[b] = (bl[0], bl[1])
    return tracks
