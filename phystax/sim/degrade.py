"""Camera degradations. Each level is a named, reproducible preset so the
benchmark can plot the perception tax as a dose-response curve."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Degradation:
    level: int
    noise_sigma: float = 0.0      # gaussian sensor noise, 0..255 scale
    blur_subframes: int = 1       # >1 = motion blur via temporal supersampling
    downscale: float = 1.0        # render small then upscale (resolution loss)
    occluder: bool = False        # a static pillar crossing the frame
    flicker: float = 0.0          # per-frame brightness jitter (fraction)
    drop_prob: float = 0.0        # chance a frame repeats the previous one
    jpeg_quality: int = 100       # compression artefacts


LEVELS: dict[int, Degradation] = {
    0: Degradation(0),
    1: Degradation(1, noise_sigma=6, blur_subframes=3, jpeg_quality=70),
    2: Degradation(2, noise_sigma=12, blur_subframes=5, downscale=0.6, flicker=0.06,
                   jpeg_quality=45),
    3: Degradation(3, noise_sigma=18, blur_subframes=7, downscale=0.45, occluder=True,
                   flicker=0.12, drop_prob=0.05, jpeg_quality=30),
}


def occluder_rect(w: int, h: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    x0 = int(rng.uniform(0.35, 0.6) * w)
    return x0, 0, x0 + int(0.07 * w), h


def apply_frame(img: np.ndarray, d: Degradation, rng: np.random.Generator,
                occ: tuple[int, int, int, int] | None) -> np.ndarray:
    out = img.astype(np.float32)
    if d.flicker:
        out *= 1.0 + rng.uniform(-d.flicker, d.flicker)
    if d.downscale < 1.0:
        h, w = out.shape[:2]
        small = cv2.resize(out, (int(w * d.downscale), int(h * d.downscale)),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if d.noise_sigma:
        out += rng.normal(0, d.noise_sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if occ is not None:
        x0, y0, x1, y1 = occ
        out[y0:y1, x0:x1] = (out[y0:y1, x0:x1] * 0.15 + np.array([70, 60, 55]) * 0.85
                             ).astype(np.uint8)
    if d.jpeg_quality < 100:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, d.jpeg_quality])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out
