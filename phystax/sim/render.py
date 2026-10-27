"""Render a ground-truth trajectory into an MP4 that SAM 3.1 will measure.

The look is domain-randomized on purpose: textured backgrounds, shaded
objects, and same-coloured static distractors. On flat coloured discs a
colour threshold would track perfectly and SAM would be decoration; here a
naive tracker fails, which is what makes SAM load-bearing (and is itself a
benchmark condition).

If `assets/backgrounds/*.png` or `assets/sprites/<noun>.png` exist (CC0
photos, with sprites cut out by SAM 3.1 image segmentation), they are used
instead of the procedural fallbacks. Muse Image is not used: Meta's
Geographic Use Policy bars it in India.
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import FPS, FRAME_H, FRAME_W
from .degrade import LEVELS, Degradation, apply_frame, occluder_rect
from .engine import Trajectory
from .ffmpeg_util import ENC_ARGS, get_ffmpeg

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
BODY_RADIUS = {"planet": 26, "marble": 11, "ball": 13, "block": 20}


@dataclass
class RenderResult:
    path: str
    width: int
    height: int
    n_frames: int
    truth_px: np.ndarray        # (n_frames, n_bodies, 2) image coords, y down
    style_seed: int


def _background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(ASSET_DIR, "backgrounds", "*.png")))
    if files:
        img = cv2.imread(files[int(rng.integers(len(files)))], cv2.IMREAD_COLOR)
        return cv2.resize(img, (w, h))
    # procedural: smooth colour field + random patches
    low = rng.uniform(40, 200, (6, 8, 3)).astype(np.float32)
    img = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(int(rng.integers(6, 14))):
        c = rng.uniform(30, 230, 3)
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 140)), y + int(rng.integers(10, 90))),
                      c.tolist(), -1)
    grain = rng.normal(0, 8, (h, w, 1)).astype(np.float32)
    grey = img.mean(axis=2, keepdims=True)
    img = 0.5 * img + 0.5 * grey                     # desaturate: tabletop, not rave
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + grain
    return np.clip(img, 0, 255).astype(np.uint8)


def _palette(rng: np.random.Generator, n: int) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in rng.uniform(40, 240, 3)) for _ in range(n)]


def _draw_body(img: np.ndarray, name: str, cx: float, cy: float, color, rng_tex: int) -> None:
    sprite_path = os.path.join(ASSET_DIR, "sprites", f"{name}.png")
    r = BODY_RADIUS.get(name, 14)
    if os.path.exists(sprite_path):
        spr = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        spr = cv2.resize(spr, (2 * r, 2 * r))
        x0, y0 = int(cx - r), int(cy - r)
        _alpha_paste(img, spr, x0, y0)
        return
    if name == "block":
        p0 = (int(cx - r), int(cy - r * 0.7))
        p1 = (int(cx + r), int(cy + r * 0.7))
        cv2.rectangle(img, p0, p1, color, -1, cv2.LINE_AA)
        cv2.rectangle(img, p0, p1, tuple(int(c * 0.5) for c in color), 2, cv2.LINE_AA)
        for k in range(3):   # wood-grain stripes
            yy = int(cy - r * 0.5 + k * r * 0.5)
            cv2.line(img, (p0[0] + 3, yy), (p1[0] - 3, yy),
                     tuple(int(c * 0.8) for c in color), 1, cv2.LINE_AA)
        return
    # shaded sphere
    for i in range(r, 0, -1):
        shade = 0.55 + 0.45 * (1 - i / r)
        cv2.circle(img, (int(cx - (r - i) * 0.3), int(cy - (r - i) * 0.3)), i,
                   tuple(int(min(255, c * shade)) for c in color), -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx - r * 0.35), int(cy - r * 0.35)), max(2, r // 4),
               (245, 245, 245), -1, cv2.LINE_AA)


def _alpha_paste(dst: np.ndarray, spr: np.ndarray, x0: int, y0: int) -> None:
    h, w = spr.shape[:2]
    H, W = dst.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0]
    if s.shape[2] == 4:
        a = s[:, :, 3:4] / 255.0
        dst[ya:yb, xa:xb] = (a * s[:, :, :3] + (1 - a) * dst[ya:yb, xa:xb]).astype(np.uint8)
    else:
        dst[ya:yb, xa:xb] = s[:, :, :3]


def world_to_px(xy: np.ndarray, h: int = FRAME_H) -> np.ndarray:
    out = xy.copy()
    out[..., 1] = h - xy[..., 1]
    return out


def render(traj: Trajectory, out_path: str, degradation: int | Degradation = 0,
           style_seed: int = 0, family: str = "", fps: int = FPS) -> RenderResult:
    d = LEVELS[degradation] if isinstance(degradation, int) else degradation
    rng = np.random.default_rng(style_seed)
    w, h = FRAME_W, FRAME_H
    bg = _background(rng, w, h)
    colors = _palette(rng, len(traj.bodies))
    # static distractors that share the moving bodies' colours
    for c in colors:
        for _ in range(2):
            cv2.circle(bg, (int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))),
                       int(rng.integers(8, 18)), c, -1, cv2.LINE_AA)
    if family == "nonlinear_spring":
        cv2.line(bg, (0, h // 2 + 22), (w, h // 2 + 22), (90, 90, 90), 3, cv2.LINE_AA)
    occ = occluder_rect(w, h, rng) if d.occluder else None
    noise_rng = np.random.default_rng(style_seed + 1)

    truth = world_to_px(traj.xy, h)
    n = len(traj.t)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        [get_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *ENC_ARGS, out_path],
        stdin=subprocess.PIPE, stderr=err_file)
    prev = None
    try:
        for f in range(n):
            if prev is not None and d.drop_prob and noise_rng.random() < d.drop_prob:
                proc.stdin.write(prev.tobytes())
                continue
            acc = np.zeros((h, w, 3), np.float32)
            sub = max(1, d.blur_subframes)
            for s in range(sub):
                frac = s / sub
                g = min(n - 1, f + 1)
                pos = truth[f] * (1 - frac) + truth[g] * frac
                img = bg.copy()
                if family == "nonlinear_spring":   # draw the tether
                    cv2.line(img, (w // 2, h // 2), (int(pos[0, 0]), int(pos[0, 1])),
                             (200, 200, 200), 2, cv2.LINE_AA)
                for i, name in enumerate(traj.bodies):
                    _draw_body(img, name, pos[i, 0], pos[i, 1], colors[i], style_seed)
                acc += img
            frame = apply_frame((acc / sub).astype(np.uint8), d, noise_rng, occ)
            proc.stdin.write(frame.tobytes())
            prev = frame
        proc.stdin.close()
    except BrokenPipeError:
        pass                                   # reported below with ffmpeg's own message
    code = proc.wait()
    if code != 0 or not os.path.exists(out_path):
        err_file.seek(0)
        msg = err_file.read().decode("utf-8", "replace").strip()[:400]
        raise RuntimeError(f"ffmpeg failed (exit {code}) writing {out_path}: {msg}")
    err_file.close()
    return RenderResult(out_path, w, h, n, truth, style_seed)


def first_frame(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read {path}")
    return img
