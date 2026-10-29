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


LAB_COLORS = {"ball": (40, 50, 200), "marble": (160, 120, 40), "planet": (60, 90, 120),
              "block": (40, 110, 170)}   # BGR: red rubber ball, steel-blue marble, slate planet, wood block


def render_style() -> str:
    """'lab' (default): a believable bench scene. 'abstract': the original
    test-pattern look, kept as a controlled appearance condition."""
    return os.environ.get("PHYSTAX_RENDER_STYLE", "lab")


def _background_lab(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(ASSET_DIR, "backgrounds", "*.png")))
    if files:
        return cv2.resize(cv2.imread(files[int(rng.integers(len(files)))], cv2.IMREAD_COLOR), (w, h))
    tone = rng.uniform(205, 228)
    wall = np.full((h, w, 3), tone, np.float32) * np.array([0.98, 0.99, 1.0], np.float32)
    wall += cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 3)[..., None]
    yy, xx = np.mgrid[0:h, 0:w]
    light = 1.0 - 0.18 * (((xx - w * 0.4) / w) ** 2 + ((yy - h * 0.3) / h) ** 2)   # soft lamp falloff
    wall *= light[..., None]
    bx0, by0, bx1, by1 = int(w * 0.08), int(h * 0.06), int(w * 0.92), int(h * 0.94)
    wall[by0:by1, bx0:bx1] = np.array([236, 240, 238], np.float32) * light[by0:by1, bx0:bx1, None]
    for x in range(bx0, bx1, 16):   # graph paper: fine grid every 16 px, bold every 80 px
        cv2.line(wall, (x, by0), (x, by1), (205, 215, 210) if (x - bx0) % 80 else (165, 180, 172), 1)
    for y in range(by0, by1, 16):
        cv2.line(wall, (bx0, y), (bx1, y), (205, 215, 210) if (y - by0) % 80 else (165, 180, 172), 1)
    cv2.rectangle(wall, (bx0, by0), (bx1, by1), (120, 130, 125), 2)
    cv2.rectangle(wall, (bx0 - 34, by0), (bx0 - 6, by1), (120, 200, 230), -1)      # ruler
    for i, y in enumerate(range(by0, by1, 16)):
        cv2.line(wall, (bx0 - 6, y), (bx0 - (18 if i % 5 == 0 else 12), y), (40, 40, 40), 1)
    wall += rng.normal(0, 2.0, wall.shape).astype(np.float32)                      # sensor grain
    return np.clip(wall, 0, 255).astype(np.uint8)


def _draw_body_lab(img: np.ndarray, name: str, cx: float, cy: float) -> None:
    r = BODY_RADIUS.get(name, 14)
    color = np.array(LAB_COLORS.get(name, (60, 60, 160)), float)
    ov = img.copy()   # soft contact shadow, offset away from the lamp
    cv2.ellipse(ov, (int(cx + r * 0.35), int(cy + r * 0.45)), (int(r * 1.05), int(r * 0.9)), 0, 0, 360,
                (90, 95, 92), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
    if name == "block":
        p0, p1 = (int(cx - r), int(cy - r * 0.7)), (int(cx + r), int(cy + r * 0.7))
        cv2.rectangle(img, p0, p1, tuple(int(c) for c in color), -1, cv2.LINE_AA)
        for k in range(4):
            yy = int(cy - r * 0.55 + k * r * 0.37)
            cv2.line(img, (p0[0] + 2, yy), (p1[0] - 2, yy), tuple(int(c * 0.75) for c in color), 1, cv2.LINE_AA)
        cv2.rectangle(img, p0, p1, tuple(int(c * 0.5) for c in color), 1, cv2.LINE_AA)
        return
    for i in range(r, 0, -1):      # shading lit from the upper left
        f = i / r
        shade = 0.45 + 0.55 * (1 - f) ** 0.6
        cv2.circle(img, (int(cx - (r - i) * 0.35), int(cy - (r - i) * 0.35)), i,
                   tuple(int(min(255, c * shade + 25 * (1 - f))) for c in color), -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx - r * 0.38), int(cy - r * 0.38)), max(1, r // 5), (250, 250, 250), -1, cv2.LINE_AA)


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
    style = render_style()
    if style == "lab":
        bg = _background_lab(rng, w, h)
        colors = [LAB_COLORS.get(b, (60, 60, 160)) for b in traj.bodies]
        for b in traj.bodies[-1:]:   # two look-alike decoys resting on the board's lower edge
            for _ in range(2):
                _draw_body_lab(bg, b, float(rng.integers(80, w - 80)), float(h * 0.94 - BODY_RADIUS.get(b, 14) - 2))
    else:
        bg = _background(rng, w, h)
        colors = _palette(rng, len(traj.bodies))
        for c in colors:   # static distractors that share the moving bodies' colours
            for _ in range(2):
                cv2.circle(bg, (int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))),
                           int(rng.integers(8, 18)), c, -1, cv2.LINE_AA)
    if family == "nonlinear_spring":
        cv2.line(bg, (0, h // 2 + 22), (w, h // 2 + 22), (90, 90, 90), 3, cv2.LINE_AA)
        if style == "lab":
            cv2.rectangle(bg, (w // 2 - 6, h // 2 - 30), (w // 2 + 6, h // 2 + 22), (80, 80, 80), -1)
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
                    if style == "lab":
                        _draw_body_lab(img, name, pos[i, 0], pos[i, 1])
                    else:
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
