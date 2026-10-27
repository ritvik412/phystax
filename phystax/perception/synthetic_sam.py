"""DEVELOPMENT STAND-IN ONLY. Never report numbers produced with this backend.

Produces SAM-shaped tracks (centroid + per-frame confidence + dropouts +
occasional glitches) from simulator truth so the agent loop, the UI and the
benchmark plumbing can be built before Meta Model API credentials arrive.
Enable with PHYSTAX_SAM_BACKEND=synthetic. Every result written while it is
on is tagged sam_backend="synthetic" and the report refuses to mix them in.

Its error model is a guess (error grows with degradation; confidence is
noisily anti-correlated with error). Whether real SAM 3.1 behaves like this
is exactly what bench/measure_perception.py measures on Day 2.
"""
from __future__ import annotations

import numpy as np

from .tracks import Track

LEVEL_SIGMA = {0: 0.8, 1: 1.8, 2: 3.5, 3: 6.0}       # centroid jitter, px
LEVEL_DROPOUT = {0: 0.0, 1: 0.01, 2: 0.04, 3: 0.10}
LEVEL_GLITCH = {0: 0.0, 1: 0.005, 2: 0.015, 3: 0.04}  # frames snapped onto a distractor


def synthetic_track(body: str, truth_world: np.ndarray, level: int, seed: int) -> Track:
    rng = np.random.default_rng(seed)
    n = len(truth_world)
    sigma = LEVEL_SIGMA.get(level, 6.0)
    err_scale = sigma * rng.lognormal(0.0, 0.5, n)             # heteroscedastic noise
    xy = truth_world + rng.normal(0, 1, (n, 2)) * err_scale[:, None]
    glitch = rng.random(n) < LEVEL_GLITCH.get(level, 0.04)
    xy[glitch] += rng.uniform(-120, 120, (glitch.sum(), 2))
    drop = rng.random(n) < LEVEL_DROPOUT.get(level, 0.1)
    err = np.linalg.norm(xy - truth_world, axis=1)
    conf = 1.0 / (1.0 + np.exp((err - 2.0 * sigma) / sigma)) # high when error is small
    conf = np.clip(conf * 0.6 + 0.35 + rng.normal(0, 0.05, n), 0.0, 1.0)
    conf[glitch] = np.clip(conf[glitch] - 0.3, 0.0, 1.0)
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    x[drop] = y[drop] = np.nan
    c = conf.copy()
    c[drop] = np.nan
    return Track(body, n, x, y, np.full(n, 300.0), c, 1)
