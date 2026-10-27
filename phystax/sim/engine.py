"""Deterministic RK4 integrator. Produces the ground-truth trajectory that
both the renderer (camera conditions) and the oracle condition read from."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FPS
from .laws import LawFamily, State


@dataclass
class Trajectory:
    t: np.ndarray            # (n_frames,)
    xy: np.ndarray           # (n_frames, n_bodies, 2) world px, y up
    v: np.ndarray            # (n_frames, n_bodies, 2)
    bodies: list[str]

    def to_jsonable(self) -> dict:
        return {
            "t": self.t.round(4).tolist(),
            "bodies": {
                name: {"x": self.xy[:, i, 0].round(2).tolist(),
                       "y": self.xy[:, i, 1].round(2).tolist()}
                for i, name in enumerate(self.bodies)
            },
        }


def simulate(family: LawFamily, hidden: dict, experiment: dict,
             n_frames: int, fps: int = FPS, substeps: int = 8) -> Trajectory:
    exp = family.validate_experiment(experiment)
    s: State = family.initial_state(exp).astype(float)
    n = s.shape[0]
    dt = 1.0 / (fps * substeps)
    fixed = np.array([i in family.fixed_bodies for i in range(n)])

    def deriv(state: State) -> State:
        a = family.accel(state, hidden)
        d = np.zeros_like(state)
        d[:, :2] = state[:, 2:]
        d[:, 2:] = a
        d[fixed] = 0.0
        return d

    ts, xs, vs = [], [], []
    with np.errstate(over="ignore", invalid="ignore"):   # bad trial params during fitting
        return _integrate(family, deriv, s, n_frames, fps, substeps, dt, ts, xs, vs)


def _integrate(family, deriv, s, n_frames, fps, substeps, dt, ts, xs, vs) -> Trajectory:
    for f in range(n_frames):
        ts.append(f / fps)
        xs.append(s[:, :2].copy())
        vs.append(s[:, 2:].copy())
        for _ in range(substeps):
            k1 = deriv(s)
            k2 = deriv(s + 0.5 * dt * k1)
            k3 = deriv(s + 0.5 * dt * k2)
            k4 = deriv(s + dt * k3)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if family.constrain is not None:
                s = family.constrain(s)
        if not np.all(np.isfinite(s)):
            break
    return Trajectory(np.array(ts), np.array(xs), np.array(vs), list(family.bodies))
