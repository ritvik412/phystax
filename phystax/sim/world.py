"""The World: the only object that knows the hidden law.

The agent process never imports this module's internals. It interacts only
through `run_experiment` (returns a video plus, in the oracle condition,
exact coordinates) and `run_test` (rendered only after the agent's
prediction is committed to the ledger). Run it in a separate process or
behind an HTTP boundary in the live demo so the secret is not even in the
agent's address space.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..config import FPS, MAX_EXPERIMENT_FRAMES, RUNS_DIR
from .engine import Trajectory, simulate
from .laws import FAMILIES, LawFamily, public_spec
from .render import RenderResult, render


@dataclass
class Experiment:
    exp_id: str
    params: dict
    n_frames: int
    video_path: str
    traj: Trajectory
    render: RenderResult
    render_level: int = 0


@dataclass
class World:
    family_name: str
    seed: int
    degradation: int = 0
    run_dir: str = ""
    hidden: dict = field(default_factory=dict)
    experiments: list[Experiment] = field(default_factory=list)
    test_params: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.family: LawFamily = FAMILIES[self.family_name]
        rng = np.random.default_rng(self.seed)
        self.hidden = self.family.sample_hidden(rng)
        self.test_params = self._sample_tests(rng, 2)
        self.run_dir = self.run_dir or os.path.join(
            RUNS_DIR, f"{self.family_name}_s{self.seed}_d{self.degradation}")
        os.makedirs(self.run_dir, exist_ok=True)

    # ---- public surface ---------------------------------------------------
    def spec(self) -> dict:
        return public_spec(self.family)

    def secret_commitment(self) -> str:
        """Hash of the hidden law, published at episode start so anyone can
        later verify the world did not change the law after the prediction."""
        blob = json.dumps({"family": self.family_name, "seed": self.seed,
                           "hidden": self.hidden}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def run_experiment(self, params: dict, duration_s: float) -> Experiment:
        n = int(min(MAX_EXPERIMENT_FRAMES, max(2, round(duration_s * FPS))))
        exp_id = f"e{len(self.experiments):02d}"
        traj = simulate(self.family, self.hidden, params, n)
        path = os.path.join(self.run_dir, f"{exp_id}.mp4")
        rr = render(traj, path, self.degradation, style_seed=self.seed * 1000 + len(self.experiments),
                    family=self.family_name)
        exp = Experiment(exp_id, self.family.validate_experiment(params), n, path, traj, rr,
                         self.degradation)
        self.experiments.append(exp)
        return exp

    def test_conditions(self) -> list[dict]:
        """Held-out initial conditions chosen by the world, not the agent."""
        return [dict(p) for p in self.test_params]

    def run_test(self, index: int, duration_s: float) -> Experiment:
        n = int(min(MAX_EXPERIMENT_FRAMES, round(duration_s * FPS)))
        traj = simulate(self.family, self.hidden, self.test_params[index], n)
        path = os.path.join(self.run_dir, f"test{index}.mp4")
        rr = render(traj, path, self.degradation, style_seed=self.seed * 1000 + 900 + index,
                    family=self.family_name)
        return Experiment(f"test{index}", self.test_params[index], n, path, traj, rr,
                          self.degradation)

    def reveal(self) -> dict:
        return {"family": self.family_name, "seed": self.seed, "hidden": self.hidden}

    # ---- internals ----------------------------------------------------------
    def _sample_tests(self, rng: np.random.Generator, n: int) -> list[dict]:
        """Held-out tests must be well-posed: the moving body stays in frame and
        does not hit a contact constraint, so a correct law predicts it exactly."""
        from ..config import FRAME_H, FRAME_W
        out, tries = [], 0
        while len(out) < n and tries < 60:
            tries += 1
            cand = self._random_experiment(rng)
            tr = simulate(self.family, self.hidden, cand, int(4.0 * FPS))
            moving = [i for i in range(len(self.family.bodies)) if i not in self.family.fixed_bodies]
            xy = tr.xy[:, moving, :]
            inside = True
            if self.family_name != "power_drag":     # a projectile may leave the frame; truth still defined
                inside = (xy[..., 0] > 15).all() and (xy[..., 0] < FRAME_W - 15).all() and \
                         (xy[..., 1] > 15).all() and (xy[..., 1] < FRAME_H - 15).all()
            if self.family_name == "central_power":
                r = np.linalg.norm(tr.xy[:, 1] - tr.xy[:, 0], axis=1)
                inside = inside and r.min() > 45
            if inside and len(tr.t) == int(4.0 * FPS):
                out.append(cand)
        while len(out) < n:                       # fall back rather than hang
            out.append(self._random_experiment(rng))
        return out

    def _random_experiment(self, rng: np.random.Generator) -> dict:
        return {p.name: float(rng.uniform(p.lo + 0.15 * (p.hi - p.lo),
                                          p.hi - 0.15 * (p.hi - p.lo)))
                for p in self.family.experiment_params}
