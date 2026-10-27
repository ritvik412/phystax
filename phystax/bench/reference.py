"""Reference fitter: a non-LLM scientist that is *told* the correct functional
form and only fits constants by least squares on the measured positions.

It splits the total perception tax into two parts:
    measurement tax = reference error(camera) - reference error(oracle)
    reasoning tax   = agent error - reference error   (same channel)
and it tests the confidence-weighting hypothesis without LLM variance:
weights = SAM confidence (sam_conf) vs uniform (sam) on identical tracks.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import json
import os

from ..grader.grade import KEY_PARAMS, commitment_hash, grade, probe_times
from ..grader.ledger import Ledger
from ..sim.engine import simulate
from ..sim.world import World
from .conditions import measure, sam_backend

DESIGNS = {   # a fixed, reasonable experiment design per family
    "central_power": [
        {"start_radius_px": 100, "launch_speed_px_s": 160, "launch_angle_deg": 90},
        {"start_radius_px": 140, "launch_speed_px_s": 135, "launch_angle_deg": 95},
        {"start_radius_px": 185, "launch_speed_px_s": 115, "launch_angle_deg": 85},
    ],
    "power_drag": [
        {"launch_speed_px_s": 250, "launch_angle_deg": 60},
        {"launch_speed_px_s": 450, "launch_angle_deg": 45},
        {"launch_speed_px_s": 650, "launch_angle_deg": 30},
    ],
    "nonlinear_spring": [
        {"displacement_px": 60, "initial_velocity_px_s": 0},
        {"displacement_px": 150, "initial_velocity_px_s": 0},
        {"displacement_px": -210, "initial_velocity_px_s": 80},
    ],
}

def _to_hidden(family: str, u: np.ndarray) -> dict:
    if family == "central_power":
        return {"alpha": u[0], "k": float(np.exp(u[1]))}
    if family == "power_drag":
        return {"g": float(np.exp(u[0])), "beta": u[1], "gamma": float(np.exp(u[2]))}
    return {"k": float(np.exp(u[0])), "c": u[1] * 1e-4, "b": u[2]}


def _starts(family: str) -> list[np.ndarray]:
    """Multi-start: power-law exponents create local minima."""
    if family == "central_power":
        # for each exponent guess, set k so a 120 px circular orbit has v ~ 120 px/s
        return [np.array([a, np.log(120.0 ** 2 * 120.0 ** (a - 1))]) for a in (1.5, 2.0, 2.5, 3.0)]
    if family == "power_drag":
        return [np.array([np.log(300), b, np.log(300 / 400 ** b)]) for b in (1.3, 1.9, 2.5)]
    return [np.array([np.log(15), 1.0, 0.2])]


def fit(world: World, measured: list[tuple[dict, dict]], use_conf: bool) -> dict:
    fam = world.family
    moving = [i for i in range(len(fam.bodies)) if i not in fam.fixed_bodies]

    def residuals(u):
        h = _to_hidden(world.family_name, u)
        res = []
        for params, m in measured:
            n = len(m["t"])
            try:
                tr = simulate(fam, h, params, n)
            except Exception:
                return np.full(sum(len(mm["t"]) * 2 * len(moving) for _, mm in measured), 1e4)
            for i in moving:
                b = fam.bodies[i]
                x = np.array([np.nan if v is None else v for v in m["bodies"][b]["x"]], float)
                y = np.array([np.nan if v is None else v for v in m["bodies"][b]["y"]], float)
                w = np.ones(n)
                if use_conf and "conf" in m["bodies"][b]:
                    w = np.array([0.0 if c is None else c for c in m["bodies"][b]["conf"]], float)
                k = min(n, len(tr.t))
                ok = np.isfinite(x[:k]) & np.isfinite(y[:k])
                rx = np.where(ok, (tr.xy[:k, i, 0] - np.nan_to_num(x[:k])) * w[:k], 0.0)
                ry = np.where(ok, (tr.xy[:k, i, 1] - np.nan_to_num(y[:k])) * w[:k], 0.0)
                rx = np.pad(rx, (0, n - k), constant_values=1e3)
                ry = np.pad(ry, (0, n - k), constant_values=1e3)
                res.extend([rx, ry])
        r = np.concatenate(res)
        return np.where(np.isfinite(r), r, 1e4)

    best = None
    for u0 in _starts(world.family_name):
        sol = least_squares(residuals, u0, loss="soft_l1", f_scale=5.0, max_nfev=300)
        if best is None or sol.cost < best.cost:
            best = sol
    return _to_hidden(world.family_name, best.x)


def run_reference(family: str, seed: int, condition: str, degradation: int = 0,
                  noise_sigma_px: float = 3.0, run_dir: str | None = None) -> dict:
    """Same episode shape as the agent loop (ledger, commit-before-test,
    result.json), so the UI and the report treat both the same way."""
    world = World(family, seed, degradation, run_dir=run_dir or "")
    ledger_path = os.path.join(world.run_dir, "ledger.jsonl")
    if os.path.exists(ledger_path):          # one ledger per episode: re-runs start fresh
        os.remove(ledger_path)
    ledger = Ledger(ledger_path)
    backend = sam_backend() if condition.startswith("sam") else None
    ledger.append("episode_start", {"spec": world.spec(), "condition": condition,
                                    "model": "reference-fitter (told the true form)",
                                    "degradation": degradation, "sam_backend": backend,
                                    "secret_law_sha256": world.secret_commitment()})
    measured, diags, shown_all = [], [], {}
    for k, params in enumerate(DESIGNS[family]):
        exp = world.run_experiment(params, 3.0)
        m, d = measure(exp, condition, world.family.fixed_bodies, noise_sigma_px, seed=k)
        measured.append((exp.params, m))
        diags.append({"exp_id": exp.exp_id, **d})
        shown_all[exp.exp_id] = {"params": exp.params, **m}
        ledger.append("experiment", {"exp_id": exp.exp_id, "params": exp.params, "duration_s": 3.0,
                                     "video": exp.video_path, "condition": condition})
    os.makedirs(os.path.join(world.run_dir, "sandbox"), exist_ok=True)
    json.dump(shown_all, open(os.path.join(world.run_dir, "sandbox", "experiments.json"), "w"))
    est = fit(world, measured, use_conf=(condition == "sam_conf"))

    times = probe_times(4.0)
    fam = world.family
    preds = []
    for i, p in enumerate(world.test_conditions()):
        tr = simulate(fam, est, p, int(4.0 * 30))
        idx = [min(len(tr.t) - 1, int(round(t * 30))) for t in times]
        for b_i, b in enumerate(fam.bodies):
            if b_i in fam.fixed_bodies:
                continue
            preds.append({"test_index": i, "body": b,
                          "x": tr.xy[idx, b_i, 0].tolist(), "y": tr.xy[idx, b_i, 1].tolist()})
    commitment = {"law_equation": f"true form of {family} (given)",
                  "parameters": [{"name": k, "value": float(v)} for k, v in est.items()],
                  "predictions": preds, "self_confidence": 1.0,
                  "rationale": "least squares on measured positions"}
    c_hash = commitment_hash(commitment)
    ledger.append("commit", {"commitment": commitment, "sha256": c_hash})
    tests = [world.run_test(i, 4.0) for i in range(len(world.test_params))]
    ledger.append("test_reveal", {"videos": [t.video_path for t in tests], "commit_sha256": c_hash})
    score = grade(commitment, family, world.hidden, [t.traj for t in tests], times, 30)
    ledger.append("grade", score)
    ledger.append("secret_reveal", world.reveal())
    ok, why = ledger.verify()
    result = {"family": family, "seed": seed, "condition": condition, "degradation": degradation,
              "model": "reference", "sam_backend": backend, "estimate": est,
              "hidden": world.hidden, "score": score, "perception": diags,
              "key_params": KEY_PARAMS[family], "n_experiments": len(world.experiments),
              "ledger_ok": ok and ledger.commit_precedes_test(), "ledger_msg": why,
              "run_dir": world.run_dir, "probe_times": times}
    json.dump(result, open(os.path.join(world.run_dir, "result.json"), "w"), indent=2, default=float)
    return result
