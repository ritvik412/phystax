"""Reference fitter: a non-LLM scientist that is TOLD the true form and only
fits constants by weighted, robust least squares on measured positions.

It makes the perception tax decomposable. For an agent episode on channel c
(camera) with agent-chosen experiments D_agent, and a fixed designed set D_ref:

    total     = agent(c, D_agent)            - ref(oracle, D_ref)
              = [agent(c, D_agent) - ref(c, D_agent)]     reasoning tax:
                                                          same data, the agent
                                                          had to find the form
              + [ref(c, D_agent)   - ref(c, D_ref)]       design gap:
                                                          agent's experiments vs
                                                          a fixed sensible design
              + [ref(c, D_ref)     - ref(oracle, D_ref)]  measurement tax:
                                                          camera vs exact numbers

The three terms telescope, so they sum to the total exactly. `refit_episode`
computes ref(c, D_agent) from an agent run's own measured data.

Confidence weighting (sam_conf): if runs/calibration.json exists (written by
measure_perception from real SAM errors), each frame gets weight 1/sigma(conf),
the calibrated expected pixel error for that confidence. Otherwise raw
confidence is used as the weight. The result records which one was used.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
from scipy.optimize import least_squares

from ..config import FPS, RUNS_DIR
from ..grader.grade import commitment_hash, grade, probe_times
from ..grader.ledger import Ledger
from ..sim.engine import simulate
from ..sim.laws import key_params
from ..sim.world import World
from .conditions import measure, sam_backend

DESIGNS = {   # a fixed, sensible design per family
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
TEST_S = 4.0


# ---- parameter transforms -------------------------------------------------
def _encode(form, phys: dict) -> np.ndarray:
    u = []
    for name, tr in form.fit.items():
        v = phys[name]
        u.append(np.log(max(v, 1e-300)) if tr[0] == "log" else v / tr[1])
    return np.array(u, float)


def _decode(form, u: np.ndarray) -> dict:
    out = {}
    for (name, tr), x in zip(form.fit.items(), u):
        out[name] = float(np.exp(x)) if tr[0] == "log" else float(x * tr[1])
    out["form"] = form.name
    return out


# ---- confidence weights ---------------------------------------------------
def load_calibration(path: str | None = None) -> list | None:
    path = path or os.path.join(RUNS_DIR, "calibration.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path)).get("bins")


def _weights(conf: list | None, n: int, mode: str, calib: list | None) -> np.ndarray:
    if mode == "uniform" or conf is None:
        return np.ones(n)
    c = np.array([np.nan if v is None else v for v in conf], float)
    if mode == "calibrated" and calib:
        sigma = np.full(n, np.nan)
        for lo, hi, med_err in calib:
            sel = (c >= lo) & (c <= hi)
            sigma[sel] = max(med_err, 0.5)
        sigma[np.isnan(sigma)] = max(b[2] for b in calib)
        return 1.0 / sigma
    return np.nan_to_num(c, nan=0.0)          # raw confidence as weight


def fit(world: World, measured: list[tuple[dict, dict]], weighting: str = "uniform",
        calib: list | None = None, return_solution: bool = False):
    fam = world.family
    form = fam.forms[world.hidden["form"]]
    moving = [i for i in range(len(fam.bodies)) if i not in fam.fixed_bodies]
    total_len = sum(len(m["t"]) * 2 * len(moving) for _, m in measured)

    def residuals(u):
        h = _decode(form, u)
        res = []
        for params, m in measured:
            n = len(m["t"])
            try:
                tr = simulate(fam, h, params, n)
            except Exception:
                return np.full(total_len, 1e4)
            for i in moving:
                b = fam.bodies[i]
                series = m["bodies"][b]
                x = np.array([np.nan if v is None else v for v in series["x"]], float)
                y = np.array([np.nan if v is None else v for v in series["y"]], float)
                w = _weights(series.get("conf"), n, weighting, calib)
                k = min(n, len(tr.t))
                ok = np.isfinite(x[:k]) & np.isfinite(y[:k])
                rx = np.where(ok, (tr.xy[:k, i, 0] - np.nan_to_num(x[:k])) * w[:k], 0.0)
                ry = np.where(ok, (tr.xy[:k, i, 1] - np.nan_to_num(y[:k])) * w[:k], 0.0)
                res += [np.pad(rx, (0, n - k), constant_values=1e3),
                        np.pad(ry, (0, n - k), constant_values=1e3)]
        r = np.concatenate(res)
        return np.where(np.isfinite(r), r, 1e4)

    best = None
    for start in form.starts:
        sol = least_squares(residuals, _encode(form, start), loss="soft_l1", f_scale=5.0,
                            max_nfev=300)
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < 1e-8 * total_len:     # exact fit found; other starts cannot do better
            break
    est = _decode(form, best.x)
    return (est, best) if return_solution else est


def param_samples(world: World, sol, n: int = 40, seed: int = 0) -> list[dict]:
    """Laplace approximation: cov = s^2 (J^T J)^-1 in the fitting space, with
    s^2 the residual variance. Sampling it and simulating each sample gives
    prediction intervals that reflect how well the data pinned the constants."""
    form = world.family.forms[world.hidden["form"]]
    J = np.asarray(sol.jac, float)
    m, k = J.shape
    s2 = 2.0 * sol.cost / max(m - k, 1)
    cov = np.linalg.pinv(J.T @ J) * s2
    if not np.all(np.isfinite(cov)):
        return []
    rng = np.random.default_rng(seed)
    U = rng.multivariate_normal(sol.x, cov, size=n, check_valid="ignore", method="eigh")
    return [_decode(form, u) for u in U]


def _predict(world: World, est: dict, samples: list[dict] | None = None) -> tuple[list, list]:
    """Point predictions from `est`; central 80% intervals from `samples`
    (10th-90th percentile of simulated positions), widened to contain the point."""
    times = probe_times(TEST_S)
    fam = world.family
    n = int(TEST_S * FPS)
    preds = []
    for i, p in enumerate(world.test_conditions()):
        tr = simulate(fam, est, p, n)
        idx = [min(len(tr.t) - 1, int(round(t * FPS))) for t in times]
        sims = []
        for h in samples or []:
            st = simulate(fam, h, p, n)
            if len(st.t) == n and np.all(np.isfinite(st.xy)):
                sims.append(st.xy[idx])
        for b_i, b in enumerate(fam.bodies):
            if b_i in fam.fixed_bodies:
                continue
            x, y = tr.xy[idx, b_i, 0], tr.xy[idx, b_i, 1]
            if len(sims) >= 5:
                S = np.stack(sims)[:, :, b_i, :]
                lo, hi = np.quantile(S, 0.1, axis=0), np.quantile(S, 0.9, axis=0)
                x_lo, x_hi = np.minimum(lo[:, 0], x), np.maximum(hi[:, 0], x)
                y_lo, y_hi = np.minimum(lo[:, 1], y), np.maximum(hi[:, 1], y)
            else:
                x_lo = x_hi = x
                y_lo = y_hi = y
            preds.append({"test_index": i, "body": b, "x": x.tolist(), "y": y.tolist(),
                          "x_lo": list(map(float, x_lo)), "x_hi": list(map(float, x_hi)),
                          "y_lo": list(map(float, y_lo)), "y_hi": list(map(float, y_hi))})
    return preds, times


def _true_tests(world: World):
    return [simulate(world.family, world.hidden, p, int(TEST_S * FPS))
            for p in world.test_conditions()]


def weighting_for(condition: str) -> str:
    if condition != "sam_conf":
        return "uniform"
    return "calibrated" if load_calibration() else "raw_conf"


def random_design(world: World, n: int = 3, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(10_000 + seed)
    return [{p.name: float(rng.uniform(p.lo, p.hi)) for p in world.family.experiment_params}
            for _ in range(n)]


def run_reference(family: str, seed: int, condition: str, degradation: int = 0,
                  noise_sigma_px: float = 3.0, run_dir: str | None = None,
                  design: str = "designed", form: str | None = None) -> dict:
    """Same episode shape as the agent loop (ledger, commit-before-test,
    result.json), so the UI and the report treat both the same way."""
    world = World(family, seed, degradation, run_dir=run_dir or "", form=form)
    ledger_path = os.path.join(world.run_dir, "ledger.jsonl")
    if os.path.exists(ledger_path):          # one ledger per episode: re-runs start fresh
        os.remove(ledger_path)
    ledger = Ledger(ledger_path)
    backend = sam_backend() if condition.startswith("sam") else None
    weighting = weighting_for(condition)
    ledger.append("episode_start", {"family": family, "spec": world.spec(), "condition": condition,
                                    "model": f"reference-fitter (told the true form), {design} design",
                                    "degradation": degradation, "sam_backend": backend,
                                    "secret_law_sha256": world.secret_commitment()})
    plan = DESIGNS[family] if design == "designed" else random_design(world, 3, seed)
    measured, diags, shown_all = [], [], {}
    for k, params in enumerate(plan):
        exp = world.run_experiment(params, 3.0)
        m, d = measure(exp, condition, world.family.fixed_bodies, noise_sigma_px, seed=k)
        measured.append((exp.params, m))
        diags.append({"exp_id": exp.exp_id, **d})
        shown_all[exp.exp_id] = {"params": exp.params, **m}
        ledger.append("experiment", {"exp_id": exp.exp_id, "params": exp.params, "duration_s": 3.0,
                                     "video": exp.video_path, "condition": condition})
    os.makedirs(os.path.join(world.run_dir, "sandbox"), exist_ok=True)
    json.dump(shown_all, open(os.path.join(world.run_dir, "sandbox", "experiments.json"), "w"))
    est, sol = fit(world, measured, weighting, load_calibration(), return_solution=True)
    preds, times = _predict(world, est, param_samples(
        world, sol, n=int(os.environ.get("PHYSTAX_INTERVAL_SAMPLES", "40")), seed=seed))
    commitment = {"law_equation": world.family.forms[world.hidden["form"]].equation + " (given)",
                  "parameters": [{"name": k, "value": float(v)} for k, v in est.items() if k != "form"],
                  "predictions": preds, "self_confidence": 1.0,
                  "rationale": f"least squares on measured positions, weighting={weighting}"}
    c_hash = commitment_hash(commitment)
    ledger.append("commit", {"commitment": commitment, "sha256": c_hash})
    tests = [world.run_test(i, TEST_S) for i in range(len(world.test_params))]
    ledger.append("test_reveal", {"videos": [t.video_path for t in tests], "commit_sha256": c_hash})
    score = grade(commitment, family, world.hidden, [t.traj for t in tests], times, FPS)
    ledger.append("grade", score)
    ledger.append("secret_reveal", world.reveal())
    ok, why = ledger.verify()
    result = {"family": family, "seed": seed, "condition": condition, "degradation": degradation,
              "model": "reference", "design": design, "weighting": weighting,
              "sam_backend": backend, "estimate": est, "hidden": world.hidden, "score": score,
              "true_form": world.hidden["form"], "perception": diags,
              "key_params": key_params(family, world.hidden["form"]),
              "n_experiments": len(world.experiments),
              "ledger_ok": ok and ledger.commit_precedes_test(), "ledger_msg": why,
              "run_dir": world.run_dir, "probe_times": times}
    json.dump(result, open(os.path.join(world.run_dir, "result.json"), "w"), indent=2, default=float)
    return result


def refit_episode(run_dir: str) -> dict:
    """ref(c, D_agent): fit the true form to exactly the data an agent saw."""
    res = json.load(open(os.path.join(run_dir, "result.json")))
    data = json.load(open(os.path.join(run_dir, "sandbox", "experiments.json")))
    with tempfile.TemporaryDirectory() as tmp:
        world = World(res["family"], res["seed"], res["degradation"], run_dir=tmp)
        measured = [(v["params"], v) for _, v in sorted(data.items())]
        if not measured:
            return {"nrmse": None, "error": "agent ran no experiments"}
        est = fit(world, measured, weighting_for(res["condition"]), load_calibration())
        preds, times = _predict(world, est)
        score = grade({"parameters": [{"name": k, "value": v} for k, v in est.items() if k != "form"],
                       "predictions": preds}, res["family"], world.hidden, _true_tests(world), times, FPS)
    return {"nrmse": score["prediction_nrmse"], "pass": score["pass"],
            "param_rel_err": {k: v["rel_err"] for k, v in score["parameters"].items()}}
