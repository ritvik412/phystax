"""P1. The critical window of Gaussian ellipsoid fitting.

Question (left open in "The Strict Threshold for Gaussian Ellipsoid Fitting",
Meta AI Research, Oct 2026): given n i.i.d. standard Gaussian points in R^d,
an origin-centred ellipsoid {x : x^T S x = 1, S positive definite} through all
of them exists with high probability for n below about d^2/4 and fails above
it. The behaviour at the threshold itself is unresolved. At finite d the
transition has a centre n50(d) and a width w(d); how do they approach the
asymptotic picture?

Method. Fitting is a semidefinite feasibility problem. We solve the robust
version  max t  s.t.  x_i^T S x_i = 1 (all i),  S - t I is PSD,  t <= 1:
a fit exists iff the optimum t* > 0, and t* is a margin, so solver tolerance
can be checked against it. For each d we estimate P(fit | n) on a grid of n
from repeated trials, then fit a logistic curve P = 1 / (1 + exp((n - n50)/w)).

Protocol (pre-registered, sealed in a ledger):
  1. measure small d (calibration set);
  2. seal predictions of n50 and w for larger, not-yet-computed d from three
     predictors: the asymptotic d^2/4, a finite-size fit to the calibration
     set, and (optionally) Muse Spark given the calibration table;
  3. measure the larger d; grade every sealed prediction against the measured
     value and its bootstrap interval.
This is numerical evidence about finite-size behaviour, not a proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

# One maths thread per worker process: many workers x many BLAS threads
# overloads shared machines (a load average in the thousands, observed).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from ..grader.ledger import Ledger

TOL = 1e-6        # a fit "exists" if the robust margin exceeds this


def _j(x):
    """JSON-normalise (int keys -> str) so payloads hash and log deterministically."""
    return json.loads(json.dumps(x, default=float))


def fit_margin(X: np.ndarray, solver: str = "CLARABEL") -> float:
    """Robust ellipsoid-fitting margin t* (> 0 iff an ellipsoid fits)."""
    import cvxpy as cp
    n, d = X.shape
    S = cp.Variable((d, d), symmetric=True)
    t = cp.Variable()
    cons = [S - t * np.eye(d) >> 0, cp.sum(cp.multiply(X @ S, X), axis=1) == 1, t <= 1]
    prob = cp.Problem(cp.Maximize(t), cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        prob.solve(solver="SCS", eps=1e-7, max_iters=50000)
    if prob.status in ("infeasible", "infeasible_inaccurate"):
        return -1.0
    return float(t.value) if t.value is not None else -1.0


def _trial(args) -> tuple[int, int, int, float]:
    d, n, seed = args
    X = np.random.default_rng(seed).standard_normal((n, d))
    return d, n, seed, fit_margin(X)


def measure(ds, ratios=None, trials: int = 40, workers: int = 8, seed0: int = 0) -> dict:
    """P(fit | d, n) on a grid of n = ratio * d^2 around the threshold."""
    ratios = ratios if ratios is not None else np.round(np.arange(0.14, 0.42, 0.02), 3)
    jobs = []
    for d in ds:
        ns = sorted({max(d + 1, int(round(r * d * d))) for r in ratios})
        for n in ns:
            jobs += [(d, n, seed0 + 100_000 * d + 1000 * n + k) for k in range(trials)]
    out: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for d, n, seed, m in ex.map(_trial, jobs, chunksize=4):
            out.setdefault(int(d), {}).setdefault(int(n), []).append(m)
    return out


def logistic_fit(ns: np.ndarray, k: np.ndarray, m: int) -> tuple[float, float]:
    """Binomial maximum-likelihood fit of P(fit) = 1 / (1 + exp((n - n50)/w))."""
    p = np.clip(k / m, 1e-3, 1 - 1e-3)

    def nll(theta):
        n50, logw = theta
        q = 1 / (1 + np.exp((ns - n50) / np.exp(logw)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return np.sqrt(-(k * np.log(q) + (m - k) * np.log(1 - q)) + 1e-12)

    n0 = float(ns[np.argmin(np.abs(p - 0.5))])
    sol = least_squares(nll, [n0, np.log(max(1.0, 0.05 * n0))])
    return float(sol.x[0]), float(np.exp(sol.x[1]))


def summarise(raw: dict, n_boot: int = 300, seed: int = 0) -> dict:
    """Per d: n50, width, bootstrap 95% intervals, and the empirical curve."""
    rng = np.random.default_rng(seed)
    res = {}
    for d, by_n in sorted(raw.items()):
        ns = np.array(sorted(by_n))
        succ = [np.array(by_n[n]) > TOL for n in ns]
        m = len(succ[0])
        k = np.array([s.sum() for s in succ])
        n50, w = logistic_fit(ns, k, m)
        boots = []
        for _ in range(n_boot):
            kb = np.array([rng.choice(s, m).sum() for s in succ])
            boots.append(logistic_fit(ns, kb, m))
        b = np.array(boots)
        res[int(d)] = {"n50": n50, "width": w, "n50_ci": np.quantile(b[:, 0], [0.025, 0.975]).tolist(),
                       "width_ci": np.quantile(b[:, 1], [0.025, 0.975]).tolist(),
                       "n50_over_d2": n50 / d ** 2, "trials": m,
                       "curve": {"n": ns.tolist(), "p_fit": (k / m).tolist()}}
    return res


def finite_size_predictor(summary: dict):
    """Fit n50(d) = d^2/4 + a d + b and width(d) = c d^beta on the calibration set."""
    ds = np.array(sorted(summary), float)
    n50 = np.array([summary[int(d)]["n50"] for d in ds])
    wid = np.array([summary[int(d)]["width"] for d in ds])
    A = np.c_[ds, np.ones_like(ds)]
    (a, b), *_ = np.linalg.lstsq(A, n50 - ds ** 2 / 4, rcond=None)
    (beta, logc), *_ = np.linalg.lstsq(np.c_[np.log(ds), np.ones_like(ds)], np.log(wid), rcond=None)
    return {"form": "n50 = d^2/4 + a*d + b ; width = c*d^beta", "a": float(a), "b": float(b),
            "c": float(np.exp(logc)), "beta": float(beta)}


def predict(params: dict, d: int) -> dict:
    return {"n50": d * d / 4 + params["a"] * d + params["b"], "width": params["c"] * d ** params["beta"]}


def spark_predictor(calib: dict, targets: list[int], model_key: str = "spark") -> dict:
    """Muse Spark sees the calibration table and seals its own predictions.
    Never raises: a failed reply is recorded as a failure, and the protocol
    continues with the other predictors."""
    from ..agent.loop import _extract_json
    from ..agent.providers import ChatModel
    try:
        m = ChatModel(model_key)
    except Exception as exc:
        return {"failure": f"{type(exc).__name__}: {exc}"}
    rows = [{"d": int(d), "n50": round(v["n50"], 2), "width": round(v["width"], 2)} for d, v in sorted(calib.items(), key=lambda kv: int(kv[0]))]
    msgs = [{"role": "user", "content":
             "Random Gaussian points in R^d; an origin-centred ellipsoid through n of them exists with high "
             "probability for n below roughly d^2/4 as d grows. At finite d the probability of a fit drops from 1 "
             "to 0 around a centre n50(d) with width w(d). Measured values: " + json.dumps(rows) +
             f". Propose a formula for n50(d) and w(d) and predict both for d = {targets}. These predictions will "
             "be sealed and graded against new measurements. Answer with JSON only, exactly: "
             '{"formula": str, "rationale": str, "predictions": [{"d": int, "n50": number, "width": number}]}'}]
    last = ""
    for _ in range(3):
        try:
            msg = m.chat(msgs, response_format={"type": "json_object"}, max_tokens=12000)
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
        last = msg.content or ""
        out = _extract_json(last)
        if out and isinstance(out.get("predictions"), list) and out["predictions"]:
            try:
                preds = {int(p["d"]): {"n50": float(p["n50"]), "width": float(p["width"])} for p in out["predictions"]}
            except (KeyError, TypeError, ValueError):
                preds = {}
            if all(t in preds for t in targets):
                return {"formula": out.get("formula", ""), "rationale": out.get("rationale", ""),
                        "predictions": preds, "model": m.model_id, "cost_usd": m.cost_usd}
        msgs += [{"role": "assistant", "content": last},
                 {"role": "user", "content": f"Reply with only the JSON object, with a prediction for every d in {targets}."}]
    return {"failure": f"no usable JSON after 3 attempts: {last[:300]}", "model": m.model_id}


def _ledger_entries(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def run_protocol(calib_ds, target_ds, trials: int = 40, workers: int = 8, use_spark: bool = False,
                 out_dir: str = "runs/open_problems/ellipsoid", resume: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    lpath = os.path.join(out_dir, "ledger.jsonl")
    prior = _ledger_entries(lpath)
    kinds = [e["kind"] for e in prior]
    if resume and "calibration" in kinds and "sealed_predictions" not in kinds:
        ledger = Ledger(lpath)
        calib = {int(d): v for d, v in next(e for e in prior if e["kind"] == "calibration")["payload"].items()}
        ledger.append("resumed", {"from": "calibration", "reason": "sealing step had failed; calibration reused"})
        t0 = time.time()
        return _seal_and_measure(ledger, calib, calib_ds, target_ds, trials, workers, use_spark, out_dir, t0)
    if prior and not resume:
        raise SystemExit(f"{lpath} already exists; delete the folder for a fresh run, or pass --resume")
    ledger = Ledger(lpath)
    ledger.append("protocol", {"problem": "gaussian ellipsoid fitting: finite-size critical window",
                               "calibration_d": list(calib_ds), "target_d": list(target_ds), "trials": trials,
                               "tolerance": TOL, "predictors": ["asymptotic d^2/4", "finite-size fit"]
                               + (["Muse Spark"] if use_spark else [])})
    t0 = time.time()
    calib = summarise(measure(calib_ds, trials=trials, workers=workers))
    ledger.append("calibration", _j({d: {k: v for k, v in r.items() if k != "curve"} for d, r in calib.items()}))
    return _seal_and_measure(ledger, calib, calib_ds, target_ds, trials, workers, use_spark, out_dir, t0)


def _seal_and_measure(ledger, calib, calib_ds, target_ds, trials, workers, use_spark, out_dir, t0) -> dict:
    fs = finite_size_predictor(calib)
    sealed = {"asymptotic": {d: {"n50": d * d / 4, "width": None} for d in target_ds},
              "finite_size_fit": {"params": fs, **{d: predict(fs, d) for d in target_ds}}}
    if use_spark:
        sealed["muse_spark"] = spark_predictor(calib, list(target_ds))
    sealed = _j(sealed)
    sha = hashlib.sha256(json.dumps(sealed, sort_keys=True).encode()).hexdigest()
    ledger.append("sealed_predictions", {"sha256": sha, "predictions": sealed})
    target = summarise(measure(target_ds, trials=trials, workers=workers, seed0=7_000_000))
    ledger.append("target_measurements", _j({d: {k: v for k, v in r.items() if k != "curve"} for d, r in target.items()}))
    grades = {}
    for name, pred in sealed.items():
        rows = {}
        for d in target_ds:
            p = pred["predictions"].get(str(d)) if name == "muse_spark" else pred.get(str(d))
            if not p:
                continue
            meas = target[d]
            lo, hi = meas["n50_ci"]
            rows[d] = {"pred_n50": p["n50"], "meas_n50": meas["n50"], "err_n50": p["n50"] - meas["n50"],
                       "inside_ci": bool(lo <= p["n50"] <= hi)}
            if p.get("width") is not None:
                rows[d]["pred_width"], rows[d]["meas_width"] = p["width"], meas["width"]
        grades[name] = rows
    ledger.append("grade", _j(grades))
    ok, why = ledger.verify()
    result = {"calibration": calib, "target": target, "sealed": sealed, "sealed_sha256": sha, "grades": grades,
              "ledger_ok": ok and why == "ok", "wall_s": time.time() - t0}
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=1, default=float)
    return result


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="P1: finite-size critical window of Gaussian ellipsoid fitting")
    ap.add_argument("--calib", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--target", type=int, nargs="+", default=[26, 32, 40])
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--workers", type=int, default=min(64, max(1, (os.cpu_count() or 8) // 3)),
                    help="worker processes (each single-threaded); keep well below the core count on shared machines")
    ap.add_argument("--spark", action="store_true", help="also seal Muse Spark's predictions")
    ap.add_argument("--out", default="runs/open_problems/ellipsoid")
    ap.add_argument("--resume", action="store_true", help="reuse a finished calibration if sealing had failed")
    a = ap.parse_args()
    r = run_protocol(a.calib, a.target, a.trials, a.workers, a.spark, a.out, resume=a.resume)
    print("calibration n50/d^2:", {d: round(v["n50_over_d2"], 4) for d, v in r["calibration"].items()})
    print("target      n50/d^2:", {d: round(v["n50_over_d2"], 4) for d, v in r["target"].items()})
    for name, rows in r["grades"].items():
        print(f"{name:16s}", {d: f"pred {v['pred_n50']:.1f} vs {v['meas_n50']:.1f} ({'in' if v['inside_ci'] else 'out'})"
                              for d, v in rows.items()})
    print("sealed", r["sealed_sha256"][:16], "ledger ok:", r["ledger_ok"], f"({r['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
