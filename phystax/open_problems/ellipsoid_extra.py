"""Seal an additional, theory-motivated predictor for P1, and grade every
sealed predictor once the run has finished.

Conic-geometry conjecture (sealed by `seal`): the PSD cone's statistical
dimension is d(d+1)/4 (noted in Arslan's paper as the origin of the 1/4
constant). Random convex feasibility problems change phase at the statistical
dimension with a window that grows like the square root of the ambient
dimension (Amelunxen-Lotz-McCoy-Tropp), here proportional to d. Rule:
    n50(d) = d(d+1)/4,   width(d) = w0 * d,
    w0 = median of width/d over the calibration dimensions of the same run.
The rule is fixed when sealed; w0 is read from calibration only, never from
the targets.

    python -m phystax.open_problems.ellipsoid_extra seal     # BEFORE targets are measured
    python -m phystax.open_problems.ellipsoid_extra grade    # after the run finishes
Grading adds the Brier score of each predictor's full transition curve against
every measured trial, with the asymptotic step function at d^2/4 as baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from ..grader.ledger import Ledger

RULE = {"name": "statistical_dimension", "n50": "d*(d+1)/4", "width": "w0*d",
        "w0": "median(width/d) over calibration d"}


def _main_ledger_kinds(out_dir):
    p = os.path.join(out_dir, "ledger.jsonl")
    return [json.loads(l)["kind"] for l in open(p)] if os.path.exists(p) else []


def seal(out_dir: str, force_posthoc: bool = False) -> dict:
    kinds = _main_ledger_kinds(out_dir)
    posthoc = "target_measurements" in kinds or os.path.exists(os.path.join(out_dir, "result.json"))
    if posthoc and not force_posthoc:
        raise SystemExit("Targets are already measured: sealing now would not be a prediction. "
                         "Use --posthoc to record it, clearly labelled as post hoc.")
    led = Ledger(os.path.join(out_dir, "extra_sealed.jsonl"))
    payload = {"rule": RULE, "sealed_at": time.time(), "posthoc": posthoc,
               "main_ledger_state": kinds}
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    led.append("sealed_rule", {**payload, "sha256": sha})
    print(f"sealed {RULE['name']} ({'POST HOC' if posthoc else 'before targets'}) sha {sha[:16]}; "
          f"main ledger had: {kinds}")
    return payload


def _logistic(n, n50, w):
    return 1 / (1 + np.exp((np.asarray(n, float) - n50) / w))


def brier(curve: dict, trials: int, q) -> float:
    """Mean Brier score over all trials: each n contributes k successes and
    (trials - k) failures against forecast q(n)."""
    p = np.asarray(curve["p_fit"], float)
    q = np.asarray(q, float)
    return float(np.mean(p * (1 - q) ** 2 + (1 - p) * q ** 2))


def grade(out_dir: str) -> dict:
    res = json.load(open(os.path.join(out_dir, "result.json")))
    calib, target = res["calibration"], res["target"]
    w0 = float(np.median([v["width"] / int(d) for d, v in calib.items()]))
    preds = {}
    for d in target:
        di = int(d)
        preds.setdefault("asymptotic_step", {})[d] = {"n50": di * di / 4, "width": None}
        preds.setdefault("statistical_dimension", {})[d] = {"n50": di * (di + 1) / 4, "width": w0 * di}
        fs = res["sealed"]["finite_size_fit"].get(d)
        if fs:
            preds.setdefault("finite_size_fit", {})[d] = fs
        ms = res["sealed"].get("muse_spark", {}).get("predictions", {}).get(d)
        if ms:
            preds.setdefault("muse_spark", {})[d] = ms
    extra = os.path.join(out_dir, "extra_sealed.jsonl")
    sealed_rule = [json.loads(l) for l in open(extra)] if os.path.exists(extra) else []
    table = {}
    for name, by_d in preds.items():
        for d, p in by_d.items():
            m = target[d]
            n = m["curve"]["n"]
            q = (np.asarray(n, float) < p["n50"]).astype(float) if p.get("width") is None \
                else _logistic(n, p["n50"], p["width"])
            lo, hi = m["n50_ci"]
            table.setdefault(name, {})[d] = {
                "pred_n50": p["n50"], "meas_n50": m["n50"], "n50_inside_95ci": bool(lo <= p["n50"] <= hi),
                "pred_width": p.get("width"), "meas_width": m["width"],
                "brier": brier(m["curve"], m["trials"], q)}
    out = {"w0_from_calibration": w0, "table": table,
           "statistical_dimension_sealed_before_targets": bool(sealed_rule) and not sealed_rule[0]["payload"]["posthoc"]}
    json.dump(out, open(os.path.join(out_dir, "grade_all.json"), "w"), indent=1)
    print(f"w0 (calibration) = {w0:.3f};  statistical-dimension rule sealed before targets: "
          f"{out['statistical_dimension_sealed_before_targets']}")
    print("| predictor | d | predicted n50 | measured n50 [95% CI] | inside | predicted width | measured width | Brier |")
    print("|---|---|---|---|---|---|---|---|")
    for name, by_d in table.items():
        for d, r in sorted(by_d.items(), key=lambda kv: int(kv[0])):
            ci = target[d]["n50_ci"]
            pw = "–" if r["pred_width"] is None else f"{r['pred_width']:.2f}"
            print(f"| {name} | {d} | {r['pred_n50']:.1f} | {r['meas_n50']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | "
                  f"{'yes' if r['n50_inside_95ci'] else 'no'} | {pw} | {r['meas_width']:.2f} | {r['brier']:.4f} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seal", "grade"])
    ap.add_argument("--out", default="runs/open_problems/ellipsoid")
    ap.add_argument("--posthoc", action="store_true")
    a = ap.parse_args()
    seal(a.out, a.posthoc) if a.cmd == "seal" else grade(a.out)


if __name__ == "__main__":
    main()
