"""Prediction intervals, proper scoring, pass@K reporting."""
import os

import numpy as np

os.environ.setdefault("PHYSTAX_RUNS", "/tmp/phystax_test_runs")
os.environ["PHYSTAX_INTERVAL_SAMPLES"] = "16"

from phystax.bench.reference import run_reference  # noqa: E402
from phystax.bench.report import table_uncertainty_and_passk  # noqa: E402
from phystax.grader.grade import interval_score  # noqa: E402


def test_interval_score_is_proper_shape():
    y = np.array([5.0])
    assert interval_score(np.array([4.0]), np.array([6.0]), y)[0] == 2.0          # inside: width only
    assert interval_score(np.array([6.0]), np.array([8.0]), y)[0] == 2.0 + 10.0   # miss by 1 at alpha=0.2
    # an honest wider interval beats a falsely narrow one that misses
    assert interval_score(np.array([3.0]), np.array([7.0]), y)[0] < interval_score(np.array([5.5]), np.array([5.6]), y)[0]


def test_reference_intervals_on_noisy_data(tmp_path):
    r = run_reference("nonlinear_spring", 3, "noisy_oracle", 0, run_dir=str(tmp_path / "ep"))
    s = r["score"]
    assert s["coverage_80"] is not None and s["interval_score_norm"] is not None
    pred = r["score"]["per_test"][0]
    assert pred["interval_score_norm"] >= 0
    # intervals must contain the point prediction by construction
    commit = [l for l in open(os.path.join(r["run_dir"], "ledger.jsonl")) if '"commit"' in l][0]
    import json
    c = json.loads(commit)["payload"]["commitment"]["predictions"][0]
    assert all(lo <= x <= hi for lo, x, hi in zip(c["x_lo"], c["x"], c["x_hi"]))


def test_passk_table():
    rows = [{"model": "spark", "condition": "sam", "family": "power_drag", "degradation": 2, "seed": 0,
             "pass": p, "coverage_80": 0.7, "interval_score": 0.1} for p in (False, True, False)]
    t = table_uncertainty_and_passk(rows, [])
    assert "| spark | sam | 0.7000 | – | 0.1000 | 0.0% | 100.0% (3) | 3 |" in t


def test_ar1_inflation_grows_with_correlation():
    from types import SimpleNamespace as NS
    from phystax.bench.reference import ar1_inflation
    rng = np.random.default_rng(1)
    white = rng.normal(0, 1, 400)
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0, 1)
    assert ar1_inflation(NS(fun=white)) < 1.3
    assert ar1_inflation(NS(fun=ar)) > 8
    offset = 0.8 + rng.normal(0, 0.1, 400)          # constant tracking bias + small jitter
    assert ar1_inflation(NS(fun=offset)) > 20
