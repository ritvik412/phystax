"""P1 ellipsoid protocol: the SDP margin, the logistic fit and the sealing order."""
import json
import os

import numpy as np


def test_margin_sign_matches_geometry():
    from phystax.open_problems.ellipsoid import fit_margin
    rng = np.random.default_rng(0)
    assert fit_margin(rng.standard_normal((4, 6))) > 1e-6          # few points: an ellipsoid fits
    assert fit_margin(rng.standard_normal((40, 6))) <= 1e-6        # n > d(d+1)/2 = 21: impossible


def test_logistic_fit_recovers_centre():
    from phystax.open_problems.ellipsoid import logistic_fit
    ns = np.arange(10, 40, 2.0)
    p = 1 / (1 + np.exp((ns - 25) / 3))
    n50, w = logistic_fit(ns, np.round(p * 200), 200)
    assert abs(n50 - 25) < 0.5 and abs(w - 3) < 0.6


def test_protocol_seals_before_measuring_targets(tmp_path):
    from phystax.open_problems.ellipsoid import run_protocol
    r = run_protocol([5, 6], [7], trials=6, workers=2, out_dir=str(tmp_path))
    kinds = [json.loads(l)["kind"] for l in open(os.path.join(tmp_path, "ledger.jsonl"))]
    assert kinds.index("sealed_predictions") < kinds.index("target_measurements") < kinds.index("grade")
    assert r["ledger_ok"]


def test_predictive_credit_three_arms(tmp_path, monkeypatch):
    """Scripted donor: one sandbox call, then a forecast. Better cards give
    better forecasts here by construction, so credit must come out positive."""
    import sys
    from types import SimpleNamespace as NS
    sys.path.insert(0, os.path.dirname(__file__))
    from test_loop_offline import FakeModel
    from phystax.agent.loop import run_episode
    from phystax.bench.reference import run_reference
    from phystax.open_problems import predictive_credit as P
    runs = str(tmp_path)
    monkeypatch.setenv("PHYSTAX_RUNS", runs)
    run_episode("nonlinear_spring", 0, "oracle", FakeModel(), degradation=0,
                run_dir=os.path.join(runs, "nonlinear_spring_s0_d0", "oracle_fake"))
    run_reference("nonlinear_spring", 0, "oracle", 0,
                  run_dir=os.path.join(runs, "reference", "nonlinear_spring_d0_s0_oracle_designed"))

    class Donor:
        def __init__(self):
            self.n = 0
        def chat(self, messages, tools=None, response_format=None, max_tokens=0):
            text = messages[0]["content"]
            if tools and self.n == 0:
                self.n += 1
                return NS(content="", tool_calls=[NS(id="c1", function=NS(name="run_python", arguments='{"code": "print(1)"}'))])
            if tools:
                return NS(content="done", tool_calls=None)
            off = 200.0 if "previous scientist" not in text else (20.0 if '"rationale"' in text else 0.5)
            pts = [320.0 + off] * 6
            return NS(content=json.dumps({"predictions": [{"test_index": i, "body": "block", "x": pts, "y": [240.0] * 6}
                                                          for i in range(2)]}), tool_calls=None)

    s = P.run(lambda: Donor(), runs=runs, workers=2, out=os.path.join(runs, "pc.jsonl"))
    assert s["episodes"] == 1
    assert s["predictive_credit"]["median"] > 0 and s["gap_to_reference"]["median"] > 0


def test_resume_after_failed_seal(tmp_path):
    """A crash after calibration resumes without re-measuring calibration."""
    from phystax.grader.ledger import Ledger
    from phystax.open_problems.ellipsoid import _j, run_protocol, summarise, measure
    out = str(tmp_path)
    led = Ledger(os.path.join(out, "ledger.jsonl"))
    led.append("protocol", {"problem": "test"})
    cal = summarise(measure([5, 6], trials=6, workers=2))
    led.append("calibration", _j({d: {k: v for k, v in r.items() if k != "curve"} for d, r in cal.items()}))
    r = run_protocol([5, 6], [7], trials=6, workers=2, out_dir=out, resume=True)
    kinds = [json.loads(l)["kind"] for l in open(os.path.join(out, "ledger.jsonl"))]
    assert kinds == ["protocol", "calibration", "resumed", "sealed_predictions", "target_measurements", "grade"]
    assert r["ledger_ok"]
