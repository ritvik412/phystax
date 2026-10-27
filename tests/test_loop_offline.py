"""End-to-end plumbing test of the two-step loop with a scripted fake model."""
import json
import os
from types import SimpleNamespace as NS

os.environ.setdefault("PHYSTAX_RUNS", "/tmp/phystax_test_runs")

from phystax.agent.loop import run_episode  # noqa: E402


class FakeModel:
    """Turn 1: two experiments. Turn 2: a fit. Turn 3: stop. Commit: JSON."""
    def __init__(self):
        self.spec = NS(name="fake", model_id="fake-0")
        self.usage = NS(calls=0)
        self.turn = 0

    def chat(self, messages, tools=None, response_format=None, max_tokens=0):
        self.usage.calls += 1
        if response_format is not None:
            return NS(content=json.dumps({
                "law_equation": "x'' = -k x - c x^3 - b x'",
                "parameters": [{"name": "k", "value": 10.0}, {"name": "c", "value": 0.0},
                               {"name": "b", "value": 0.1}],
                "predictions": [{"test_index": i, "body": "block", "x": [320.0] * 6, "y": [240.0] * 6}
                                for i in range(2)],
                "self_confidence": 0.2, "rationale": "scripted"}), tool_calls=None)
        self.turn += 1
        if self.turn == 1:
            calls = [NS(id=f"c{i}", function=NS(name="run_experiment", arguments=json.dumps(
                {"params": {"displacement_px": d, "initial_velocity_px_s": 0}, "duration_s": 2.0})))
                for i, d in enumerate((80, 160))]
            return NS(content="", tool_calls=calls)
        if self.turn == 2:
            code = "print(sorted(data.keys()), len(data['e00']['t']))"
            return NS(content="", tool_calls=[NS(id="c9", function=NS(name="run_python",
                                                                       arguments=json.dumps({"code": code})))])
        return NS(content="DONE", tool_calls=None)


def test_two_step_loop_and_ledger():
    r = run_episode("nonlinear_spring", 0, "oracle", FakeModel(), degradation=0, tag="_test")
    assert r["n_experiments"] == 2
    assert r["ledger_ok"], r["ledger_msg"]
    assert r["score"]["prediction_nrmse"] is not None
    out = open(os.path.join(r["run_dir"], "sandbox", "_cell.py")).read()
    assert "data" in out
