"""Tests for the scientific machinery: forms, decomposition, weights, budget."""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("PHYSTAX_RUNS", "/tmp/phystax_test_runs")

from phystax import budget  # noqa: E402
from phystax.bench.reference import _weights, refit_episode, run_reference  # noqa: E402
from phystax.sim.laws import FAMILIES  # noqa: E402
from phystax.sim.world import World  # noqa: E402


def test_every_form_is_reachable_and_secret():
    for name, fam in FAMILIES.items():
        seen = {World(name, s).hidden["form"] for s in range(24)}
        assert seen == set(fam.forms), (name, seen)
        spec = json.dumps(World(name, 0).spec())
        assert name not in spec                  # internal id hints at the form
        for form in fam.forms.values():          # no form names or constants leak
            assert form.name not in spec, (name, form.name)
            for k in form.key_params:
                assert f'"{k}"' not in spec


@pytest.mark.parametrize("family,form", [("power_drag", "mixed"), ("nonlinear_spring", "dry"),
                                         ("power_drag", "power"), ("nonlinear_spring", "viscous")])
def test_reference_recovers_each_form_from_oracle(family, form, tmp_path):
    r = run_reference(family, 1, "oracle", 0, run_dir=str(tmp_path / f"{family}_{form}"), form=form)
    assert r["true_form"] == form
    assert r["score"]["prediction_nrmse"] < 1e-3, r["score"]
    assert r["ledger_ok"]


def test_refit_on_agent_data_matches_reference(tmp_path):
    # treat a reference run as if an agent produced it: refitting its own data
    # must reproduce its score (the design-gap term is then exactly zero)
    r = run_reference("nonlinear_spring", 2, "noisy_oracle", 0, run_dir=str(tmp_path / "ep"))
    again = refit_episode(r["run_dir"])
    assert abs(again["nrmse"] - r["score"]["prediction_nrmse"]) < 1e-6


def test_decomposition_telescopes():
    agent, on_agent, ref_c, ref_o = 0.31, 0.12, 0.05, 0.01
    terms = (ref_c - ref_o) + (on_agent - ref_c) + (agent - on_agent)
    assert abs(terms - (agent - ref_o)) < 1e-12


def test_calibrated_weights_follow_bins():
    calib = [[0.0, 0.5, 8.0], [0.5, 1.0, 2.0]]
    w = _weights([0.2, 0.9, None], 3, "calibrated", calib)
    assert w[0] == pytest.approx(1 / 8.0) and w[1] == pytest.approx(1 / 2.0)
    assert w[2] == pytest.approx(1 / 8.0)          # missing conf -> worst bin
    assert np.allclose(_weights([0.2, 0.9], 2, "uniform", calib), 1.0)


def test_budget_cap_blocks_before_spending(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSTAX_RUNS", str(tmp_path))
    monkeypatch.setenv("PHYSTAX_BUDGET_USD", "1.00")
    budget.record("meta", 0.95, "chat", "muse-spark-1.3")
    budget.check("meta", 0.04, "small call")              # fits under the cap
    with pytest.raises(budget.BudgetExceeded):
        budget.check("meta", 0.10, "too big")
    assert budget.chat_cost("muse-spark-1.3-contributor", 1_000_000, 0, 1_000_000) == pytest.approx(0.30)
    assert budget.sam_video_cost(10) == pytest.approx(0.01)   # 50-frame minimum
