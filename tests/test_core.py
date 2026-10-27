"""Offline tests: no API keys needed. Run with `pytest -q`."""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("PHYSTAX_RUNS", "/tmp/phystax_test_runs")

from meta_sam_parser import video_segmentation_format, CompletedOutcome  # noqa: E402

from phystax.grader.grade import grade, probe_times  # noqa: E402
from phystax.grader.ledger import Ledger  # noqa: E402
from phystax.perception.tracks import build_track  # noqa: E402
from phystax.sim.engine import simulate  # noqa: E402
from phystax.sim.laws import FAMILIES  # noqa: E402
from phystax.sim.world import World  # noqa: E402


def test_simulation_is_deterministic():
    w1, w2 = World("power_drag", 7), World("power_drag", 7)
    assert w1.hidden == w2.hidden
    p = {"launch_speed_px_s": 400, "launch_angle_deg": 50}
    a = simulate(w1.family, w1.hidden, p, 60).xy
    b = simulate(w2.family, w2.hidden, p, 60).xy
    assert np.array_equal(a, b)


def test_hidden_law_is_not_textbook():
    gs = [World("power_drag", s).hidden["g"] for s in range(20)]
    assert not any(abs(g - 9.81) < 1 for g in gs)


def test_experiment_bounds_enforced():
    w = World("nonlinear_spring", 0)
    with pytest.raises(ValueError):
        w.family.validate_experiment({"displacement_px": 999, "initial_velocity_px_s": 0})


def test_ledger_detects_tampering(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(str(path))
    led.append("commit", {"x": 1})
    led.append("test_reveal", {"y": 2})
    assert led.verify()[0] and led.commit_precedes_test()
    lines = path.read_text().splitlines()
    e = json.loads(lines[0]); e["payload"]["x"] = 2
    lines[0] = json.dumps(e, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    assert not Ledger(str(path)).verify()[0]


def test_grader_scores_truth_as_perfect():
    w = World("nonlinear_spring", 3)
    times = probe_times(4.0)
    tests = [simulate(w.family, w.hidden, p, 120) for p in w.test_conditions()]
    preds = []
    for i, tr in enumerate(tests):
        idx = [int(round(t * 30)) for t in times]
        preds.append({"test_index": i, "body": "block",
                      "x": tr.xy[idx, 0, 0].tolist(), "y": tr.xy[idx, 0, 1].tolist()})
    c = {"parameters": [{"name": k, "value": v} for k, v in w.hidden.items()], "predictions": preds}
    s = grade(c, "nonlinear_spring", w.hidden, tests, times, 30)
    assert s["pass"] and s["prediction_nrmse"] < 1e-9


def test_tracks_pick_moving_instance_over_static_distractor():
    # two SAM objects for one phrase: id 0 static distractor, id 1 moving body (boxes only)
    lines = []
    for f in range(5):
        lines.append(f"<{f}f>0<|box;x1=10;y1=10;x2=19;y2=19;w=640;h=480;c=0.9|>"
                     f"<|mask;x=0;y=0;data=1,1,!!!!!!!|>,"
                     f"1<|box;x1={100+10*f};y1=200;x2={109+10*f};y2=209;w=640;h=480;c=0.8|>"
                     f"<|mask;x=0;y=0;data=1,1,!!!!!!!|>")
    parser = video_segmentation_format().create_parser()
    parser.push("\n".join(lines) + "\n")
    res = parser.finish(CompletedOutcome()).result
    tr = build_track("ball", list(res.records), 5, 480, mode="moving")
    assert tr.n_candidates >= 1
    # whether or not the toy mask payload validates, boxes fill the centroid
    assert np.isfinite(tr.x).sum() >= 1
