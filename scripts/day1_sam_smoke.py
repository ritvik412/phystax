"""Day 1 smoke test: one rendered clip per family -> SAM 3.1 -> parsed tracks.

Answers the Day-1 unknowns with real numbers: latency per clip, frames billed,
whether `c` confidence comes back, how many instances a phrase returns
(distractors), and centroid error against simulator truth.

    export MODEL_API_KEY=...
    python scripts/day1_sam_smoke.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phystax.bench.reference import DESIGNS  # noqa: E402
from phystax.config import FRAME_H  # noqa: E402
from phystax.perception.sam_client import estimated_cost_usd, segment_video  # noqa: E402
from phystax.perception.tracks import build_track, centroid_error_px  # noqa: E402
from phystax.sim.world import World  # noqa: E402

report = []
for fam in ["power_drag", "nonlinear_spring", "central_power"]:
    w = World(fam, seed=0, degradation=0, run_dir=f"runs/smoke/{fam}")
    exp = w.run_experiment(DESIGNS[fam][1], 3.0)
    for i, body in enumerate(w.family.bodies):
        call = segment_video(exp.video_path, body, include_confidence=True, use_cache=False)
        mode = "largest" if i in w.family.fixed_bodies else "moving"
        tr = build_track(body, call.records, exp.n_frames, FRAME_H, mode)
        err = centroid_error_px(tr, exp.traj.xy[:, i, :])
        confs = [r.confidence for r in call.records if getattr(r, "confidence", None) is not None]
        row = {"family": fam, "body": body, "latency_s": round(call.latency_s, 2),
               "frames_billed": call.frames_billed, "est_cost_usd": estimated_cost_usd(exp.n_frames),
               "records": len(call.records), "instances": tr.n_candidates,
               "confidence_returned": bool(confs), "diagnostics": [str(d) for d in call.diagnostics][:3],
               **err}
        report.append(row)
        print(json.dumps(row))
json.dump(report, open("runs/smoke/report.json", "w"), indent=2)
