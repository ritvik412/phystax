"""Which SAM 3.1 prompting mode can actually measure our objects?

Day-1 finding: concept prompts "ball", "planet", "marble" returned zero
records on our renders, while "block" was tracked at 0.81 px. This probe
re-renders the same smoke clips (deterministic per seed) and measures, per
body: several noun phrases (concept mode) and a frame-0 box prompt (tracking
mode, box from the known experiment set-up). It prints instances found,
coverage, centroid error vs truth, and confidence/error correlation.

    python scripts/probe_sam.py            # prints the plan and estimated cost
    python scripts/probe_sam.py --yes      # runs it (about $0.40 at list price)

Results: runs/probe/probe.json and runs/probe/<family>_<mode>.png contact sheets.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from phystax import budget  # noqa: E402
from phystax.bench.conditions import setup_boxes  # noqa: E402
from phystax.bench.reference import DESIGNS  # noqa: E402
from phystax.config import FRAME_H, FRAME_W  # noqa: E402
from phystax.perception.sam_client import segment_video  # noqa: E402
from phystax.perception.tracks import build_track, centroid_error_px, track_for_object  # noqa: E402
from phystax.sim.world import World  # noqa: E402

PHRASES = {
    "ball": ["ball", "circle", "yellow circle", "sphere", "dot"],
    "block": ["block", "rectangle"],
    "planet": ["planet", "circle", "sphere"],
    "marble": ["marble", "circle", "dot"],
}


def sheet(video, truth_px, track_xy, path, frames=(0, 30, 60, 89)):
    cap = cv2.VideoCapture(video)
    imgs = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        imgs.append(im)
    tiles = []
    for f in [f for f in frames if f < len(imgs)]:
        im = imgs[f].copy()
        for (xt, yt), (xs, ys) in zip(truth_px[f], track_xy[f]):
            cv2.drawMarker(im, (int(xt), int(yt)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            if np.isfinite(xs):
                cv2.circle(im, (int(xs), int(FRAME_H - ys)), 12, (0, 200, 0), 2)
        cv2.putText(im, f"f{f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (320, 240)))
    cv2.imwrite(path, np.hstack(tiles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    plan = []
    for fam in ["power_drag", "nonlinear_spring", "central_power"]:
        w = World(fam, seed=0, degradation=0, run_dir=f"runs/probe/{fam}")
        exp = w.run_experiment(DESIGNS[fam][1], 3.0)
        for i, b in enumerate(w.family.bodies):
            for ph in PHRASES.get(b, [b]):
                plan.append((fam, w, exp, i, b, "concept", ph))
        plan.append((fam, w, exp, None, None, "box", setup_boxes(w.family, exp.params, FRAME_W, FRAME_H)))
    est = sum(budget.sam_video_cost(p[2].n_frames) for p in plan)
    print(f"{len(plan)} SAM requests, estimated ${est:.2f} (cached ones cost nothing). {budget.summary()}")
    if not a.yes:
        print("Re-run with --yes to spend it.")
        return
    results = []
    for fam, w, exp, i, b, mode, prompt in plan:
        call = segment_video(exp.video_path, prompt, include_confidence=True, n_frames=exp.n_frames,
                             object_prompt=(mode == "box"))
        bodies = [(i, b)] if mode == "concept" else list(enumerate(w.family.bodies))
        xy_all = np.full((exp.n_frames, len(bodies), 2), np.nan)
        for k, (bi, bn) in enumerate(bodies):
            if mode == "concept":
                tr = build_track(bn, call.records, exp.n_frames, FRAME_H,
                                 "largest" if bi in w.family.fixed_bodies else "moving")
            else:
                tr = track_for_object(bn, call.records, bi + 1, exp.n_frames, FRAME_H)
            xy_all[:, k, 0], xy_all[:, k, 1] = tr.x, tr.y
            err = centroid_error_px(tr, exp.traj.xy[:, bi, :])
            row = {"family": fam, "body": bn, "mode": mode, "prompt": prompt if mode == "concept" else "frame-0 box",
                   "instances": tr.n_candidates, "latency_s": round(call.latency_s, 1), "usd": round(call.usd, 4), **err}
            results.append(row)
            print(json.dumps(row))
        tag = f"{fam}_{mode}_{prompt.replace(' ', '_')}" if mode == "concept" else f"{fam}_box"
        sheet(exp.video_path, exp.render.truth_px[:, [bi for bi, _ in bodies], :], xy_all,
              f"runs/probe/{tag}.png")
    json.dump(results, open("runs/probe/probe.json", "w"), indent=2)
    print("wrote runs/probe/probe.json and contact sheets;", budget.summary())


if __name__ == "__main__":
    main()
