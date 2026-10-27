"""Experiment 1 (do this before any agent run): how good is SAM 3.1 as an
instrument, and is its confidence informative?

For each family x degradation level x seed, render the reference design,
measure with SAM and with the classical tracker, and record centroid RMSE,
coverage, and Spearman(confidence, error). If confidence does not
anti-correlate with error, the "confidence recovers the tax" claim is dead
on arrival and you should find that out on Day 2, not Day 8.

    python -m phystax.bench.measure_perception --seeds 3 --levels 0 1 2 3
"""
import argparse
import csv
import os

from ..config import FRAME_H
from ..perception import classical
from ..perception.tracks import build_track, centroid_error_px
from ..sim.world import World
from .reference import DESIGNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(DESIGNS))
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-sam", action="store_true", help="classical only (offline)")
    ap.add_argument("--out", default="runs/perception.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for fam in a.families:
        for lvl in a.levels:
            for seed in range(a.seeds):
                w = World(fam, seed, lvl, run_dir=f"runs/perception/{fam}_d{lvl}_s{seed}")
                for k, params in enumerate(DESIGNS[fam]):
                    exp = w.run_experiment(params, 3.0)
                    cl = classical.track_video(exp.video_path, w.family.bodies, FRAME_H)
                    for i, b in enumerate(w.family.bodies):
                        base = {"family": fam, "level": lvl, "seed": seed, "exp": k, "body": b}
                        rows.append({**base, "tracker": "classical",
                                     **centroid_error_px(cl[b], exp.traj.xy[:, i, :])})
                        if not a.no_sam:
                            from ..perception.sam_client import segment_video
                            call = segment_video(exp.video_path, b, include_confidence=True)
                            mode = "largest" if i in w.family.fixed_bodies else "moving"
                            tr = build_track(b, call.records, exp.n_frames, FRAME_H, mode)
                            rows.append({**base, "tracker": "sam3.1", "instances": tr.n_candidates,
                                         **centroid_error_px(tr, exp.traj.xy[:, i, :])})
                print(fam, lvl, seed, "done")
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
