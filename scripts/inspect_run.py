"""See exactly what an episode measured, frame by frame.

    python scripts/inspect_run.py runs/ui/<job_id>          # or any run dir with result.json
    python scripts/inspect_run.py runs/reference/<dir> --frames 0 15 30 45 60 89

For every experiment in the run's ledger it writes, into <run_dir>/inspect/:
  <exp>_sheet.png  frames at the chosen timestamps: red cross = simulator truth,
                   green circle = what the agent was shown (SAM / classical / oracle),
                   with the per-frame confidence if the channel exposed it
  <exp>_frames.csv frame, t, body, x_true, y_true, x_shown, y_shown, conf, err_px
No API calls: truth is recomputed from the seed, the shown data is read from
sandbox/experiments.json, the video from disk.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from phystax.config import FRAME_H  # noqa: E402
from phystax.grader.ledger import Ledger  # noqa: E402
from phystax.sim.engine import simulate  # noqa: E402
from phystax.sim.world import World  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 15, 30, 45, 60, 75, 89])
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.run_dir, "result.json")))
    shown_all = json.load(open(os.path.join(a.run_dir, "sandbox", "experiments.json")))
    exps = [e["payload"] for e in Ledger(os.path.join(a.run_dir, "ledger.jsonl")).read()
            if e["kind"] == "experiment"]
    out_dir = os.path.join(a.run_dir, "inspect")
    os.makedirs(out_dir, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        world = World(res["family"], res["seed"], res["degradation"], run_dir=tmp)
    fam = world.family
    print(f"{res['family']} seed={res['seed']} level={res['degradation']} channel={res['condition']} "
          f"true form={world.hidden['form']}")
    for e in exps:
        exp_id, video = e["exp_id"], e["video"]
        shown = shown_all.get(exp_id)
        if shown is None or not os.path.exists(video):
            print(f"  {exp_id}: missing data or video ({video})")
            continue
        n = len(shown["t"])
        truth = simulate(fam, world.hidden, e["params"], n).xy
        cap = cv2.VideoCapture(video)
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        rows, tiles = [], []
        for f in range(min(n, len(frames))):
            for i, b in enumerate(fam.bodies):
                s = shown["bodies"][b]
                xs, ys = s["x"][f], s["y"][f]
                conf = s.get("conf", [None] * n)[f]
                xt, yt = truth[f, i]
                err = None if xs is None else float(np.hypot(xs - xt, ys - yt))
                rows.append([f, round(f / 30, 4), b, round(xt, 2), round(yt, 2), xs, ys, conf,
                             None if err is None else round(err, 3)])
        for f in [f for f in a.frames if f < len(frames)]:
            im = frames[f].copy()
            for i, b in enumerate(fam.bodies):
                xt, yt = truth[f, i]
                cv2.drawMarker(im, (int(xt), int(FRAME_H - yt)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                s = shown["bodies"][b]
                if s["x"][f] is not None:
                    p = (int(s["x"][f]), int(FRAME_H - s["y"][f]))
                    cv2.circle(im, p, 12, (0, 200, 0), 2)
                    c = s.get("conf", [None] * n)[f]
                    if c is not None:
                        cv2.putText(im, f"{c:.2f}", (p[0] + 14, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                else:
                    cv2.putText(im, f"{b}: not seen", (10, 44 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(im, f"{exp_id} frame {f}  t={f / 30:.2f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(cv2.resize(im, (320, 240)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
        cv2.imwrite(os.path.join(out_dir, f"{exp_id}_sheet.png"), sheet)
        with open(os.path.join(out_dir, f"{exp_id}_frames.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "t", "body", "x_true", "y_true", "x_shown", "y_shown", "conf", "err_px"])
            w.writerows(rows)
        errs = [r[-1] for r in rows if r[-1] is not None]
        seen = sum(r[5] is not None for r in rows) / max(len(rows), 1)
        print(f"  {exp_id}: params={e['params']} seen={seen:.0%} median_err_px="
              f"{(np.median(errs) if errs else float('nan')):.2f} -> {out_dir}/{exp_id}_sheet.png")


if __name__ == "__main__":
    main()
