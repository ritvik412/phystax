"""Instrument study (H-A): SAM 3.1 as a measuring device for moving objects.

Replicates the Oct 29 single-clip probe across seeds and camera levels:
  * prompt mode: frame-0 box (one request per clip) vs a generic shape word
    ("circle" / "rectangle"; specific nouns proved brittle)
  * error split: interior frames vs frames where SAM's box touches the border
  * confidence: per mode, Spearman with error after removing frames' time
    trend, and an interior-frame calibration (confidence bin -> median error)
    written from BOX mode only (concept-mode confidence tracked frame index)
  * classical background subtraction as the non-foundation baseline

    python -m phystax.bench.measure_perception --seeds 3 --levels 0 1 2 3      # ~$4
    python -m phystax.bench.measure_perception --seeds 1 --levels 0 3 --no-sam  # offline

Outputs: runs/perception.csv (one row per clip x body x tracker),
runs/calibration.json (box mode), and a printed summary by level and mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from ..config import FRAME_H, FRAME_W
from ..perception import classical
from ..perception.tracks import Track, build_track, track_for_object
from ..sim.render import BODY_RADIUS
from ..sim.world import World
from .conditions import CONCEPT_WORD, setup_boxes
from .reference import DESIGNS


def _rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def _partial_rho(c, e, t):
    if len(c) < 8:
        return float("nan")
    Z = np.column_stack([np.ones(len(c)), _rank(t)])
    rc, re = _rank(c), _rank(e)
    ec = rc - Z @ np.linalg.lstsq(Z, rc, rcond=None)[0]
    ee = re - Z @ np.linalg.lstsq(Z, re, rcond=None)[0]
    return float(np.corrcoef(ec, ee)[0, 1]) if ec.std() > 0 and ee.std() > 0 else float("nan")


def score(tr: Track, truth: np.ndarray, radius: float) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    x, y = truth[:, 0], truth[:, 1]
    inside = (x > 0) & (x < FRAME_W) & (y > 0) & (y < FRAME_H)
    edge = np.minimum.reduce([x, FRAME_W - x, y, FRAME_H - y]) < radius
    seen = np.isfinite(tr.x)
    err = np.hypot(tr.x - x, tr.y - y)
    interior = inside & ~edge & seen
    border = inside & edge & seen
    trunc = tr.truncated if tr.truncated is not None else np.zeros(len(x), bool)
    row = {
        "coverage_visible": float(seen[inside].mean()) if inside.any() else None,
        "rmse_interior_px": float(np.sqrt(np.mean(err[interior] ** 2))) if interior.any() else None,
        "rmse_border_px": float(np.sqrt(np.mean(err[border] ** 2))) if border.any() else None,
        "n_border_frames": int(border.sum()),
        "trunc_flag_recall": float(trunc[border].mean()) if border.any() else None,     # border frames SAM's box flags
        "trunc_flag_false": float(trunc[interior].mean()) if interior.any() else None,  # interior frames wrongly flagged
    }
    conf_ok = interior & np.isfinite(tr.conf)
    if conf_ok.sum() >= 8:
        row["partial_rho_conf_err"] = _partial_rho(tr.conf[conf_ok], err[conf_ok], np.flatnonzero(conf_ok))
    return row, err, interior, conf_ok


def write_calibration(pairs, path, n_bins=8) -> dict:
    c = np.array([p[0] for p in pairs])
    e = np.array([p[1] for p in pairs])
    edges = np.unique(np.quantile(c, np.linspace(0, 1, n_bins + 1)))
    bins = [[float(lo), float(hi), float(np.median(e[(c >= lo) & (c <= hi)]))]
            for lo, hi in zip(edges[:-1], edges[1:]) if ((c >= lo) & (c <= hi)).any()]
    rc, re = _rank(c), _rank(e)
    out = {"mode": "box", "frames": "interior only", "bins": bins, "n_frames": int(len(c)),
           "spearman_conf_vs_error": float(np.corrcoef(rc, re)[0, 1]),
           "note": "usable as weights only if bin errors fall as confidence rises"}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(DESIGNS))
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-sam", action="store_true", help="classical only (offline)")
    ap.add_argument("--out", default="runs/perception.csv")
    ap.add_argument("--calibration", default="runs/calibration.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows, box_pairs = [], []
    for fam in a.families:
        for lvl in a.levels:
            for seed in range(a.seeds):
                w = World(fam, seed, lvl, run_dir=f"runs/perception/{fam}_d{lvl}_s{seed}")
                for k, params in enumerate(DESIGNS[fam]):
                    exp = w.run_experiment(params, 3.0)
                    trackers: dict[str, dict[str, Track]] = {
                        "classical": classical.track_video(exp.video_path, w.family.bodies, FRAME_H)}
                    if not a.no_sam:
                        from ..perception.sam_client import segment_video
                        box = segment_video(exp.video_path, setup_boxes(w.family, exp.params, FRAME_W, FRAME_H),
                                            include_confidence=True, n_frames=exp.n_frames, object_prompt=True)
                        trackers["sam_box"] = {b: track_for_object(b, box.records, i + 1, exp.n_frames, FRAME_H)
                                               for i, b in enumerate(w.family.bodies)}
                        trackers["sam_concept"] = {}
                        for i, b in enumerate(w.family.bodies):
                            call = segment_video(exp.video_path, CONCEPT_WORD.get(b, b), include_confidence=True,
                                                 n_frames=exp.n_frames)
                            trackers["sam_concept"][b] = build_track(
                                b, call.records, exp.n_frames, FRAME_H,
                                "largest" if i in w.family.fixed_bodies else "moving")
                    for tname, tracks in trackers.items():
                        for i, b in enumerate(w.family.bodies):
                            r, err, interior, conf_ok = score(tracks[b], exp.traj.xy[:, i, :], BODY_RADIUS.get(b, 14))
                            rows.append({"family": fam, "level": lvl, "seed": seed, "exp": k, "body": b,
                                         "tracker": tname, "form": w.hidden["form"], **r})
                            if tname == "sam_box":
                                box_pairs += list(zip(tracks[b].conf[conf_ok].tolist(), err[conf_ok].tolist()))
                print(f"{fam} level {lvl} seed {seed} done")
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)\n")
    summary(rows)
    if len(box_pairs) >= 50:
        cal = write_calibration(box_pairs, a.calibration)
        print(f"\nbox-mode calibration: n={cal['n_frames']} interior frames, "
              f"spearman(conf, error) = {cal['spearman_conf_vs_error']:.3f} -> {a.calibration}")
        for lo, hi, med in cal["bins"]:
            print(f"  confidence {lo:.3f}-{hi:.3f}: median error {med:.2f} px")


def summary(rows: list[dict]) -> None:
    g = defaultdict(list)
    for r in rows:
        g[(r["level"], r["tracker"])].append(r)
    print("| level | tracker | n | coverage (visible) | interior RMSE px | border RMSE px | border frames flagged | partial rho(conf,err) |")
    print("|---|---|---|---|---|---|---|---|")

    def med(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
        return f"{np.median(v):.2f}" if v else "–"

    for (lvl, tname), rs in sorted(g.items()):
        print(f"| {lvl} | {tname} | {len(rs)} | {med(rs, 'coverage_visible')} | {med(rs, 'rmse_interior_px')} | "
              f"{med(rs, 'rmse_border_px')} | {med(rs, 'trunc_flag_recall')} | {med(rs, 'partial_rho_conf_err')} |")


if __name__ == "__main__":
    main()
