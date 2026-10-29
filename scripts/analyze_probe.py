"""Frame-level analysis of the SAM probe. Costs nothing: it replays cached SAM
responses (cache_only=True) and never calls the API.

    python scripts/analyze_probe.py                 # both render styles
    python scripts/analyze_probe.py --style lab

Why this exists. Pixel RMSE is the wrong summary for dynamics: a constant
offset does not change a fitted acceleration at all, while a slow drift or
frame-to-frame jitter does. For every probe track this splits the error into
  bias      mean (dx, dy) over in-frame frames         -> harmless for g, k, alpha
  drift     linear-in-time part of the error           -> biases velocities
  jitter    what remains after removing bias + drift   -> limits accelerations
and checks two confounders behind the confidence/error correlation:
  edge      frames where the object is within one radius of the image border
            (partly cut off, so its mask centroid is pulled inward)
  time      SAM's confidence and the error can both trend over the clip; the
            partial Spearman controls for frame index and edge distance.
Correlations come with a moving-block bootstrap CI, because neighbouring
frames are not independent.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from phystax.bench.conditions import setup_boxes  # noqa: E402
from phystax.bench.reference import DESIGNS  # noqa: E402
from phystax.config import FRAME_H, FRAME_W  # noqa: E402
from phystax.sim.render import BODY_RADIUS  # noqa: E402

PHRASES = {"ball": ["ball", "circle", "yellow circle", "sphere", "dot"], "block": ["block", "rectangle"],
           "planet": ["planet", "circle", "sphere"], "marble": ["marble", "circle", "dot"]}


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    if len(a) < 5:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial_spearman(a, b, controls):
    """Spearman of a and b after regressing both (as ranks) on the controls."""
    if len(a) < 8:
        return float("nan")
    Z = np.column_stack([np.ones(len(a))] + [rank(c) for c in controls])
    ra, rb = rank(a), rank(b)
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]
    eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    if ea.std() == 0 or eb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])


def block_bootstrap_ci(a, b, block=10, n_boot=1000, seed=0):
    n = len(a)
    if n < 2 * block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n // block + 1)])[:n]
        vals.append(spearman(a[idx], b[idx]))
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) else (np.nan, np.nan)


def decompose(err_xy, t):
    """bias / drift / jitter of a 2-D error series (px)."""
    bias = err_xy.mean(0)
    X = np.c_[np.ones_like(t), t - t.mean()]
    resid = np.empty_like(err_xy)
    drift = np.zeros(2)
    for k in range(2):
        coef = np.linalg.lstsq(X, err_xy[:, k], rcond=None)[0]
        drift[k] = coef[1]
        resid[:, k] = err_xy[:, k] - X @ coef
    return bias, drift, float(np.sqrt(np.mean(np.sum(resid ** 2, 1))))


def analyze_track(name, truth_world, tr, radius):
    t = np.arange(len(truth_world)) / 30.0
    x, y = truth_world[:, 0], truth_world[:, 1]
    inside = (x > 0) & (x < FRAME_W) & (y > 0) & (y < FRAME_H)
    edge_d = np.minimum.reduce([x, FRAME_W - x, y, FRAME_H - y])
    seen = np.isfinite(tr.x)
    ok = inside & seen
    out = {"track": name, "in_frame_frames": int(inside.sum()),
           "coverage_of_visible": float(seen[inside].mean()) if inside.any() else None,
           "seen_while_out_of_frame": int((seen & ~inside).sum())}
    if ok.sum() < 5:
        return out
    e = np.c_[tr.x[ok] - x[ok], tr.y[ok] - y[ok]]
    mag = np.hypot(e[:, 0], e[:, 1])
    bias, drift, jitter = decompose(e, t[ok])
    edge = edge_d[ok] < radius
    out.update(rmse_px=float(np.sqrt(np.mean(mag ** 2))), bias_px=[round(float(v), 3) for v in bias],
               drift_px_per_s=[round(float(v), 3) for v in drift], jitter_rms_px=round(jitter, 3),
               edge_frames=int(edge.sum()),
               err_edge_px=float(mag[edge].mean()) if edge.any() else None,
               err_interior_px=float(mag[~edge].mean()) if (~edge).any() else None)
    c = tr.conf[ok]
    good = np.isfinite(c)
    if good.sum() >= 8:
        cc, mm, ff, dd = c[good], mag[good], np.arange(len(c))[good], edge_d[ok][good]
        out.update(conf_mean=float(cc.mean()), spearman_conf_err=spearman(cc, mm),
                   spearman_ci95=block_bootstrap_ci(cc, mm),
                   partial_spearman_ctrl_time_edge=partial_spearman(cc, mm, [ff, dd]),
                   spearman_conf_time=spearman(cc, ff), spearman_err_time=spearman(mm, ff))
    return out


def main() -> None:
    from phystax.perception.sam_client import SamError, segment_video
    from phystax.perception.tracks import build_track, track_for_object
    from phystax.sim.world import World
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["lab", "abstract", "both"], default="both")
    a = ap.parse_args()
    results = []
    for style in (["lab", "abstract"] if a.style == "both" else [a.style]):
        os.environ["PHYSTAX_RENDER_STYLE"] = style
        for fam in ["power_drag", "nonlinear_spring", "central_power"]:
            w = World(fam, seed=0, degradation=0, run_dir=f"runs/probe_analysis/{style}/{fam}")
            exp = w.run_experiment(DESIGNS[fam][1], 3.0)
            prompts = [(i, b, "concept", ph) for i, b in enumerate(w.family.bodies) for ph in PHRASES.get(b, [b])]
            prompts.append((None, None, "box", setup_boxes(w.family, exp.params, FRAME_W, FRAME_H)))
            for i, b, mode, prompt in prompts:
                try:
                    call = segment_video(exp.video_path, prompt, include_confidence=True, n_frames=exp.n_frames,
                                         object_prompt=(mode == "box"), cache_only=True)
                except SamError as err:
                    print(f"[{style}] {fam} {mode} {prompt[:30]!r}: not in cache ({err}); "
                          "re-render differs from the probe clip or the probe was not run")
                    continue
                bodies = [(i, b)] if mode == "concept" else list(enumerate(w.family.bodies))
                for bi, bn in bodies:
                    if mode == "concept":
                        tr = build_track(bn, call.records, exp.n_frames, FRAME_H,
                                         "largest" if bi in w.family.fixed_bodies else "moving")
                    else:
                        tr = track_for_object(bn, call.records, bi + 1, exp.n_frames, FRAME_H)
                    r = analyze_track(f"{style}/{fam}/{bn}/{mode}:{prompt if mode == 'concept' else 'box'}",
                                      exp.traj.xy[:, bi, :], tr, BODY_RADIUS.get(bn, 14))
                    r.update(style=style, family=fam, body=bn, mode=mode, instances=tr.n_candidates)
                    results.append(r)
    os.makedirs("runs/probe_analysis", exist_ok=True)
    json.dump(results, open("runs/probe_analysis/frames.json", "w"), indent=2, default=float)
    print("| track | vis. cov | RMSE | bias (dx,dy) | jitter | edge vs interior err | rho(conf,err) [95% CI] | partial rho | rho(conf,t) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "rmse_px" not in r:
            print(f"| {r['track']} | {r.get('coverage_of_visible')} | not detected | | | | | | |")
            continue
        ci = r.get("spearman_ci95", (np.nan, np.nan))
        print(f"| {r['track']} | {r['coverage_of_visible']:.2f} | {r['rmse_px']:.2f} | {r['bias_px']} | {r['jitter_rms_px']:.2f} | "
              f"{r['err_edge_px'] if r['err_edge_px'] is None else round(r['err_edge_px'], 2)} vs {r['err_interior_px']:.2f} | "
              f"{r.get('spearman_conf_err', float('nan')):.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
              f"{r.get('partial_spearman_ctrl_time_edge', float('nan')):.2f} | {r.get('spearman_conf_time', float('nan')):.2f} |")
    print("\nwrote runs/probe_analysis/frames.json (no API calls were made)")


if __name__ == "__main__":
    main()
