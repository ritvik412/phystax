"""Build the static replay site from finished runs. No API calls, no keys.

    python scripts/build_site.py                 # -> site/  (open site/index.html)
    python scripts/build_site.py --episode runs/power_drag_s1_d2/sam_box_Muse_Spark_1.3

Reads runs/reference.jsonl, runs/agent.jsonl, runs/perception.csv,
runs/calibration.json, runs/real/**/pendulum.json and one agent episode
(ledger, measured tracks, videos). Writes site/index.html, site/data.js and
site/media/*.mp4. Upload the site/ folder to a static Hugging Face Space.
Everything shown is read from these files; nothing is typed in by hand except
the naming-probe table, which is the recorded Oct 29 probe result.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
from collections import defaultdict
from statistics import median

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
sys.path.insert(0, ROOT)
TEMPLATE = os.path.join(ROOT, "phystax", "web", "site", "index.html")

# Recorded probe (docs/RESEARCH_AUDIT.md, Oct 29): same frames, different words.
NAMING_PROBE = {
    "rows": ["ball", "planet", "block", "marble", "circle / dot / rectangle", "box on frame 0"],
    "lab": [True, True, False, False, True, True],
    "abstract": [False, False, True, False, True, True],
}


def jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def boot_ci(xs, n_boot=2000, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return [float(xs[0]), float(xs[0])] if len(xs) else [None, None]
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(xs, (n_boot, len(xs))), axis=1)
    return [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]


def decomposition(ref, agent):
    r_idx = {(r["family"], r["level"], r["seed"], r["condition"]): r["nrmse"]
             for r in ref if r.get("nrmse") is not None and r.get("design", "designed") == "designed"}
    acc = defaultdict(lambda: defaultdict(list))
    for a in agent:
        if a.get("nrmse") is None:
            continue
        k = (a["family"], a["degradation"], a["seed"])
        rc, ro = r_idx.get(k + (a["condition"],)), r_idx.get(k + ("oracle",))
        oa = (a.get("ref_on_agent") or {}).get("nrmse")
        if None in (rc, ro, oa):
            continue
        d = acc[(a["model"], a["condition"])]
        d["measurement"].append(rc - ro)
        d["design"].append(oa - rc)
        d["reasoning"].append(a["nrmse"] - oa)
        d["total"].append(a["nrmse"] - ro)
    out = []
    for (m, c), d in sorted(acc.items()):
        out.append({"model": m, "channel": c, "n": len(d["total"]),
                    **{k: {"median": float(median(v)), "ci": boot_ci(v)} for k, v in d.items()}})
    return out


def agent_summary(agent):
    g = defaultdict(list)
    for a in agent:
        if a.get("nrmse") is not None:
            g[(a["model"], a["condition"])].append(a)
    return [{"model": m, "channel": c, "n": len(rs),
             "median_error": float(median(r["nrmse"] for r in rs)),
             "pass_rate": sum(bool(r["pass"]) for r in rs) / len(rs),
             "coverage": float(median(r["coverage_80"] for r in rs if r.get("coverage_80") is not None))
             if any(r.get("coverage_80") is not None for r in rs) else None,
             "cost_usd": float(sum(r.get("cost_usd", 0) or 0 for r in rs))}
            for (m, c), rs in sorted(g.items())]


def coverage(ref):
    g = defaultdict(list)
    for r in ref:
        g[r["condition"]].append(r)
    out = []
    for c, rs in sorted(g.items()):
        cov = [r["coverage_80"] for r in rs if r.get("coverage_80") is not None]
        iid = [r["coverage_80_iid"] for r in rs if r.get("coverage_80_iid") is not None]
        out.append({"channel": c, "n": len(rs), "corrected": float(median(cov)) if cov else None,
                    "textbook": float(median(iid)) if iid else None,
                    "pass_rate": sum(bool(r.get("pass")) for r in rs) / len(rs)})
    return out


def perception(path):
    if not os.path.exists(path):
        return []
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        v = r.get("rmse_interior_px")
        if v not in (None, "", "None"):
            g[(int(r["level"]), r["tracker"])].append(float(v))
    return [{"level": lvl, "tracker": t, "median_px": float(median(v)), "n": len(v)}
            for (lvl, t), v in sorted(g.items())]


def pendulums(pattern):
    out = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        r = json.load(open(p))
        parts = p.split(os.sep)
        label = "/".join(parts[-3:-1]) if "pendulum_" in parts[-3] else parts[-2]
        d = r["damping_test"]
        out.append({"label": label, "T0": r["T0_s"], "T0_se": r["T0_se_s"], "cycles": r["cycles"],
                    "amplitude_deg": r["amplitude_deg"], "fps": r.get("fps"),
                    "isochronism_dbic": r["isochronism_test"]["bic_isochronous"]
                    - r["isochronism_test"]["bic_nonlinear_pendulum"],
                    "damping_bic": {k: v["bic"] for k, v in d.items()}, "damping_best": r["damping_best"],
                    "damping_params": {k: v["params"] for k, v in d.items()},
                    "L_eff": r["effective_length_if_g_9.81"], "series": r.get("series")})
    return out


def pick_episode(explicit=None):
    cands = [explicit] if explicit else [os.path.dirname(p) for p in glob.glob("runs/*_s*_d*/*/result.json")]
    best = None
    for d in cands:
        try:
            res = json.load(open(os.path.join(d, "result.json")))
        except (OSError, json.JSONDecodeError):
            continue
        if res.get("model") in (None, "reference") or res.get("score", {}).get("prediction_nrmse") is None:
            continue
        key = (res["condition"] != "sam_box", not res["score"]["pass"], res["score"]["prediction_nrmse"])
        if best is None or key < best[0]:
            best = (key, d, res)
    return (best[1], best[2]) if best else (None, None)


def episode_payload(run_dir, res, media_dir):
    ledger = jsonl(os.path.join(run_dir, "ledger.jsonl"))
    shown = json.load(open(os.path.join(run_dir, "sandbox", "experiments.json")))
    kinds = defaultdict(list)
    for e in ledger:
        kinds[e["kind"]].append(e)
    os.makedirs(media_dir, exist_ok=True)

    def copy(video):
        name = os.path.basename(video)
        src = video if os.path.exists(video) else os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(media_dir, name))
            return "media/" + name
        return None

    exps = []
    for e in kinds["experiment"]:
        p = e["payload"]
        exps.append({"id": p["exp_id"], "params": p["params"], "video": copy(p["video"]),
                     "measured": shown.get(p["exp_id"], {}).get("bodies", {})})
    commit = kinds["commit"][-1]["payload"] if kinds["commit"] else {}
    reveal = kinds["test_reveal"][-1]["payload"] if kinds["test_reveal"] else {"videos": []}
    grade = kinds["grade"][-1]["payload"] if kinds["grade"] else {}
    secret = kinds["secret_reveal"][-1]["payload"] if kinds["secret_reveal"] else {}
    start = kinds["episode_start"][0]["payload"] if kinds["episode_start"] else {}
    sealed_at = kinds["commit"][-1]["ts"] if kinds["commit"] else None
    revealed_at = kinds["test_reveal"][-1]["ts"] if kinds["test_reveal"] else None
    return {"sealed_at": sealed_at, "revealed_at": revealed_at,
            "system": start.get("spec", {}).get("system"), "description": start.get("spec", {}).get("description"),
            "channel": res["condition"], "model": res["model"], "secret_hash": start.get("secret_law_sha256"),
            "experiments": exps, "commit": {"sha256": commit.get("sha256"), **(commit.get("commitment") or {})},
            "tests": [copy(v) for v in reveal.get("videos", [])], "probe_times": res.get("probe_times", []),
            "grade": {"error": grade.get("prediction_nrmse"), "pass": grade.get("pass"),
                      "coverage": grade.get("coverage_80")},
            "truth": {"form": secret.get("hidden", {}).get("form"),
                      "constants": {k: v for k, v in secret.get("hidden", {}).items() if k != "form"}},
            "ledger_entries": len(ledger), "ledger_ok": res.get("ledger_ok"),
            "cost_usd": res.get("cost_usd")}


def open_problem_p1(base="runs/open_problems/ellipsoid"):
    rp, gp = os.path.join(base, "result.json"), os.path.join(base, "grade_all.json")
    if not os.path.exists(rp):
        return None
    r = json.load(open(rp))
    g = json.load(open(gp)) if os.path.exists(gp) else None
    pts = lambda block: [{"d": int(d), "n50": v["n50"], "ci": v["n50_ci"], "width": v["width"],  # noqa: E731
                          "trials": v["trials"]} for d, v in sorted(block.items(), key=lambda kv: int(kv[0]))]
    return {"calibration": pts(r["calibration"]), "target": pts(r["target"]),
            "sealed_sha256": r["sealed_sha256"], "grade": g,
            "spark": r["sealed"].get("muse_spark", {}).get("formula")}


def open_problem_p2(path="runs/open_problems/predictive_credit.jsonl"):
    if not os.path.exists(path):
        return None
    from phystax.open_problems.predictive_credit import summarise
    return summarise(jsonl(path))


def live_runs(base="runs/live"):
    """Live runs from the current pipeline (joint physical models, fixed seal)."""
    out = []
    for p in sorted(glob.glob(os.path.join(base, "*", "result.json"))):
        r = json.load(open(p))
        if r.get("model_text") and os.path.exists(os.path.join(os.path.dirname(p), "clip.mp4")):
            out.append((os.path.basename(os.path.dirname(p)), r))
    return out


def live_example(out_dir, choice=None, base="runs/live"):
    runs = live_runs(base)
    if choice:
        runs = [x for x in runs if x[0] == choice]
        if not runs:
            raise SystemExit(f"no current-pipeline live run named {choice}; see --list-live")
    if not runs:
        return None
    rid, r = runs[-1] if choice else min(runs, key=lambda x: (not x[1]["grade"]["pass"], x[1]["grade"]["error_share_of_path"]))
    d = os.path.join(base, rid)
    os.makedirs(os.path.join(out_dir, "media"), exist_ok=True)
    shutil.copy(os.path.join(d, "clip.mp4"), os.path.join(out_dir, "media", "live.mp4"))
    keep = ("size", "fps", "frames", "track", "split_frame", "first_frame", "last_frame", "model_text",
            "chosen_by", "rationale", "sealed", "sha256", "sealed_at", "grade", "scale", "boxes")
    return {"id": rid, "video": "media/live.mp4", **{k: r.get(k) for k in keep}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--live", default=None, help="id of the live run to feature (see --list-live)")
    ap.add_argument("--list-live", action="store_true")
    a = ap.parse_args()
    if a.list_live:
        for rid, r in live_runs():
            g = r["grade"]
            print(f"{rid}  {r['model_text'][:40]:40s}  error {100 * g['error_share_of_path']:5.1f}%  "
                  f"{'pass' if g['pass'] else 'fail'}  by {r['chosen_by']}")
        return
    ref, agent = jsonl("runs/reference.jsonl"), jsonl("runs/agent.jsonl")
    spend = sum(e.get("usd", 0) for e in jsonl("runs/spend.jsonl"))
    calib = json.load(open("runs/calibration.json")) if os.path.exists("runs/calibration.json") else None
    os.makedirs(a.out, exist_ok=True)
    run_dir, res = pick_episode(a.episode)
    data = {
        "built": __import__("time").strftime("%Y-%m-%d"),
        "spend_usd": round(spend, 2),
        "counts": {"reference_fits": len(ref), "agent_episodes": sum(1 for r in agent if r.get("nrmse") is not None)},
        "decomposition": decomposition(ref, agent),
        "agents": agent_summary(agent),
        "coverage": coverage(ref),
        "perception": perception("runs/perception.csv"),
        "calibration": calib,
        "naming": NAMING_PROBE,
        "pendulums": pendulums("runs/real/**/pendulum.json"),
        "episode": episode_payload(run_dir, res, os.path.join(a.out, "media")) if run_dir else None,
        "p1": open_problem_p1(),
        "p2": open_problem_p2(),
        "live": live_example(a.out, a.live),
        "repo": "https://github.com/ritvik412/phystax",
    }
    with open(os.path.join(a.out, "data.js"), "w") as f:
        f.write("window.PHYSTAX = " + json.dumps(data, default=float) + ";\n")
    shutil.copy(TEMPLATE, os.path.join(a.out, "index.html"))
    vend = os.path.join(os.path.dirname(TEMPLATE), "vendor")
    if os.path.isdir(vend):
        shutil.copytree(vend, os.path.join(a.out, "vendor"), dirs_exist_ok=True)
    print(f"site -> {a.out}/ (episode: {run_dir or 'none found'}; "
          f"{data['counts']['reference_fits']} reference fits, {data['counts']['agent_episodes']} agent episodes, "
          f"{len(data['pendulums'])} real pendulum clips)")


if __name__ == "__main__":
    main()
