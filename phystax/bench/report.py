"""Turn runs/*.jsonl into the tables for the README and the UI.

    python -m phystax.bench.report                 # real results only
    python -m phystax.bench.report --allow-synthetic   # plumbing check, stamped as such

Tables:
  1. Measurement tax: reference fitter error per channel (no LLM involved)
  2. Agent results: per model x channel
  3. Decomposition: total tax = measurement tax + reasoning tax
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from statistics import median

CHANNELS = ["oracle", "noisy_oracle", "sam", "sam_conf", "sam_box", "sam_box_conf", "classical"]


def load(path: str, allow_synthetic: bool) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if "error" not in r and r.get("nrmse") is not None]
    if not allow_synthetic:
        rows = [r for r in rows if r.get("sam_backend") != "synthetic"]
    return rows


def fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{100 * v:.1f}%" if pct else f"{v:.4f}"


def table_reference(rows: list[dict]) -> str:
    cell = defaultdict(list)
    for r in rows:
        cell[(r["family"], r["level"], r["condition"])].append(r)
    keys = sorted({(r["family"], r["level"]) for r in rows})
    chans = [c for c in CHANNELS if any(k[2] == c for k in cell)]
    out = ["| family | level | " + " | ".join(chans) + " |",
           "|---|---|" + "---|" * len(chans)]
    for fam, lvl in keys:
        vals = []
        for c in chans:
            rs = cell.get((fam, lvl, c), [])
            if not rs:
                vals.append("–")
                continue
            m = median(r["nrmse"] for r in rs)
            p = sum(bool(r["pass"]) for r in rs) / len(rs)
            vals.append(f"{fmt(m)} ({fmt(p, True)} pass, n={len(rs)})")
        out.append(f"| {fam} | {lvl} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def table_agent(rows: list[dict]) -> str:
    cell = defaultdict(list)
    for r in rows:
        cell[(r["model"], r["degradation"], r["condition"])].append(r)
    out = ["| model | level | channel | median nRMSE | pass | mean experiments | n |",
           "|---|---|---|---|---|---|---|"]
    for (m, lvl, c), rs in sorted(cell.items()):
        out.append(f"| {m} | {lvl} | {c} | {fmt(median(r['nrmse'] for r in rs))} | "
                   f"{fmt(sum(bool(r['pass']) for r in rs) / len(rs), True)} | "
                   f"{sum(r['n_experiments'] for r in rs) / len(rs):.1f} | {len(rs)} |")
    return "\n".join(out)


def _boot_median_ci(xs, n_boot=2000, seed=0):
    import numpy as np
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return (float(xs[0]), float(xs[0])) if len(xs) else (None, None)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(xs, (n_boot, len(xs)), replace=True), axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def _cell(xs):
    if not xs:
        return "–"
    lo, hi = _boot_median_ci(xs)
    return f"{median(xs):.4f} [{lo:.4f}, {hi:.4f}]"


def table_decomposition(ref: list[dict], agent: list[dict]) -> str:
    """Exact telescoping split per episode, medians with 95% bootstrap CIs
    (episodes are paired: same universe under every channel)."""
    r_idx = {(r["family"], r["level"], r["seed"], r["condition"]): r["nrmse"]
             for r in ref if r.get("design", "designed") == "designed"}
    acc = defaultdict(lambda: {"measure": [], "design": [], "reason": [], "total": []})
    for a in agent:
        k = (a["family"], a["degradation"], a["seed"])
        ref_c, ref_o = r_idx.get(k + (a["condition"],)), r_idx.get(k + ("oracle",))
        on_agent = (a.get("ref_on_agent") or {}).get("nrmse")
        if None in (ref_c, ref_o, on_agent):
            continue
        d = acc[(a["model"], a["condition"])]
        d["measure"].append(ref_c - ref_o)
        d["design"].append(on_agent - ref_c)
        d["reason"].append(a["nrmse"] - on_agent)
        d["total"].append(a["nrmse"] - ref_o)
    out = ["| model | channel | measurement tax | design gap | reasoning tax | total | n |",
           "|---|---|---|---|---|---|---|"]
    for (m, c), d in sorted(acc.items()):
        out.append(f"| {m} | {c} | {_cell(d['measure'])} | {_cell(d['design'])} | "
                   f"{_cell(d['reason'])} | {_cell(d['total'])} | {len(d['total'])} |")
    out.append("\nMedian [95% bootstrap CI] of normalised prediction error differences. "
               "The three terms sum to the total per episode.")
    return "\n".join(out)


def table_uncertainty_and_passk(agent: list[dict], ref: list[dict]) -> str:
    """Calibration of committed 80% intervals (coverage should be about 0.80;
    the interval score is a proper scoring rule, lower is better) and pass@K
    over repeated attempts on the same universe (needs --repeats K > 1)."""
    groups = defaultdict(list)
    for r in agent:
        groups[(r["model"], r["condition"])].append(r)
    for r in ref:
        groups[("reference", r["condition"])].append(r)
    out = ["| scientist | channel | 80% coverage | coverage, iid-noise intervals | interval score | pass@1 | pass@K (K) | n |",
           "|---|---|---|---|---|---|---|---|"]
    for (m, c), rs in sorted(groups.items()):
        cov = [r["coverage_80"] for r in rs if r.get("coverage_80") is not None]
        cov_iid = [r["coverage_80_iid"] for r in rs if r.get("coverage_80_iid") is not None]
        isc = [r["interval_score"] for r in rs if r.get("interval_score") is not None]
        universes = defaultdict(list)
        for r in rs:
            universes[(r["family"], r.get("degradation", r.get("level")), r["seed"])].append(bool(r["pass"]))
        k = max((len(v) for v in universes.values()), default=1)
        p1 = sum(v[0] for v in universes.values()) / max(len(universes), 1)
        pk = sum(any(v) for v in universes.values()) / max(len(universes), 1)
        out.append(f"| {m} | {c} | {fmt(median(cov)) if cov else '–'} | {fmt(median(cov_iid)) if cov_iid else '–'} | "
                   f"{fmt(median(isc)) if isc else '–'} | "
                   f"{fmt(p1, True)} | {fmt(pk, True)} ({k}) | {len(rs)} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="runs/reference.jsonl")
    ap.add_argument("--agent", default="runs/agent.jsonl")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--out", default="runs/report.md")
    a = ap.parse_args()
    ref = load(a.reference, a.allow_synthetic)
    agent = load(a.agent, a.allow_synthetic)
    parts = []
    if a.allow_synthetic:
        parts.append("> **SYNTHETIC SAM BACKEND INCLUDED: plumbing check, not a result.**\n")
    parts += ["## 1. Measurement tax (reference fitter, no LLM)\n",
              "Median normalised prediction error on held-out tests (lower is better).\n",
              table_reference(ref) if ref else "_no reference runs yet_", "",
              "## 2. Agents\n", table_agent(agent) if agent else "_no agent runs yet_", "",
              "## 3. Decomposition (total = measurement + design + reasoning)\n",
              table_decomposition(ref, agent) if (ref and agent) else "_needs both grids_", "",
              "## 4. Uncertainty and repeatability\n",
              table_uncertainty_and_passk(agent, ref) if (ref or agent) else "_no runs yet_"]
    text = "\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
