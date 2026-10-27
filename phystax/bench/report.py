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

CHANNELS = ["oracle", "noisy_oracle", "sam", "sam_conf", "classical"]


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


def table_decomposition(ref: list[dict], agent: list[dict]) -> str:
    r_idx = {(r["family"], r["level"], r["seed"], r["condition"]): r["nrmse"] for r in ref}
    acc = defaultdict(lambda: {"measure": [], "reason": [], "total": []})
    for a in agent:
        k = (a["family"], a["degradation"], a["seed"])
        ref_c = r_idx.get(k + (a["condition"],))
        ref_o = r_idx.get(k + ("oracle",))
        if ref_c is None or ref_o is None:
            continue
        d = acc[(a["model"], a["condition"])]
        d["measure"].append(ref_c - ref_o)
        d["reason"].append(a["nrmse"] - ref_c)
        d["total"].append(a["nrmse"] - ref_o)
    out = ["| model | channel | measurement tax | reasoning tax | total tax | n |",
           "|---|---|---|---|---|---|"]
    for (m, c), d in sorted(acc.items()):
        out.append(f"| {m} | {c} | {fmt(median(d['measure']))} | {fmt(median(d['reason']))} | "
                   f"{fmt(median(d['total']))} | {len(d['total'])} |")
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
              "## 3. Decomposition (total = measurement + reasoning)\n",
              table_decomposition(ref, agent) if (ref and agent) else "_needs both grids_"]
    text = "\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
