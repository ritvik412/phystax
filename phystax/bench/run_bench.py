"""Run the Perception Tax benchmark.

Reference fitter (no LLM; isolates measurement tax and tests confidence weighting):
    python -m phystax.bench.run_bench reference --seeds 8 --levels 0 2 3

Agents (the headline table; costs credits):
    python -m phystax.bench.run_bench agent --models spark spark-open \
        --conditions oracle sam sam_conf --seeds 4 --levels 2

Freeze the protocol first: `protocol` hashes this exact grid into
runs/protocol.json so the README can show the benchmark was pre-registered
before results came in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback

from .. import budget
from .reference import DESIGNS, refit_episode, run_reference


def _save(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def cmd_protocol(a) -> None:
    proto = vars(a).copy()
    proto.pop("func", None)
    blob = json.dumps(proto, sort_keys=True)
    out = {"protocol": proto, "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    os.makedirs("runs", exist_ok=True)
    json.dump(out, open("runs/protocol.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


def cmd_reference(a) -> None:
    for fam in a.families:
        for lvl in a.levels:
            for seed in range(a.seed_start, a.seed_start + a.seeds):
                for cond in a.conditions:
                    try:
                        r = run_reference(fam, seed, cond, lvl, design=a.design,
                                          run_dir=f"runs/reference/{fam}_d{lvl}_s{seed}_{cond}_{a.design}")
                        s = r["score"]
                        row = {"kind": "reference", "family": fam, "level": lvl, "seed": seed,
                               "condition": cond, "nrmse": s["prediction_nrmse"],
                               "design": a.design, "weighting": r["weighting"],
                               "true_form": r["true_form"],
                               "sam_backend": r["sam_backend"],
                               "pass": s["pass"],
                               "param_rel_err": {k: v["rel_err"] for k, v in s["parameters"].items()}}
                    except Exception as exc:
                        row = {"kind": "reference", "family": fam, "level": lvl, "seed": seed,
                               "condition": cond, "error": repr(exc)}
                    print(row)
                    _save([row], a.out)


def cmd_agent(a) -> None:
    from ..agent.loop import run_episode
    from ..agent.providers import ChatModel
    reviewer = ChatModel("wonder") if a.review else None
    for key in a.models:
        model = ChatModel(key)
        for fam in a.families:
            for lvl in a.levels:
                for seed in range(a.seed_start, a.seed_start + a.seeds):
                    for cond in a.conditions:
                        try:
                            r = run_episode(fam, seed, cond, model, lvl, review=a.review,
                                            reviewer=reviewer)
                            row = {"kind": "agent", "model": key, **{k: r[k] for k in (
                                "family", "seed", "degradation", "condition", "n_experiments",
                                "wall_s", "ledger_ok", "sam_backend")},
                                "nrmse": r["score"]["prediction_nrmse"], "pass": r["score"]["pass"],
                                "true_form": r["true_form"], "cost_usd": r["cost_usd"],
                                "ref_on_agent": refit_episode(r["run_dir"]),
                                "param_rel_err": {k: v["rel_err"] for k, v in r["score"]["parameters"].items()},
                                "usage": r["usage"]}
                        except Exception as exc:
                            row = {"kind": "agent", "model": key, "family": fam, "seed": seed,
                                   "degradation": lvl, "condition": cond, "error": repr(exc),
                                   "trace": traceback.format_exc()[-800:]}
                        print({k: v for k, v in row.items() if k != "trace"})
                        _save([row], a.out)
                        if isinstance(row.get("error"), str) and "BudgetExceeded" in row["error"]:
                            print("Budget cap reached; stopping the grid.", budget.summary())
                            return
    print(budget.summary())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    for name, fn, default_conds in [
        ("protocol", cmd_protocol, ["oracle", "noisy_oracle", "sam", "sam_conf", "classical"]),
        ("reference", cmd_reference, ["oracle", "noisy_oracle", "sam", "sam_conf", "classical"]),
        ("agent", cmd_agent, ["oracle", "sam", "sam_conf"]),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--families", nargs="+", default=list(DESIGNS))
        p.add_argument("--levels", nargs="+", type=int, default=[0, 2, 3])
        p.add_argument("--seeds", type=int, default=4)
        p.add_argument("--seed-start", type=int, default=0)
        p.add_argument("--conditions", nargs="+", default=default_conds)
        p.add_argument("--models", nargs="+", default=["spark", "spark-open"])
        p.add_argument("--review", action="store_true")
        p.add_argument("--design", choices=["designed", "random"], default="designed")
        p.add_argument("--out", default=f"runs/{name}.jsonl")
        p.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
