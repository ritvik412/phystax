"""P2. Predictive Credit of an AI scientist's sealed law.

Protocol from Ning, Li, Kong and Li (CMU), "Predictive Credit: Measuring What
Scientific Explanations Add to Experimental Forecasts" (arXiv 2610.00314),
applied to PhysTax's agent episodes. A fresh forecaster (Muse Spark, with a
Python sandbox to integrate equations) predicts the held-out test trajectories
of an episode it never saw, in three matched arms:

  description   the system description and the test starting conditions only
  agent_card    + the exploring agent's sealed law, constants and rationale
  reference     + the true form with constants least-squares fitted on the same
                  channel's measurements (positive control)

Graded by the same deterministic grader as the episodes. Per episode:
  predictive credit   = error(description) - error(agent_card)
  remaining gap       = error(agent_card) - error(reference)
Paired bootstrap intervals over episodes. No language model judges anything.

    python -m phystax.open_problems.predictive_credit --workers 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..agent.loop import TEST_DURATION_S, _extract_json
from ..agent.sandbox import run as run_python
from ..grader.grade import grade, probe_times
from ..sim.engine import simulate
from ..sim.laws import FAMILIES
from ..sim.world import World

ARMS = ["description", "agent_card", "reference"]
FPS = 30
TOOL = [{"type": "function", "function": {
    "name": "run_python", "description": "Run Python 3 (numpy, scipy available). Print what you need.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["code"],
                   "properties": {"code": {"type": "string"}}}}}]


def find_episodes(runs: str = "runs") -> list[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(runs, "*_s*_d*", "*", "result.json"))):
        r = json.load(open(p))
        if r.get("model") not in (None, "reference") and r.get("score", {}).get("prediction_nrmse") is not None:
            out.append(os.path.dirname(p))
    return out


def _commit_of(run_dir: str) -> dict:
    for line in open(os.path.join(run_dir, "ledger.jsonl")):
        e = json.loads(line)
        if e["kind"] == "commit":
            return e["payload"]["commitment"]
    return {}


def cards(run_dir: str, runs: str = "runs") -> dict:
    res = json.load(open(os.path.join(run_dir, "result.json")))
    fam = FAMILIES[res["family"]]
    commit = _commit_of(run_dir)
    agent = {"law": commit.get("law_equation", ""),
             "constants": {p["name"]: p["value"] for p in commit.get("parameters", [])},
             "rationale": (commit.get("rationale") or "")[:1200]}
    ref_dir = os.path.join(runs, "reference",
                           f"{res['family']}_d{res['degradation']}_s{res['seed']}_{res['condition']}_designed")
    reference = None
    if os.path.exists(os.path.join(ref_dir, "result.json")):
        rr = json.load(open(os.path.join(ref_dir, "result.json")))
        est = rr["estimate"]
        reference = {"law": fam.forms[est["form"]].equation,
                     "constants": {k: v for k, v in est.items() if k != "form"}}
    return {"res": res, "agent": agent, "reference": reference}


def _prompt(spec: dict, tests: list, times: list, bodies: list, card: dict | None) -> str:
    txt = ("You forecast where objects will be in a physics experiment you did not run.\n"
           f"System: {json.dumps(spec)}\n")
    if card:
        txt += ("A previous scientist studied this system and reports this law (positions in pixels, y up, time "
                f"in seconds): {json.dumps(card)}\n")
    txt += (f"Predict, for each test below, the position of each body {bodies} at times {times} s after launch.\n"
            f"Tests (starting conditions): {json.dumps(tests)}\n"
            "Use run_python to integrate the law numerically if you have one. When done, answer with JSON only: "
            '{"predictions": [{"test_index": int, "body": str, "x": [6 numbers], "y": [6 numbers]}]}')
    return txt


def forecast(model, run_dir: str, arm: str, info: dict, max_tool_turns: int = 6) -> dict:
    res = info["res"]
    with tempfile.TemporaryDirectory() as tmp:
        world = World(res["family"], res["seed"], res["degradation"], run_dir=tmp)
        fam = world.family
        moving = [b for i, b in enumerate(fam.bodies) if i not in fam.fixed_bodies]
        times = probe_times(TEST_DURATION_S)
        card = None if arm == "description" else (info["agent"] if arm == "agent_card" else info["reference"])
        if arm == "reference" and card is None:
            return {"arm": arm, "error": "no reference run for this episode"}
        msgs = [{"role": "user", "content": _prompt(world.spec(), world.test_conditions(), times, moving, card)}]
        sandbox = os.path.join(tmp, "sandbox")
        os.makedirs(sandbox, exist_ok=True)
        for _ in range(max_tool_turns):
            msg = model.chat(msgs, tools=TOOL)
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                break
            msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name,
                                                               "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                try:
                    code = json.loads(c.function.arguments).get("code", "")
                except json.JSONDecodeError:
                    code = ""
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": run_python(code, sandbox)})
        msgs.append({"role": "user", "content": "Now give the JSON forecast only."})
        obj = None
        for _ in range(2):
            out = model.chat(msgs, response_format={"type": "json_object"})
            obj = _extract_json(out.content or "")
            if obj and obj.get("predictions"):
                break
            msgs += [{"role": "assistant", "content": out.content or ""},
                     {"role": "user", "content": "That was not valid JSON with a predictions list. JSON only."}]
        preds = (obj or {}).get("predictions", [])
        truth = [simulate(fam, world.hidden, p, int(TEST_DURATION_S * FPS)) for p in world.test_conditions()]
        score = grade({"parameters": [], "predictions": preds}, res["family"], world.hidden, truth, times, FPS)
    return {"arm": arm, "error": score["prediction_nrmse"], "pass": score["pass"], "n_predictions": len(preds)}


def _boot(xs, n=2000, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return [float(xs.mean()) if len(xs) else None] * 2
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(xs, (n, len(xs))), axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def run(model_factory, runs: str = "runs", workers: int = 4, out: str = "runs/open_problems/predictive_credit.jsonl",
        limit: int | None = None) -> dict:
    eps = find_episodes(runs)[:limit]
    infos = {d: cards(d, runs) for d in eps}
    jobs = [(d, arm) for d in eps for arm in ARMS]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def work(job):
        d, arm = job
        try:
            r = forecast(model_factory(), d, arm, infos[d])
        except Exception as exc:
            r = {"arm": arm, "error": None, "failure": f"{type(exc).__name__}: {exc}"}
        r.update(episode=d, channel=infos[d]["res"]["condition"], agent_error=infos[d]["res"]["score"]["prediction_nrmse"])
        with open(out, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, jobs))
    return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        if r.get("error") is not None:
            by.setdefault(r["episode"], {})[r["arm"]] = r["error"]
            by[r["episode"]]["channel"] = r["channel"]
    credit = [v["description"] - v["agent_card"] for v in by.values() if {"description", "agent_card"} <= v.keys()]
    gap = [v["agent_card"] - v["reference"] for v in by.values() if {"agent_card", "reference"} <= v.keys()]
    med = lambda a: [float(np.median([v[a] for v in by.values() if a in v]))] if any(a in v for v in by.values()) else [None]  # noqa: E731
    out = {"episodes": len(by),
           "median_error": {a: med(a)[0] for a in ARMS},
           "predictive_credit": {"median": float(np.median(credit)) if credit else None, "ci95": _boot(credit),
                                 "n": len(credit), "share_positive": float(np.mean(np.array(credit) > 0)) if credit else None},
           "gap_to_reference": {"median": float(np.median(gap)) if gap else None, "ci95": _boot(gap), "n": len(gap)}}
    return out


def main() -> None:
    from ..agent.providers import ChatModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spark")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="runs/open_problems/predictive_credit.jsonl")
    a = ap.parse_args()
    s = run(lambda: ChatModel(a.model), workers=a.workers, out=a.out, limit=a.limit)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
