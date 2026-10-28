"""One episode = explore with tools, then pre-register a prediction with no
tools and a strict JSON schema, then (only then) render and grade the test.

Why two steps: Meta's endpoint accepts tool_choice="auto" only, so we cannot
force a final "commit" tool call. Instead exploration ends when the model
stops calling tools (or the budget runs out), and the commit is a separate
call with tools removed and response_format set. The commitment is hashed
into the ledger before the world renders the test clip.
"""
from __future__ import annotations

import json
import os
import time

from ..config import FPS, MAX_AGENT_TURNS, RUNS_DIR
from ..grader.grade import COMMIT_SCHEMA, commitment_hash, grade, probe_times
from ..bench.conditions import sam_backend
from ..grader.ledger import Ledger
from ..sim.world import World
from . import prompts
from .tools import TOOLS, ToolBox

TEST_DURATION_S = 4.0


def _assistant_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
    return d


def run_episode(family: str, seed: int, condition: str, model, degradation: int = 0,
                review: bool = False, reviewer=None, noise_sigma_px: float = 3.0,
                tag: str = "", run_dir: str | None = None) -> dict:
    """`model` is a providers.ChatModel (or anything with .chat() and .spec)."""
    world = World(family, seed, degradation,
                  run_dir=run_dir or os.path.join(RUNS_DIR, f"{family}_s{seed}_d{degradation}",
                                                  f"{condition}_{model.spec.name.replace(' ', '_')}{tag}"))
    ledger_path = os.path.join(world.run_dir, "ledger.jsonl")
    if os.path.exists(ledger_path):          # one ledger per episode: re-runs start fresh
        os.remove(ledger_path)
    ledger = Ledger(ledger_path)
    ledger.append("episode_start", {"family": family, "spec": world.spec(), "condition": condition,
                                    "model": model.spec.model_id, "degradation": degradation,
                                    "sam_backend": sam_backend() if condition.startswith("sam") else None,
                                    "secret_law_sha256": world.secret_commitment()})
    tools = ToolBox(world, condition, ledger, os.path.join(world.run_dir, "sandbox"),
                    noise_sigma_px)

    messages = [{"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": prompts.task_message(world.spec(), condition)}]
    t0 = time.time()

    # ---- step 1: exploration (tool_choice stays "auto") ------------------
    for _ in range(MAX_AGENT_TURNS):
        msg = model.chat(messages, tools=TOOLS)
        messages.append(_assistant_dict(msg))
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            break
        for tc in calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result = f"ERROR: arguments were not valid JSON ({exc}); resend."
            else:
                result = tools.call(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # ---- step 2: pre-registered commitment (no tools, strict schema) -----
    times = probe_times(TEST_DURATION_S)
    moving = [b for i, b in enumerate(world.family.bodies) if i not in world.family.fixed_bodies]
    messages.append({"role": "user", "content": prompts.commit_message(
        world.test_conditions(), times, moving)})
    commitment = _commit(model, messages)

    if review and reviewer is not None:
        verdict = _review(reviewer, tools.data, commitment)
        ledger.append("review", verdict)
        if verdict.get("verdict") == "revise" and verdict.get("issues"):
            messages.append({"role": "assistant", "content": json.dumps(commitment)})
            messages.append({"role": "user", "content":
                             "An independent referee raised these issues. Address them and "
                             "re-commit (exploration is still closed): " + json.dumps(verdict["issues"])})
            commitment = _commit(model, messages)

    c_hash = commitment_hash(commitment)
    ledger.append("commit", {"commitment": commitment, "sha256": c_hash})

    # ---- step 3: the world runs the held-out tests, then grades ----------
    tests = [world.run_test(i, TEST_DURATION_S) for i in range(len(world.test_params))]
    ledger.append("test_reveal", {"videos": [t.video_path for t in tests], "commit_sha256": c_hash})
    score = grade(commitment, family, world.hidden, [t.traj for t in tests], times, FPS)
    ledger.append("grade", score)
    ledger.append("secret_reveal", world.reveal())

    ok, why = ledger.verify()
    result = {
        "family": family, "seed": seed, "degradation": degradation, "condition": condition,
        "model": model.spec.name, "review": review,
        "sam_backend": sam_backend() if condition.startswith("sam") else None,
        "n_experiments": len(world.experiments), "wall_s": round(time.time() - t0, 1),
        "score": score, "usage": vars(model.usage) | {"log": None},
        "perception": tools.diagnostics, "ledger_ok": ok and ledger.commit_precedes_test(),
        "ledger_msg": why, "run_dir": world.run_dir, "probe_times": times,
        "true_form": world.hidden["form"], "cost_usd": round(getattr(model, "cost_usd", 0.0), 4),
    }
    json.dump(result, open(os.path.join(world.run_dir, "result.json"), "w"), indent=2, default=str)
    return result


def _commit(model, messages: list[dict]) -> dict:
    msg = model.chat(messages, response_format=COMMIT_SCHEMA)
    try:
        return json.loads(msg.content)
    except (json.JSONDecodeError, TypeError):
        text = (msg.content or "").strip().strip("`").removeprefix("json").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"law_equation": "", "parameters": [], "predictions": [],
                    "self_confidence": 0.0, "rationale": f"unparseable commit: {text[:200]}"}


def _review(reviewer, data: dict, commitment: dict) -> dict:
    brief = {k: {"params": v["params"], "n_frames": len(v["t"])} for k, v in data.items()}
    msg = reviewer.chat([
        {"role": "system", "content": prompts.REVIEWER_SYSTEM},
        {"role": "user", "content": json.dumps({"experiments": brief, "draft": commitment})},
    ], response_format={"type": "json_object"})
    try:
        return json.loads(msg.content)
    except (json.JSONDecodeError, TypeError):
        return {"verdict": "accept", "issues": [], "note": "unparseable review"}
