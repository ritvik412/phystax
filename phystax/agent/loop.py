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
    commitment = _commit(model, messages, ledger, len(world.test_params), len(times), moving)

    if review and reviewer is not None:
        verdict = _review(reviewer, tools.data, commitment)
        ledger.append("review", verdict)
        if verdict.get("verdict") == "revise" and verdict.get("issues"):
            messages.append({"role": "assistant", "content": json.dumps(commitment)})
            messages.append({"role": "user", "content":
                             "An independent referee raised these issues. Address them and "
                             "re-commit (exploration is still closed): " + json.dumps(verdict["issues"])})
            commitment = _commit(model, messages, ledger, len(world.test_params), len(times), moving)

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


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object even if the model wrapped it in prose or code fences."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            obj = json.loads(t[a:b + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _shape_problems(c: dict, n_tests: int, n_times: int, bodies: list[str]) -> list[str]:
    """Format checks only (never the answer): one prediction per test and body,
    each coordinate list exactly as long as the probe-time list."""
    probs = []
    preds = c.get("predictions") or []
    seen = {(p.get("test_index"), p.get("body")) for p in preds}
    for i in range(n_tests):
        for b in bodies:
            if (i, b) not in seen:
                probs.append(f"missing prediction for test_index={i}, body='{b}'")
    for p in preds:
        for k in ("x", "y", "x_lo", "x_hi", "y_lo", "y_hi"):
            v = p.get(k)
            if not isinstance(v, list) or len(v) != n_times:
                probs.append(f"test_index={p.get('test_index')} body='{p.get('body')}': '{k}' must have "
                             f"{n_times} numbers, got {len(v) if isinstance(v, list) else 'none'}")
    return probs


def _commit(model, messages: list[dict], ledger=None, n_tests: int = 2, n_times: int = 6,
            bodies: list[str] | None = None, max_attempts: int = 3) -> dict:
    """Ask for the sealed prediction as schema-constrained JSON. Long tool-using
    sessions sometimes answer in prose instead; we extract JSON if present,
    otherwise ask again (format feedback only), and log every attempt."""
    bodies = bodies or []
    msgs = list(messages)
    last_text = ""
    for attempt in range(1, max_attempts + 1):
        fmt = COMMIT_SCHEMA if attempt < max_attempts else {"type": "json_object"}
        msg = model.chat(msgs, response_format=fmt)
        last_text = msg.content or ""
        obj = _extract_json(last_text)
        probs = ["reply was not a JSON object"] if obj is None else _shape_problems(obj, n_tests, n_times, bodies)
        if ledger is not None:
            ledger.append("commit_attempt", {"attempt": attempt, "parsed": obj is not None,
                                             "problems": probs[:10], "raw_head": last_text[:400]})
        if obj is not None and not probs:
            return obj
        msgs = msgs + [{"role": "assistant", "content": last_text},
                       {"role": "user", "content":
                        "Your reply could not be accepted: " + "; ".join(probs[:6]) +
                        ". Reply with ONLY the JSON object (no prose, no code fences) with keys "
                        "law_equation, parameters, predictions, self_confidence, rationale."}]
        if obj is not None and probs and attempt == max_attempts:
            return obj        # well-formed JSON with shape problems: grade what we have
    return {"law_equation": "", "parameters": [], "predictions": [],
            "self_confidence": 0.0, "rationale": f"unparseable commit after {max_attempts} attempts: {last_text[:200]}"}


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
