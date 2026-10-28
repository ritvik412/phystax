"""Day 1: check every chat-API assumption the agent loop makes, for a few cents.

    python scripts/day1_llm_smoke.py            # Muse Spark (contributor, falls back to standard)
    python scripts/day1_llm_smoke.py wonder     # the reviewer model

Checks, in order: plain chat, `reasoning_effort` accepted, a tool call with
tool_choice left at "auto", tool result round trip, strict json_schema output
with no tools, and the price actually charged. Each check prints PASS or FAIL
with the API's own error text, so a failure tells us exactly what to change.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phystax import budget  # noqa: E402
from phystax.agent.providers import ChatModel  # noqa: E402
from phystax.config import AGENT_MODELS  # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "spark"
m = ChatModel(key)
TOOLS = [{"type": "function", "function": {
    "name": "run_experiment", "description": "Run one experiment.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["speed"],
                   "properties": {"speed": {"type": "number"}}}}}]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["value", "unit"],
    "properties": {"value": {"type": "number"}, "unit": {"type": "string"}}}}}


def check(name, fn):
    try:
        print(f"PASS {name}: {fn()}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:300]}")
        return False


def plain():
    r = m.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=200)
    return repr((r.content or "")[:40]) + f" via {m.model_id}"


def no_effort():
    saved = AGENT_MODELS[key]
    m.spec = type(saved)(**{**saved.__dict__, "reasoning_effort": None})
    try:
        return repr((m.chat([{"role": "user", "content": "Say ok"}], max_tokens=100).content or "")[:20])
    finally:
        m.spec = saved


msgs = [{"role": "user", "content": "Call run_experiment once with speed 3.5, then stop."}]


def tool_call():
    r = m.chat(msgs, tools=TOOLS, max_tokens=800)
    calls = r.tool_calls or []
    assert calls, f"no tool call; content={r.content!r}"
    msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.function.name,
                                                       "arguments": c.function.arguments}} for c in calls]})
    for c in calls:
        msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps({"distance_px": 41.2})})
    return f"{len(calls)} call(s): {calls[0].function.name}({calls[0].function.arguments})"


def round_trip():
    r = m.chat(msgs, tools=TOOLS, max_tokens=800)
    return repr((r.content or "")[:80])


def structured():
    msgs.append({"role": "user", "content": "Report the measured distance as JSON."})
    r = m.chat(msgs, response_format=SCHEMA, max_tokens=800)
    obj = json.loads(r.content)
    assert set(obj) == {"value", "unit"}, obj
    return obj


ok = [check("plain chat (with reasoning_effort)", plain), check("chat without reasoning_effort", no_effort),
      check("tool call, tool_choice=auto", tool_call), check("tool result round trip", round_trip),
      check("strict json_schema, no tools", structured)]
print(f"\nmodel used: {m.model_id}; tokens in={m.usage.prompt_tokens} out={m.usage.completion_tokens} "
      f"reasoning={m.usage.reasoning_tokens}; cost ${m.cost_usd:.4f}")
print(budget.summary())
print("ALL PASS" if all(ok) else "Some checks failed: paste this output back.")
