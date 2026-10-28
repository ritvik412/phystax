"""One chat-completions surface for every agent model.

Notes verified against the Meta cookbook / docs:
  * the OpenAI SDK does not read MODEL_API_KEY by itself; pass it explicitly
  * Meta tool calling accepts tool_choice="auto" only (others return 400),
    so the loop never forces a tool; it ends exploration by dropping tools
  * response_format json_schema gives schema-constrained output
  * Baseten: use the hackathon slug meta-models/muse-spark-open-1026;
    reasoning text arrives in message.reasoning_content
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .. import budget
from ..config import AGENT_MODELS, AgentModel, api_key, base_url


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    log: list = field(default_factory=list)


class ChatModel:
    def __init__(self, key: str):
        from openai import OpenAI
        self.spec: AgentModel = AGENT_MODELS[key]
        self.client = OpenAI(base_url=base_url(self.spec.provider),
                             api_key=api_key(self.spec.provider))
        self.usage = Usage()
        self.model_id = self.spec.model_id
        self.pool = budget.pool(self.spec.provider)
        self.cost_usd = 0.0

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             response_format: dict | None = None, max_tokens: int = 16000):
        budget.check(self.pool, budget.chat_estimate(self.model_id, messages), f"chat {self.model_id}")
        kwargs = {"model": self.model_id, "messages": messages, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools          # tool_choice left at its "auto" default
        if response_format:
            kwargs["response_format"] = response_format
        if self.spec.reasoning_effort:
            kwargs["reasoning_effort"] = self.spec.reasoning_effort
        t0 = time.time()
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # retry transient errors only
                status = getattr(exc, "status_code", None)
                if status in (401, 403, 404) and self.spec.fallback_id and self.model_id != self.spec.fallback_id:
                    print(f"[providers] {self.model_id} refused ({status}); falling back to {self.spec.fallback_id}")
                    self.model_id = kwargs["model"] = self.spec.fallback_id
                    continue
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        u = getattr(resp, "usage", None)
        self.usage.calls += 1
        self.usage.seconds += time.time() - t0
        if u:
            self.usage.prompt_tokens += u.prompt_tokens or 0
            self.usage.completion_tokens += u.completion_tokens or 0
            det = getattr(u, "completion_tokens_details", None)
            self.usage.reasoning_tokens += getattr(det, "reasoning_tokens", 0) or 0
            pdet = getattr(u, "prompt_tokens_details", None)
            cached = getattr(pdet, "cached_tokens", 0) or 0
            usd = budget.chat_cost(self.model_id, u.prompt_tokens or 0, cached, u.completion_tokens or 0)
            self.cost_usd += usd
            budget.record(self.pool, usd, "chat", self.model_id,
                          {"in": u.prompt_tokens, "cached": cached, "out": u.completion_tokens})
        return resp.choices[0].message
