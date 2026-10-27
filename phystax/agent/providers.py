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

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             response_format: dict | None = None, max_tokens: int = 16000):
        kwargs = {"model": self.spec.model_id, "messages": messages, "max_tokens": max_tokens}
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
        return resp.choices[0].message
