"""Spend meter with a hard cap, persisted across runs.

Every billable call is checked *before* it is sent (estimated cost) and
recorded *after* (actual usage). The ledger lives in runs/spend.jsonl, so the
cap holds across scripts, the UI and repeated benchmark runs.

Prices are USD per million tokens (input, cached input, output), copied from
the Meta Model API Models tab on Oct 28, 2026. Update them here if they change.
Caps: PHYSTAX_BUDGET_USD (Meta, default 120 of the 150 credits) and
PHYSTAX_BASETEN_BUDGET_USD (default 90 of the 100 Baseten credits).
"""
from __future__ import annotations

import json
import os
import threading
import time

PRICES = {  # model_id: (input, cached_input, output) per 1M tokens
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.002, 0.20),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.2-contributor": (0.10, 0.002, 0.20),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "muse-glimmer": (0.0, 0.0, 0.0),            # self-hosted: no credits used
    # Baseten list price; confirm on your Baseten dashboard before the full grid
    "meta-models/muse-spark-open-1026": (1.25, 1.25, 4.25),
}
SAM_VIDEO_PER_FRAME = 0.20 / 1000
SAM_IMAGE_EACH = 2.50 / 1000
SAM_MIN_FRAMES = 50
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def _path() -> str:
    d = os.environ.get("PHYSTAX_RUNS", "runs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "spend.jsonl")


def pool(provider: str) -> str:
    return "baseten" if provider == "baseten" else ("meta" if provider == "meta" else "other")


_RUN_START: dict = {}


def cap(p: str) -> float:
    """Total cap (PHYSTAX_BUDGET_USD, a running total across all runs), further
    limited by PHYSTAX_RUN_BUDGET_USD: the most THIS process may add on top of
    what was already spent when it started. Use the run cap for one-off commands."""
    if p == "meta":
        total = float(os.environ.get("PHYSTAX_BUDGET_USD", "120"))
    elif p == "baseten":
        total = float(os.environ.get("PHYSTAX_BASETEN_BUDGET_USD", "90"))
    else:
        return float("inf")
    run = os.environ.get("PHYSTAX_RUN_BUDGET_USD")
    if run:
        if p not in _RUN_START:
            _RUN_START[p] = spent(p)
        total = min(total, _RUN_START[p] + float(run))
    return total


def spent(p: str | None = None) -> float:
    path = _path()
    if not os.path.exists(path):
        return 0.0
    total = 0.0
    with open(path) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if p is None or e.get("pool") == p:
                    total += e.get("usd", 0.0)
    return total


def check(p: str, estimate_usd: float, what: str) -> None:
    s = spent(p)
    if s + estimate_usd > cap(p):
        raise BudgetExceeded(
            f"{what}: would bring {p} spend to ${s + estimate_usd:.2f}, over the cap ${cap(p):.2f}. "
            f"Raise PHYSTAX_{'BASETEN_' if p == 'baseten' else ''}BUDGET_USD if you mean it.")


def record(p: str, usd: float, kind: str, model: str, detail: dict | None = None) -> None:
    with _lock, open(_path(), "a") as f:
        f.write(json.dumps({"ts": time.time(), "pool": p, "kind": kind, "model": model,
                            "usd": round(usd, 6), **(detail or {})}) + "\n")


def chat_cost(model_id: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    pin, pcache, pout = PRICES.get(model_id, (1.25, 0.15, 4.25))
    fresh = max(0, prompt_tokens - cached_tokens)
    return (fresh * pin + cached_tokens * pcache + completion_tokens * pout) / 1e6


def chat_estimate(model_id: str, messages: list, max_out: int = 4000) -> float:
    chars = len(json.dumps(messages, default=str))
    return chat_cost(model_id, int(chars / 3.5), 0, max_out)


def sam_video_cost(n_frames: int) -> float:
    return max(n_frames, SAM_MIN_FRAMES) * SAM_VIDEO_PER_FRAME


def summary() -> str:
    return (f"Meta ${spent('meta'):.2f} / ${cap('meta'):.0f} cap, "
            f"Baseten ${spent('baseten'):.2f} / ${cap('baseten'):.0f} cap")


if __name__ == "__main__":
    print(summary())
