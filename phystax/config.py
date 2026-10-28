"""Central configuration. Every model id and endpoint lives here so the
benchmark can swap providers without touching agent code."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .env import load_dotenv

load_dotenv()

META_BASE_URL = "https://api.meta.ai/v1"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"

SAM_MODEL = "sam-3.1"


@dataclass(frozen=True)
class AgentModel:
    name: str          # label used in tables
    provider: str      # "meta" | "baseten" | "together" | "local"
    model_id: str
    vision: bool       # can it look at frames? (Glimmer: False today)
    reasoning_effort: str | None = "medium"
    fallback_id: str | None = None   # used if model_id is refused (e.g. region-limited tier)


# Contributor tier: $0.10/$0.20 per 1M tokens vs $1.25/$4.25 standard, but
# "available in select countries". While free credits are in use the content
# may be used for product improvement anyway, so contributor costs nothing
# extra in privacy. If the API refuses it, we fall back to standard once.
AGENT_MODELS: dict[str, AgentModel] = {
    "spark": AgentModel("Muse Spark 1.3", "meta", "muse-spark-1.3-contributor", True,
                        fallback_id="muse-spark-1.3"),
    "spark-std": AgentModel("Muse Spark 1.3 (std)", "meta", "muse-spark-1.3", True),
    "wonder": AgentModel("Muse Wonder 1.0", "meta", "muse-wonder-1.0", True),
    # Hackathon override from the participant guide: use the -1026 slug.
    "spark-open": AgentModel("Muse Spark Open (Baseten)", "baseten",
                             "meta-models/muse-spark-open-1026", True, "high"),
    # Text + tools only; fill in the endpoint you actually serve it from.
    "glimmer": AgentModel("Muse Glimmer 30B", "local",
                          os.environ.get("GLIMMER_MODEL_ID", "meta-models/Muse-Glimmer-30B"),
                          False, None),
}

REVIEWER_MODEL = "wonder"

# Experiment and render budget
FPS = 30
FRAME_W, FRAME_H = 640, 480
MAX_EXPERIMENT_FRAMES = 150     # 5 s per clip
SAM_MIN_BILLED_FRAMES = 50      # API bills at least 50 frames per video request
MAX_EXPERIMENTS_PER_EPISODE = 6
MAX_AGENT_TURNS = 24

RUNS_DIR = os.environ.get("PHYSTAX_RUNS", "runs")
CACHE_DIR = os.environ.get("PHYSTAX_CACHE", ".cache")


def api_key(provider: str) -> str:
    env = {"meta": "MODEL_API_KEY", "baseten": "BASETEN_API_KEY",
           "together": "TOGETHER_API_KEY", "local": "LOCAL_API_KEY"}[provider]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"{env} is not set")
    return key


def base_url(provider: str) -> str:
    return {"meta": META_BASE_URL, "baseten": BASETEN_BASE_URL,
            "together": "https://api.together.xyz/v1",
            "local": os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1")}[provider]
