"""Lab UI server: start an episode, watch experiments with SAM overlays,
see the locked prediction, the test reveal, the grade, and verify the ledger.
Also replays any finished run from disk (the demo-day safety net).

    uvicorn phystax.web.app:app --port 8000
"""
from __future__ import annotations

import glob
import json
import os
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..bench.conditions import CONDITIONS, sam_backend
from ..config import AGENT_MODELS, RUNS_DIR
from ..grader.ledger import Ledger
from ..sim.laws import FAMILIES, public_spec

STATIC = os.path.join(os.path.dirname(__file__), "static")
app = FastAPI(title="PhysTax Lab")
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


class RunRequest(BaseModel):
    family: str = "power_drag"
    seed: int = 0
    degradation: int = 2
    condition: str = "sam_conf"
    agent: str = "reference"          # "reference" or a key of AGENT_MODELS
    review: bool = False


def _keys_present(provider: str) -> bool:
    env = {"meta": "MODEL_API_KEY", "baseten": "BASETEN_API_KEY",
           "together": "TOGETHER_API_KEY", "local": "LOCAL_BASE_URL"}[provider]
    return bool(os.environ.get(env))


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/config")
def config():
    return {
        "families": {k: public_spec(f) for k, f in FAMILIES.items()},
        "conditions": CONDITIONS,
        "agents": [{"key": "reference", "name": "Reference fitter (no LLM)", "available": True}] + [
            {"key": k, "name": m.name, "available": _keys_present(m.provider)}
            for k, m in AGENT_MODELS.items()],
        "sam_backend": sam_backend(),
        "sam_available": sam_backend() == "synthetic" or _keys_present("meta"),
    }


def _work(job_id: str, req: RunRequest, run_dir: str) -> None:
    try:
        if req.agent == "reference":
            from ..bench.reference import run_reference
            res = run_reference(req.family, req.seed, req.condition, req.degradation, run_dir=run_dir)
        else:
            from ..agent.loop import run_episode
            from ..agent.providers import ChatModel
            model = ChatModel(req.agent)
            reviewer = ChatModel("wonder") if req.review else None
            res = run_episode(req.family, req.seed, req.condition, model, req.degradation,
                              review=req.review, reviewer=reviewer, run_dir=run_dir)
        with _lock:
            JOBS[job_id].update(status="done", result_ok=bool(res.get("ledger_ok")))
    except Exception as exc:
        with _lock:
            JOBS[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}",
                                trace=traceback.format_exc()[-1500:])


@app.post("/api/runs")
def start_run(req: RunRequest):
    if req.family not in FAMILIES or req.condition not in CONDITIONS:
        raise HTTPException(400, "unknown family or condition")
    if req.agent != "reference" and req.agent not in AGENT_MODELS:
        raise HTTPException(400, "unknown agent")
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = os.path.join(RUNS_DIR, "ui", job_id)
    os.makedirs(run_dir, exist_ok=True)
    with _lock:
        JOBS[job_id] = {"id": job_id, "status": "running", "run_dir": run_dir,
                        "request": req.model_dump(), "started": time.time()}
    threading.Thread(target=_work, args=(job_id, req, run_dir), daemon=True).start()
    return {"id": job_id}


def _run_dir(job_id: str) -> str:
    if job_id in JOBS:
        return JOBS[job_id]["run_dir"]
    d = os.path.join(RUNS_DIR, "ui", job_id)
    if not os.path.isdir(d):
        raise HTTPException(404, "no such run")
    return d


def _safe_ledger(entries: list[dict], finished: bool) -> list[dict]:
    # the secret is only written after grading, but never show it mid-run
    return [e for e in entries if finished or e["kind"] != "secret_reveal"]


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    d = _run_dir(job_id)
    job = JOBS.get(job_id, {"id": job_id, "status": "done", "run_dir": d})
    lpath = os.path.join(d, "ledger.jsonl")
    entries = Ledger(lpath).read() if os.path.exists(lpath) else []
    rpath = os.path.join(d, "result.json")
    result = json.load(open(rpath)) if os.path.exists(rpath) else None
    epath = os.path.join(d, "sandbox", "experiments.json")
    experiments = {}
    if os.path.exists(epath):
        try:
            experiments = json.load(open(epath))
        except json.JSONDecodeError:
            experiments = {}          # being rewritten right now; next poll gets it
    finished = result is not None
    return {"job": job, "ledger": _safe_ledger(entries, finished),
            "experiments": experiments, "result": result}


@app.get("/api/runs")
def list_runs():
    out = []
    for r in sorted(glob.glob(os.path.join(RUNS_DIR, "ui", "*", "result.json")), reverse=True)[:50]:
        res = json.load(open(r))
        out.append({"id": os.path.basename(os.path.dirname(r)), "family": res["family"],
                    "condition": res["condition"], "model": res["model"],
                    "degradation": res["degradation"], "nrmse": res["score"]["prediction_nrmse"],
                    "pass": res["score"]["pass"], "sam_backend": res.get("sam_backend")})
    return out


@app.get("/api/runs/{job_id}/verify")
def verify(job_id: str):
    led = Ledger(os.path.join(_run_dir(job_id), "ledger.jsonl"))
    ok, why = led.verify()
    return {"chain_ok": ok, "detail": why, "commit_before_test": led.commit_precedes_test()}


@app.get("/files/{job_id}/{name}")
def files(job_id: str, name: str):
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".mp4"):
        raise HTTPException(400, "bad file name")
    p = os.path.join(_run_dir(job_id), name)
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="video/mp4")
