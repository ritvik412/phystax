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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


# ---------------------------------------------------------------- live mode
LIVE: dict[str, dict] = {}


@app.get("/live")
def live_page():
    return FileResponse(os.path.join(STATIC, "live.html"))


def _live_work(job_id: str, src: str, out_dir: str, args: dict) -> None:
    from ..real.live import run_live, spark_chooser
    try:
        chooser = spark_chooser() if args.pop("agent") else None
        res = run_live(src, out_dir=out_dir, chooser=chooser, **args)
        with _lock:
            LIVE[job_id].update(status="done", result=res)
    except Exception as exc:
        with _lock:
            LIVE[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}",
                                trace=traceback.format_exc()[-1500:])


@app.post("/api/live")
async def start_live(video: UploadFile = File(...), tap_frame: int = Form(0), tap_x: float = Form(...),
                     tap_y: float = Form(...), view_w: float = Form(...), view_h: float = Form(...),
                     split: float = Form(0.45), object_cm: float = Form(0.0), agent: bool = Form(True),
                     tracker: str = Form("sam")):
    if tracker not in ("sam", "classical"):
        raise HTTPException(400, "tracker must be sam or classical")
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out_dir = os.path.join(RUNS_DIR, "live", job_id)
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(video.filename or "")[1].lower() or ".webm"
    src = os.path.join(out_dir, "upload" + (ext if ext in (".webm", ".mp4", ".mov", ".mkv") else ".webm"))
    with open(src, "wb") as f:
        f.write(await video.read())
    args = {"tap_frame": tap_frame, "tap_x": tap_x, "tap_y": tap_y, "view_w": view_w, "view_h": view_h,
            "object_cm": object_cm or None, "tracker": tracker,
            "agent": agent and tracker == "sam"}
    with _lock:
        LIVE[job_id] = {"id": job_id, "status": "running", "started": time.time()}
    threading.Thread(target=_live_work, args=(job_id, src, out_dir, args), daemon=True).start()
    return {"id": job_id}


@app.post("/api/live/prepare")
async def live_prepare(video: UploadFile = File(...)):
    """Step 1: store and normalise the clip; the UI then shows exactly this clip
    for tapping, so the tapped frame is the frame SAM is prompted on."""
    from ..real.live import prepare_clip
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out_dir = os.path.join(RUNS_DIR, "live", job_id)
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(video.filename or "")[1].lower()
    src = os.path.join(out_dir, "upload" + (ext if ext in (".webm", ".mp4", ".mov", ".mkv") else ".webm"))
    with open(src, "wb") as f:
        f.write(await video.read())
    try:
        meta = prepare_clip(src, out_dir)
    except Exception as exc:
        raise HTTPException(400, f"could not read this video: {exc}")
    return {"id": job_id, "clip": f"/livefiles/{job_id}/clip.mp4", **meta}


@app.post("/api/live/{job_id}/run")
async def live_run(job_id: str, tap_frame: int = Form(...), tap_x: float = Form(...), tap_y: float = Form(...),
                   view_w: float = Form(...), view_h: float = Form(...), object_cm: float = Form(0.0),
                   agent: bool = Form(True), tracker: str = Form("sam")):
    out_dir = os.path.join(RUNS_DIR, "live", job_id)
    if "/" in job_id or ".." in job_id or not os.path.exists(os.path.join(out_dir, "clip.json")):
        raise HTTPException(404, "prepare the clip first")
    if tracker not in ("sam", "classical"):
        raise HTTPException(400, "tracker must be sam or classical")
    args = {"tap_frame": tap_frame, "tap_x": tap_x, "tap_y": tap_y, "view_w": view_w, "view_h": view_h,
            "object_cm": object_cm or None, "tracker": tracker, "agent": agent and tracker == "sam"}
    with _lock:
        LIVE[job_id] = {"id": job_id, "status": "running", "started": time.time()}
    threading.Thread(target=_live_work, args=(job_id, None, out_dir, args), daemon=True).start()
    return {"id": job_id}


@app.get("/api/live/{job_id}")
def get_live(job_id: str):
    if job_id in LIVE:
        return LIVE[job_id]
    p = os.path.join(RUNS_DIR, "live", job_id, "result.json")
    if os.path.exists(p):
        return {"id": job_id, "status": "done", "result": json.load(open(p))}
    raise HTTPException(404, "no such live run")


@app.get("/livefiles/{job_id}/clip.mp4")
def live_clip(job_id: str):
    if "/" in job_id or ".." in job_id:
        raise HTTPException(400, "bad id")
    p = os.path.join(RUNS_DIR, "live", job_id, "clip.mp4")
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="video/mp4")
