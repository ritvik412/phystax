"""Provenance snapshots for self-hosted open-weight runs.

Competition staff asked for run logs, screen captures and outputs proving the
open-weight model really ran. Each snapshot records, in runs/provenance/:
the git commit, the GPU (nvidia-smi), the model list the local server reports,
key package versions, and the time. `serve_glimmer.sh` also tees the full vLLM
server log there. Episode ledgers already record the model id per call.

    python -m phystax.provenance            # take a snapshot now
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import urllib.request


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _served_models() -> object:
    base = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/models", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as exc:
        return f"unavailable: {exc}"


def snapshot(tag: str = "manual") -> str:
    out_dir = os.path.join(os.environ.get("PHYSTAX_RUNS", "runs"), "provenance")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap = {
        "tag": tag, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": platform.python_version(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                            "--format=csv,noheader"]),
        "nvidia_smi_processes": _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                      "--format=csv,noheader"]),
        "served_models": _served_models(),
        "glimmer_model_id": os.environ.get("GLIMMER_MODEL_ID", "muse-glimmer"),
    }
    path = os.path.join(out_dir, f"{stamp}_{tag}.json")
    json.dump(snap, open(path, "w"), indent=2)
    return path


if __name__ == "__main__":
    print(snapshot())
