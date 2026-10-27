"""Pre-registered prediction schema and the deterministic grader.

No LLM judge anywhere in the scoring path: the grader compares numbers the
agent committed against what the hidden law actually produced.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..sim.engine import Trajectory

# Which hidden constants we score, per family (the agent is told these names).
KEY_PARAMS = {
    "central_power": ["alpha"],
    "power_drag": ["g", "beta"],
    "nonlinear_spring": ["k", "c", "b"],
}

COMMIT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "prediction_commitment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["law_equation", "parameters", "predictions", "self_confidence", "rationale"],
            "properties": {
                "law_equation": {"type": "string",
                                 "description": "the acceleration law you inferred, e.g. a = -k r^-alpha r_hat"},
                "parameters": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "value"],
                    "properties": {"name": {"type": "string"}, "value": {"type": "number"}}}},
                "predictions": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["test_index", "body", "x", "y"],
                    "properties": {
                        "test_index": {"type": "integer"},
                        "body": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "number"}},
                        "y": {"type": "array", "items": {"type": "number"}}}}},
                "self_confidence": {"type": "number"},
                "rationale": {"type": "string"},
            },
        },
    },
}


def probe_times(duration_s: float, n: int = 6, fps: int = 30) -> list[float]:
    """Probe times aligned to frame boundaries inside the test clip."""
    frames = np.linspace(round(0.5 * fps), round(duration_s * fps) - 1, n).round().astype(int)
    return [round(float(f) / fps, 4) for f in frames]


def commitment_hash(c: dict) -> str:
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def grade(commitment: dict, family: str, hidden: dict, tests: list[Trajectory],
          times: list[float], fps: int) -> dict:
    # trajectory prediction error, normalised by the size of the true path
    per_test = []
    for i, traj in enumerate(tests):
        idx = [min(len(traj.t) - 1, int(round(t * fps))) for t in times]
        for p in commitment.get("predictions", []):
            if p.get("test_index") != i or p.get("body") not in traj.bodies:
                continue
            b = traj.bodies.index(p["body"])
            true = traj.xy[idx, b, :]
            pred = np.stack([np.asarray(p["x"], float)[:len(idx)],
                             np.asarray(p["y"], float)[:len(idx)]], 1)
            if pred.shape != true.shape:
                per_test.append({"test_index": i, "body": p["body"], "nrmse": None,
                                 "error": "wrong number of probe points"})
                continue
            extent = np.linalg.norm(traj.xy[:, b, :].max(0) - traj.xy[:, b, :].min(0)) or 1.0
            rmse = float(np.sqrt(np.mean(np.sum((pred - true) ** 2, 1))))
            per_test.append({"test_index": i, "body": p["body"], "rmse_px": rmse,
                             "nrmse": rmse / extent})

    # parameter recovery for the scored constants
    given = {p["name"]: p["value"] for p in commitment.get("parameters", [])}
    params = {}
    for k in KEY_PARAMS[family]:
        if k in given and hidden[k] != 0:
            params[k] = {"true": hidden[k], "est": given[k],
                         "rel_err": abs(given[k] - hidden[k]) / abs(hidden[k])}
        else:
            params[k] = {"true": hidden[k], "est": given.get(k), "rel_err": None}

    nr = [t["nrmse"] for t in per_test if t.get("nrmse") is not None]
    return {
        "prediction_nrmse": float(np.mean(nr)) if nr else None,
        "per_test": per_test,
        "parameters": params,
        "pass": bool(nr) and float(np.mean(nr)) < 0.05,   # within 5% of path size
    }
