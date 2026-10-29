"""Pre-registered prediction schema and the deterministic grader.

No LLM judge anywhere in the scoring path: the grader compares numbers the
agent committed against what the hidden law actually produced.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..sim.engine import Trajectory
from ..sim.laws import key_params


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
                    "required": ["test_index", "body", "x", "y", "x_lo", "x_hi", "y_lo", "y_hi"],
                    "properties": {
                        "test_index": {"type": "integer"},
                        "body": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "number"}},
                        "y": {"type": "array", "items": {"type": "number"}},
                        # central 80% prediction interval for each coordinate
                        "x_lo": {"type": "array", "items": {"type": "number"}},
                        "x_hi": {"type": "array", "items": {"type": "number"}},
                        "y_lo": {"type": "array", "items": {"type": "number"}},
                        "y_hi": {"type": "array", "items": {"type": "number"}}}}},
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


INTERVAL_ALPHA = 0.2          # committed intervals are central 80% intervals


def interval_score(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = INTERVAL_ALPHA) -> np.ndarray:
    """Gneiting & Raftery (2007) interval score, a proper scoring rule: width,
    plus 2/alpha times the miss distance when the truth falls outside. Lower is
    better; it rewards intervals that are both narrow and honest."""
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


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
            row = {"test_index": i, "body": p["body"], "rmse_px": rmse, "nrmse": rmse / extent}
            if all(k in p for k in ("x_lo", "x_hi", "y_lo", "y_hi")):
                n = len(idx)
                bounds = [np.asarray(p[k], float)[:n] for k in ("x_lo", "x_hi", "y_lo", "y_hi")]
                if all(len(v) == n for v in bounds):
                    xs = interval_score(bounds[0], bounds[1], true[:, 0])
                    ys = interval_score(bounds[2], bounds[3], true[:, 1])
                    inside = ((true[:, 0] >= np.minimum(bounds[0], bounds[1])) & (true[:, 0] <= np.maximum(bounds[0], bounds[1])),
                              (true[:, 1] >= np.minimum(bounds[2], bounds[3])) & (true[:, 1] <= np.maximum(bounds[2], bounds[3])))
                    row["interval_score_norm"] = float(np.mean(np.concatenate([xs, ys]))) / extent
                    row["coverage"] = float(np.mean(np.concatenate(inside)))
            per_test.append(row)

    # parameter recovery for the true form's constants. Agents are never told
    # the form or its parameter names, so for agents this only scores when they
    # happen to name a constant the same way; the primary score is prediction.
    given = {p["name"]: p["value"] for p in commitment.get("parameters", [])}
    params = {}
    for k in key_params(family, hidden["form"]):
        if k in given and hidden[k] != 0:
            params[k] = {"true": hidden[k], "est": given[k],
                         "rel_err": abs(given[k] - hidden[k]) / abs(hidden[k])}
        else:
            params[k] = {"true": hidden[k], "est": given.get(k), "rel_err": None}

    nr = [t["nrmse"] for t in per_test if t.get("nrmse") is not None]
    isc = [t["interval_score_norm"] for t in per_test if "interval_score_norm" in t]
    cov = [t["coverage"] for t in per_test if "coverage" in t]
    return {
        "prediction_nrmse": float(np.mean(nr)) if nr else None,
        "interval_score_norm": float(np.mean(isc)) if isc else None,
        "coverage_80": float(np.mean(cov)) if cov else None,   # calibrated -> about 0.8
        "per_test": per_test,
        "parameters": params,
        "pass": bool(nr) and float(np.mean(nr)) < 0.05,   # within 5% of path size
        "true_form": hidden["form"],
    }
