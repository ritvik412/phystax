"""How the agent gets to *see* an experiment. Same world, same experiment,
different measurement channel. The gap between channels is the perception
tax.

  oracle        exact simulator coordinates
  noisy_oracle  exact coordinates + iid gaussian noise (sigma matched to SAM's
                measured error) -> separates "noise" from SAM-specific failure
                modes such as dropouts, identity swaps and occlusion bias
  sam           SAM 3.1 tracks, confidence withheld
  sam_conf      SAM 3.1 tracks with per-frame confidence
  classical     median background subtraction (the non-foundation-model baseline)
"""
from __future__ import annotations

import os

import numpy as np

from ..config import FRAME_H
from ..perception import classical
from ..perception.tracks import build_track, centroid_error_px
from ..sim.world import Experiment

CONDITIONS = ["oracle", "noisy_oracle", "sam", "sam_conf", "classical"]


def sam_backend() -> str:
    """'api' (real SAM 3.1) or 'synthetic' (offline stand-in, never reportable)."""
    return os.environ.get("PHYSTAX_SAM_BACKEND", "api")


def measure(exp: Experiment, condition: str, fixed_bodies: set[int],
            noise_sigma_px: float = 3.0, seed: int = 0) -> tuple[dict, dict]:
    """Returns (measurement shown to the agent, diagnostics kept from the agent)."""
    t = exp.traj.t
    truth = exp.traj.xy                       # (n, bodies, 2) world px
    bodies = exp.traj.bodies
    out = {"t": [round(float(v), 4) for v in t], "bodies": {}}
    diag = {"condition": condition, "per_body": {}}

    if condition in ("oracle", "noisy_oracle"):
        rng = np.random.default_rng(seed)
        for i, b in enumerate(bodies):
            xy = truth[:, i, :].copy()
            if condition == "noisy_oracle":
                xy += rng.normal(0, noise_sigma_px, xy.shape)
            out["bodies"][b] = {"x": xy[:, 0].round(2).tolist(), "y": xy[:, 1].round(2).tolist()}
        return out, diag

    if condition == "classical":
        tracks = classical.track_video(exp.video_path, bodies, FRAME_H)
        for i, b in enumerate(bodies):
            out["bodies"][b] = tracks[b].to_jsonable(include_conf=False)
            diag["per_body"][b] = centroid_error_px(tracks[b], truth[:, i, :])
        return out, diag

    if condition in ("sam", "sam_conf") and sam_backend() == "synthetic":
        from ..perception.synthetic_sam import synthetic_track
        diag["sam_backend"] = "synthetic"
        for i, b in enumerate(bodies):
            tr = synthetic_track(b, truth[:, i, :], exp.render_level, seed * 31 + i)
            out["bodies"][b] = tr.to_jsonable(include_conf=(condition == "sam_conf"))
            diag["per_body"][b] = centroid_error_px(tr, truth[:, i, :])
        return out, diag

    if condition in ("sam", "sam_conf"):
        from ..perception.sam_client import segment_video
        diag["sam_backend"] = "api"
        for i, b in enumerate(bodies):
            call = segment_video(exp.video_path, b, include_confidence=True)   # cached; same call for both
            mode = "largest" if i in fixed_bodies else "moving"
            tr = build_track(b, call.records, exp.n_frames, FRAME_H, mode)
            out["bodies"][b] = tr.to_jsonable(include_conf=(condition == "sam_conf"))
            d = centroid_error_px(tr, truth[:, i, :])
            d.update({"candidates": tr.n_candidates, "sam_latency_s": call.latency_s,
                      "cached": call.cached, "frames_billed": call.frames_billed})
            diag["per_body"][b] = d
        return out, diag

    raise ValueError(f"unknown condition {condition}")
