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

CONDITIONS = ["oracle", "noisy_oracle", "sam", "sam_conf", "sam_box", "sam_box_conf", "classical"]

# Concept channels use generic shape words: in the Oct 29 probe, specific nouns
# ("ball", "planet", "block", "marble") were found in only 4 of 12 attempts and
# flipped with the rendering style, while shape words were found every time.
CONCEPT_WORD = {"ball": "circle", "marble": "circle", "planet": "circle", "block": "rectangle"}

# Box half-sizes (px) used to point SAM at each body on frame 0. The position
# comes from the experiment set-up (where the experimenter placed the object),
# never from the simulated trajectory.
BOX_HALF = {"planet": (30, 30), "marble": (15, 15), "ball": (17, 17), "block": (24, 18)}


def setup_boxes(family, params: dict, frame_w: int, frame_h: int):
    """Frame-0 boxes from the known initial state of the experiment."""
    from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt
    s0 = family.initial_state(family.validate_experiment(params))
    objs = []
    for i, name in enumerate(family.bodies):
        x, y = float(s0[i, 0]), frame_h - float(s0[i, 1])
        hx, hy = BOX_HALF.get(name, (16, 16))
        box = PromptBox(left=int(max(0, x - hx)), top=int(max(0, y - hy)),
                        right=int(min(frame_w, x + hx)), bottom=int(min(frame_h, y + hy)))
        objs.append(PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, box=box, points=[])]))
    return build_object_prompt(size=(frame_w, frame_h), objects=objs)


def drop_truncated_enabled() -> bool:
    """Drop frames where SAM's box touches the border. Day-2 probe: those frames
    carry 5-8x the error of interior frames. PHYSTAX_DROP_TRUNCATED=0 to keep them."""
    return os.environ.get("PHYSTAX_DROP_TRUNCATED", "1") != "0"


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

    if condition in ("sam_box", "sam_box_conf"):
        from ..config import FRAME_W
        from ..perception.sam_client import segment_video
        from ..perception.tracks import track_for_object
        from ..sim.laws import FAMILIES
        family = next(f for f in FAMILIES.values() if f.bodies == bodies)
        prompt = setup_boxes(family, exp.params, FRAME_W, FRAME_H)
        call = segment_video(exp.video_path, prompt, include_confidence=True,
                             n_frames=exp.n_frames, object_prompt=True)  # one request, all bodies
        diag["sam_backend"] = "api"
        diag["prompt"] = prompt
        from ..perception.tracks import drop_truncated
        for i, b in enumerate(bodies):
            tr = track_for_object(b, call.records, i + 1, exp.n_frames, FRAME_H)
            if drop_truncated_enabled():
                tr = drop_truncated(tr)
            out["bodies"][b] = tr.to_jsonable(include_conf=(condition == "sam_box_conf"))
            d = centroid_error_px(tr, truth[:, i, :])
            d.update({"sam_latency_s": call.latency_s, "cached": call.cached})
            diag["per_body"][b] = d
        return out, diag

    if condition in ("sam", "sam_conf"):
        from ..perception.sam_client import segment_video
        diag["sam_backend"] = "api"
        for i, b in enumerate(bodies):
            call = segment_video(exp.video_path, CONCEPT_WORD.get(b, b), include_confidence=True,
                                 n_frames=exp.n_frames)   # cached; one call serves sam and sam_conf
            mode = "largest" if i in fixed_bodies else "moving"
            tr = build_track(b, call.records, exp.n_frames, FRAME_H, mode)
            if drop_truncated_enabled():
                from ..perception.tracks import drop_truncated
                tr = drop_truncated(tr)
            out["bodies"][b] = tr.to_jsonable(include_conf=(condition == "sam_conf"))
            d = centroid_error_px(tr, truth[:, i, :])
            d.update({"candidates": tr.n_candidates, "sam_latency_s": call.latency_s,
                      "cached": call.cached, "frames_billed": call.frames_billed})
            diag["per_body"][b] = d
        return out, diag

    raise ValueError(f"unknown condition {condition}")
