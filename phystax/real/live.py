"""Live experiments on your own footage: drop, toss, swing or roll something
in front of a camera, tap it, and let the scientist predict the rest.

Protocol (the order is the point):
  1. Track the tapped object through the clip with SAM 3.1 (point prompt on
     the tapped frame, propagated forward). Border-touching frames are dropped.
  2. Split the track at `split`: the scientist sees ONLY the first part.
  3. For each image axis it chooses a law from the candidate bank (constant
     velocity, uniform acceleration, damped oscillation, exponential
     relaxation): by BIC, or by Muse Spark given the BIC table and the data.
  4. Predict every later frame from the chosen laws, with 80% bands from the
     fit residuals (inflated for serially correlated tracking error).
  5. Seal: hash the prediction into a ledger. Only after that is the second
     part of the track read, and the prediction graded against it.
Optional object size (cm) turns pixel accelerations into m/s² using SAM's
mask width, e.g. to read g from a dropped ball.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

import numpy as np

from ..grader.ledger import Ledger
from ..sim.ffmpeg_util import ENC_ARGS, get_ffmpeg
from .fit import BANK

LAW_TEXT = {"linear": "constant velocity", "const_acc": "uniform acceleration",
            "damped_osc": "damped oscillation", "exp_decay": "exponential relaxation"}

# Fixed in advance, never tuned on outcomes: the scientist sees the first half
# of the tracked motion and predicts the second half.
SEAL_FRACTION = 0.5

# Joint 2-D models for objects moving freely in front of a camera. Each pairs
# a law per image axis, so the choice is between physical situations rather
# than between curve shapes fitted to each axis independently.
JOINT_MODELS = {
    "free_flight": {"x": "linear", "y": "const_acc", "text": "free flight: steady sideways, accelerating vertically"},
    "constant_velocity": {"x": "linear", "y": "linear", "text": "constant velocity: no net force"},
    "uniform_acceleration": {"x": "const_acc", "y": "const_acc", "text": "uniform acceleration in both directions (sliding, rolling, tilted)"},
    "swing": {"x": "damped_osc", "y": "damped_osc", "text": "swinging: a damped oscillation"},
}


def normalise_video(src: str, dst: str, width: int = 640, fps: int = 30, max_s: float = 8.0) -> tuple[int, int, int]:
    """Any browser recording (WebM/MP4, variable frame rate) -> H.264 MP4 at a
    constant frame rate, which SAM accepts and which makes t = frame / fps exact."""
    subprocess.run([get_ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-t", str(max_s),
                    "-vf", f"fps={fps},scale={width}:-2", "-an", *ENC_ARGS, dst], check=True)
    import cv2
    cap = cv2.VideoCapture(dst)
    w, h, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    cap.release()
    return w, h, n


def point_prompt(frame: int, x: int, y: int, w: int, h: int) -> str:
    from meta_sam_parser import FramePrompt, PromptObject, PromptPoint, build_object_prompt
    return build_object_prompt(size=(w, h), objects=[PromptObject(id=1, prompts=[
        FramePrompt(frame=frame, box=None, points=[PromptPoint(x=int(x), y=int(y), label="positive")])])])


def sam_track(video: str, frame: int, x: int, y: int, w: int, h: int, n: int):
    from ..perception.sam_client import segment_video
    from ..perception.tracks import boxes_for_object, drop_truncated, track_for_object
    call = segment_video(video, point_prompt(frame, x, y, w, h), include_confidence=True,
                         n_frames=n, object_prompt=True)
    raw = track_for_object("object", call.records, 1, n, h)
    tr = drop_truncated(raw)
    meta = {"latency_s": call.latency_s, "usd": call.usd, "cached": call.cached,
            "frames_tracked": int(np.isfinite(raw.x).sum()), "frames_at_border": int(np.nansum(raw.truncated))}
    return tr, meta, boxes_for_object(call.records, 1, n)


def _band(fit, tt, t_future):
    """80% prediction band from prefix residuals, inflated for lag-1 correlation."""
    s = np.sqrt(fit.rss / max(fit.n - fit.k, 1))
    res = getattr(fit, "_resid", None)
    rho = 0.0
    if res is not None and len(res) > 3 and np.dot(res, res) > 0:
        rho = float(np.clip(np.dot(res[:-1], res[1:]) / np.dot(res, res), 0, 0.95))
    infl = np.sqrt((1 + rho) / (1 - rho))
    grow = 1 + (t_future - tt[-1]) / max(tt[-1] - tt[0], 1e-6)       # widen with extrapolation distance
    return 1.2816 * s * infl * grow


def _fit_axis(t, s, model):
    f = BANK[model](t, s)
    f._resid = s - f.pred if f.pred is not None and len(f.pred) == len(s) else None
    return f


def _predict(model, f, t, t0):
    p = f.params
    if model == "linear":
        return p["s0"] + p["v"] * t
    if model == "const_acc":
        return p["s0"] + p["v"] * t + 0.5 * p["a"] * t ** 2
    tt = t - t0
    if model == "damped_osc":
        return p["c"] + np.exp(-p["b"] * tt) * (p["A"] * np.cos(p["w"] * tt) + p["B"] * np.sin(p["w"] * tt))
    return p["c"] + p["A"] * np.exp(-p["k"] * tt)                     # exp_decay


def choose_laws(t, x, y, chooser=None) -> dict:
    """Joint-model BIC table (sum over both axes). `chooser(table, t, x, y)` may
    pick instead (Muse Spark); it must return one of JOINT_MODELS."""
    table = {}
    for name, m in JOINT_MODELS.items():
        try:
            table[name] = float(BANK[m["x"]](t, x).bic + BANK[m["y"]](t, y).bic)
        except Exception:
            table[name] = float("inf")
    pick, by, why = min(table, key=table.get), "BIC", ""
    if chooser is not None:
        try:
            c = chooser(table, t, x, y)
            if c and c.get("model") in JOINT_MODELS:
                pick, by, why = c["model"], "Muse Spark", c.get("rationale", "")
        except Exception as exc:
            why = f"agent unavailable ({type(exc).__name__}); lowest BIC used"
    return {"model": pick, "x": JOINT_MODELS[pick]["x"], "y": JOINT_MODELS[pick]["y"],
            "table": table, "by": by, "rationale": why}


def prepare_clip(video_src: str, out_dir: str) -> dict:
    """Step 1: normalise the upload so the frame the user taps is exactly the
    frame SAM will be prompted on (browser recordings have variable timing)."""
    os.makedirs(out_dir, exist_ok=True)
    w, h, n = normalise_video(video_src, os.path.join(out_dir, "clip.mp4"))
    meta = {"size": [w, h], "frames": n, "fps": 30.0}
    json.dump(meta, open(os.path.join(out_dir, "clip.json"), "w"))
    return meta


def run_live(video_src: str | None, tap_frame: int, tap_x: float, tap_y: float, view_w: float, view_h: float,
             out_dir: str, split: float = SEAL_FRACTION, object_cm: float | None = None,
             tracker: str = "sam", chooser=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    clip = os.path.join(out_dir, "clip.mp4")
    if video_src is not None:                      # one-shot use; the UI calls prepare_clip first
        prepare_clip(video_src, out_dir)
    meta_clip = json.load(open(os.path.join(out_dir, "clip.json")))
    (w, h), n, fps = meta_clip["size"], meta_clip["frames"], meta_clip["fps"]
    split = SEAL_FRACTION
    x_px, y_px = tap_x * w / view_w, tap_y * h / view_h          # tap in displayed pixels -> video pixels
    ledger = Ledger(os.path.join(out_dir, "ledger.jsonl"))
    ledger.append("live_start", {"frames": n, "fps": fps, "size": [w, h], "tap": [x_px, y_px, tap_frame],
                                 "split": split, "tracker": tracker})
    boxes = [None] * n
    if tracker == "sam":
        tr, meta, boxes = sam_track(clip, tap_frame, int(x_px), int(y_px), w, h, n)
        xs, ys_up, widths = tr.x, tr.y, np.sqrt(np.nan_to_num(tr.area, nan=np.nan))
    else:                                               # offline / fallback: classical blob tracker
        from ..perception.classical import track_video
        tr = track_video(clip, ["object"], h)["object"]
        xs, ys_up, widths, meta = tr.x, tr.y, np.sqrt(np.nan_to_num(tr.area, nan=np.nan)), {"tracker": "classical"}
    t_all = np.arange(n) / fps
    ok = np.isfinite(xs) & np.isfinite(ys_up) & (np.arange(n) >= tap_frame)
    idx = np.flatnonzero(ok)
    if len(idx) < 12:
        raise ValueError(f"SAM tracked the object in only {len(idx)} frames after the tap. Tap it on a frame where it is "
                         "clearly visible and already moving freely, and keep it inside the frame.")
    cut = idx[0] + int(split * (idx[-1] - idx[0]))
    pre, post = idx[idx <= cut], idx[idx > cut]
    if len(pre) < 8 or len(post) < 4:
        raise ValueError("the motion is too short to split; record a little longer")
    t_pre = t_all[pre]
    laws = choose_laws(t_pre, xs[pre], ys_up[pre], chooser)
    # speed readout (px/s, or m/s with a size), from the tracked frames
    v = np.full(n, np.nan)
    good = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys_up))
    if len(good) > 2:
        vx, vy = np.gradient(xs[good], t_all[good]), np.gradient(ys_up[good], t_all[good])
        v[good] = np.hypot(vx, vy)
    preds, bands, fits = {}, {}, {}
    for a, s in (("x", xs), ("y", ys_up)):
        f = _fit_axis(t_pre, s[pre], laws[a])
        fits[a] = {k: float(v) for k, v in f.params.items()}
        preds[a] = _predict(laws[a], f, t_all[post], t_pre[0]).tolist()
        bands[a] = _band(f, t_pre, t_all[post]).tolist()
    sealed = {"model": laws["model"], "laws": {"x": laws["x"], "y": laws["y"]}, "chosen_by": laws["by"], "constants": fits,
              "frames": post.tolist(), "x": preds["x"], "y": preds["y"], "band_x": bands["x"], "band_y": bands["y"]}
    sha = hashlib.sha256(json.dumps(sealed, sort_keys=True).encode()).hexdigest()
    sealed_at = time.time()
    ledger.append("commit", {"sha256": sha, "prediction": sealed})
    # ---- only now is the rest of the track read ----
    ax, ay = xs[post], ys_up[post]
    err = np.hypot(np.array(preds["x"]) - ax, np.array(preds["y"]) - ay)
    extent = float(np.hypot(np.ptp(xs[idx]), np.ptp(ys_up[idx]))) or 1.0
    inside = (np.abs(np.array(preds["x"]) - ax) <= np.array(bands["x"])) & \
             (np.abs(np.array(preds["y"]) - ay) <= np.array(bands["y"]))
    grade = {"rmse_px": float(np.sqrt(np.mean(err ** 2))), "error_share_of_path": float(np.sqrt(np.mean(err ** 2)) / extent),
             "coverage_80": float(inside.mean()), "pass": bool(np.sqrt(np.mean(err ** 2)) / extent < 0.05)}
    scale = None
    if object_cm and np.isfinite(widths[pre]).any():
        px_per_cm = float(np.nanmedian(widths[pre])) / float(object_cm)      # sqrt(mask area) ~ object width
        scale = {"px_per_m": px_per_cm * 100}
        if laws["y"] == "const_acc":
            scale["vertical_acceleration_m_s2"] = float(-fits["y"]["a"] / (px_per_cm * 100))
    ledger.append("reveal", {"commit_sha256": sha, "grade": grade})
    ok_chain, why = ledger.verify()
    result = {"clip": "clip.mp4", "size": [w, h], "fps": fps, "frames": n, "tap": [x_px, y_px, tap_frame],
              "track": {"x": [None if not np.isfinite(v) else round(float(v), 2) for v in xs],
                        "y": [None if not np.isfinite(v) else round(float(v), 2) for v in ys_up]},
              "split_frame": int(cut), "first_frame": int(idx[0]), "last_frame": int(idx[-1]),
              "model": laws["model"], "model_text": JOINT_MODELS[laws["model"]]["text"],
              "laws": {k: laws[k] for k in ("x", "y")}, "law_text": {a: LAW_TEXT[laws[a]] for a in ("x", "y")},
              "chosen_by": laws["by"], "rationale": laws.get("rationale", ""), "bic": laws["table"],
              "boxes": boxes, "speed": [None if not np.isfinite(s_) else round(float(s_), 1) for s_ in v],
              "seal_fraction": SEAL_FRACTION,
              "sealed": sealed, "sha256": sha, "sealed_at": sealed_at, "grade": grade, "scale": scale,
              "tracker_meta": meta, "ledger_ok": ok_chain and why == "ok"}
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=1, default=float)
    return result


def spark_chooser(model_key: str = "spark"):
    """Muse Spark picks the physical situation from the BIC table and a data sample."""
    from ..agent.providers import ChatModel
    m = ChatModel(model_key)
    m.spec = type(m.spec)(**{**m.spec.__dict__, "reasoning_effort": "low"})
    schema = {"type": "json_schema", "json_schema": {"name": "situation", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["model", "rationale"],
        "properties": {"model": {"type": "string", "enum": list(JOINT_MODELS)}, "rationale": {"type": "string"}}}}}

    def choose(table, t, x, y):
        step = max(1, len(t) // 25)
        sample = [{"t": round(float(a), 3), "x": round(float(b), 1), "y": round(float(c), 1)}
                  for a, b, c in zip(t[::step], x[::step], y[::step])]
        msg = m.chat([{"role": "user", "content":
                       "You are an experimental physicist. An object was filmed with a still camera; positions are "
                       "image pixels (x right, y up), time in seconds. This is only the first half of the motion. "
                       "Which physical situation is it? Options: " + json.dumps({k: v["text"] for k, v in JOINT_MODELS.items()})
                       + f". Lower BIC means a better fit to these frames: {json.dumps(table)}. Data: {json.dumps(sample)}. "
                       "Choose what will best predict how the motion continues; prefer the physically plausible "
                       "situation over a flexible one that only fits noise. Reply as JSON, rationale in one sentence."}],
                     response_format=schema, max_tokens=3000)
        return json.loads(msg.content)
    return choose
