"""Measure physics in a REAL video clip with SAM 3.1 (your phone clip, an IRIS
or Delfys75 video converted to MP4).

    # free fall, concept prompt, scale from a known object size in the frame
    python scripts/measure_clip.py drop.mp4 --concept "tennis ball" --experiment free_fall \
        --px-per-m 812
    # pendulum, box prompt on frame 0 (left,top,right,bottom in pixels), measured string length
    python scripts/measure_clip.py swing.mp4 --box 900 400 960 460 --experiment pendulum \
        --px-per-m 1500 --length-m 0.62
    # unknown: let the candidate bank decide (BIC) on the motion's principal axis
    python scripts/measure_clip.py clip.mp4 --concept "ball" --experiment auto --px-per-m 900

Writes runs/clips/<name>/: track.csv (frame, t, x, y in px and m, confidence),
sheet.png (frames with the track), result.json (fit, uncertainty, model choice).
Large clips are downscaled to <=1920 px wide and trimmed with --max-seconds
before upload to keep SAM cost and file size down; the scale is adjusted.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from phystax import budget  # noqa: E402
from phystax.real.fit import fit_free_fall, fit_pendulum, principal_axis, select_model  # noqa: E402
from phystax.sim.ffmpeg_util import ENC_ARGS, get_ffmpeg  # noqa: E402


def probe(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n


def prepare(src, out, max_w, max_s):
    w, h, fps, n = probe(src)
    scale = min(1.0, max_w / w)
    vf = f"scale={int(w * scale) // 2 * 2}:-2"
    cmd = [get_ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-t", str(max_s), "-vf", vf,
           "-an", *ENC_ARGS, out]
    subprocess.run(cmd, check=True)
    return scale, fps


def analyze(t, x_px, y_px, frame_h, px_per_m, experiment, length_m=None):
    """Pixels -> metres (y up) -> fit. Separated from SAM so it is testable."""
    x_m = x_px / px_per_m
    y_m = (frame_h - y_px) / px_per_m
    ok = np.isfinite(x_m) & np.isfinite(y_m)
    t, x_m, y_m = t[ok], x_m[ok], y_m[ok]
    out = {"n_points": int(ok.sum())}
    if experiment == "free_fall":
        r = fit_free_fall(t, y_m)
        out.update(g=r["g"], g_se=r["g_se"], model="const_acc")
    elif experiment == "pendulum":
        r = fit_pendulum(t, x_m, y_m, length_m)
        out.update({k: r[k] for k in ("g", "g_se", "L", "L_from_arc", "damping_b", "omega_d", "amplitude_rad")})
        out["model"] = "damped_pendulum"
    else:
        s, d = principal_axis(x_m, y_m)
        sel = select_model(t, s)
        b = sel["best"]
        out.update(model=b.model, params={k: float(v) for k, v in b.params.items()},
                   stderr={k: float(v) for k, v in b.stderr.items()}, margin_bic=sel["margin_bic"],
                   bic={k: float(f.bic) for k, f in sel["all"].items()}, axis=[float(d[0]), float(d[1])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--concept", help="one short noun phrase, e.g. 'tennis ball'")
    g.add_argument("--box", nargs=4, type=int, metavar=("L", "T", "R", "B"), help="frame-0 box in ORIGINAL pixels")
    ap.add_argument("--experiment", choices=["free_fall", "pendulum", "auto"], default="auto")
    ap.add_argument("--px-per-m", type=float, required=True, help="scale in the ORIGINAL video")
    ap.add_argument("--length-m", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None, help="override (slow-mo files often report 30)")
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--yes", action="store_true", help="actually call SAM")
    a = ap.parse_args()

    from phystax.perception.sam_client import segment_video
    from phystax.perception.tracks import build_track, track_for_object

    name = os.path.splitext(os.path.basename(a.video))[0]
    out_dir = os.path.join("runs", "clips", name)
    os.makedirs(out_dir, exist_ok=True)
    clip = os.path.join(out_dir, "clip.mp4")
    scale, fps0 = prepare(a.video, clip, a.max_width, a.max_seconds)
    w, h, fps_enc, n = probe(clip)
    fps = a.fps or fps0
    est = budget.sam_video_cost(n)
    print(f"{name}: {w}x{h}, {n} frames at {fps:.1f} fps (scale {scale:.3f}); SAM cost ~${est:.3f}. {budget.summary()}")
    if not a.yes:
        print("Re-run with --yes to call SAM.")
        return
    if a.concept:
        call = segment_video(clip, a.concept, include_confidence=True, n_frames=n)
        tr = build_track(a.concept, call.records, n, h, mode="moving")
    else:
        from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt
        L, T, R, B = (int(v * scale) for v in a.box)
        prompt = build_object_prompt(size=(w, h), objects=[PromptObject(id=1, prompts=[
            FramePrompt(frame=0, box=PromptBox(left=L, top=T, right=R, bottom=B), points=[])])])
        call = segment_video(clip, prompt, include_confidence=True, n_frames=n, object_prompt=True)
        tr = track_for_object("object", call.records, 1, n, h)
    t = np.arange(n) / fps
    x_px, y_px = tr.x, h - tr.y            # back to image coordinates
    res = analyze(t, x_px, y_px, h, a.px_per_m * scale, a.experiment, a.length_m)
    res.update(video=a.video, frames=n, fps=fps, coverage=float(np.isfinite(tr.x).mean()),
               sam_latency_s=call.latency_s, sam_usd=call.usd)
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=2, default=float)
    with open(os.path.join(out_dir, "track.csv"), "w") as f:
        f.write("frame,t,x_px,y_px,conf\n")
        for i in range(n):
            f.write(f"{i},{t[i]:.5f},{x_px[i]:.2f},{y_px[i]:.2f},{tr.conf[i]:.3f}\n")
    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    tiles = []
    for f in np.linspace(0, len(frames) - 1, 6).astype(int):
        im = frames[f].copy()
        pts = [(int(x), int(y)) for x, y in zip(x_px[:f + 1], y_px[:f + 1]) if np.isfinite(x)]
        for p, q in zip(pts, pts[1:]):
            cv2.line(im, p, q, (0, 200, 0), 2)
        cv2.putText(im, f"t={t[f]:.3f}s", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        tiles.append(cv2.resize(im, (480, int(480 * h / w))))
    cv2.imwrite(os.path.join(out_dir, "sheet.png"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    print(json.dumps({k: v for k, v in res.items() if k != "bic"}, indent=2, default=float))
    print("->", out_dir, budget.summary())


if __name__ == "__main__":
    main()
