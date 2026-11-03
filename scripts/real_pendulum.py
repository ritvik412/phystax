"""Real-footage pendulum analysis: everything scale-free, nothing from metadata
unless you pass it.

    python scripts/real_pendulum.py pend45_01_proxy.mp4 --fps 20 --rope-length 0.50
    python scripts/real_pendulum.py data/iris/Pendulum/pendulum_45/01.mp4      # full 4K/60 fps

Measures, from the bob's track:
  * the period at every cycle (sub-frame peak timing) and its amplitude
  * law test 1, isochronism: does the period depend on amplitude as the full
    pendulum equation predicts, T(A) = T0 (1 + A^2/16 + 11 A^4/3072), or is it
    constant (small-angle law)? Compared by BIC.
  * law test 2, damping: which envelope describes the amplitude decay:
    exponential (viscous), linear (dry friction), hyperbolic (quadratic air
    drag), or viscous+dry? Compared by BIC.
  * g = L (2 pi / T0)^2 when a rope length is given (reported for both
    readings of "rope length": to the bob's surface and to its centre).
Tracker: colour segmentation of the bob (HSV range, default tennis-ball
yellow-green). Writes a figure and a JSON next to the clip under runs/real/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from phystax.real.fit import circle_fit  # noqa: E402


def track(path, hsv_lo, hsv_hi, max_width=960):
    cap = cv2.VideoCapture(path)
    fps = cap.get(5)
    xs, ys = [], []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if im.shape[1] > max_width:
            im = cv2.resize(im, (max_width, int(im.shape[0] * max_width / im.shape[1])))
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        m = cv2.morphologyEx(cv2.inRange(hsv, hsv_lo, hsv_hi), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        k, _, st, ce = cv2.connectedComponentsWithStats(m)
        if k < 2 or st[1:, 4].max() < 20:
            xs.append(np.nan)
            ys.append(np.nan)
            continue
        i = 1 + int(np.argmax(st[1:, 4]))
        xs.append(ce[i][0])
        ys.append(ce[i][1])
    return np.array(xs), np.array(ys), fps


def analyse(x, y, fps, rope_length=None, bob_radius=0.0335):
    t = np.arange(len(x)) / fps
    moving = np.flatnonzero(np.abs(np.diff(x)) > 1.0)
    start = int(moving[0]) if len(moving) else 0          # release
    ok = np.isfinite(x) & (np.arange(len(x)) >= start)
    X, Y, T = x[ok], -y[ok], t[ok]
    xc, yc, R = circle_fit(X, Y)
    th = np.arctan2(X - xc, -(Y - yc))
    pk, _ = find_peaks(th, distance=max(2, int(0.8 * fps)))
    tp, ap = [], []
    for i in pk:
        if 0 < i < len(th) - 1:
            a, b, c = th[i - 1], th[i], th[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den != 0 else 0.0
            tp.append(T[i] + off / fps)
            ap.append(b - 0.25 * (a - c) * off)
    tp, ap = np.array(tp), np.abs(np.array(ap))
    Tc, Am = np.diff(tp), 0.5 * (ap[1:] + ap[:-1])
    n = len(Tc)
    bic = lambda rss, k, m: m * np.log(rss / m) + k * np.log(m)  # noqa: E731
    corr = 1 + Am ** 2 / 16 + 11 * Am ** 4 / 3072
    T0 = float(np.median(Tc / corr))
    rss_nl = float(np.sum((Tc - T0 * corr) ** 2))
    rss_iso = float(np.sum((Tc - np.median(Tc)) ** 2))
    se_T0 = float(np.std(Tc - T0 * corr) / np.sqrt(n))
    s = tp - tp[0]
    env = {"exponential (viscous)": (lambda p, u: p[0] * np.exp(-p[1] * u), [ap[0], 0.01]),
           "linear (dry friction)": (lambda p, u: p[0] - p[1] * u, [ap[0], 0.001]),
           "hyperbolic (quadratic air drag)": (lambda p, u: p[0] / (1 + p[1] * u), [ap[0], 0.01]),
           "viscous + dry": (lambda p, u: (p[0] + p[2] / p[1]) * np.exp(-p[1] * u) - p[2] / p[1],
                             [ap[0], 0.01, 1e-4])}
    fits = {}
    for k, (f, p0) in env.items():
        r = least_squares(lambda p: f(p, s) - ap, p0)
        fits[k] = {"bic": float(bic(float(np.sum(r.fun ** 2)), len(p0), len(ap))), "params": r.x.tolist()}
    best = min(fits, key=lambda k: fits[k]["bic"])
    out = {"cycles": int(n), "amplitude_deg": [float(np.degrees(ap[0])), float(np.degrees(ap[-1]))],
           "duration_s": float(s[-1]), "T0_s": T0, "T0_se_s": se_T0,
           "isochronism_test": {"bic_nonlinear_pendulum": float(bic(rss_nl, 1, n)),
                                "bic_isochronous": float(bic(rss_iso, 1, n))},
           "damping_test": fits, "damping_best": best,
           "effective_length_if_g_9.81": float(9.81 * (T0 / (2 * np.pi)) ** 2)}
    if rope_length:
        out["g_rope_to_surface"] = float(rope_length * (2 * np.pi / T0) ** 2)
        out["g_rope_to_centre"] = float((rope_length + bob_radius) * (2 * np.pi / T0) ** 2)
    return out, (tp, ap, Tc, Am, T0, env, fits, s)


def figure(path, extra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tp, ap, Tc, Am, T0, env, fits, s = extra
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(s, np.degrees(ap), "o", ms=3, color="#444", label="measured amplitude")
    uu = np.linspace(0, s[-1], 300)
    for k, (f, _) in env.items():
        if k in ("exponential (viscous)", "hyperbolic (quadratic air drag)", "linear (dry friction)"):
            ax[0].plot(uu, np.degrees(f(np.array(fits[k]["params"]), uu)), lw=1.5,
                       label=f"{k}  BIC {fits[k]['bic']:.0f}")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("amplitude (deg)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Which damping law?", loc="left", fontsize=10)
    ax[1].plot(np.degrees(Am), Tc, "o", ms=3, color="#444", label="measured period")
    aa = np.linspace(Am.min(), Am.max(), 200)
    ax[1].plot(np.degrees(aa), T0 * (1 + aa ** 2 / 16 + 11 * aa ** 4 / 3072), lw=1.5, label="full pendulum law")
    ax[1].axhline(np.median(Tc), lw=1, ls="--", color="#999", label="small-angle law (constant)")
    ax[1].set_xlabel("amplitude (deg)")
    ax[1].set_ylabel("period (s)")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("Does the period depend on amplitude?", loc="left", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=None, help="override the container's fps")
    ap.add_argument("--rope-length", type=float, default=None, help="metres, e.g. IRIS rope_length")
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=[22, 70, 70])
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=[48, 255, 255])
    a = ap.parse_args()
    x, y, fps = track(a.video, tuple(a.hsv_lo), tuple(a.hsv_hi))
    fps = a.fps or fps
    res, extra = analyse(x, y, fps, a.rope_length)
    res.update(video=a.video, fps=fps, tracked_fraction=float(np.isfinite(x).mean()), tracker="colour (HSV)")
    out_dir = os.path.join("runs", "real", os.path.splitext(os.path.basename(a.video))[0])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "pendulum.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "damping_test"}, indent=2))
    print("damping BIC:", {k: round(v["bic"], 1) for k, v in res["damping_test"].items()})
    try:
        figure(os.path.join(out_dir, "pendulum.png"), extra)
    except ImportError:
        print("(figure skipped: pip install matplotlib)")
    print("->", out_dir)


if __name__ == "__main__":
    main()
