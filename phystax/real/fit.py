"""Physics fits for REAL video tracks (pixels -> metres -> laws).

Used on real footage (IRIS, Delfys75, phone clips) where there is no simulator
and no exact trajectory, only ground-truth parameters measured independently.

Candidate bank (the model-selection problem IRIS poses):
  linear      s = s0 + v t                                 (constant velocity)
  const_acc   s = s0 + v t + a t^2 / 2                     (free fall, sliding)
  damped_osc  s = c + exp(-b t) (A cos w t + B sin w t)    (spring, pendulum, torsion)
  exp_decay   s = c + A exp(-k t)                          (first-order relaxation)
Model choice is by BIC on the motion's principal axis: the bank is fitted, the
lowest BIC wins, and the margin is reported so a weak preference is visible.

Pendulum: the pivot and the length come from a circle fitted to the 2-D path
(algebraic Kasa fit); theta(t) is then fitted as a damped oscillation, giving
w0^2 = w_d^2 + b^2 and g = L w0^2 (with the standard finite-amplitude period
correction). Every estimate carries a standard error from the fit covariance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Fit:
    model: str
    params: dict
    stderr: dict
    rss: float
    n: int
    k: int
    bic: float
    pred: np.ndarray = field(repr=False, default=None)


def _clean(t, s):
    t, s = np.asarray(t, float), np.asarray(s, float)
    ok = np.isfinite(t) & np.isfinite(s)
    return t[ok], s[ok]


def _bic(rss: float, n: int, k: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def _lin_fit(name, X, t, s, names):
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    pred = X @ coef
    rss = float(np.sum((s - pred) ** 2))
    n, k = len(s), X.shape[1]
    s2 = rss / max(n - k, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return Fit(name, dict(zip(names, coef)), dict(zip(names, se)), rss, n, k, _bic(rss, n, k), pred)


def fit_linear(t, s):
    t, s = _clean(t, s)
    return _lin_fit("linear", np.c_[np.ones_like(t), t], t, s, ["s0", "v"])


def fit_const_acc(t, s):
    t, s = _clean(t, s)
    f = _lin_fit("const_acc", np.c_[np.ones_like(t), t, 0.5 * t ** 2], t, s, ["s0", "v", "a"])
    return f


def _nonlin(name, model, p0, t, s, names, bounds=(-np.inf, np.inf)):
    res = least_squares(lambda p: model(p, t) - s, p0, bounds=bounds, max_nfev=4000)
    pred = model(res.x, t)
    rss = float(np.sum((s - pred) ** 2))
    n, k = len(s), len(p0)
    s2 = rss / max(n - k, 1)
    try:
        cov = s2 * np.linalg.pinv(res.jac.T @ res.jac)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        se = np.full(k, np.nan)
    return Fit(name, dict(zip(names, res.x)), dict(zip(names, se)), rss, n, k, _bic(rss, n, k), pred)


def fit_damped_osc(t, s):
    """s = c + exp(-b t)(A cos w t + B sin w t). Frequency seeded from the FFT,
    damping from a coarse grid, so the nonlinear fit starts in the right basin."""
    t, s = _clean(t, s)
    tt = t - t[0]
    dt = np.median(np.diff(tt))
    spec = np.abs(np.fft.rfft(s - s.mean()))
    freqs = np.fft.rfftfreq(len(s), dt)
    w0 = 2 * np.pi * freqs[1:][np.argmax(spec[1:])] if len(spec) > 2 else 1.0

    def model(p, x):
        c, A, B, b, w = p
        return c + np.exp(-b * x) * (A * np.cos(w * x) + B * np.sin(w * x))

    best = None
    for b0 in (0.0, 0.05, 0.3, 1.0):
        f = _nonlin("damped_osc", model, [s.mean(), s[0] - s.mean(), 0.0, b0, w0], tt, s,
                    ["c", "A", "B", "b", "w"])
        if best is None or f.rss < best.rss:
            best = f
    return best


def fit_exp_decay(t, s):
    t, s = _clean(t, s)
    tt = t - t[0]

    def model(p, x):
        c, A, k = p
        return c + A * np.exp(-k * x)

    return _nonlin("exp_decay", model, [s[-1], s[0] - s[-1], 1.0 / max(tt[-1], 1e-6)], tt, s,
                   ["c", "A", "k"], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]))


BANK = {"linear": fit_linear, "const_acc": fit_const_acc,
        "damped_osc": fit_damped_osc, "exp_decay": fit_exp_decay}


def principal_axis(x, y):
    """Project a 2-D track on its direction of largest motion."""
    P = np.c_[x, y]
    ok = np.all(np.isfinite(P), axis=1)
    c = P[ok].mean(0)
    _, _, vt = np.linalg.svd(P[ok] - c, full_matrices=False)
    d = vt[0]
    return (P - c) @ d, d


def select_model(t, s, bank=BANK) -> dict:
    fits = {name: fn(t, s) for name, fn in bank.items()}
    ranked = sorted(fits.values(), key=lambda f: f.bic)
    margin = ranked[1].bic - ranked[0].bic if len(ranked) > 1 else float("inf")
    return {"best": ranked[0], "margin_bic": float(margin), "all": fits}


def circle_fit(x, y):
    """Algebraic (Kasa) circle fit: returns centre (xc, yc) and radius R."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (xc, yc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(xc), float(yc), float(np.sqrt(c + xc ** 2 + yc ** 2))


def fit_pendulum(t, x_m, y_m, length_m: float | None = None) -> dict:
    """Pendulum from a metric 2-D track (y up). If the string length is
    measured, pass it: the period is precise but the arc curvature is a weak
    estimate of L for small swings."""
    t, x_m, y_m = np.asarray(t, float), np.asarray(x_m, float), np.asarray(y_m, float)
    xc, yc, R = circle_fit(x_m, y_m)
    if yc < np.nanmean(y_m):          # pivot must be above the bob
        pass
    theta = np.arctan2(x_m - xc, -(y_m - yc))
    f = fit_damped_osc(t, theta)
    b, w = f.params["b"], abs(f.params["w"])
    w0_sq = w ** 2 + b ** 2
    amp = float(np.nanmax(np.abs(theta - f.params["c"])))
    L = length_m if length_m is not None else R
    corr = (1 + amp ** 2 / 16) ** 2          # finite-amplitude period correction, squared
    g = L * w0_sq * corr
    se_w = f.stderr.get("w", np.nan)
    se_g = g * np.sqrt((2 * se_w / max(w, 1e-9)) ** 2 + ((0.002 / L) if length_m else 0.0) ** 2)
    return {"g": float(g), "g_se": float(se_g), "L": float(L), "L_from_arc": float(R),
            "pivot": (xc, yc), "damping_b": float(b), "omega_d": float(w),
            "amplitude_rad": amp, "fit": f}


def fit_free_fall(t, y_m) -> dict:
    """g from the vertical coordinate of a released object (y up)."""
    f = fit_const_acc(t, y_m)
    return {"g": float(-f.params["a"]), "g_se": float(f.stderr["a"]), "fit": f}
