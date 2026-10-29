"""Real-video fitting bank: recover known physics from noisy synthetic tracks."""
import numpy as np

from phystax.real.fit import fit_free_fall, fit_pendulum, principal_axis, select_model

rng = np.random.default_rng(0)


def test_free_fall_recovers_g_and_wins_selection():
    t = np.arange(0, 0.45, 1 / 240)
    y = 1.0 - 0.5 * 9.81 * t ** 2 + rng.normal(0, 0.001, t.size)   # metres, 1 mm noise
    r = fit_free_fall(t, y)
    assert abs(r["g"] - 9.81) < 0.05
    assert select_model(t, y)["best"].model == "const_acc"


def test_pendulum_recovers_g_with_and_without_measured_length():
    g, L, b = 9.81, 0.6, 0.05
    t = np.arange(0, 6, 1 / 60)
    w0 = np.sqrt(g / L)
    th = np.radians(12) * np.exp(-b * t) * np.cos(np.sqrt(w0 ** 2 - b ** 2) * t)
    x = L * np.sin(th) + rng.normal(0, 0.0007, t.size)
    y = 1.5 - L * np.cos(th) + rng.normal(0, 0.0007, t.size)
    r_known = fit_pendulum(t, x, y, length_m=L)
    assert abs(r_known["g"] - g) / g < 0.01, r_known["g"]
    r_arc = fit_pendulum(t, x, y)
    assert abs(r_arc["L_from_arc"] - L) / L < 0.05


def test_selection_prefers_oscillator_for_spring_motion():
    t = np.arange(0, 4, 1 / 60)
    s = 0.05 * np.exp(-0.2 * t) * np.cos(2 * np.pi * 1.3 * t) + rng.normal(0, 0.0005, t.size)
    out = select_model(t, s)
    assert out["best"].model == "damped_osc" and out["margin_bic"] > 10
    assert abs(abs(out["best"].params["w"]) - 2 * np.pi * 1.3) < 0.05


def test_principal_axis_handles_tilted_motion():
    t = np.linspace(0, 1, 50)
    s = 2 * t ** 2
    x, y = s * np.cos(0.4), s * np.sin(0.4)
    p, d = principal_axis(x, y)
    assert abs(abs(d[0]) - np.cos(0.4)) < 1e-6


def test_measure_clip_analyze_pixels_to_physics():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "measure_clip", os.path.join(os.path.dirname(__file__), "..", "scripts", "measure_clip.py"))
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    px_per_m, H = 800.0, 1080
    t = np.arange(0, 0.4, 1 / 240)
    y_m = 1.2 - 0.5 * 9.81 * t ** 2
    y_px = H - y_m * px_per_m + rng.normal(0, 0.5, t.size)      # half-pixel tracking noise
    x_px = np.full_like(t, 700.0)
    r = mc.analyze(t, x_px, y_px, H, px_per_m, "free_fall")
    assert abs(r["g"] - 9.81) < 0.1
    r_auto = mc.analyze(t, x_px, y_px, H, px_per_m, "auto")
    assert r_auto["model"] == "const_acc"


def test_qfi_separates_law_form_from_parameter():
    from phystax.real.fit import qfi
    t = np.arange(0, 0.5, 1 / 60)
    good = 0.5 * 9.81 * t ** 2 + rng.normal(0, 1e-4, t.size)
    wrong_g = 0.5 * 2.0 * t ** 2 + rng.normal(0, 1e-4, t.size)       # right form, wrong parameter
    curved = 0.5 * 9.81 * t ** 2 + 30 * t ** 4                         # wrong form
    a, b, c = qfi(t, good), qfi(t, wrong_g), qfi(t, curved)
    assert a["qfi"] < 10 and abs(a["a"] - 9.81) < 0.05
    assert b["qfi"] < 10 and abs(b["a"] - 2.0) < 0.05                    # form fine, a wrong
    assert c["qfi"] > 100
