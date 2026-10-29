"""Hidden-law families.

Each family is a physical system with TWO structurally different candidate
laws ("forms"). Per seed the world picks one form and draws its constants,
so the agent must do model selection (which law?) and system identification
(which constants?), not only fit known parameters.

Why two forms matter (and how this differs from NewtonBench): NewtonBench
shifts canonical laws to a fixed menu of variants (e.g. r^-1.5, r^-2.6) and
grades the symbolic form with an LLM judge. Here constants and exponents are
continuous per seed (no finite menu to memorise), forms differ in structure
(power law vs screened exponential, power drag vs linear+quadratic drag,
viscous vs dry friction), and grading is purely numeric: a wrong form shows
up as wrong predictions on held-out tests, with no judge in the loop.

World units are pixels and seconds; y points up. The agent sees only the
family's public description, the experiment parameters and measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

State = np.ndarray  # (n_bodies, 4): x, y, vx, vy
CX, CY = 320.0, 240.0


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    unit: str
    doc: str


@dataclass
class Form:
    """One candidate law. `fit` maps a parameter to its fitting transform:
    ("log",) fits log(value); ("lin", s) fits value / s. `starts` are
    multi-start initial guesses in physical units (power laws have local minima)."""
    name: str
    equation: str
    accel: Callable[[State, dict], np.ndarray]
    sample: Callable[[np.random.Generator], dict]
    fit: dict[str, tuple]
    starts: list[dict]
    key_params: list[str]


@dataclass
class LawFamily:
    name: str
    public_description: str
    bodies: list[str]                       # noun phrases SAM is prompted with
    experiment_params: list[ParamSpec]
    forms: dict[str, Form]
    initial_state: Callable[[dict], State]
    fixed_bodies: set[int] = field(default_factory=set)
    constrain: Callable[[State], State] | None = None

    def accel(self, s: State, hidden: dict) -> np.ndarray:
        return self.forms[hidden["form"]].accel(s, hidden)

    def sample_hidden(self, rng: np.random.Generator, form: str | None = None) -> dict:
        name = form or sorted(self.forms)[int(rng.integers(len(self.forms)))]
        h = self.forms[name].sample(rng)
        h["form"] = name
        return h

    def validate_experiment(self, exp: dict) -> dict:
        clean = {}
        for p in self.experiment_params:
            if p.name not in exp:
                raise ValueError(f"missing experiment parameter '{p.name}'")
            v = float(exp[p.name])
            if not (p.lo <= v <= p.hi):
                raise ValueError(f"{p.name}={v} outside [{p.lo}, {p.hi}] {p.unit}")
            clean[p.name] = v
        return clean


# ---------------------------------------------------------------------------
# Family 1: a marble around a fixed planet
CONTACT_R = 37.0   # planet radius + marble radius: the marble lands and sticks


def _central_constrain(s: State) -> State:
    d = s[1, :2] - s[0, :2]
    r = np.linalg.norm(d)
    if r < CONTACT_R:
        s = s.copy()
        s[1, :2] = s[0, :2] + d / max(r, 1e-9) * CONTACT_R
        s[1, 2:] = 0.0
    return s


def _central_init(e: dict) -> State:
    th = np.deg2rad(e["launch_angle_deg"])
    return np.array([[CX, CY, 0.0, 0.0],
                     [CX + e["start_radius_px"], CY,
                      e["launch_speed_px_s"] * np.cos(th), e["launch_speed_px_s"] * np.sin(th)]])


def _radial(s: State, mag_fn) -> np.ndarray:
    a = np.zeros((s.shape[0], 2))
    d = s[1, :2] - s[0, :2]
    r = max(np.linalg.norm(d), 8.0)
    a[1] = -mag_fn(r) * d / r
    return a


def _power_sample(rng):
    alpha = float(rng.uniform(1.3, 3.1))
    v_c = float(rng.uniform(90, 160))              # circular speed at r = 120 px
    return {"alpha": alpha, "k": v_c ** 2 * 120.0 ** (alpha - 1)}


def _screened_sample(rng):
    lam = float(rng.uniform(70, 260))              # screening length, px
    v_c = float(rng.uniform(90, 160))
    return {"lam": lam, "k": v_c ** 2 * 120.0 * np.exp(120.0 / lam)}


CENTRAL = LawFamily(
    name="central_power",
    public_description=(
        "A heavy planet sits fixed at the centre of the frame. A light marble is launched "
        "from a chosen distance to the right of the planet, at a chosen speed and angle. "
        "Some attractive central force acts between them. Its dependence on distance is unknown."),
    bodies=["planet", "marble"],
    experiment_params=[
        ParamSpec("start_radius_px", 60, 200, "px", "initial planet-marble distance"),
        ParamSpec("launch_speed_px_s", 0, 260, "px/s", "initial marble speed"),
        ParamSpec("launch_angle_deg", 0, 180, "deg", "0 = away from planet, 90 = upward"),
    ],
    forms={
        "power": Form("power", "a = -k r^(-alpha) r_hat",
                      lambda s, h: _radial(s, lambda r: h["k"] * r ** (-h["alpha"])),
                      _power_sample, {"alpha": ("lin", 1.0), "k": ("log",)},
                      [{"alpha": a, "k": 120.0 ** 2 * 120.0 ** (a - 1)} for a in (1.5, 2.0, 2.5, 3.0)],
                      ["alpha"]),
        "screened": Form("screened", "a = -k exp(-r/lam) / r^2 r_hat",
                         lambda s, h: _radial(s, lambda r: h["k"] * np.exp(-r / h["lam"]) / r ** 2),
                         _screened_sample, {"lam": ("log",), "k": ("log",)},
                         [{"lam": l, "k": 120.0 ** 3 * np.exp(120.0 / l)} for l in (80.0, 140.0, 240.0)],
                         ["lam"]),
    },
    initial_state=_central_init, fixed_bodies={0}, constrain=_central_constrain,
)


# ---------------------------------------------------------------------------
# Family 2: a projectile with alien gravity and unknown air resistance
def _drag_init(e: dict) -> State:
    th = np.deg2rad(e["launch_angle_deg"])
    sp = e["launch_speed_px_s"]
    return np.array([[40.0, 40.0, sp * np.cos(th), sp * np.sin(th)]])


def _drag_accel(mag_fn, g):
    def f(s: State) -> np.ndarray:
        v = s[0, 2:]
        speed = np.linalg.norm(v)
        drag = -mag_fn(speed) * v / speed if speed > 1e-9 else np.zeros(2)
        return np.array([[drag[0], drag[1] - g]])
    return f


def _pdrag_sample(rng):
    g, beta, v_t = float(rng.uniform(120, 520)), float(rng.uniform(1.15, 2.6)), float(rng.uniform(250, 600))
    return {"g": g, "beta": beta, "gamma": g / v_t ** beta}


def _mdrag_sample(rng):
    g, v_t, f = float(rng.uniform(120, 520)), float(rng.uniform(250, 600)), float(rng.uniform(0.2, 0.8))
    return {"g": g, "c1": f * g / v_t, "c2": (1 - f) * g / v_t ** 2}   # terminal speed v_t


DRAG = LawFamily(
    name="power_drag",
    public_description=(
        "A ball is launched from the lower-left corner at a chosen speed and angle. It falls "
        "under a uniform downward pull of unknown strength and is slowed by air resistance "
        "whose dependence on speed is unknown."),
    bodies=["ball"],
    experiment_params=[
        ParamSpec("launch_speed_px_s", 80, 700, "px/s", "initial speed"),
        ParamSpec("launch_angle_deg", 5, 85, "deg", "above horizontal"),
    ],
    forms={
        "power": Form("power", "a = -g y_hat - gamma |v|^beta v_hat",
                      lambda s, h: _drag_accel(lambda v: h["gamma"] * v ** h["beta"], h["g"])(s),
                      _pdrag_sample, {"g": ("log",), "beta": ("lin", 1.0), "gamma": ("log",)},
                      [{"g": 300.0, "beta": b, "gamma": 300.0 / 400.0 ** b} for b in (1.3, 1.9, 2.5)],
                      ["g", "beta"]),
        "mixed": Form("mixed", "a = -g y_hat - (c1 |v| + c2 |v|^2) v_hat",
                      lambda s, h: _drag_accel(lambda v: h["c1"] * v + h["c2"] * v * v, h["g"])(s),
                      _mdrag_sample, {"g": ("log",), "c1": ("log",), "c2": ("log",)},
                      [{"g": 300.0, "c1": f * 300 / 400, "c2": (1 - f) * 300 / 400 ** 2} for f in (0.3, 0.7)],
                      ["g", "c1", "c2"]),
    },
    initial_state=_drag_init,
)


# ---------------------------------------------------------------------------
# Family 3: a block on a rail tethered by a nonlinear spring
def _osc_init(e: dict) -> State:
    return np.array([[CX + e["displacement_px"], CY, e["initial_velocity_px_s"], 0.0]])


def _viscous(s, h):
    x, v = s[0, 0] - CX, s[0, 2]
    return np.array([[-h["k"] * x - h["c"] * x ** 3 - h["b"] * v, 0.0]])


def _dry(s, h):
    x, v = s[0, 0] - CX, s[0, 2]
    # Coulomb friction, smoothed over ~3 px/s so the ODE stays integrable
    return np.array([[-h["k"] * x - h["c"] * x ** 3 - h["mu"] * np.tanh(v / 3.0), 0.0]])


def _visc_sample(rng):
    k = float(rng.uniform(4, 40))
    return {"k": k, "c": float(rng.uniform(-0.5, 2.0)) * k / 220.0 ** 2, "b": float(rng.uniform(0.0, 0.6))}


def _dry_sample(rng):
    k = float(rng.uniform(4, 40))
    return {"k": k, "c": float(rng.uniform(-0.5, 2.0)) * k / 220.0 ** 2, "mu": float(rng.uniform(5, 60))}


OSC = LawFamily(
    name="nonlinear_spring",
    public_description=(
        "A block slides on a horizontal rail and is tethered to the centre of the frame by a "
        "spring whose restoring force is not necessarily linear in displacement. Some energy "
        "loss may be present; its form is unknown. You choose the initial displacement and velocity."),
    bodies=["block"],
    experiment_params=[
        ParamSpec("displacement_px", -220, 220, "px", "initial offset from centre"),
        ParamSpec("initial_velocity_px_s", -300, 300, "px/s", "initial velocity"),
    ],
    forms={
        "viscous": Form("viscous", "x'' = -k x - c x^3 - b x'", _viscous, _visc_sample,
                        {"k": ("log",), "c": ("lin", 1e-4), "b": ("lin", 1.0)},
                        [{"k": k, "c": 1e-4, "b": 0.2} for k in (8.0, 25.0)], ["k", "c", "b"]),
        "dry": Form("dry", "x'' = -k x - c x^3 - mu sign(x')", _dry, _dry_sample,
                    {"k": ("log",), "c": ("lin", 1e-4), "mu": ("lin", 10.0)},
                    [{"k": k, "c": 1e-4, "mu": mu} for k in (8.0, 25.0) for mu in (5.0, 20.0, 50.0)],
                    ["k", "c", "mu"]),
    },
    initial_state=_osc_init,
)

FAMILIES: dict[str, LawFamily] = {f.name: f for f in (CENTRAL, DRAG, OSC)}


PUBLIC_NAMES = {"central_power": "orbit", "power_drag": "projectile",
                "nonlinear_spring": "tethered_block"}   # internal ids hint at forms; never shown


def public_spec(family: LawFamily) -> dict:
    """Everything the agent may know: no internal id, form names or parameter names."""
    return {
        "system": PUBLIC_NAMES[family.name],
        "description": family.public_description,
        "bodies": family.bodies,
        "units": "positions in pixels (origin bottom-left, y up), time in seconds",
        "camera_frame_px": [640, 480],
        "note": "the camera only sees positions inside the frame; anything outside is not measured",
        "experiment_parameters": [
            {"name": p.name, "min": p.lo, "max": p.hi, "unit": p.unit, "doc": p.doc}
            for p in family.experiment_params],
    }


def key_params(family: str, form: str) -> list[str]:
    return FAMILIES[family].forms[form].key_params
