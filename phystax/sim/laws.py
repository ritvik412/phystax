"""Hidden-law families.

Each family is a physical system whose force law has randomized, hidden
constants (and non-textbook exponents), so the agent cannot recall the answer
from pretraining. World units are pixels and seconds; y points up.

The agent only ever sees: the family's public description, the experiment
parameters it may set (with bounds), and measurements of what happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

State = np.ndarray  # shape (n_bodies, 4): x, y, vx, vy


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    unit: str
    doc: str


@dataclass
class LawFamily:
    name: str
    public_description: str          # what the agent is told
    bodies: list[str]                # SAM-friendly noun phrase per body (set by asset pack)
    experiment_params: list[ParamSpec]
    hidden_params: list[ParamSpec]   # sampled per episode, never shown
    fixed_bodies: set[int] = field(default_factory=set)
    accel: Callable[[State, dict], np.ndarray] = None          # (state, hidden) -> (n,2)
    initial_state: Callable[[dict], State] = None              # (experiment) -> state
    constrain: Callable[[State], State] | None = None          # contact rules after each step

    def sample_hidden(self, rng: np.random.Generator) -> dict:
        return {p.name: float(rng.uniform(p.lo, p.hi)) for p in self.hidden_params}

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


CX, CY = 320.0, 240.0  # frame centre in world px (render maps y-up to y-down)


# --- Family 1: central force with a non-integer power law --------------------
def _central_accel(s: State, h: dict) -> np.ndarray:
    a = np.zeros((s.shape[0], 2))
    d = s[1, :2] - s[0, :2]
    r = max(np.linalg.norm(d), 8.0)            # softening so close passes stay finite
    a[1] = -h["k"] * r ** (-h["alpha"]) * d / r
    return a


CONTACT_R = 37.0   # planet radius + marble radius in px: the marble lands and sticks


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
    return np.array([
        [CX, CY, 0.0, 0.0],
        [CX + e["start_radius_px"], CY,
         e["launch_speed_px_s"] * np.cos(th), e["launch_speed_px_s"] * np.sin(th)],
    ])


CENTRAL = LawFamily(
    name="central_power",
    public_description=(
        "A heavy planet sits fixed at the centre of the frame. A light marble is "
        "launched from a chosen distance to the right of the planet, at a chosen "
        "speed and angle. Some attractive central force acts between them. Its "
        "dependence on distance is unknown."
    ),
    bodies=["planet", "marble"],   # noun phrases SAM is prompted with
    experiment_params=[
        ParamSpec("start_radius_px", 60, 200, "px", "initial planet-marble distance"),
        ParamSpec("launch_speed_px_s", 0, 260, "px/s", "initial marble speed"),
        ParamSpec("launch_angle_deg", 0, 180, "deg", "0 = away from planet, 90 = upward"),
    ],
    hidden_params=[
        ParamSpec("alpha", 1.3, 3.1, "", "distance exponent"),
        ParamSpec("k", 0, 0, "px^(1+alpha)/s^2", "strength (set from alpha)"),
    ],
    fixed_bodies={0},
    accel=_central_accel,
    initial_state=_central_init,
    constrain=_central_constrain,
)


def _central_sample(rng: np.random.Generator) -> dict:
    alpha = float(rng.uniform(1.3, 3.1))
    # pick k so a circular orbit at r=120 px has speed ~ 90..160 px/s
    v_c = float(rng.uniform(90, 160))
    k = v_c ** 2 * 120.0 ** (alpha - 1)
    return {"alpha": alpha, "k": k}


CENTRAL.sample_hidden = _central_sample  # type: ignore[method-assign]


# --- Family 2: projectile with alien gravity and power-law drag ---------------
def _drag_accel(s: State, h: dict) -> np.ndarray:
    v = s[0, 2:]
    speed = np.linalg.norm(v)
    drag = -h["gamma"] * speed ** h["beta"] * (v / speed) if speed > 1e-9 else np.zeros(2)
    return np.array([[drag[0], drag[1] - h["g"]]])


def _drag_init(e: dict) -> State:
    th = np.deg2rad(e["launch_angle_deg"])
    sp = e["launch_speed_px_s"]
    return np.array([[40.0, 40.0, sp * np.cos(th), sp * np.sin(th)]])


DRAG = LawFamily(
    name="power_drag",
    public_description=(
        "A ball is launched from the lower-left corner at a chosen speed and "
        "angle. It falls under a uniform downward pull of unknown strength and "
        "is slowed by air resistance whose dependence on speed is unknown."
    ),
    bodies=["ball"],
    experiment_params=[
        ParamSpec("launch_speed_px_s", 80, 700, "px/s", "initial speed"),
        ParamSpec("launch_angle_deg", 5, 85, "deg", "above horizontal"),
    ],
    hidden_params=[
        ParamSpec("g", 120, 520, "px/s^2", "downward acceleration"),
        ParamSpec("beta", 1.15, 2.6, "", "drag speed exponent"),
        ParamSpec("gamma", 0, 0, "", "drag coefficient (set from beta)"),
    ],
    accel=_drag_accel,
    initial_state=_drag_init,
)


def _drag_sample(rng: np.random.Generator) -> dict:
    g = float(rng.uniform(120, 520))
    beta = float(rng.uniform(1.15, 2.6))
    v_term = float(rng.uniform(250, 600))     # terminal speed in px/s
    gamma = g / v_term ** beta
    return {"g": g, "beta": beta, "gamma": gamma}


DRAG.sample_hidden = _drag_sample  # type: ignore[method-assign]


# --- Family 3: damped hardening/softening oscillator (Duffing-type) -----------
def _osc_accel(s: State, h: dict) -> np.ndarray:
    x = s[0, 0] - CX
    vx = s[0, 2]
    ax = -h["k"] * x - h["c"] * x ** 3 - h["b"] * vx
    return np.array([[ax, 0.0]])


def _osc_init(e: dict) -> State:
    return np.array([[CX + e["displacement_px"], CY, e["initial_velocity_px_s"], 0.0]])


OSC = LawFamily(
    name="nonlinear_spring",
    public_description=(
        "A block slides on a frictionless horizontal rail and is tethered to the "
        "centre of the frame by a spring whose restoring force is not "
        "necessarily linear in displacement. Some damping may be present. You "
        "choose the initial displacement and velocity."
    ),
    bodies=["block"],
    experiment_params=[
        ParamSpec("displacement_px", -220, 220, "px", "initial offset from centre"),
        ParamSpec("initial_velocity_px_s", -300, 300, "px/s", "initial velocity"),
    ],
    hidden_params=[
        ParamSpec("k", 4, 40, "1/s^2", "linear stiffness"),
        ParamSpec("c", -3e-4, 2e-3, "1/(px^2 s^2)", "cubic stiffness"),
        ParamSpec("b", 0.0, 0.6, "1/s", "linear damping"),
    ],
    accel=_osc_accel,
    initial_state=_osc_init,
)

def _osc_sample(rng: np.random.Generator) -> dict:
    k = float(rng.uniform(4, 40))
    # cubic term between mildly softening and strongly hardening at |x| = 220 px,
    # bounded so the softening case never loses its restoring force in frame
    c = float(rng.uniform(-0.5, 2.0)) * k / 220.0 ** 2
    b = float(rng.uniform(0.0, 0.6))
    return {"k": k, "c": c, "b": b}


OSC.sample_hidden = _osc_sample  # type: ignore[method-assign]

FAMILIES: dict[str, LawFamily] = {f.name: f for f in (CENTRAL, DRAG, OSC)}


def public_spec(family: LawFamily) -> dict:
    """Everything the agent is allowed to know about a family."""
    return {
        "family": family.name,
        "description": family.public_description,
        "bodies": family.bodies,
        "units": "positions in pixels (origin bottom-left, y up), time in seconds",
        "experiment_parameters": [
            {"name": p.name, "min": p.lo, "max": p.hi, "unit": p.unit, "doc": p.doc}
            for p in family.experiment_params
        ],
    }
