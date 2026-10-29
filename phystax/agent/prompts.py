import json

CHANNEL_NOTES = {
    "oracle": "Measurements are exact simulator coordinates.",
    "noisy_oracle": "Measurements are coordinates with random measurement noise.",
    "sam": ("Measurements come from a camera: an object-segmentation model tracked each body "
            "in the rendered video. Expect noise, missing frames (null) and occasional glitches."),
    "sam_conf": ("Measurements come from a camera: an object-segmentation model tracked each body "
                 "in the rendered video. Each frame carries the model's confidence `conf` in [0,1]; "
                 "null means the body was not found. Use the confidence however you judge best."),
    "sam_box": ("Measurements come from a camera: an object-tracking model was pointed at each body's "
                "starting position in the first frame and tracked it. Expect noise and possible lost frames (null)."),
    "sam_box_conf": ("Measurements come from a camera: an object-tracking model was pointed at each body's "
                     "starting position and tracked it. Each frame carries the tracker's confidence `conf`; in "
                     "calibration, high-confidence frames had several times lower error. null = not measured."),
    "classical": ("Measurements come from a camera using background subtraction. Expect noise, "
                  "missing frames (null) and possible confusion between objects."),
}

SYSTEM = """You are an experimental physicist working in a universe whose laws are unknown to you.
Textbook constants and exponents may be wrong here. Discover the law from evidence only.

Method:
1. Design informative experiments (vary parameters deliberately; you have a limited budget).
2. Use run_python to fit candidate laws to the measured data (numerical derivatives are noisy;
   prefer fitting simulated trajectories of a candidate ODE to positions).
3. Compare competing hypotheses quantitatively before settling on one.
4. Stop exploring when further experiments will not change your conclusion, then say DONE.

Rules: you cannot see the true law or any test outcome. Report numbers, not guesses."""


def task_message(spec: dict, condition: str) -> str:
    return (f"System under study:\n{json.dumps(spec, indent=2)}\n\n"
            f"Measurement channel: {CHANNEL_NOTES[condition]}\n\n"
            "The functional form of the law is unknown: several structurally different laws "
            "are plausible. At the end you will state the law you infer, its constants, and "
            "predict positions for held-out initial conditions. Begin.")


def commit_message(tests: list[dict], times: list[float], bodies: list[str]) -> str:
    return (
        "Exploration is closed. Pre-register your prediction now; it will be hashed and locked "
        "before the test runs.\n"
        f"Held-out test initial conditions (by test_index): {json.dumps(tests)}\n"
        f"Predict x and y (world px, y up) for each moving body in {bodies} at exactly these times "
        f"(seconds): {times}. One prediction object per (test_index, body), with len(x)=len(y)={len(times)}.\n"
        "Give the law as an equation, list its constants with short names, and use numbers you "
        "computed from your fits, not estimates from memory. For every predicted point also give a "
        "central 80% prediction interval (x_lo, x_hi, y_lo, y_hi): the truth should fall inside about "
        "80% of the time. Intervals are scored with a proper scoring rule, so too narrow and too wide "
        "both cost you; derive them from your fit uncertainty and measurement noise."
    )


REVIEWER_SYSTEM = """You are an independent referee. You see an experimentalist's evidence and draft
conclusion about an unknown physical law. Check it hard: is the functional form supported over the
alternatives? Were the parameter fits stable across experiments? Are predictions consistent with the
fitted law? Reply as JSON {"verdict": "accept"|"revise", "issues": [..]}. Be concise."""
