"""Tools the experimentalist may call during exploration. There is no tool
that reveals the law, the test outcome, or simulator state in camera
conditions; the only path to the world is `run_experiment`."""
from __future__ import annotations

import json
import os

from ..bench.conditions import measure
from ..config import MAX_EXPERIMENTS_PER_EPISODE
from ..grader.ledger import Ledger
from ..sim.world import World
from . import sandbox

TOOLS = [
    {"type": "function", "function": {
        "name": "run_experiment",
        "description": ("Run one experiment with the parameters you choose and record it. "
                        "Returns a summary; the full measured time series is saved as "
                        "data['<exp_id>'] in experiments.json for run_python."),
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["params", "duration_s"],
                       "properties": {
                           "params": {"type": "object", "description": "experiment parameters by name"},
                           "duration_s": {"type": "number", "description": "clip length, 1.7 to 5 s"}}}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Execute Python with numpy, scipy (opt = scipy.optimize, solve_ivp) "
                        "available and `data` preloaded from experiments.json. Print results."),
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["code"], "properties": {"code": {"type": "string"}}}}},
]


class ToolBox:
    def __init__(self, world: World, condition: str, ledger: Ledger, workdir: str,
                 noise_sigma_px: float = 3.0):
        self.world, self.condition, self.ledger = world, condition, ledger
        self.workdir = workdir
        self.noise_sigma_px = noise_sigma_px
        self.data: dict = {}
        self.diagnostics: list = []
        os.makedirs(workdir, exist_ok=True)
        self._flush()

    def _flush(self) -> None:
        json.dump(self.data, open(os.path.join(self.workdir, "experiments.json"), "w"))

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "run_experiment":
                return self._run_experiment(args)
            if name == "run_python":
                out = sandbox.run(args.get("code", ""), self.workdir)
                self.ledger.append("python", {"code": args.get("code", ""), "out_head": out[:500]})
                return out
            return f"unknown tool {name}"
        except Exception as exc:            # recoverable: tell the model what went wrong
            return f"ERROR: {type(exc).__name__}: {exc}"

    def _run_experiment(self, args: dict) -> str:
        if len(self.world.experiments) >= MAX_EXPERIMENTS_PER_EPISODE:
            return f"ERROR: experiment budget of {MAX_EXPERIMENTS_PER_EPISODE} used up; fit and commit."
        dur = float(args.get("duration_s", 3.0))
        dur = max(1.7, min(5.0, dur))        # >= 50 frames: the SAM billing floor
        exp = self.world.run_experiment(args.get("params", {}), dur)
        shown, diag = measure(exp, self.condition, self.world.family.fixed_bodies,
                              self.noise_sigma_px, seed=len(self.world.experiments))
        self.data[exp.exp_id] = {"params": exp.params, **shown}
        self._flush()
        self.diagnostics.append({"exp_id": exp.exp_id, **diag})
        self.ledger.append("experiment", {"exp_id": exp.exp_id, "params": exp.params,
                                          "duration_s": dur, "video": exp.video_path,
                                          "condition": self.condition})
        summary = {"exp_id": exp.exp_id, "params": exp.params, "n_frames": exp.n_frames,
                   "fps": round(1 / (shown["t"][1] - shown["t"][0])) if len(shown["t"]) > 1 else None,
                   "bodies": {}}
        for b, series in shown["bodies"].items():
            xs = [v for v in series["x"] if v is not None]
            summary["bodies"][b] = {
                "frames_measured": f"{len(xs)}/{exp.n_frames}",
                "first_xy": [series["x"][0], series["y"][0]],
                "last_xy": [series["x"][-1], series["y"][-1]],
            }
            if "conf" in series:
                cs = [c for c in series["conf"] if c is not None]
                summary["bodies"][b]["mean_conf"] = round(sum(cs) / len(cs), 3) if cs else None
        left = MAX_EXPERIMENTS_PER_EPISODE - len(self.world.experiments)
        summary["experiments_left"] = left
        return json.dumps(summary)
