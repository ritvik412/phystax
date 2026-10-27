"""Run agent-written Python (numpy/scipy/sympy fits) in a separate process.

The working directory holds only the measurements the agent was given
(`experiments.json`), never simulator state. This is a local subprocess
sandbox for development; for the public demo, swap `run` for a Modal
sandbox (or the cookbook's sandboxed-execution recipe) so containment is
enforced by the platform, not by convention.
"""
from __future__ import annotations

import os
import subprocess
import sys

PRELUDE = (
    "import json, math\n"
    "import numpy as np\n"
    "import scipy.optimize as opt\n"
    "from scipy.integrate import solve_ivp\n"
    "data = json.load(open('experiments.json'))\n"
)


def run(code: str, workdir: str, timeout_s: int = 60, max_out: int = 6000) -> str:
    os.makedirs(workdir, exist_ok=True)
    script = os.path.join(workdir, "_cell.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(PRELUDE + code)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0",
           "OMP_NUM_THREADS": "2", "HOME": workdir}
    for var in ("SYSTEMROOT", "TEMP", "TMP"):      # Windows needs these to start Python
        if var in os.environ:
            env[var] = os.environ[var]
    try:
        p = subprocess.run([sys.executable, "-I", script], cwd=workdir, env=env,
                           capture_output=True, text=True, timeout=timeout_s)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    except subprocess.TimeoutExpired:
        out = f"[timeout after {timeout_s}s]"
    if len(out) > max_out:
        out = out[:max_out // 2] + "\n...[truncated]...\n" + out[-max_out // 2:]
    return out or "[no output; print what you want to see]"
