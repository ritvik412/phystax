"""Environment self-check. Run first on any new machine:

    python -m phystax.doctor
"""
from __future__ import annotations

import os
import sys


def check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[ ok ] {name}: {detail}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    results = []
    results.append(check("python", lambda: sys.version.split()[0]))

    def libs():
        import cv2, numpy, scipy
        return f"numpy {numpy.__version__}, scipy {scipy.__version__}, opencv {cv2.__version__}"
    results.append(check("scientific stack", libs))

    def parser():
        import meta_sam_parser as m
        return "meta_sam_parser imports; decode_mask_to_raster present=" + str(hasattr(m, "decode_mask_to_raster"))
    results.append(check("SAM parser", parser))

    def ff():
        from .sim.ffmpeg_util import get_ffmpeg
        return get_ffmpeg()
    results.append(check("ffmpeg with libx264", ff))

    def render():
        import tempfile
        from .sim.engine import simulate
        from .sim.laws import FAMILIES
        from .sim.render import render as r
        from .sim.world import World
        w = World("power_drag", 0)
        tr = simulate(w.family, w.hidden, {"launch_speed_px_s": 300, "launch_angle_deg": 45}, 20)
        with tempfile.TemporaryDirectory() as d:
            out = r(tr, os.path.join(d, "t.mp4"), 0, family="power_drag")
            return f"rendered {out.n_frames} frames, {os.path.getsize(out.path)} bytes"
    results.append(check("render + encode", render))

    def keys():
        have = {k: bool(os.environ.get(k)) for k in ("MODEL_API_KEY", "BASETEN_API_KEY")}
        return ", ".join(f"{k}={'set' if v else 'not set'}" for k, v in have.items()) + \
               f"; PHYSTAX_SAM_BACKEND={os.environ.get('PHYSTAX_SAM_BACKEND', 'api')}"
    check("API keys (informational)", keys)

    print("\nAll required checks passed." if all(results) else "\nFix the FAIL lines above, then rerun.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
