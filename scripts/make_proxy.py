"""Make a small, shareable copy of a video: downscaled, optionally trimmed or
resampled, with the MP4 index at the front (+faststart) so it stays readable.
Chat uploads are cut off around 12 MB, and an MP4 with its index at the end is
unreadable once truncated.

    python scripts/make_proxy.py data/iris/Pendulum/pendulum_45/01.mp4 pend45_01_proxy.mp4 \
        --width 640 --fps 20 --crf 28
    python scripts/make_proxy.py in.mp4 out.mp4 --start 0.3 --duration 1.2   # trim

Prints the original and proxy properties and the proxy size. Timing note: if
you change --fps, physics must use the PROXY's frame times (the script prints them).
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from phystax.sim.ffmpeg_util import get_ffmpeg  # noqa: E402


def props(p):
    c = cv2.VideoCapture(p)
    out = (int(c.get(3)), int(c.get(4)), c.get(5), int(c.get(7)))
    c.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None)
    a = ap.parse_args()
    w, h, fps, n = props(a.src)
    print(f"original: {w}x{h} @ {fps:.3f} fps, {n} frames ({n / fps:.2f} s)")
    vf = f"scale={a.width}:-2" + (f",fps={a.fps}" if a.fps else "")
    cmd = [get_ffmpeg(), "-y", "-loglevel", "error"]
    if a.start is not None:
        cmd += ["-ss", str(a.start)]
    cmd += ["-i", a.src]
    if a.duration is not None:
        cmd += ["-t", str(a.duration)]
    cmd += ["-vf", vf, "-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", a.dst]
    subprocess.run(cmd, check=True)
    w2, h2, fps2, n2 = props(a.dst)
    print(f"proxy:    {w2}x{h2} @ {fps2:.3f} fps, {n2} frames, {os.path.getsize(a.dst) / 1e6:.1f} MB -> {a.dst}")


if __name__ == "__main__":
    main()
