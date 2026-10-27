"""Find an ffmpeg that can really encode H.264, and fail with a readable
message when none can.

Why this exists: on shared servers `ffmpeg` on PATH is sometimes a wrapper,
an old build without libx264, or a different program entirely. Instead of a
BrokenPipeError deep inside the renderer, we smoke-test each candidate by
encoding two tiny frames and use the first that works.

Search order: $PHYSTAX_FFMPEG, every `ffmpeg` on PATH, then the static binary
shipped by the `imageio-ffmpeg` package (installed via requirements.txt).
"""
from __future__ import annotations

import functools
import os
import subprocess
import tempfile

ENC_ARGS = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", "18", "-movflags", "+faststart"]


class FFmpegError(RuntimeError):
    pass


def _on_path(name: str) -> list[str]:
    exts = [""] if os.name != "nt" else os.environ.get("PATHEXT", ".EXE").split(";")
    found = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for e in exts:
            p = os.path.join(d, name + e)
            if os.path.isfile(p) and os.access(p, os.X_OK) and p not in found:
                found.append(p)
    return found


def _candidates() -> list[tuple[str, str]]:
    out = []
    if os.environ.get("PHYSTAX_FFMPEG"):
        out.append(("$PHYSTAX_FFMPEG", os.environ["PHYSTAX_FFMPEG"]))
    out += [("PATH", p) for p in _on_path("ffmpeg")]
    try:
        import imageio_ffmpeg
        out.append(("imageio-ffmpeg", imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:
        pass
    return out


def _smoke(exe: str) -> str | None:
    """None if `exe` encodes a tiny H.264 mp4, else a one-line reason."""
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "t.mp4")
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "64x64", "-r", "30", "-i", "-", *ENC_ARGS, target]
        try:
            p = subprocess.run(cmd, input=bytes(64 * 64 * 3 * 2), capture_output=True, timeout=30)
        except Exception as exc:
            return f"could not run: {type(exc).__name__}: {exc}"
        if p.returncode != 0 or not os.path.exists(target) or os.path.getsize(target) == 0:
            msg = (p.stderr or b"").decode("utf-8", "replace").strip().replace("\n", " ")[:200]
            return f"exit {p.returncode}: {msg or 'no output file'}"
    return None


@functools.lru_cache(maxsize=1)
def get_ffmpeg() -> str:
    problems = []
    for source, exe in _candidates():
        why = _smoke(exe)
        if why is None:
            return exe
        problems.append(f"  - {exe} (from {source}): {why}")
    raise FFmpegError(
        "No working ffmpeg with libx264 was found.\n"
        + ("\n".join(problems) if problems else "  (no ffmpeg candidates at all)")
        + "\nFix: `pip install imageio-ffmpeg` (bundles a static ffmpeg), or install ffmpeg "
          "and point PHYSTAX_FFMPEG at it.")
