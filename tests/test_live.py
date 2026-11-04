"""Live mode end to end, offline: a rendered toss uploaded as WebM, classical
tracker in place of SAM, BIC as the chooser. Checks the seal ordering too."""
import json
import os
import subprocess
import time

import numpy as np

os.environ.setdefault("PHYSTAX_RUNS", "/tmp/phystax_test_runs")


def _toss_webm(path):
    from phystax.sim.engine import simulate
    from phystax.sim.ffmpeg_util import get_ffmpeg
    from phystax.sim.laws import FAMILIES
    from phystax.sim.render import render
    fam = FAMILIES["power_drag"]
    hidden = {"form": "power", "g": 300.0, "beta": 1.5, "gamma": 1e-6}     # nearly drag-free toss
    tr = simulate(fam, hidden, {"launch_speed_px_s": 420, "launch_angle_deg": 62}, 90)
    mp4 = path.replace(".webm", "_src.mp4")
    render(tr, mp4, 0, style_seed=3, family="power_drag")
    subprocess.run([get_ffmpeg(), "-y", "-loglevel", "error", "-i", mp4, "-c:v", "libvpx", "-b:v", "2M", path], check=True)
    return tr


def test_live_toss_offline(tmp_path):
    from fastapi.testclient import TestClient
    from phystax.web.app import app
    webm = str(tmp_path / "toss.webm")
    tr = _toss_webm(webm)
    x0, y0 = tr.xy[0, 0]
    c = TestClient(app)
    with open(webm, "rb") as f:
        rid = c.post("/api/live", files={"video": ("toss.webm", f, "video/webm")},
                     data={"tap_frame": 0, "tap_x": x0, "tap_y": 480 - y0, "view_w": 640, "view_h": 480,
                           "split": 0.45, "agent": "false", "tracker": "classical"}).json()["id"]
    for _ in range(120):
        j = c.get(f"/api/live/{rid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.5)
    assert j["status"] == "done", j
    R = j["result"]
    assert R["laws"]["y"] == "const_acc"                       # the vertical motion is a parabola
    assert R["ledger_ok"]
    led = [json.loads(l)["kind"] for l in open(os.path.join(os.environ["PHYSTAX_RUNS"], "live", rid, "ledger.jsonl"))]
    assert led.index("commit") < led.index("reveal")           # sealed before the rest was read
    assert R["grade"]["error_share_of_path"] < 0.05, R["grade"]
    assert c.get(f"/livefiles/{rid}/clip.mp4").status_code == 200


def test_live_two_step_flow(tmp_path):
    """prepare (normalise) -> tap on the normalised clip -> run; fixed seal point."""
    from fastapi.testclient import TestClient
    from phystax.real.live import SEAL_FRACTION
    from phystax.web.app import app
    webm = str(tmp_path / "toss2.webm")
    tr = _toss_webm(webm)
    x0, y0 = tr.xy[3, 0]
    c = TestClient(app)
    with open(webm, "rb") as f:
        prep = c.post("/api/live/prepare", files={"video": ("toss2.webm", f, "video/webm")}).json()
    assert prep["frames"] >= 80 and prep["size"] == [640, 480]
    c.post(f"/api/live/{prep['id']}/run", data={"tap_frame": 3, "tap_x": x0, "tap_y": 480 - y0, "view_w": 640,
                                                 "view_h": 480, "agent": "false", "tracker": "classical"})
    for _ in range(120):
        j = c.get(f"/api/live/{prep['id']}").json()
        if j["status"] != "running":
            break
        time.sleep(0.5)
    R = j["result"]
    assert R["model"] == "free_flight", R["bic"]
    assert R["seal_fraction"] == SEAL_FRACTION == 0.5
    assert R["grade"]["error_share_of_path"] < 0.05
    assert len(R["speed"]) == R["frames"]
