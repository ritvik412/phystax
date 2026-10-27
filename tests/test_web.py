"""UI server smoke test with the synthetic SAM backend (no keys)."""
import os
import time

os.environ.setdefault("PHYSTAX_RUNS", "/tmp/phystax_test_runs")
os.environ["PHYSTAX_SAM_BACKEND"] = "synthetic"

from fastapi.testclient import TestClient  # noqa: E402

from phystax.web.app import app  # noqa: E402


def test_reference_run_through_ui():
    c = TestClient(app)
    assert c.get("/api/config").json()["sam_backend"] == "synthetic"
    rid = c.post("/api/runs", json={"family": "nonlinear_spring", "condition": "sam_conf",
                                    "degradation": 1, "agent": "reference"}).json()["id"]
    for _ in range(240):
        d = c.get(f"/api/runs/{rid}").json()
        if d["result"] or d["job"]["status"] == "error":
            break
        time.sleep(0.5)
    assert d["result"], d["job"]
    assert d["result"]["sam_backend"] == "synthetic"
    v = c.get(f"/api/runs/{rid}/verify").json()
    assert v["chain_ok"] and v["commit_before_test"]
    assert c.get(f"/files/{rid}/e00.mp4").status_code == 200
    assert c.get(f"/files/{rid}/..%2Fledger.jsonl").status_code == 404
