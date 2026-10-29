"""Verify a self-hosted Glimmer before benchmarking it, and keep the evidence.

    LOCAL_BASE_URL=http://localhost:8000/v1 python scripts/glimmer_check.py

Runs the same chat / tool-call / JSON-schema checks as day1_llm_smoke.py
against the local server, then writes a provenance snapshot (GPU, git commit,
served model list) next to the vLLM server log in runs/provenance/.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LOCAL_API_KEY", "unused")

from phystax.provenance import snapshot  # noqa: E402

print("provenance before:", snapshot("glimmer_check_before"))
sys.argv = [sys.argv[0], "glimmer"]
try:
    runpy.run_path(os.path.join(os.path.dirname(__file__), "day1_llm_smoke.py"), run_name="__main__")
finally:
    print("provenance after:", snapshot("glimmer_check_after"))
