"""Assemble a Hugging Face Docker Space: results site at /, live experiment at
/live, simulator lab at /lab, in one container.

    python scripts/build_site.py --live <id>        # first: the site to ship
    python scripts/make_space.py                    # -> space/
    hf upload <user>/<space> space . --repo-type space

Then, in the Space's Settings -> Variables and secrets, add the secret
MODEL_API_KEY. Without it the results site and replays still work; live runs and
the lab's SAM channels need it. Spend is capped per container
(PHYSTAX_BUDGET_USD) and live runs per hour (PHYSTAX_LIVE_PER_HOUR).
"""
import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

README = """---
title: {title}
emoji: 🔭
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: An AI scientist that seals predictions and is graded by reality
---

# {title}

Results of the project at {repo}: an AI scientist (Muse Spark) that sees only
through SAM 3.1, seals its predictions in a hash-chained ledger before tests
are revealed, and is graded by what actually happens.

- `/` results and replays
- `/live` bring your own experiment (webcam or upload)
- `/lab` the simulated-universe lab
"""

DOCKERFILE = """FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV PHYSTAX_RUNS=/tmp/runs PHYSTAX_SITE_DIR=/app/site PHYSTAX_BUDGET_USD={budget} \\
    PHYSTAX_LIVE_PER_HOUR={per_hour} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EXPOSE 7860
CMD ["uvicorn", "phystax.web.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="space")
    ap.add_argument("--site", default="site")
    ap.add_argument("--title", default="PhysTax")
    ap.add_argument("--repo", default="https://github.com/ritvik412/phystax")
    ap.add_argument("--budget", type=float, default=30.0, help="USD cap per container lifetime")
    ap.add_argument("--per-hour", type=int, default=40, help="live runs allowed per hour")
    a = ap.parse_args()
    site = os.path.join(ROOT, a.site) if not os.path.isabs(a.site) else a.site
    if not os.path.exists(os.path.join(site, "index.html")):
        raise SystemExit("build the site first: python scripts/build_site.py")
    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    shutil.copytree(os.path.join(ROOT, "phystax"), os.path.join(out, "phystax"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(site, os.path.join(out, "site"))
    shutil.copy(os.path.join(ROOT, "requirements.txt"), out)
    shutil.copy(os.path.join(ROOT, "LICENSE"), out)
    open(os.path.join(out, "README.md"), "w").write(README.format(title=a.title, repo=a.repo))
    open(os.path.join(out, "Dockerfile"), "w").write(DOCKERFILE.format(budget=a.budget, per_hour=a.per_hour))
    open(os.path.join(out, ".gitignore"), "w").write(".env\n__pycache__/\n")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs) / 1e6
    print(f"space -> {out}/ ({size:.1f} MB). Upload with: hf upload <user>/<space> {a.out} . --repo-type space")
    print("Then add the secret MODEL_API_KEY in the Space settings.")


if __name__ == "__main__":
    main()
