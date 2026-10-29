"""Look at the IRIS real-video benchmark BEFORE downloading it.

IRIS (Khanbayov et al., 2026; CC BY 4.0): 220 real 4K/60 fps videos, 8 phenomena,
independently measured ground-truth parameters with uncertainty, fixed 7/1/2
splits. Dataset: huggingface.co/datasets/rasulkhanbayov/IRIS

    pip install huggingface_hub
    python scripts/iris_inspect.py                       # file tree + sizes, downloads nothing big
    python scripts/iris_inspect.py --metadata            # also fetch json/csv/md files (small)
    python scripts/iris_inspect.py --get "pendulum/*"    # fetch one phenomenon (check size first!)

Data goes to data/iris/ (gitignored). Paste the printed tree back so the loader
can be written against the real layout instead of a guess.
"""
import argparse
import os
from collections import defaultdict

REPO = "rasulkhanbayov/IRIS"


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", action="store_true")
    ap.add_argument("--get", nargs="*", default=None, help="glob patterns to download")
    ap.add_argument("--out", default="data/iris")
    a = ap.parse_args()
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    total = sum(sz for _, sz in files)
    by_top = defaultdict(lambda: [0, 0])
    for name, sz in files:
        top = name.split("/")[0] if "/" in name else "(root)"
        by_top[top][0] += 1
        by_top[top][1] += sz
    print(f"{REPO}: {len(files)} files, {total / 1e9:.2f} GB total")
    for top, (n, sz) in sorted(by_top.items()):
        print(f"  {top:40s} {n:5d} files  {sz / 1e9:8.3f} GB")
    exts = defaultdict(int)
    for name, _ in files:
        exts[os.path.splitext(name)[1] or "(none)"] += 1
    print("file types:", dict(exts))
    print("first 40 paths:")
    for name, sz in sorted(files)[:40]:
        print(f"  {sz / 1e6:9.1f} MB  {name}")
    if a.metadata:
        p = snapshot_download(REPO, repo_type="dataset", local_dir=a.out,
                              allow_patterns=["*.json", "*.csv", "*.md", "*.txt", "*.yaml"])
        print("metadata ->", p)
    if a.get:
        p = snapshot_download(REPO, repo_type="dataset", local_dir=a.out, allow_patterns=a.get)
        print("downloaded ->", p)


if __name__ == "__main__":
    main()
