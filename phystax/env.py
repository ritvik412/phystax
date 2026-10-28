"""Load `.env` from the repo root into os.environ (no extra dependency).
Existing environment variables always win, so `export X=...` overrides the file."""
from __future__ import annotations

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(path: str | None = None) -> None:
    for candidate in ([path] if path else [os.path.join(os.getcwd(), ".env"), os.path.join(_ROOT, ".env")]):
        if not candidate or not os.path.exists(candidate):
            continue
        with open(candidate, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
        return
