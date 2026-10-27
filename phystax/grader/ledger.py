"""Append-only, hash-chained experiment ledger (JSONL).

Every event (episode start with the world's secret-law hash, each
experiment, each tool call, the pre-registered prediction, the test reveal,
the grade) is chained to the previous entry's hash. Editing or reordering
any line breaks `verify()`. This is the runtime analogue of the Muse Code
append-only audit-log recipe, and it is what lets a judge check that the
prediction was locked before the test was rendered.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._last = GENESIS
        self._seq = 0
        if os.path.exists(path):
            for e in self.read():
                self._last, self._seq = e["hash"], e["seq"] + 1

    def append(self, kind: str, payload: dict[str, Any]) -> dict:
        entry = {"seq": self._seq, "ts": time.time(), "kind": kind,
                 "payload": payload, "prev": self._last}
        entry["hash"] = _digest(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last, self._seq = entry["hash"], self._seq + 1
        return entry

    def read(self) -> list[dict]:
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, e in enumerate(self.read()):
            if e["seq"] != i:
                return False, f"sequence gap at line {i}"
            if e["prev"] != prev:
                return False, f"broken chain at seq {i}"
            if _digest(e) != e["hash"]:
                return False, f"tampered entry at seq {i}"
            prev = e["hash"]
        return True, "ok"

    def commit_precedes_test(self) -> bool:
        kinds = [e["kind"] for e in self.read()]
        return "commit" in kinds and "test_reveal" in kinds and \
            kinds.index("commit") < kinds.index("test_reveal")
