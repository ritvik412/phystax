"""SAM 3.1 over the Meta Model API, parsed with the official `meta_sam_parser`.

Facts this module relies on (from dev.meta.ai/docs/sam/segmenting and the
meta-sam protocol spec):
  * one concept (short noun phrase) per request; lists return zero matches
  * metadata values are strings: include_confidence="true", mask_encoding="one_bit"
  * mask payloads are base85, opaque: always decode with decode_mask_to_raster
  * a completed response with empty output_text is a valid zero-match result
  * video is billed per frame with a 50-frame minimum per request
  * results can vary slightly run to run -> we cache by content hash so the
    benchmark replays deterministically
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    video_segmentation_format,
)

from ..config import CACHE_DIR, SAM_MIN_BILLED_FRAMES, SAM_MODEL, api_key, base_url


@dataclass
class SamCall:
    phrase: str
    video_path: str
    records: list                 # SegmentationBoxRecord | SegmentationMaskRecord
    diagnostics: list
    latency_s: float
    frames_billed: int | None
    cached: bool


def _client():
    from openai import OpenAI
    return OpenAI(base_url=base_url("meta"), api_key=api_key("meta"))


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse(output_text: str):
    parser = video_segmentation_format().create_parser()
    parser.push(output_text)
    return parser.finish(CompletedOutcome()).result


def segment_video(video_path: str, phrase: str, include_confidence: bool = True,
                  use_cache: bool = True) -> SamCall:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(f"{_sha(video_path)}|{phrase}|{include_confidence}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"sam_{key}.json")
    if use_cache and os.path.exists(cache_path):
        blob = json.load(open(cache_path))
        res = _parse(blob["output_text"])
        return SamCall(phrase, video_path, list(res.records), list(res.diagnostics),
                       blob["latency_s"], blob.get("frames_billed"), True)

    client = _client()
    t0 = time.time()
    with open(video_path, "rb") as f:
        file = client.files.create(purpose="user_data", file=f)
    metadata = {"mask_encoding": "one_bit"}
    if include_confidence:
        metadata["include_confidence"] = "true"
    resp = client.responses.create(
        model=SAM_MODEL,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": phrase},
            {"type": "input_video", "file_id": file.id},
        ]}],
        metadata=metadata,
    )
    latency = time.time() - t0
    text = resp.output_text or ""
    usage = getattr(resp, "usage", None)
    billed = getattr(usage, "video_frames_processed", None) if usage else None
    json.dump({"output_text": text, "latency_s": latency, "frames_billed": billed},
              open(cache_path, "w"))
    res = _parse(text)
    return SamCall(phrase, video_path, list(res.records), list(res.diagnostics),
                   latency, billed, False)


def estimated_cost_usd(n_frames: int, n_requests: int = 1) -> float:
    return n_requests * max(n_frames, SAM_MIN_BILLED_FRAMES) * 0.20 / 1000.0


__all__ = ["SamCall", "segment_video", "estimated_cost_usd",
           "SegmentationBoxRecord", "SegmentationMaskRecord"]
