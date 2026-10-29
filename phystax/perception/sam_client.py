"""SAM 3.1 over the Meta Model API.

This follows Meta's reference client (meta-model-cookbook/07_sam/01_api_basics/
python/sam_api.py) line for line where it matters:

  * upload with files.create(file=(name, bytes, mime), purpose="user_data");
    the id must start with "file-"
  * responses.create(model="sam-3.1", stream=True, input=[message with
    input_text = ONE short noun phrase, and input_video = {file_id}])
  * the API streams response.content_part.done but not
    response.output_text.done, which the parser needs to close a lane, so we
    synthesize it (same as the reference `stream_events`)
  * parse with meta_sam_parser.parse_responses_stream(..., video_segmentation_format())
    and reject results whose outcome is not "completed" or that carry
    parser diagnostics of severity "error"

Additions for the benchmark:
  * metadata {"include_confidence": "true", "mask_encoding": "one_bit"}
    (string values, per dev.meta.ai/docs/sam/segmenting). If the API rejects
    metadata, PHYSTAX_SAM_METADATA=0 turns it off; Day 1 tells us which.
  * every call's event stream is cached by (video hash, phrase, options) and
    replayed through the same parser, so a benchmark rerun is bit-identical
    even though SAM results "can vary slightly between runs"
  * cost is checked against the budget cap before the call and recorded after
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from meta_sam_parser import (
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    parse_responses_stream,
    video_segmentation_format,
)

from .. import budget
from ..config import CACHE_DIR, SAM_MODEL, api_key, base_url


class SamError(RuntimeError):
    pass


@dataclass
class SamCall:
    phrase: str
    video_path: str
    records: list
    diagnostics: list
    latency_s: float
    frames_billed: int | None
    cached: bool
    usd: float


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _use_metadata() -> bool:
    return os.environ.get("PHYSTAX_SAM_METADATA", "1") != "0"


async def _synthesize_done(stream, sink: list):
    """Port of the reference `stream_events`, also recording every event."""
    completed: set = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed:
                completed.add(lane)
                sink.append(data)
                yield data
            continue
        sink.append(data)
        yield data
        part = data.get("part")
        if (data.get("type") == "response.content_part.done" and lane not in completed
                and isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)):
            completed.add(lane)
            done = {"type": "response.output_text.done", "item_id": data.get("item_id"),
                    "output_index": data.get("output_index"),
                    "content_index": data.get("content_index"), "text": part["text"]}
            sink.append(done)
            yield done


async def _replay(events: list):
    for e in events:
        yield e


async def _parse(events_iter) -> Any:
    parsed = parse_responses_stream(events_iter, video_segmentation_format())
    async with parsed:
        async for _ in parsed:
            pass
        return await parsed.final_result()


def _validate(result) -> None:
    if result.outcome.status != "completed":
        raise SamError(f"segmentation incomplete: {getattr(result.outcome, 'reason', '?')} "
                       f"({getattr(result.outcome, 'detail', None)})")
    errors = [d for d in result.diagnostics if getattr(d, "severity", "") == "error"]
    if errors:
        raise SamError("parser rejected records: " +
                       "; ".join(f"line {d.line}: {d.message}" for d in errors[:5]))


async def _call_api(video_path: str, phrase: str, include_confidence: bool,
                    object_prompt: bool = False) -> tuple[Any, list]:
    from openai import AsyncOpenAI
    events: list = []
    async with AsyncOpenAI(base_url=base_url("meta"), api_key=api_key("meta")) as api:
        with open(video_path, "rb") as f:
            uploaded = await api.files.create(
                file=(os.path.basename(video_path), f.read(), "video/mp4"), purpose="user_data")
        if not str(uploaded.id).startswith("file-"):
            raise SamError(f"Files API returned an unexpected id: {uploaded.id!r}")
        kwargs: dict = {
            "model": SAM_MODEL, "stream": True,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": phrase.strip()},
                {"type": "input_video", "file_id": uploaded.id}]}],
        }
        if _use_metadata():
            kwargs["metadata"] = {"mask_encoding": "one_bit"}
            if include_confidence:
                kwargs["metadata"]["include_confidence"] = "true"
        if object_prompt:   # track the prompted objects from frame 0 onwards only
            kwargs.setdefault("metadata", {})["propagation_direction"] = "forward"
        stream = await api.responses.create(**kwargs)
        async with stream, aclosing(_synthesize_done(stream, events)) as ev:
            result = await _parse(ev)
    return result, events


def _frames_billed(events: list) -> int | None:
    for e in reversed(events):
        if e.get("type") == "response.completed":
            usage = (e.get("response") or {}).get("usage") or {}
            for k in ("video_frames_processed", "frames", "video_frames"):
                if k in usage:
                    return usage[k]
    return None


def segment_video(video_path: str, phrase: str, include_confidence: bool = True,
                  use_cache: bool = True, n_frames: int | None = None,
                  retries: int = 3, object_prompt: bool = False,
                  cache_only: bool = False) -> SamCall:
    """`phrase` is either ONE short noun phrase (concept mode) or, with
    object_prompt=True, object-prompt text from meta_sam_parser.build_object_prompt
    (box/point mode). The two must never be combined in one request."""
    if not object_prompt and (not phrase.strip() or "," in phrase):
        raise SamError("SAM takes ONE short noun phrase per request (no lists)")
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(
        f"{_sha(video_path)}|{phrase}|{include_confidence}|{_use_metadata()}|{object_prompt}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"sam_{key}.json")

    if use_cache and os.path.exists(cache_path):
        blob = json.load(open(cache_path))
        result = asyncio.run(_parse(_replay(blob["events"])))
        return SamCall(phrase, video_path, list(result.records), list(result.diagnostics),
                       blob["latency_s"], blob.get("frames_billed"), True, 0.0)

    if cache_only:
        raise SamError(f"not cached (cache_only): {os.path.basename(video_path)} / {phrase[:40]}")
    est = budget.sam_video_cost(n_frames or 150)
    budget.check("meta", est, f"SAM on {os.path.basename(video_path)}")
    last_exc = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            result, events = asyncio.run(_call_api(video_path, phrase, include_confidence, object_prompt))
            break
        except SamError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise SamError(f"SAM request rejected ({status}): {exc}") from exc
            last_exc = exc
            if attempt == retries:
                raise SamError(f"SAM failed after {retries + 1} attempts: {exc}") from exc
            time.sleep(2 ** attempt + random.random())
    latency = time.time() - t0
    _validate(result)
    billed = _frames_billed(events)
    usd = budget.sam_video_cost(billed or n_frames or 150)
    budget.record("meta", usd, "sam_video", SAM_MODEL,
                  {"frames": billed or n_frames, "phrase": phrase})
    json.dump({"events": events, "latency_s": latency, "frames_billed": billed},
              open(cache_path, "w"))
    return SamCall(phrase, video_path, list(result.records), list(result.diagnostics),
                   latency, billed, False, usd)


def estimated_cost_usd(n_frames: int, n_requests: int = 1) -> float:
    return n_requests * budget.sam_video_cost(n_frames)


__all__ = ["SamCall", "SamError", "segment_video", "estimated_cost_usd",
           "SegmentationBoxRecord", "SegmentationMaskRecord"]
