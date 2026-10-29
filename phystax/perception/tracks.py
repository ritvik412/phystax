"""Turn segmentation records into measured trajectories.

A measured track has, per frame: centroid (world px, y up), mask area and
SAM's confidence, or None where the object was not found. Concept prompts
return *every* instance (including same-looking static distractors), so we
pick the instance that behaves like the body we asked about.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import SegmentationBoxRecord, SegmentationMaskRecord, decode_mask_to_raster


@dataclass
class Track:
    body: str
    n_frames: int
    x: np.ndarray          # world px, NaN when missing
    y: np.ndarray
    area: np.ndarray
    conf: np.ndarray       # NaN when missing or not requested
    n_candidates: int      # how many instances SAM returned for the phrase
    truncated: np.ndarray | None = None   # True where SAM's box touches the image border

    def coverage(self) -> float:
        return float(np.mean(np.isfinite(self.x)))

    def to_jsonable(self, include_conf: bool) -> dict:
        def clean(a, nd):
            return [None if not np.isfinite(v) else round(float(v), nd) for v in a]
        out = {"x": clean(self.x, 2), "y": clean(self.y, 2)}
        if include_conf:
            out["conf"] = clean(self.conf, 3)
        return out


def _centroid(rec: SegmentationMaskRecord) -> tuple[float, float, int]:
    m = rec.mask
    raster = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
    ys, xs = np.nonzero(raster)
    if len(xs) == 0:
        b = rec.bounds
        return (b.left + b.right) / 2, (b.top + b.bottom) / 2, 0
    return rec.bounds.left + xs.mean() + 0.5, rec.bounds.top + ys.mean() + 0.5, len(xs)


BORDER_PX = 2   # a box within this many pixels of the edge means the object is cut off


def _touches_border(b, frame_w: int, frame_h: int) -> bool:
    return (b.left <= BORDER_PX or b.top <= BORDER_PX or
            b.right >= frame_w - 1 - BORDER_PX or b.bottom >= frame_h - 1 - BORDER_PX)


def records_to_tracks(records: list, n_frames: int, frame_h: int, frame_w: int = 640) -> dict[str, dict]:
    """Group records by SAM object id -> per-frame (cx, cy, area, conf, truncated).
    `truncated` uses only SAM's own box, never ground truth, so it is a fair
    rule for an agent or fitter to apply."""
    by_obj: dict[str, dict] = defaultdict(lambda: {
        "x": np.full(n_frames, np.nan), "y": np.full(n_frames, np.nan),
        "area": np.full(n_frames, np.nan), "conf": np.full(n_frames, np.nan),
        "trunc": np.zeros(n_frames, bool)})
    boxes = {}
    for r in records:
        if r.frame is None:
            continue
        f = r.frame.frame_index
        if f >= n_frames:
            continue
        if isinstance(r, SegmentationBoxRecord):
            boxes[(r.object_id, f)] = r
        elif isinstance(r, SegmentationMaskRecord):
            cx, cy, area = _centroid(r)
            o = by_obj[r.object_id]
            o["trunc"][f] |= _touches_border(r.bounds, frame_w, frame_h)
            o["x"][f], o["y"][f], o["area"][f] = cx, frame_h - cy, area
            if r.confidence is not None:
                o["conf"][f] = r.confidence
    for (oid, f), b in boxes.items():   # boxes fill frames that lacked a mask
        o = by_obj[oid]
        o["trunc"][f] |= _touches_border(b, frame_w, frame_h)
        if not np.isfinite(o["x"][f]):
            o["x"][f] = (b.left + b.right) / 2
            o["y"][f] = frame_h - (b.top + b.bottom) / 2
            o["area"][f] = (b.right - b.left) * (b.bottom - b.top)
        if b.confidence is not None and not np.isfinite(o["conf"][f]):
            o["conf"][f] = b.confidence
    return dict(by_obj)


def pick_instance(candidates: dict[str, dict], mode: str = "moving") -> str | None:
    """'moving': longest path (the launched body, not a static distractor).
    'largest': biggest median area (a fixed anchor among small distractors)."""
    best, score = None, -1.0
    for oid, o in candidates.items():
        ok = np.isfinite(o["x"])
        if ok.sum() < 2:
            continue
        if mode == "largest":
            s = float(np.nanmedian(o["area"])) * ok.mean()
        else:
            xy = np.stack([o["x"][ok], o["y"][ok]], 1)
            s = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        if s > score:
            best, score = oid, s
    return best


def build_track(body: str, records: list, n_frames: int, frame_h: int,
                mode: str = "moving") -> Track:
    cands = records_to_tracks(records, n_frames, frame_h)
    oid = pick_instance(cands, mode)
    if oid is None:
        nan = np.full(n_frames, np.nan)
        return Track(body, n_frames, nan, nan.copy(), nan.copy(), nan.copy(), len(cands))
    o = cands[oid]
    return Track(body, n_frames, o["x"], o["y"], o["area"], o["conf"], len(cands), o["trunc"])


def centroid_error_px(track: Track, truth_world: np.ndarray) -> dict:
    """Measurement error against simulator truth (n_frames, 2), world px."""
    ok = np.isfinite(track.x)
    if not ok.any():
        return {"rmse_px": None, "coverage": 0.0}
    err = np.hypot(track.x[ok] - truth_world[ok, 0], track.y[ok] - truth_world[ok, 1])
    out = {"rmse_px": float(np.sqrt(np.mean(err ** 2))), "p95_px": float(np.percentile(err, 95)),
           "coverage": float(ok.mean())}
    c = track.conf[ok]
    if np.isfinite(c).sum() > 3:
        good = np.isfinite(c)
        out["conf_error_spearman"] = float(_spearman(c[good], err[good]))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def track_for_object(body: str, records: list, object_id: int, n_frames: int, frame_h: int) -> Track:
    """Box/point-prompt mode: the response reports each object under the id
    the request gave it, so no instance picking is needed."""
    cands = records_to_tracks(records, n_frames, frame_h)
    o = cands.get(object_id) or cands.get(str(object_id))
    if o is None:
        nan = np.full(n_frames, np.nan)
        return Track(body, n_frames, nan, nan.copy(), nan.copy(), nan.copy(), len(cands))
    return Track(body, n_frames, o["x"], o["y"], o["area"], o["conf"], len(cands), o["trunc"])


def drop_truncated(tr: Track) -> Track:
    """Remove frames where the object is cut off by the border (centroid biased inward)."""
    if tr.truncated is None or not tr.truncated.any():
        return tr
    m = tr.truncated
    x, y, a, c = tr.x.copy(), tr.y.copy(), tr.area.copy(), tr.conf.copy()
    x[m] = y[m] = a[m] = c[m] = np.nan
    return Track(tr.body, tr.n_frames, x, y, a, c, tr.n_candidates, tr.truncated)
