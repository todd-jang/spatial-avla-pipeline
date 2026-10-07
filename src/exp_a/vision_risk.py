"""S_A vision-risk: map RT-DETR detections to a [0,1] anomaly score.

Consumes outputs/detections/<clip_id>__<model>.json (written by
scripts/run_rtdetr_detect.py) plus configs/vision_risk_weights.json and
produces a per-frame anomaly score in [0,1] that S_A records as
event_trigger.details.vision_anomaly_score.

Score rule (documented here so the report can quote it):

    risk(frame) = max over boxes of class_weight(class) * conf

where class_weight comes from the weights config. Without the max, a
dense-but-benign scene (many cars, all weight 0.30 * ~0.9) would still sit
below the S_B trigger band; taking the max lets a single high-risk object
(person, motorcycle, bus) dominate the frame as intended. No detections ->
0.0; no detections file -> the caller keeps the incoming state untouched
(mock lane).

A small positive epsilon floor is NOT applied: an empty frame (night, out of
view) must score 0 so the moving average stays honest instead of drifting.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_WEIGHTS = ROOT / "configs/vision_risk_weights.json"
DEFAULT_DETECTIONS = ROOT / "outputs/detections"


@lru_cache(maxsize=8)
def _weights(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=16)
def _detections(clip_id: str, model: str, det_dir: str) -> dict | None:
    p = Path(det_dir) / f"{clip_id}__{model}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def has_detections(clip_id: str, model: str = "rtdetr-l",
                   det_dir: str = DEFAULT_DETECTIONS.as_posix()) -> bool:
    """True when a detection JSON exists for this clip (file may be absent
    only in mock mode / before the batch detection run)."""
    return _detections(clip_id, model, det_dir) is not None


def _frame_boxes(clip_id: str, frame_idx: int, model: str,
                 det_dir: str = DEFAULT_DETECTIONS.as_posix()) -> list[dict]:
    doc = _detections(clip_id, model, det_dir)
    if doc is None:
        return []

    # The detector writes frames keyed by zero-based frame_idx.
    f = doc["frames"].get(str(frame_idx))
    return list(f["boxes"]) if f else []


def anomaly_score(state: dict, weights_path: str = DEFAULT_WEIGHTS.as_posix(),
                  det_dir: str = DEFAULT_DETECTIONS.as_posix(),
                  model: str = "rtdetr-l") -> float:
    """[0,1] risk score from this frame's detections; 0.0 when unavailable."""
    clip = state.get("clip_id")
    if not clip:
        return 0.0
    frame_idx = int(state.get("frame_idx", 0))
    cfg = _weights(weights_path)
    w = cfg.get("classes", {})
    default_w = float(cfg.get("default_weight", 0.0))

    boxes = _frame_boxes(clip, frame_idx, model, det_dir)
    if not boxes:
        return 0.0
    best = 0.0
    for b in boxes:
        conf = float(b["conf"])
        if conf <= 0:
            continue
        base = w.get(b.get("class_name"), default_w)
        best = max(best, base * conf)
    return round(min(1.0, best), 4)