"""S_A node: record the frame's vision anomaly score into the state.

Exp A wiring: when a per-clip detection JSON exists under
outputs/detections/ (written by scripts/run_rtdetr_detect.py), the node
overwrites event_trigger.details.vision_anomaly_score with the frame's
RT-DETR risk score (see src.exp_a.vision_risk). When the detections file is
absent (mock mode, tests, Colab before detection runs) the incoming value is
kept untouched, preserving the FrameAudioPacketState v1.0 contract and the
historical mock behaviour.
"""
from __future__ import annotations

from src.exp_a.vision_risk import anomaly_score, has_detections  # noqa: E402


def s_a_fn(state: dict) -> dict:
    s = dict(state)
    ev = s.setdefault("event_trigger", {})
    d = ev.setdefault("details", {})
    h = s.setdefault("_hist", {"va_list": [], "dd_list": []})

    incoming = float(d.get("vision_anomaly_score", 0.2))
    if has_detections(s.get("clip_id", "")):
        va = anomaly_score(s)
    else:
        va = incoming

    d["vision_anomaly_score"] = va
    d["lead_lost"] = bool(d.get("lead_lost", False))
    d["lane_lost_count"] = int(d.get("lane_lost_count", 0))
    h["va_list"].append(va)
    if len(h["va_list"]) > 300:
        h["va_list"].pop(0)
    s.setdefault("img", {"clean_b64": ""})
    s.setdefault("percept", {"has_lead": not d["lead_lost"]})
    s.setdefault("doa", {})
    return s