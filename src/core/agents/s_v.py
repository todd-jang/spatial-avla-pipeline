"""Frame-level vision and siren scores for the S_V pipeline agent."""
from __future__ import annotations

import copy
import os

from src.core.audio.p_siren_provider import P_sirenProvider
from src.core.mae_score_provider import MaeScoreProvider
from src.core.mae_vision import (
    SIREN_HOP_SECONDS,
    SIREN_THRESHOLD_FRAME,
    SIREN_THRESHOLD_MODES,
    SIREN_WINDOW_SECONDS,
    VIDEO_FPS,
    MaeVisionSirenScorer,
    siren_detected,
)
from src.exp_a.vision_risk import anomaly_score, has_detections

UI_FAST = True
MAE_SIREN_THRESHOLD_MODE = os.environ.get(
    "AVLA_SIREN_THRESHOLD_MODE", "balanced"
)
if MAE_SIREN_THRESHOLD_MODE not in SIREN_THRESHOLD_MODES:
    raise ValueError(
        f"Invalid AVLA_SIREN_THRESHOLD_MODE: {MAE_SIREN_THRESHOLD_MODE!r}"
    )
SIREN_MIN_CONSECUTIVE_WINDOWS = 5

_MAE_SCORES = MaeScoreProvider()
_P_SIREN = P_sirenProvider()
_MAE_SIREN = MaeVisionSirenScorer()


def _mean(values: list[float], window: int = 5) -> float:
    selected = values[-window:]
    return sum(selected) / len(selected) if selected else 0.0


def _push(history: dict, key: str, value: float, limit: int = 300) -> list[float]:
    values = list(history.get(key, []))
    values.append(float(value))
    history[key] = values[-limit:]
    return history[key]


def _window_streak(
    history: dict, clip_id: str, frame_idx: int, candidate: bool | None
) -> int:
    if history.get("siren_streak_clip_id") != clip_id:
        history["siren_streak_clip_id"] = clip_id
        history["siren_streak_window_idx"] = None
        history["siren_streak_count"] = 0

    window_idx = int(frame_idx / (VIDEO_FPS * SIREN_HOP_SECONDS))
    previous_idx = history.get("siren_streak_window_idx")
    if previous_idx == window_idx:
        return int(history.get("siren_streak_count", 0))

    count = int(history.get("siren_streak_count", 0))
    if previous_idx is None or window_idx != previous_idx + 1:
        count = 0
    count = count + 1 if candidate is True else 0
    history["siren_streak_window_idx"] = window_idx
    history["siren_streak_count"] = count
    return count


def s_v_fn(state: dict) -> dict:
    clip_id = str(state.get("clip_id", ""))
    frame_idx = int(state.get("frame_idx", 0))
    history = copy.deepcopy(state.get("_hist") or {})
    event_trigger = copy.deepcopy(state.get("event_trigger") or {})
    details = dict(event_trigger.get("details") or {})
    meta_exec = copy.deepcopy(state.get("meta_exec") or {})
    score_status = meta_exec.setdefault("score_status", {})
    errors = meta_exec.setdefault("errors", [])

    if clip_id and has_detections(clip_id):
        try:
            rtdetr_score = anomaly_score(state)
            score_status["rtdetr"] = "available"
        except Exception as exc:
            rtdetr_score = None
            score_status["rtdetr"] = "error"
            errors.append({
                "path": "S_V.rtdetr",
                "code": "SCORE_FAILED",
                "msg": str(exc),
            })
    else:
        rtdetr_score = None
        score_status["rtdetr"] = "detections_unavailable"

    mae_score = (
        _MAE_SCORES.frame_value(clip_id, frame_idx)
        if clip_id
        else None
    )
    if mae_score is not None:
        history_mae = _push(history, "mae_list", mae_score)
        details["vision_anomaly_mae_ma"] = round(_mean(history_mae), 4)
        score_status["mae"] = "available"
    else:
        details["vision_anomaly_mae_ma"] = None
        score_status["mae"] = "frame_unavailable"

    bridge_siren_status = score_status.get("siren")
    if bridge_siren_status is not None or "p_siren" in details:
        has_p_siren = (
            bridge_siren_status == "available"
            if bridge_siren_status is not None
            else True
        )
        p_siren = float(details.get("p_siren", 0.0))
    else:
        has_p_siren = bool(
            clip_id and _P_SIREN.has_frame_value(clip_id, frame_idx)
        )
        p_siren = (
            float(_P_SIREN.frame_value(clip_id, frame_idx))
            if has_p_siren
            else 0.0
        )
    history_siren = _push(history, "siren_mae_list", p_siren)
    details["p_siren_ma5"] = round(_mean(history_siren), 4)
    details["p_siren_delta"] = (
        round(p_siren - history_siren[-2], 4)
        if len(history_siren) > 1
        else None
    )
    details["p_siren"] = p_siren
    details["audio_confidence"] = p_siren
    score_status["siren"] = "available" if has_p_siren else "report_unavailable"

    siren_score = None
    siren_scope = "frame" if state.get("audio_path") is not None else "clip"
    siren_candidate = None
    siren_detected_flag = False
    if _MAE_SIREN.available:
        try:
            score_frame = getattr(_MAE_SIREN, "score_frame", None)
            if callable(score_frame):
                siren_score = score_frame(
                    state,
                    window_seconds=SIREN_WINDOW_SECONDS,
                    hop_seconds=SIREN_HOP_SECONDS,
                    fps=VIDEO_FPS,
                )
                siren_scope = "frame"
            else:
                siren_score = _MAE_SIREN.score(state)

            if siren_score is not None:
                siren_candidate = siren_detected(
                    siren_score,
                    MAE_SIREN_THRESHOLD_MODE,
                    scope=siren_scope,
                )
                if siren_scope == "frame" and callable(score_frame):
                    streak = _window_streak(
                        history, clip_id, frame_idx, siren_candidate
                    )
                    siren_detected_flag = (
                        streak >= SIREN_MIN_CONSECUTIVE_WINDOWS
                    )
                else:
                    siren_detected_flag = siren_candidate
                score_status["siren_template"] = "available"
            else:
                score_status["siren_template"] = "audio_unavailable"
        except Exception as exc:
            score_status["siren_template"] = "error"
            errors.append({
                "path": "S_V.siren_template",
                "code": "SCORE_FAILED",
                "msg": str(exc),
            })
    else:
        score_status["siren_template"] = "template_unavailable"

    threshold = (
        SIREN_THRESHOLD_FRAME
        if siren_scope == "frame"
        else SIREN_THRESHOLD_MODES[MAE_SIREN_THRESHOLD_MODE]
    )
    details.update({
        "vision_anomaly_score": rtdetr_score,
        "vision_anomaly_mae": mae_score,
        "p_siren": p_siren,
        "siren_score_mae": siren_score,
        "siren_score_template": siren_score,
        "siren_detected_mae": bool(siren_detected_flag),
        "siren_strength_mae": (
            "strong" if siren_score is not None and siren_score >= 0.3
            else "weak" if siren_detected_flag
            else "none"
        ),
        "siren_threshold_mode": MAE_SIREN_THRESHOLD_MODE,
        "siren_threshold_scope": siren_scope,
        "siren_threshold": threshold,
        "siren_candidate": siren_candidate,
        "siren_consecutive_windows": int(
            history.get("siren_streak_count", 0)
        ),
    })
    event_trigger["details"] = details

    meta_exec["metrics"] = {
        "jitter_rms": 0,
        "steering_smoothness_dtheta_dt_rms": 0,
        "path_curvature_rms": 0,
        "event_alignment_ms": 0,
        "direction_match_rate": 0,
        "direction_mention_rate": 0,
    }
    meta_exec["golden_fixtures"] = {
        "multi_frame_required": True,
        "sequences": ["seq_lead_lost_15f"],
        **({"ui_skip_check": True} if UI_FAST else {}),
    }
    meta_exec["siren_frame_rule"] = {
        "window_seconds": SIREN_WINDOW_SECONDS,
        "hop_seconds": SIREN_HOP_SECONDS,
        "threshold": SIREN_THRESHOLD_FRAME,
        "min_consecutive_windows": SIREN_MIN_CONSECUTIVE_WINDOWS,
    }

    return {
        "_hist": history,
        "event_trigger": event_trigger,
        "meta_exec": meta_exec,
    }
