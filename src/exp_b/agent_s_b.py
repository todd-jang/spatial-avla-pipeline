import math
import os
from statistics import median

FPS_NOMINAL = 30

# Legacy rule was doa_change_rate_ma >= 15.0 deg/frame = 450 deg/s at 30 fps, while
    # the fastest real sweep (20 m/s, 3-5 m offset) peaks at 229-382 deg/s, so it could
    # never fire. Both numbers are kept in meta_exec.doa_trigger_rule to stay auditable.
DOA_RATE_MA_THRESHOLD_DPS = 120.0
LEGACY_RATE_MA_THRESHOLD_DEG_PER_FRAME = 15.0

# Raw MAE distance is uncalibrated (tracks scene/domain shift, not anomaly), so it
# is gated as a causal within-clip robust z. Threshold sits above the normal
# reference clip's peak (T09 z_ma ~= 4.27).
MAE_Z_THRESHOLD = 5.0
MAE_BASELINE_WINDOW = 60
MAE_MIN_HISTORY = 10
MAE_Z_SCALE = 1.4826  # MAD -> sigma (Gaussian consistency)
# Inside a stable clip MAD collapses toward zero and ordinary jitter explodes z,
# so the scale is floored; with no labelled anomalies this keeps normals quiet.
MAE_MIN_SCALE_ABS = 0.2
MAE_MIN_SCALE_REL = 0.03
# Opt-in (default off): raw MAE distance is an unlabelled, weak signal, so it only
# joins raw_trigger when AVLA_MAE_TRIGGER_ENABLED=1. The score is always computed.
MAE_TRIGGER_ENABLED = os.environ.get("AVLA_MAE_TRIGGER_ENABLED", "0") == "1"


def _ma(v, w=5):
    if not v:
        return 0.0
    x = v[-w:]
    return sum(x) / len(x)


def _mae_robust_z(h: dict) -> float:
    hist = list(h.get("mae_list", []))
    if len(hist) < MAE_MIN_HISTORY + 1:
        return 0.0
    current = float(hist[-1])
    baseline = hist[-(MAE_BASELINE_WINDOW + 1):-1]
    med = median(baseline)
    mad = median([abs(v - med) for v in baseline])
    scale = max(MAE_Z_SCALE * mad, MAE_MIN_SCALE_ABS, MAE_MIN_SCALE_REL * abs(med))
    if scale <= 1e-6:
        return 0.0
    return (current - med) / scale


def s_b_fn(state: dict) -> dict:
    s = dict(state)
    ev = s.setdefault("event_trigger", {})
    d = ev.setdefault("details", {})
    h = s.setdefault("_hist", {})
    fi = int(s.get("frame_idx", 0))
    p = float(d.get("p_siren", 0.0))
    ac = float(d.get("audio_confidence", p))

    doa_in = s.get("doa") or (s.get("audio") or {}).get("doa") or {}
    ambiguous = bool(doa_in.get("ambiguous", False))
    prev_dir = h.get("doa_prev_dir")
    cur_dir = float(doa_in.get("dir_deg", 0.0)) if not ambiguous else prev_dir
    dd = 0.0 if prev_dir is None or cur_dir is None else abs(cur_dir - float(prev_dir))
    if not ambiguous:
        h["doa_prev_dir"] = cur_dir
    dps = dd * FPS_NOMINAL

    d["p_siren"] = p
    d["audio_confidence"] = ac
    d["doa_change_rate_deg"] = dd
    d["doa_change_rate_dps"] = dps
    h.setdefault("dd_list", []).append(dd)
    if len(h["dd_list"]) > 300:
        h["dd_list"].pop(0)
    h.setdefault("dps_list", []).append(dps)
    if len(h["dps_list"]) > 300:
        h["dps_list"].pop(0)
    vam = round(_ma(h.get("va_list", []), 5), 4)
    ddm = round(_ma(h.get("dd_list", []), 5), 4)
    dpsm = round(_ma(h.get("dps_list", []), 5), 4)
    d["vision_anomaly_ma"] = vam
    d["doa_change_rate_ma"] = ddm
    d["doa_change_rate_dps_ma"] = dpsm
    s.setdefault("meta_exec", {}).setdefault("doa_trigger_rule", {}).update(
        {
            "threshold_dps_ma": DOA_RATE_MA_THRESHOLD_DPS,
            "legacy_threshold_deg_per_frame_ma": LEGACY_RATE_MA_THRESHOLD_DEG_PER_FRAME,
            "fps_nominal": FPS_NOMINAL,
            "ma_window_frames": 5,
        }
    )
    mae_z = _mae_robust_z(h)
    mae_z_list = h.setdefault("mae_z_list", [])
    mae_z_list.append(mae_z)
    if len(mae_z_list) > 300:
        mae_z_list.pop(0)
    mae_z_ma = round(_ma(mae_z_list, 5), 4)
    d["mae_anomaly_z"] = round(mae_z, 4)
    d["mae_anomaly_z_ma"] = mae_z_ma
    d["mae_anomaly_score"] = round(1.0 - math.exp(-max(mae_z_ma, 0.0) / MAE_Z_THRESHOLD), 4)
    mae_trigger = MAE_TRIGGER_ENABLED and mae_z_ma >= MAE_Z_THRESHOLD
    d["mae_trigger"] = bool(mae_trigger)
    siren_vision_trigger = (
        d.get("siren_detected_mae") is True
        and mae_z_ma >= MAE_Z_THRESHOLD
    )
    d["siren_vision_trigger"] = bool(siren_vision_trigger)
    s.setdefault("meta_exec", {}).setdefault(
        "siren_vision_trigger_rule", {}
    ).update(
        {
            "siren_score_threshold": 0.0,
            "mae_anomaly_z_ma_threshold": MAE_Z_THRESHOLD,
            "combination": "siren detected AND MAE robust-z moving average at threshold",
        }
    )
    s["meta_exec"].setdefault("mae_trigger_rule", {}).update(
        {
            "threshold_z_ma": MAE_Z_THRESHOLD,
            "baseline_window_frames": MAE_BASELINE_WINDOW,
            "min_history_frames": MAE_MIN_HISTORY,
            "ma_window_frames": 5,
            "mad_sigma_scale": MAE_Z_SCALE,
            "min_scale_abs": MAE_MIN_SCALE_ABS,
            "min_scale_rel": MAE_MIN_SCALE_REL,
            "enabled": MAE_TRIGGER_ENABLED,
            "method": "causal robust z (median/MAD) of raw MAE distance vs trailing baseline",
        }
    )
    raw = (
        p >= 0.80
        or dpsm >= DOA_RATE_MA_THRESHOLD_DPS
        or d.get("lead_lost", False)
        or int(d.get("lane_lost_count", 0)) >= 3
        or vam >= 0.75
        or mae_trigger
        or siren_vision_trigger
    )
    trigger_reason_raw = list(ev.get("trigger_reason_raw", []))
    if siren_vision_trigger:
        for reason in ("siren_detected", "vision_anomaly"):
            if reason not in trigger_reason_raw:
                trigger_reason_raw.append(reason)
    ev["trigger_reason_raw"] = trigger_reason_raw
    ev["raw_trigger"] = bool(raw)
    hy = ev.setdefault(
        "hysteresis",
        {
            "min_hold_frames": 35,
            "state": "idle",
            "active_since_frame_idx": -1,
            "active_until_frame_idx": -1,
            "last_trigger_frame_idx": -1,
        },
    )
    st = hy["state"]
    mhf = 35
    if raw and st in ("idle", "armed"):
        hy["state"] = "active"
        hy["active_since_frame_idx"] = fi
        hy["active_until_frame_idx"] = fi + mhf
        hy["last_trigger_frame_idx"] = fi
    if raw and st == "active":
        hy["active_until_frame_idx"] = max(hy["active_until_frame_idx"], fi + mhf)
        hy["last_trigger_frame_idx"] = fi
    if (not raw) and st == "active" and fi >= hy["active_until_frame_idx"]:
        hy["state"] = "idle"
        hy["active_since_frame_idx"] = -1
        hy["active_until_frame_idx"] = -1
    ev["is_triggered"] = (hy["state"] == "active")
    tr = list(ev.get("trigger_reason_raw", []))
    ev["trigger_reason"] = list(tr) if ev["is_triggered"] else []
    doa_in = s.get("doa") or (s.get("audio") or {}).get("doa") or {}
    audio = dict(s.get("audio") or {})
    audio["siren"] = {"detected": p >= 0.80, "confidence": ac,
                      "type": "siren" if p >= 0.80 else "none"}
    audio["doa"] = {
        "dir_deg": float(doa_in.get("dir_deg", 0.0)),
        "gt_dir_deg": float(doa_in.get("gt_dir_deg", -1.0)),
        "confidence": float(doa_in.get("confidence", 0.0)),
        "direction": doa_in.get("direction", "center"),
    }
    audio.setdefault("audio_features_b64", "")
    s["audio"] = audio
    return s
