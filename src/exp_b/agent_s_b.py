FPS_NOMINAL = 30

# Legacy rule was doa_change_rate_ma >= 15.0 deg/frame = 450 deg/s at 30 fps, while
    # the fastest real sweep (20 m/s, 3-5 m offset) peaks at 229-382 deg/s, so it could
    # never fire. Both numbers are kept in meta_exec.doa_trigger_rule to stay auditable.
DOA_RATE_MA_THRESHOLD_DPS = 120.0
LEGACY_RATE_MA_THRESHOLD_DEG_PER_FRAME = 15.0


def _ma(v, w=5):
    if not v:
        return 0.0
    x = v[-w:]
    return sum(x) / len(x)


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
    raw = (
        p >= 0.80
        or dpsm >= DOA_RATE_MA_THRESHOLD_DPS
        or d.get("lead_lost", False)
        or int(d.get("lane_lost_count", 0)) >= 3
        or vam >= 0.75
    )
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
