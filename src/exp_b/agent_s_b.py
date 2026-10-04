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
    dd = float(d.get("doa_change_rate_deg", 0.0))
    d["p_siren"] = p
    d["audio_confidence"] = ac
    d["doa_change_rate_deg"] = dd
    h.setdefault("dd_list", []).append(dd)
    if len(h["dd_list"]) > 300:
        h["dd_list"].pop(0)
    vam = round(_ma(h.get("va_list", []), 5), 4)
    ddm = round(_ma(h.get("dd_list", []), 5), 4)
    d["vision_anomaly_ma"] = vam
    d["doa_change_rate_ma"] = ddm
    raw = (
        p >= 0.80
        or ddm >= 15.0
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
    s.setdefault(
        "audio",
        {
            "siren": {"detected": p >= 0.80, "confidence": ac, "type": "none"},
            "doa": {"dir_deg": 0, "direction": "center"},
            "audio_features_b64": "",
        },
    )
    return s
