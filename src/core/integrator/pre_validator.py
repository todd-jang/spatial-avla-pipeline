def pre_frame_init(state: dict) -> dict:
    s = dict(state)
    ev = s.setdefault("event_trigger", {"trigger_reason": [], "trigger_reason_raw": [], "details": {}, "hysteresis": {}, "raw_trigger": False})
    ev["trigger_reason"] = []
    ev["trigger_reason_raw"] = []
    s.setdefault("_hist", {"va_list": [], "dd_list": []})
    s.setdefault("_sync", {})
    s.setdefault("meta_exec", {})
    return s


def pre_validator_sync(state: dict) -> dict:
    s = dict(state)
    _sync = s.setdefault("_sync", {})
    fi = int(s.get("frame_idx", 0))
    ts = float(s.get("timestamp_ms", s.get("ts_ms", 0.0)))
    s["frame_idx"] = fi
    s["timestamp_ms"] = ts
    s["ts_ms"] = float(s.get("ts_ms", ts))
    lf = _sync.get("last_frame_idx", -10**15)
    lt = _sync.get("last_timestamp_ms", -10**15)
    me = s.setdefault("meta_exec", {"errors": []})
    if fi < lf:
        me["errors"].append({"path": "pre_validator_sync", "code": "NON_MONOTONIC_FRAME_IDX", "msg": f"{fi}<{lf}"})
    if ts < lt:
        me["errors"].append({"path": "pre_validator_sync", "code": "NON_MONOTONIC_TIMESTAMP", "msg": f"{ts}<{lt}"})
    _sync["last_frame_idx"] = fi
    _sync["last_timestamp_ms"] = ts
    return s
