def s_a_fn(state: dict) -> dict:
    s = dict(state)
    ev = s.setdefault("event_trigger", {})
    d = ev.setdefault("details", {})
    h = s.setdefault("_hist", {"va_list": [], "dd_list": []})
    va = float(d.get("vision_anomaly_score", 0.2))
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
