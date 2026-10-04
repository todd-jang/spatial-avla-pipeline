def s_c_fn(state: dict) -> dict:
    s = dict(state)
    ev = s.setdefault("event_trigger", {})
    vp = s.setdefault("vlm_payload", {"trigger": False, "mode": "symbolic", "prompt_context": ""})
    trig = bool(ev.get("is_triggered", False))
    vp["trigger"] = trig
    vp["mode"] = "symbolic"
    rs = list(ev.get("trigger_reason", []))
    if rs:
        vp["prompt_context"] = "[AUDIO/VISION ALERT] " + (", ".join(rs))
    else:
        vp["prompt_context"] = "[AUDIO/VISION ALERT] EVENT_TRIGGERED" if trig else ""
    s.setdefault(
        "vlm",
        {
            "trigger": "event" if trig else "none",
            "frame_id_ref": s.get("frame_id", -1),
            "ts": s.get("ts_ms", 0.0),
        },
    )
    s.setdefault("img", {"raw_b64": "", "clean_b64": "", "quad_b64": ""})
    return s
