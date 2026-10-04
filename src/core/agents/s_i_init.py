def s_i_init_fn(state: dict) -> dict:
    s = dict(state)
    if "version" not in s:
        s["version"] = "1.0"
    s.setdefault("clip_id", s.get("clip_id", "siren_left_01"))
    s.setdefault("frame_id", int(s.get("frame_id", 1)))
    s.setdefault("frame_idx", int(s.get("frame_idx", 0)))
    s.setdefault("ts_ms", float(s.get("ts_ms", 0.0)))
    s.setdefault("timestamp_ms", float(s.get("timestamp_ms", s.get("ts_ms", 0.0))))
    s.setdefault("video_path", s.get("video_path", "raw_videos/sample.mp4"))
    s.setdefault("wav_path", s.get("wav_path", "raw_videos/sample.wav"))
    s.setdefault("img", {"raw_b64": "", "clean_b64": "", "quad_b64": ""})
    s.setdefault("audio_win", {"start_ms": 0, "end_ms": 320, "sr": 16000})
    s.setdefault("doa", {"dir_deg": 0, "direction": "center"})
    s.setdefault("percept", {"boxes": [], "drivable_rle_b64": "", "wheeltrack_poly": [], "center_path": [], "has_lead": False})
    s.setdefault("geom", {"depth_roi_pts": [], "bev": {"ego": {"x": 0, "y": 0, "theta": 0}, "pts": [], "path": []}, "flow": {"ttc_min": None, "ttc_valid": False}})
    s.setdefault("plan", {"steer_deg": 0, "speed_kmh": 0, "lookahead_m": 9, "ttc_min": None, "alerts": {"fcw": False, "aeb_prep": False}, "mode": "auto_sim", "mu_eff": "low"})
    s.setdefault("weather", {"road_condition": "snow", "visibility": "reduced", "headlight_active_roi": True})
    s.setdefault("vlm", {"prompt_baseline": "", "prompt_meta": "", "reasoning_baseline": "", "reasoning_meta": "", "reasoning_avla": "", "frame_id_ref": -1, "ts": 0.0, "trigger": "none"})
    s.setdefault("avla", {"wavlm_emb_path": "cache/emb/wavlm.pt", "audio_tokens_shape": [0, 2048]})
    s.setdefault("audio", {"siren": {"detected": False, "confidence": 0, "type": "none"}, "doa": {"dir_deg": 0, "direction": "center"}, "audio_features_b64": ""})
    s.setdefault("vlm_payload", {"trigger": False, "mode": "symbolic", "prompt_context": ""})
    s.setdefault("event_trigger", {
        "is_triggered": False,
        "trigger_reason": [],
        "raw_trigger": False,
        "trigger_reason_raw": [],
        "details": {"audio_confidence": 0, "doa_change_rate_deg": 0, "vision_anomaly_score": 0, "lane_lost_count": 0, "lead_lost": False, "p_siren": 0, "vision_anomaly_ma": 0, "doa_change_rate_ma": 0},
        "hysteresis": {"min_hold_frames": 35, "state": "idle", "active_since_frame_idx": -1, "active_until_frame_idx": -1, "last_trigger_frame_idx": -1},
    })
    s.setdefault("meta_exec", {
        "enabled": {"A": True, "B": True, "C": True},
        "status": "idle",
        "errors": [],
        "poc_completed_at_ms": 0,
        "poc_last_completed_agent": "",
        "poc_all_done": False,
        "poc_all_ok": False,
        "_agent_status": {},
        "_agent_finished_ts_ms": {},
    })
    s.setdefault("_hist", {"va_list": [], "dd_list": []})
    s.setdefault("_sync", {})
    return s
