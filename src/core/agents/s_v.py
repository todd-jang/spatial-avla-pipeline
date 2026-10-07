UI_FAST = True


def s_v_fn(state: dict) -> dict:
    me = {
        "metrics": {
            "jitter_rms": 0,
            "steering_smoothness_dtheta_dt_rms": 0,
            "path_curvature_rms": 0,
            "event_alignment_ms": 0,
            "direction_match_rate": 0,
            "direction_mention_rate": 0,
        }
    }
    if UI_FAST:
        me["golden_fixtures"] = {
            "multi_frame_required": True,
            "sequences": ["seq_lead_lost_15f"],
            "ui_skip_check": True,
        }
    else:
        me["golden_fixtures"] = {
            "multi_frame_required": True,
            "sequences": ["seq_lead_lost_15f"],
        }
    return {"meta_exec": me}