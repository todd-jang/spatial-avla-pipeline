import copy
import time
from pathlib import Path

from src.core.pipeline import graph_ui


class GraphBridge:
    def __init__(self, mock_path="mocks/mock_base.json", fast_mode=True):
        self.mock_path = Path(mock_path)
        self.fast_mode = fast_mode
        self._base = self._load()
        self._base_ui = copy.deepcopy(self._base)

    def _load(self):
        with open(self.mock_path, "r", encoding="utf-8") as f:
            import json

            return json.load(f)

    def invoke_frame(self, frame_idx: int, ts_ms: float, clip_id="siren_left_01"):
        t0 = time.perf_counter()
        s = copy.deepcopy(self._base_ui)
        s["clip_id"] = clip_id
        s["frame_id"] = frame_idx + 1
        s["frame_idx"] = int(frame_idx)
        s["ts_ms"] = float(ts_ms)
        s["timestamp_ms"] = float(ts_ms)
        s.setdefault(
            "event_trigger",
            {
                "is_triggered": False,
                "raw_trigger": False,
                "trigger_reason": [],
                "trigger_reason_raw": [],
                "details": {
                    "vision_anomaly_score": 0,
                    "vision_anomaly_ma": 0,
                    "doa_change_rate_deg": 0,
                    "doa_change_rate_ma": 0,
                    "lane_lost_count": 0,
                    "lead_lost": False,
                    "p_siren": 0,
                    "audio_confidence": 0,
                },
                "hysteresis": {
                    "state": "idle",
                    "min_hold_frames": 35,
                    "active_since_frame_idx": -1,
                    "active_until_frame_idx": -1,
                    "last_trigger_frame_idx": -1,
                },
            },
        )
        s.setdefault("vlm_payload", {"trigger": False, "mode": "symbolic", "prompt_context": ""})
        s.setdefault(
            "plan",
            {
                "mode": "auto_sim",
                "steer_deg": 0,
                "speed_kmh": 0,
                "alerts": {"fcw": False, "aeb_prep": False},
            },
        )
        s.setdefault(
            "meta_exec",
            {
                "status": "idle",
                "errors": [],
                "_agent_status": {},
                "_agent_finished_ts_ms": {},
            },
        )
        try:
            res = graph_ui.invoke(s)
            res.setdefault("meta_exec", {})["lat_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return res
        except Exception as e:
            s["meta_exec"]["errors"].append({"path": "bridge", "code": "GRAPH_INVOKE_EXCEPTION", "msg": str(e)})
            s["meta_exec"]["lat_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return s
