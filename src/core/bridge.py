import copy
import time
from pathlib import Path

from src.core.audio.p_siren_provider import P_sirenProvider
from src.core.pipeline import graph_ui


class GraphBridge:
    def __init__(self, mock_path="mocks/mock_base.json", fast_mode=True, p_siren_report=None):
        self.mock_path = Path(mock_path)
        self.fast_mode = fast_mode
        self._base = self._load()
        self._base_ui = copy.deepcopy(self._base)
        self.p_siren = P_sirenProvider(report_path=p_siren_report)

    def _load(self):
        with open(self.mock_path, "r", encoding="utf-8") as f:
            import json

            return json.load(f)

    def _prime(self, s, frame_idx, ts_ms, clip_id):
        """Stamp one frame's identity + measured audio onto a state dict."""
        s["clip_id"] = clip_id
        s["frame_id"] = frame_idx + 1
        s["frame_idx"] = int(frame_idx)
        s["ts_ms"] = float(ts_ms)
        s["timestamp_ms"] = float(ts_ms)
        siren_score = self.p_siren.frame_value(clip_id, int(frame_idx))
        s.setdefault(
            "event_trigger",
            {
                "is_triggered": False,
                "raw_trigger": False,
                "trigger_reason": [],
                "trigger_reason_raw": [],
                "details": {},
                "hysteresis": {
                    "state": "idle",
                    "min_hold_frames": 35,
                    "active_since_frame_idx": -1,
                    "active_until_frame_idx": -1,
                    "last_trigger_frame_idx": -1,
                },
            },
        )
        # mock_base.json ships a pre-populated event_trigger, so setdefault above
        # would silently keep its p_siren=0. Stamp the measured value explicitly:
        # unknown clips resolve to 0.0 (p_siren_provider fallback).
        details = s["event_trigger"].setdefault("details", {})
        details.setdefault("vision_anomaly_score", 0)
        details.setdefault("vision_anomaly_ma", 0)
        details.setdefault("doa_change_rate_deg", 0)
        details.setdefault("doa_change_rate_ma", 0)
        details.setdefault("lane_lost_count", 0)
        details.setdefault("lead_lost", False)
        details["p_siren"] = siren_score
        details["audio_confidence"] = siren_score
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
        return s

    def invoke_frame(self, frame_idx: int, ts_ms: float, clip_id="siren_left_01"):
        t0 = time.perf_counter()
        s = self._prime(copy.deepcopy(self._base_ui), frame_idx, ts_ms, clip_id)
        try:
            res = graph_ui.invoke(s)
            res.setdefault("meta_exec", {})["lat_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return res
        except Exception as e:
            s["meta_exec"]["errors"].append({"path": "bridge", "code": "GRAPH_INVOKE_EXCEPTION", "msg": str(e)})
            s["meta_exec"]["lat_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return s

    def invoke_frame_stateful(self, frame_idx: int, ts_ms: float, clip_id="siren_left_01",
                              prev_state=None):
        """Run frame N threading the previous frame's full state.

        Stateless ``invoke_frame`` deep-copies a fresh mock every call, which means
        S_B's hysteresis FSM and the MA windows in _hist never actually run across
        frames -- every stateless render has been a per-frame cold start (the FSM
        resets to idle and the 35-frame hold never happens). Demos that must show /
        verify the real trigger envelope (see scripts/build_demo_sequence.py) need
        the FSM continuity this method provides.

        ``prev_state`` must be the *result* of the previous call (it carries
        event_trigger.hysteresis, _hist.dd_list/va_list, etc.). Pass None on frame
        0 to start from the mock base.
        """
        t0 = time.perf_counter()
        base = prev_state if prev_state is not None else self._base_ui
        s = self._prime(copy.deepcopy(base), frame_idx, ts_ms, clip_id)
        try:
            res = graph_ui.invoke(s)
            res.setdefault("meta_exec", {})["lat_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return res
        except Exception as e:
            s["meta_exec"]["errors"].append({"path": "bridge", "code": "GRAPH_INVOKE_EXCEPTION", "msg": str(e)})
            s["meta_exec"]["lat_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return s
