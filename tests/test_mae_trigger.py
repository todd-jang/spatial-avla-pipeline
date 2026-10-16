"""Acceptance tests for the mini-MAE calibration wired into the S_B trigger.

The raw MAE distance is uncalibrated, so it is gated as a causal within-clip
robust z and is opt-in (AVLA_MAE_TRIGGER_ENABLED, default off). These tests pin:
no/short history never fires, a stable baseline stays silent, the calibrated score
is always computed, the trigger only joins raw_trigger when enabled, and the rule
is exposed in meta_exec for audit.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

import src.exp_b.agent_s_b as sb  # noqa: E402
from src.core.agents.s_i_init import s_i_init_fn  # noqa: E402
from src.exp_b.agent_s_b import s_b_fn  # noqa: E402

BASELINE = [5.0, 5.02, 4.99, 5.01, 4.98, 5.03, 5.0, 4.97, 5.02, 5.01, 4.99, 5.0]


def run_mae(values):
    s = s_i_init_fn({"frame_idx": 0})
    details = {}
    fired = False
    for i, v in enumerate(values):
        s["frame_idx"] = i
        s.setdefault("_hist", {}).setdefault("mae_list", []).append(v)
        s = s_b_fn(dict(s))
        details = s["event_trigger"]["details"]
        fired = fired or bool(s["event_trigger"]["raw_trigger"])
    return fired, details


def test_no_history_leaves_mae_idle():
    s = s_b_fn(s_i_init_fn({"frame_idx": 0}))
    details = s["event_trigger"]["details"]
    assert details["mae_anomaly_z"] == 0.0
    assert details["mae_trigger"] is False
    assert s["event_trigger"]["raw_trigger"] is False


def test_short_history_is_ignored():
    fired, details = run_mae([5.0, 5.0, 5.0, 5.0, 9.0])
    assert details["mae_anomaly_z"] == 0.0
    assert details["mae_trigger"] is False
    assert fired is False


def test_stable_baseline_never_triggers():
    fired, details = run_mae(BASELINE)
    assert details["mae_trigger"] is False
    assert fired is False


def test_disabled_by_default_keeps_score_but_never_fires():
    assert sb.MAE_TRIGGER_ENABLED is False
    fired, details = run_mae(BASELINE + [8.0] * 6)
    assert details["mae_anomaly_z_ma"] >= sb.MAE_Z_THRESHOLD
    assert details["mae_trigger"] is False
    assert fired is False


def test_enabled_flag_fires_on_sustained_spike(monkeypatch):
    monkeypatch.setattr(sb, "MAE_TRIGGER_ENABLED", True)
    fired, details = run_mae(BASELINE + [8.0] * 6)
    assert details["mae_anomaly_z_ma"] >= sb.MAE_Z_THRESHOLD
    assert details["mae_trigger"] is True
    assert fired is True


def test_single_frame_spike_is_smoothed_out(monkeypatch):
    monkeypatch.setattr(sb, "MAE_TRIGGER_ENABLED", True)
    fired, details = run_mae(BASELINE + [8.0])
    assert details["mae_trigger"] is False
    assert fired is False


def test_calibration_score_stays_in_unit_interval():
    _, details = run_mae(BASELINE + [8.0] * 6)
    assert 0.0 <= details["mae_anomaly_score"] < 1.0
    assert details["mae_anomaly_score"] > 0.5


def test_threshold_is_monkeypatchable_for_retuning(monkeypatch):
    monkeypatch.setattr(sb, "MAE_TRIGGER_ENABLED", True)
    monkeypatch.setattr(sb, "MAE_Z_THRESHOLD", 1.0)
    fired, details = run_mae(BASELINE + [5.5] * 6)
    assert details["mae_trigger"] is True
    assert fired is True


def test_trigger_rule_is_auditable():
    s = s_b_fn(s_i_init_fn({"frame_idx": 0}))
    rule = s["meta_exec"]["mae_trigger_rule"]
    assert rule["threshold_z_ma"] == sb.MAE_Z_THRESHOLD
    assert rule["baseline_window_frames"] == sb.MAE_BASELINE_WINDOW
    assert rule["min_history_frames"] == sb.MAE_MIN_HISTORY
    assert rule["mad_sigma_scale"] == pytest.approx(1.4826)
    assert rule["enabled"] == sb.MAE_TRIGGER_ENABLED
    assert "method" in rule


def test_siren_and_mae_anomaly_jointly_trigger_with_reasons():
    state = s_i_init_fn({"frame_idx": 20})
    state["event_trigger"]["details"].update(
        {"p_siren": 0.0, "siren_detected_mae": True}
    )
    state["_hist"]["mae_list"] = [5.0] * 10 + [8.0]

    result = s_b_fn(state)

    event = result["event_trigger"]
    details = event["details"]
    assert details["siren_vision_trigger"] is True
    assert event["raw_trigger"] is True
    assert event["is_triggered"] is True
    assert event["trigger_reason"] == ["siren_detected", "vision_anomaly"]


def test_siren_without_mae_anomaly_does_not_jointly_trigger():
    state = s_i_init_fn({"frame_idx": 20})
    state["event_trigger"]["details"].update(
        {"p_siren": 0.0, "siren_detected_mae": True}
    )
    state["_hist"]["mae_list"] = BASELINE + [BASELINE[-1]]

    result = s_b_fn(state)

    assert result["event_trigger"]["details"]["siren_vision_trigger"] is False
    assert result["event_trigger"]["raw_trigger"] is False
