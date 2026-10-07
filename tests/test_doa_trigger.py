"""Acceptance tests for todo 14 (DoA wired through the packet, reachable threshold)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

import src.exp_b.agent_s_b as sb  # noqa: E402
from src.core.agents.s_i_init import s_i_init_fn  # noqa: E402
from src.exp_b.agent_s_b import s_b_fn  # noqa: E402

FPS = 30


def sweep(rate_dps, frames=90, ambiguous=False, dir_override=None):
    s = s_i_init_fn({"frame_idx": 0})
    prev = 0.0
    fired = False
    peak = 0.0
    for i in range(frames):
        nxt = prev if dir_override is not None else prev + rate_dps / FPS
        s["doa"] = {
            "dir_deg": max(-90.0, min(90.0, nxt)),
            "gt_dir_deg": -1.0,
            "confidence": 0.9,
            "direction": "right",
            "ambiguous": ambiguous,
        }
        prev = s["doa"]["dir_deg"]
        s = s_b_fn(dict(s))
        s["frame_idx"] = i
        det = s["event_trigger"]["details"]
        peak = max(peak, float(det.get("doa_change_rate_dps_ma", 0.0)))
        fired = fired or bool(s["event_trigger"]["raw_trigger"])
    return fired, peak


def test_fast_pass_fires_raw_trigger():
    fired, peak = sweep(275.0)
    assert fired, f"a 275 deg/s pass must trigger, peak dps_ma={peak}"
    assert peak >= 120.0


def test_slow_sweep_does_not_fire():
    fired, peak = sweep(10.0)
    assert not fired, f"a 10 deg/s sweep must stay silent, peak dps_ma={peak}"
    assert peak < 120.0


def test_legacy_threshold_would_never_fire():
    """Negative control: the old 15.0 deg/frame rule is 450 deg/s, so 275 must miss it."""
    original = sb.DOA_RATE_MA_THRESHOLD_DPS
    sb.DOA_RATE_MA_THRESHOLD_DPS = 15.0 * FPS
    try:
        fired, peak = sweep(275.0)
    finally:
        sb.DOA_RATE_MA_THRESHOLD_DPS = original
    assert not fired, "at the legacy 450 deg/s the fast pass must stop triggering"
    assert peak < 15.0 * FPS


def test_ambiguous_frames_do_not_fabricate_a_rate_spike():
    """A hard direction jump while ambiguous must be carried forward, not counted."""
    _, clean = sweep(0.0, frames=20, dir_override=0.0)
    fired, peak = sweep(0.0, frames=20, ambiguous=True, dir_override=90.0)
    assert peak == 0.0, f"ambiguous run fabricated a rate: {peak}"
    assert peak == clean
    assert not fired


def test_do_a_threshold_constant_changed_and_legacy_recorded():
    src = (ROOT / "src/exp_b/agent_s_b.py").read_text()
    assert "DOA_RATE_MA_THRESHOLD_DPS = 120.0" in src
    assert "dpsm >= DOA_RATE_MA_THRESHOLD_DPS" in src
    assert "dpsm >= 15.0" not in src, "15.0 must not remain an active threshold"
    assert "LEGACY_RATE_MA_THRESHOLD_DEG_PER_FRAME = 15.0" in src, "legacy value must stay auditable"
    assert sb.DOA_RATE_MA_THRESHOLD_DPS == 120.0
    assert sb.LEGACY_RATE_MA_THRESHOLD_DEG_PER_FRAME == 15.0


def test_other_rules_and_min_hold_are_unchanged():
    src = (ROOT / "src/exp_b/agent_s_b.py").read_text()
    for untouched in (
        "p >= 0.80",
        'd.get("lead_lost", False)',
        'int(d.get("lane_lost_count", 0)) >= 3',
        "vam >= 0.75",
        "mhf = 35",
        '"min_hold_frames": 35',
    ):
        assert untouched in src, f"rule drifted: {untouched}"


def test_both_thresholds_are_recorded_in_meta_exec():
    s = s_i_init_fn({"frame_idx": 0})
    prev = 0.0
    state = s
    for i in range(20):
        s["doa"] = {"dir_deg": max(-90.0, min(90.0, prev + 275.0 / FPS)),
                    "gt_dir_deg": -1.0, "confidence": 0.9, "direction": "right"}
        prev = s["doa"]["dir_deg"]
        state = s_b_fn(dict(s))
        state["frame_idx"] = i
    rule = state["meta_exec"]["doa_trigger_rule"]
    assert rule["threshold_dps_ma"] == 120.0
    assert rule["legacy_threshold_deg_per_frame_ma"] == 15.0
    assert rule["fps_nominal"] == 30
    assert rule["ma_window_frames"] == 5


@pytest.mark.parametrize("rate,should_fire", [(275.0, True), (229.0, True), (10.0, False), (0.0, False)])
def test_threshold_sits_inside_the_physically_reachable_band(rate, should_fire):
    """Real siren passes peak at 229-382 deg/s, so both sides of 120 must be reachable."""
    fired, _ = sweep(rate)
    assert fired is should_fire