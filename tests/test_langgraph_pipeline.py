import copy

from mocks.mock_base import MOCK_STATE
from src.core.pipeline import graph


def _invoke(**overrides):
    s = copy.deepcopy(MOCK_STATE)
    s.update(overrides)
    return graph.invoke(s)


def test_graph_invokes_end_to_end_ok():
    r = _invoke()
    assert r["meta_exec"]["status"] == "ok"
    assert r["meta_exec"]["poc_all_ok"] is True
    assert r["meta_exec"]["poc_all_done"] is True
    assert r["meta_exec"]["errors"] == []
    st = r["meta_exec"]["_agent_status"]
    assert st == {"S_I": "done", "S_A": "done", "S_T": "done", "S_V": "done", "S_B": "done", "S_C": "done"}
    assert r["event_trigger"]["is_triggered"] is False


def test_graph_fires_on_high_p_siren():
    s = copy.deepcopy(MOCK_STATE)
    s["event_trigger"]["details"]["p_siren"] = 0.95
    r = graph.invoke(s)
    assert r["event_trigger"]["is_triggered"] is True
    assert r["event_trigger"]["hysteresis"]["state"] == "active"
    assert r["vlm_payload"]["trigger"] is True
    assert r["vlm_payload"]["prompt_context"].startswith("[AUDIO/VISION ALERT]")


def test_graph_fires_on_fast_doa_jump():
    s = copy.deepcopy(MOCK_STATE)
    s["_hist"]["doa_prev_dir"] = 0.0
    s["doa"]["dir_deg"] = 170.0
    r = graph.invoke(s)
    d = r["event_trigger"]["details"]
    assert d["doa_change_rate_dps_ma"] >= 120.0
    assert r["event_trigger"]["is_triggered"] is True
    assert r["event_trigger"]["hysteresis"]["state"] == "active"