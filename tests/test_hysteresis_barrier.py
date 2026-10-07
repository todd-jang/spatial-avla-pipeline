def test_hysteresis_hold_logic():
    from src.exp_b.agent_s_b import s_b_fn
    s = {
        'frame_idx': 0,
        'event_trigger': {'details': {'lead_lost': True, 'vision_anomaly_ma': 0.8, 'p_siren': 0.9, 'doa_change_rate_ma': 20}},
        '_hist': {'va_list': [0.8]*5, 'dd_list': [20]*5},
    }
    r = s_b_fn(s)
    assert r['event_trigger']['is_triggered'] is True


def test_poc_t_last_separate():
    from src.core.integrator.team_barrier import team_barrier
    s = {'meta_exec': {'_agent_finished_ts_ms': {'S_A': 10, 'S_C': 100}, 'errors': [], '_agent_status': {'S_A':'done','S_C':'done'}}}
    r = team_barrier(s)
    assert r['meta_exec']['poc_completed_at_ms'] == 100
