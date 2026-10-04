from langgraph.graph import StateGraph, START, END

from src.core.state import FrameAudioPacketState
from src.exp_a.agent_s_a import s_a_fn
from src.exp_b.agent_s_b import s_b_fn
from src.exp_c.agent_s_c import s_c_fn
from src.core.agents.s_i_init import s_i_init_fn
from src.core.agents.s_t import s_t_fn
from src.core.agents.s_v import s_v_fn
from src.core.integrator.pre_validator import pre_frame_init, pre_validator_sync
from src.core.integrator.team_barrier import team_barrier, team_barrier_ui
from src.core.tools.agent_wrapper import wrap_agent, wrap_agent_fast

s_i_init_n = wrap_agent("S_I", s_i_init_fn)
s_a_n = wrap_agent("S_A", s_a_fn)
s_b_n = wrap_agent("S_B", s_b_fn)
s_c_n = wrap_agent("S_C", s_c_fn)
s_t_n = wrap_agent("S_T", s_t_fn)
s_v_n = wrap_agent("S_V", s_v_fn)

s_i_init_n_ui = wrap_agent_fast("S_I", s_i_init_fn)
s_a_n_ui = wrap_agent_fast("S_A", s_a_fn)
s_b_n_ui = wrap_agent_fast("S_B", s_b_fn)
s_c_n_ui = wrap_agent_fast("S_C", s_c_fn)
s_t_n_ui = wrap_agent_fast("S_T", s_t_fn)
s_v_n_ui = wrap_agent_fast("S_V", s_v_fn)


def build_pipeline():
    b = StateGraph(FrameAudioPacketState)
    b.add_node("pre_frame_init", pre_frame_init)
    b.add_node("pre_validator_sync_pre", pre_validator_sync)
    b.add_node(
        "validator_pre",
        lambda s: __import__("src.core.validator", fromlist=[""]).contract_validate(s, "pre"),
    )
    b.add_node("S_I_init", s_i_init_n)
    b.add_node("S_A", s_a_n)
    b.add_node("S_T", s_t_n)
    b.add_node("S_V", s_v_n)
    b.add_node("aggregator_event", s_b_n)
    b.add_node("S_C", s_c_n)
    b.add_node("team_barrier", team_barrier)
    b.add_node(
        "validator_final",
        lambda s: __import__("src.core.validator", fromlist=[""]).contract_validate(s, "final"),
    )
    b.add_node("metrics", lambda s: dict(s))
    b.add_node("pre_validator_sync_post", pre_validator_sync)
    b.add_edge(START, "pre_frame_init")
    b.add_edge("pre_frame_init", "pre_validator_sync_pre")
    b.add_edge("pre_validator_sync_pre", "validator_pre")
    b.add_edge("validator_pre", "S_I_init")
    b.add_edge("S_I_init", "S_A")
    b.add_edge("S_I_init", "S_T")
    b.add_edge("S_I_init", "S_V")
    b.add_edge("S_A", "aggregator_event")
    b.add_edge("S_T", "aggregator_event")
    b.add_edge("S_V", "aggregator_event")
    b.add_edge("aggregator_event", "S_C")
    b.add_edge("S_C", "team_barrier")
    b.add_edge("team_barrier", "validator_final")
    b.add_edge("validator_final", "metrics")
    b.add_edge("metrics", "pre_validator_sync_post")
    b.add_edge("pre_validator_sync_post", END)
    return b.compile()


def build_ui_pipeline():
    b = StateGraph(FrameAudioPacketState)
    b.add_node("pre_frame_init", pre_frame_init)
    b.add_node("pre_validator_sync_pre", pre_validator_sync)
    b.add_node("S_I_init", s_i_init_n_ui)
    b.add_node("S_A", s_a_n_ui)
    b.add_node("S_T", s_t_n_ui)
    b.add_node("S_V", s_v_n_ui)
    b.add_node("aggregator_event", s_b_n_ui)
    b.add_node("S_C", s_c_n_ui)
    b.add_node("team_barrier", team_barrier_ui)
    b.add_edge(START, "pre_frame_init")
    b.add_edge("pre_frame_init", "pre_validator_sync_pre")
    b.add_edge("pre_validator_sync_pre", "S_I_init")
    b.add_edge("S_I_init", "S_A")
    b.add_edge("S_I_init", "S_T")
    b.add_edge("S_I_init", "S_V")
    b.add_edge("S_A", "aggregator_event")
    b.add_edge("S_T", "aggregator_event")
    b.add_edge("S_V", "aggregator_event")
    b.add_edge("aggregator_event", "S_C")
    b.add_edge("S_C", "team_barrier")
    b.add_edge("team_barrier", END)
    return b.compile()


graph = build_pipeline()
graph_ui = build_ui_pipeline()
