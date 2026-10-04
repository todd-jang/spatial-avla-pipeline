import time


def _now_ms() -> int:
    return int(time.time() * 1000)


def team_barrier(state: dict) -> dict:
    s = dict(state)
    me = s.setdefault("meta_exec", {})
    me.setdefault("_agent_finished_ts_ms", {})
    me.setdefault("errors", [])
    fm = me.get("_agent_finished_ts_ms", {})
    ft = [int(v) for v in fm.values()]
    lc_ts = max(ft) if ft else int(me.get("poc_completed_at_ms_candidate", _now_ms()))
    me["poc_completed_at_ms"] = int(lc_ts)
    me["poc_last_completed_agent"] = me.get("poc_last_completed_agent_candidate", "")
    me["poc_all_done"] = True
    st = me.get("_agent_status", {})
    if st:
        me["poc_all_ok"] = (len(me.get("errors", [])) == 0 and all(v == "done" for v in st.values()))
    else:
        me["poc_all_ok"] = (len(me.get("errors", [])) == 0)
    me["status"] = "ok" if me["poc_all_ok"] else "partial_or_error"
    return s


def team_barrier_ui(state: dict) -> dict:
    s = dict(state)
    me = s.setdefault("meta_exec", {})
    cand = me.get("poc_completed_at_ms_candidate", _now_ms())
    me["poc_completed_at_ms"] = int(cand)
    me["poc_last_completed_agent"] = me.get("poc_last_completed_agent_candidate", "")
    me["poc_all_done"] = True
    me["poc_all_ok"] = (len(me.get("errors", [])) == 0)
    me["status"] = "ok" if me["poc_all_ok"] else "partial_or_error"
    return s
