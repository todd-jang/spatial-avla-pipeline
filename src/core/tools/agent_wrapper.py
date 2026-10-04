import time
from copy import deepcopy
from typing import Callable, Any


def _now_ms() -> int:
    return int(time.time() * 1000)


def wrap_agent(agent_name: str, fn: Callable[[Any], Any]):
    def node(state: Any) -> Any:
        s = deepcopy(state) if isinstance(state, dict) else dict(state)
        me = s.setdefault("meta_exec", {})
        if not isinstance(me, dict):
            me = {}
            s["meta_exec"] = me
        me.setdefault("_agent_status", {})
        me.setdefault("_agent_finished_ts_ms", {})
        me.setdefault("errors", [])
        me["_agent_status"][agent_name] = "running"
        try:
            r = fn(s)
            s2 = r if isinstance(r, dict) else s
            me2 = s2.setdefault("meta_exec", {})
            if not isinstance(me2, dict):
                me2 = {}
                s2["meta_exec"] = me2
            me2.setdefault("_agent_status", {})
            me2.setdefault("_agent_finished_ts_ms", {})
            me2.setdefault("errors", [])
            me2["_agent_status"][agent_name] = "done"
            t = _now_ms()
            me2["_agent_finished_ts_ms"][agent_name] = t
            me2["poc_last_completed_agent_candidate"] = agent_name
            me2["poc_completed_at_ms_candidate"] = t
            return s2
        except Exception as e:
            me.setdefault("errors", []).append({"path": f"agent.{agent_name}", "code": "AGENT_EXCEPTION", "msg": str(e)})
            me["_agent_status"][agent_name] = "error"
            t = _now_ms()
            me["_agent_finished_ts_ms"][agent_name] = t
            me["poc_last_completed_agent_candidate"] = agent_name
            me["poc_completed_at_ms_candidate"] = t
            return s

    return node


def wrap_agent_fast(agent_name: str, fn: Callable[[Any], Any]):
    def node(state: Any) -> Any:
        s = deepcopy(state) if isinstance(state, dict) else dict(state)
        me = s.setdefault("meta_exec", {})
        if not isinstance(me, dict):
            me = {}
            s["meta_exec"] = me
        me.setdefault("_agent_status", {})
        me.setdefault("_agent_finished_ts_ms", {})
        me.setdefault("errors", [])
        me["_agent_status"][agent_name] = "running"
        try:
            r = fn(s)
            s2 = r if isinstance(r, dict) else s
            me2 = s2.setdefault("meta_exec", {})
            if not isinstance(me2, dict):
                me2 = {}
                s2["meta_exec"] = me2
            me2.setdefault("_agent_status", {})
            me2.setdefault("_agent_finished_ts_ms", {})
            me2.setdefault("errors", [])
            me2["_agent_status"][agent_name] = "done"
            t = _now_ms()
            me2["_agent_finished_ts_ms"][agent_name] = t
            me2["poc_last_completed_agent_candidate"] = agent_name
            me2["poc_completed_at_ms_candidate"] = t
            return s2
        except Exception as e:
            me.setdefault("errors", []).append({"path": f"agent.{agent_name}", "code": "AGENT_EXCEPTION", "msg": str(e)})
            me["_agent_status"][agent_name] = "error"
            t = _now_ms()
            me["_agent_finished_ts_ms"][agent_name] = t
            me["poc_last_completed_agent_candidate"] = agent_name
            me["poc_completed_at_ms_candidate"] = t
            return s

    return node
