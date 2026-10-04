import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.pipeline import graph
from mocks.mock_base import MOCK_STATE


def execute_backup_plan(state: dict) -> dict:
    s = dict(state)
    me = s.setdefault("meta_exec", {})
    me["fallback"] = {
        "triggered": True,
        "reason": "local_inference_safety_fallback",
        "mode_effective": "vision_only_local",
    }
    plan = s.setdefault("plan", {})
    plan["mode"] = "fallback"
    alerts = plan.setdefault("alerts", {})
    alerts["aeb_prep"] = True
    ev = s.setdefault("event_trigger", {})
    ev["is_triggered"] = bool(ev.get("is_triggered", False))
    return s


def main():
    s = deepcopy(MOCK_STATE)
    try:
        res = graph.invoke(s)
    except Exception as e:
        print("GRAPH ERROR:", e)
        res = execute_backup_plan(s)
    me = res.get("meta_exec", {})
    print("POC.POC_Completed_At_ms_T_last:", me.get("poc_completed_at_ms"))
    print("POC.All_OK:", me.get("poc_all_ok"))
    print("Status:", me.get("status"))
    print("Errors_Count:", len(me.get("errors", [])))
    if me.get("fallback"):
        print("Fallback:", me["fallback"])


if __name__ == "__main__":
    main()
