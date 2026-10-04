import json
from pathlib import Path
from typing import Dict, Any

try:
    from jsonschema import Draft7Validator, ValidationError
except ImportError:
    Draft7Validator = None
    ValidationError = Exception


def _load_schema() -> Dict[str, Any]:
    p = Path("contracts/frame_audio_packet_v1.0.json")
    if not p.exists():
        raise FileNotFoundError(p)
    return json.load(open(p, "r", encoding="utf-8"))


def contract_validate(pkt: Dict[str, Any], node_name: str = "unknown") -> Dict[str, Any]:
    s = dict(pkt)
    me = s.setdefault(
        "meta_exec",
        {
            "errors": [],
            "status": "idle",
            "poc_completed_at_ms": 0,
            "poc_last_completed_agent": "",
            "poc_all_done": False,
            "poc_all_ok": False,
        },
    )
    me.setdefault("errors", [])
    if Draft7Validator is not None:
        try:
            Draft7Validator(_load_schema()).validate(s)
        except ValidationError as e:
            me["status"] = "error"
            me["errors"].append(
                {
                    "path": node_name,
                    "code": "SCHEMA_VALIDATION_ERROR",
                    "msg": e.message,
                }
            )
    return s
