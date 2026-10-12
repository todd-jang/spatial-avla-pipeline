"""S_C agent: turns the event trigger into an on-screen driving-action text.

Modes
-----
* default (USE_REAL_QWEN unset/0): deterministic stub text, optional latency
  simulation via QWEN_BITS (16 -> 85 ms, 4 -> 25 ms)  [unchanged behaviour].
* USE_REAL_QWEN=1: a real Qwen-Audio-Chat (4-bit, bitsandbytes) generates the
  action sentence from the trigger context. Generations are cached per
  (trigger, prompt_context) so a 749-frame clip costs only a handful of LLM
  calls instead of one per frame. If the model cannot be loaded, the agent
  falls back to the stub and tags the result ``llm_source="fallback"`` so a
  silent fallback is always visible (see ``vlm_payload["llm_source"]``).

``python src/exp_c/agent_s_c.py`` runs a self-check and prints which backend
actually answered.
"""
import os
import re
import sys
import threading
import time

QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen-Audio-Chat")
MAX_OVERLAY_CHARS = 60  # cv2.putText tile is ~640px wide, ASCII only

_lock = threading.Lock()
_backend = {"tok": None, "model": None, "failed": None}
_cache = {}


def _use_real() -> bool:
    return os.environ.get("USE_REAL_QWEN", "0") == "1"


def _load_qwen():
    """Lazy singleton loader. Returns (tok, model) or None (and remembers why)."""
    with _lock:
        if _backend["model"] is not None:
            return _backend["tok"], _backend["model"]
        if _backend["failed"] is not None:
            return None
        try:
            print("Loading Qwen Multimodal Model in 4-bit...", flush=True)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required for 4-bit Qwen (Colab: Runtime > GPU)")
            tok = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL, device_map="cuda", quantization_config=bnb,
                trust_remote_code=True,
            ).eval()
            try:
                model.generation_config.max_new_tokens = 24
            except Exception:
                pass
            _backend["tok"], _backend["model"] = tok, model
            print("Qwen loaded OK.", flush=True)
            return tok, model
        except Exception as e:  # noqa: BLE001 - must never break the pipeline
            _backend["failed"] = f"{type(e).__name__}: {e}"
            print(f"Failed to load Qwen: {_backend['failed']}", flush=True)
            return None


def _clean(text: str) -> str:
    text = re.sub(r"[^\x20-\x7e]", "", " ".join(str(text).split()))
    text = text.strip()
    if len(text) > MAX_OVERLAY_CHARS:
        text = text[: MAX_OVERLAY_CHARS - 3].rstrip() + "..."
    return text or "NO RESPONSE"


def _build_prompt(trig: bool, context: str) -> str:
    state = "an emergency siren/vehicle event is TRIGGERED" if trig else "no emergency event is detected"
    return (
        "You are the driving assistant of an autonomous car. "
        f"Sensor summary: {state}. {context} "
        "Reply with ONE short imperative sentence in English (max 8 words) "
        "telling the car what to do."
    )


def _ask_qwen(trig: bool, context: str):
    """Return generated text, or None if the real backend is unavailable."""
    key = (trig, context)
    if key in _cache:
        return _cache[key]
    loaded = _load_qwen()
    if loaded is None:
        return None
    tok, model = loaded
    try:
        query = tok.from_list_format([{"text": _build_prompt(trig, context)}])
        resp, _ = model.chat(tok, query=query, history=None)
    except Exception as e:  # noqa: BLE001
        _backend["failed"] = f"inference {type(e).__name__}: {e}"
        print(f"Qwen inference failed: {_backend['failed']}", flush=True)
        return None
    _cache[key] = _clean(resp)
    return _cache[key]


def s_c_fn(state: dict) -> dict:
    s = dict(state)
    ev = s.setdefault("event_trigger", {})
    vp = s.setdefault("vlm_payload", {"trigger": False, "mode": "symbolic", "prompt_context": ""})
    trig = bool(ev.get("is_triggered", False))
    vp["trigger"] = trig
    vp["mode"] = "symbolic"
    rs = list(ev.get("trigger_reason", []))
    if rs:
        vp["prompt_context"] = "[AUDIO/VISION ALERT] " + (", ".join(rs))
    else:
        vp["prompt_context"] = "[AUDIO/VISION ALERT] EVENT_TRIGGERED" if trig else ""

    stub = (
        "Qwen(Local): EMERGENCY VEHICLE DETECTED. PULL OVER."
        if trig else "Qwen(Local): KEEP CURRENT LANE. SAFE."
    )
    if _use_real():
        text = _ask_qwen(trig, vp["prompt_context"])
        if text is not None:
            vp["llm_action_text"] = "Qwen: " + text
            vp["llm_source"] = "qwen"
        else:
            vp["llm_action_text"] = stub.replace("Qwen(Local)", "Qwen(Sim)")
            vp["llm_source"] = "fallback"
    else:
        vp["llm_action_text"] = stub
        vp["llm_source"] = "stub"
        # Simulate VLM latency overhead based on quantization
        qwen_bits = os.environ.get("QWEN_BITS")
        if qwen_bits == "16":
            time.sleep(0.085)  # ~85ms FP16
        elif qwen_bits == "4":
            time.sleep(0.025)  # ~25ms 4-bit

    s.setdefault(
        "vlm",
        {
            "trigger": "event" if trig else "none",
            "frame_id_ref": s.get("frame_id", -1),
            "ts": s.get("ts_ms", 0.0),
        },
    )
    s.setdefault("img", {"raw_b64": "", "clean_b64": "", "quad_b64": ""})
    return s


if __name__ == "__main__":
    ok = True
    for trig, reason in ((False, []), (True, ["siren p=0.93", "DoA left 40deg"])):
        out = s_c_fn({"event_trigger": {"is_triggered": trig, "trigger_reason": reason}})
        vp = out["vlm_payload"]
        print(f"trigger={trig!s:5} source={vp['llm_source']:8} -> {vp['llm_action_text']}")
        if _use_real() and vp["llm_source"] != "qwen":
            ok = False
    if _use_real():
        print("REAL LLM OK" if ok else f"REAL LLM FAILED: {_backend['failed']}")
        sys.exit(0 if ok else 1)

