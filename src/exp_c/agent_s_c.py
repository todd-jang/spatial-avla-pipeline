import time
import os

# --- 실제 4-bit Qwen 연동용 코드 (Colab 등에서 USE_REAL_QWEN=1 환경변수로 활성화) ---
# pip install transformers accelerate bitsandbytes
USE_REAL_QWEN = os.environ.get("USE_REAL_QWEN", "0") == "1"
qwen_model, qwen_tokenizer = None, None

if USE_REAL_QWEN:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print("Loading Qwen Multimodal Model in 4-bit...")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        # 예시로 Qwen-Audio 또는 Qwen-VL 모델 경로 사용
        model_id = "Qwen/Qwen-Audio-Chat" 
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        qwen_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, trust_remote_code=True, device_map="auto")
        qwen_model.eval()
    except Exception as e:
        print(f"Failed to load Qwen: {e}")
        USE_REAL_QWEN = False

def s_c_fn(state: dict) -> dict:
    s = dict(state)
    ev = s.setdefault("event_trigger", {})
    vp = s.setdefault("vlm_payload", {"trigger": False, "mode": "symbolic", "prompt_context": ""})
    trig = bool(ev.get("is_triggered", False))
    vp["trigger"] = trig
    vp["mode"] = "symbolic"
    rs = list(ev.get("trigger_reason", []))
    
    prompt = "[AUDIO/VISION ALERT] " + (", ".join(rs)) if rs else ("[AUDIO/VISION ALERT] EVENT_TRIGGERED" if trig else "ALL CLEAR. NO ALERT.")
    vp["prompt_context"] = prompt
    
    if USE_REAL_QWEN and qwen_model is not None:
        # 실제 Qwen 4-bit 인퍼런스
        sys_prompt = "You are a driving assistant. Based on the alert context, output a short action."
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
        outputs = qwen_model.generate(**inputs, max_new_tokens=15)
        response = qwen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        vp["llm_action_text"] = f"Qwen: {response.strip()}"
        vp["mode"] = "qwen_4bit"
    else:
        # Mock 시뮬레이션
        if trig:
            vp["llm_action_text"] = "Qwen(Local): EMERGENCY VEHICLE DETECTED. PULL OVER."
        else:
            vp["llm_action_text"] = "Qwen(Local): KEEP CURRENT LANE. SAFE."
        
        qwen_bits = os.environ.get("QWEN_BITS")
        if qwen_bits == "16":
            time.sleep(0.085)
        elif qwen_bits == "4":
            time.sleep(0.025)
            
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
