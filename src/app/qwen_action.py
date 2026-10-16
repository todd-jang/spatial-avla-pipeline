"""Asynchronous Qwen action advisories for the read-only live monitor."""
from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACTION_LABELS = {"MONITOR", "CAUTION", "ALERT_OPERATOR"}
MODEL_ID = os.environ.get("QWEN_ACTION_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-action")
_model_lock = threading.Lock()
_model_bundle = None
MIN_INFERENCE_INTERVAL_S = 5.0


def _load_model():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle
    with _model_lock:
        if _model_bundle is not None:
            return _model_bundle
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        ).to(device).eval()
        _model_bundle = tokenizer, model, device
    return _model_bundle


def _infer_action(context: str) -> str:
    tokenizer, model, device = _load_model()
    messages = [
        {
            "role": "system",
            "content": (
                "You provide a read-only driving-safety advisory for a human "
                "operator. You cannot control a vehicle. The input contains "
                "camera detections only; no live audio or siren classification "
                "is available. Return exactly one label: MONITOR, CAUTION, "
                "or ALERT_OPERATOR. Never invent a siren or issue a steering, "
                "braking, acceleration, or route command."
            ),
        },
        {"role": "user", "content": context},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        generated[0, input_ids.shape[-1]:], skip_special_tokens=True
    ).strip().upper()
    if text not in ACTION_LABELS:
        match = re.fullmatch(
            r"(?:ACTION\s*:\s*)?(MONITOR|CAUTION|ALERT_OPERATOR)[.!]?", text
        )
        if not match:
            raise ValueError(f"Qwen returned an unsupported action label: {text!r}")
        text = match.group(1)
    return text


class QwenActionAdvisor:
    """Pollable per-session advisory state backed by one shared model worker."""

    def __init__(self, infer_fn=None):
        self._infer_fn = infer_fn or _infer_action
        self._future: Future | None = None
        self._future_key = None
        self._key = None
        self._context = ""
        self._action = None
        self._error = None
        self._last_started_at = 0.0

    def update(self, triggered: bool, risk: float, labels: list[str]) -> str:
        band = min(4, max(0, int(float(risk) * 4)))
        unique_labels = sorted(set(str(label) for label in labels))[:8]
        key = bool(triggered), band, tuple(unique_labels)
        self._collect_result()
        if key != self._key:
            self._key = key
            self._context = (
                f"Vision risk band {band}/4; detections: "
                f"{', '.join(unique_labels) if unique_labels else 'none'}; "
                f"pipeline vision trigger: {'active' if triggered else 'inactive'}."
            )
            self._action = None
            self._error = None
        self._schedule_latest()
        return self.status()

    def _collect_result(self) -> None:
        if self._future is None or not self._future.done():
            return
        future, future_key = self._future, self._future_key
        self._future = None
        self._future_key = None
        try:
            action = future.result()
        except Exception as exc:
            if future_key == self._key:
                self._error = f"{type(exc).__name__}: {exc}"
            return
        if future_key == self._key:
            self._action = action

    def _schedule_latest(self) -> None:
        if (
            self._future is None
            and self._key is not None
            and self._action is None
            and self._error is None
            and time.monotonic() - self._last_started_at >= MIN_INFERENCE_INTERVAL_S
        ):
            self._future_key = self._key
            self._future = _executor.submit(self._infer_fn, self._context)
            self._last_started_at = time.monotonic()

    def status(self) -> str:
        self._collect_result()
        self._schedule_latest()
        if self._error:
            return f"Qwen advisory error: {self._error}"
        if self._action:
            return f"Qwen read-only advisory: {self._action}"
        if self._future:
            return "Qwen read-only advisory: loading / analyzing..."
        if self._key is not None:
            return "Qwen read-only advisory: waiting to refresh..."
        return "Qwen read-only advisory: waiting for camera analysis"
