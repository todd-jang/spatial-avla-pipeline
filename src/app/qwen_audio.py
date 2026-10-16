"""Optional asynchronous Qwen-Audio inference for uploaded-video audio windows."""
from __future__ import annotations

import os
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

DEFAULT_MODEL_ID = "Qwen/Qwen-Audio-Chat"
_MODEL_LOCK = threading.Lock()
_MODEL_BUNDLE = None
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-audio")


def _load_model(model_id: str):
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        loaded_id, tokenizer, model = _MODEL_BUNDLE
        if loaded_id != model_id:
            raise RuntimeError(
                f"Qwen Audio model already loaded as {loaded_id!r}; "
                f"cannot switch to {model_id!r} in the same process"
            )
        return tokenizer, model

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is not None:
            loaded_id, tokenizer, model = _MODEL_BUNDLE
            if loaded_id != model_id:
                raise RuntimeError(
                    f"Qwen Audio model already loaded as {loaded_id!r}; "
                    f"cannot switch to {model_id!r} in the same process"
                )
            return tokenizer, model

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen Audio requires a CUDA GPU; select a GPU runtime")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            quantization_config=quantization,
            trust_remote_code=True,
        ).eval()
        _MODEL_BUNDLE = model_id, tokenizer, model
        return tokenizer, model


def _infer_audio_window(
    waveform: np.ndarray, sample_rate: int, model_id: str
) -> str:
    import soundfile as sf

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Qwen Audio input must be a non-empty mono waveform")
    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate: {sample_rate}")
    if not np.isfinite(audio).all():
        raise ValueError("Qwen Audio input contains non-finite samples")

    tokenizer, model = _load_model(model_id)
    prompt = (
        "Listen to this environmental audio and assess whether an emergency "
        "vehicle siren is audible. Return a concise acoustic observation. "
        "Do not infer driving actions or claim certainty beyond the audio."
    )
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
        sf.write(audio_file.name, audio, sample_rate)
        query = tokenizer.from_list_format([
            {"audio": audio_file.name},
            {"text": prompt},
        ])
        response, _ = model.chat(tokenizer, query=query, history=None)
    text = " ".join(str(response).split())
    if not text:
        raise RuntimeError("Qwen Audio returned an empty response")
    return text[:240]


class QwenAudioAnalyzer:
    """Run one Qwen-Audio request at a time without blocking video rendering."""

    def __init__(self, enabled: bool | None = None, infer_fn=None) -> None:
        self.enabled = (
            os.environ.get("USE_QWEN_AUDIO", "0") == "1"
            if enabled is None
            else enabled
        )
        self._infer_fn = infer_fn or _infer_audio_window
        self._future: Future | None = None
        self._future_window: int | None = None
        self._pending: tuple[int, np.ndarray, int] | None = None
        self._last_submitted_window: int | None = None
        self._latest: tuple[int, str] | None = None
        self._error: str | None = None
        self._model_id = os.environ.get("QWEN_AUDIO_MODEL", DEFAULT_MODEL_ID)

    def update(
        self, window_index: int, waveform: np.ndarray, sample_rate: int
    ) -> dict:
        if not self.enabled:
            return {
                "status": "disabled",
                "source": "disabled",
                "text": "Set USE_QWEN_AUDIO=1 to enable",
                "window_index": None,
            }

        self._collect_result()
        if self._error is None and window_index != self._last_submitted_window:
            audio = np.asarray(waveform, dtype=np.float32).copy()
            if audio.size:
                self._pending = (window_index, audio, sample_rate)
                self._schedule()

        if self._error is not None:
            return {
                "status": "error",
                "source": "error",
                "text": self._error,
                "window_index": self._future_window,
            }
        if self._future is not None:
            return {
                "status": "analyzing",
                "source": "qwen_audio",
                "text": "Analyzing audio window...",
                "window_index": self._future_window,
            }
        if self._latest is not None:
            index, text = self._latest
            return {
                "status": "ready",
                "source": "qwen_audio",
                "text": text,
                "window_index": index,
            }
        return {
            "status": "waiting",
            "source": "qwen_audio",
            "text": "Waiting for audio window",
            "window_index": None,
        }

    def _schedule(self) -> None:
        if self._future is not None or self._pending is None:
            return
        window_index, waveform, sample_rate = self._pending
        self._pending = None
        self._future_window = window_index
        self._last_submitted_window = window_index
        self._future = _EXECUTOR.submit(
            self._infer_fn, waveform, sample_rate, self._model_id
        )

    def _collect_result(self) -> None:
        if self._future is None or not self._future.done():
            return
        future = self._future
        window_index = self._future_window
        self._future = None
        self._future_window = None
        try:
            self._latest = (window_index, future.result())
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._pending = None
        self._schedule()
