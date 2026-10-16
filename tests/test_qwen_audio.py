import threading
import time

import numpy as np

from src.app.qwen_audio import QwenAudioAnalyzer


def test_qwen_audio_reports_when_disabled():
    analyzer = QwenAudioAnalyzer(enabled=False)

    result = analyzer.update(0, np.zeros(32, dtype=np.float32), 16000)

    assert result == {
        "status": "disabled",
        "source": "disabled",
        "text": "Set USE_QWEN_AUDIO=1 to enable",
        "window_index": None,
    }


def test_qwen_audio_runs_in_background_and_keeps_latest_pending_window():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def infer(waveform, sample_rate, model_id):
        calls.append((waveform.copy(), sample_rate, model_id))
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=2)
        return f"window {int(waveform[0])}"

    analyzer = QwenAudioAnalyzer(enabled=True, infer_fn=infer)
    try:
        first = analyzer.update(0, np.array([0], dtype=np.float32), 16000)
        assert first["status"] == "analyzing"
        assert entered.wait(timeout=1)

        analyzer.update(1, np.array([1], dtype=np.float32), 16000)
        analyzer.update(2, np.array([2], dtype=np.float32), 16000)
        release.set()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = analyzer.update(
                2, np.array([2], dtype=np.float32), 16000
            )
            if result["status"] == "ready" and result["window_index"] == 2:
                break
            time.sleep(0.01)

        assert result == {
            "status": "ready",
            "source": "qwen_audio",
            "text": "window 2",
            "window_index": 2,
        }
        assert [int(call[0][0]) for call in calls] == [0, 2]
    finally:
        release.set()


def test_qwen_audio_errors_are_reported_without_fallback_text():
    def fail(*args):
        raise RuntimeError("model unavailable")

    analyzer = QwenAudioAnalyzer(enabled=True, infer_fn=fail)
    result = analyzer.update(0, np.ones(32, dtype=np.float32), 16000)

    deadline = time.monotonic() + 1
    while result["status"] != "error" and time.monotonic() < deadline:
        time.sleep(0.01)
        result = analyzer.update(0, np.ones(32, dtype=np.float32), 16000)

    assert result["status"] == "error"
    assert result["source"] == "error"
    assert "RuntimeError: model unavailable" in result["text"]
