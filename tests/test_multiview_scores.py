import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.app import multiview  # noqa: E402


def test_multiview_renders_vision_mae_and_siren_scores(monkeypatch):
    rendered_text = []
    original_put_text = multiview.cv2.putText

    def capture_text(image, text, *args, **kwargs):
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(multiview.cv2, "putText", capture_text)
    state = {
        "event_trigger": {
            "details": {
                "vision_anomaly_score": 0.278,
                "vision_anomaly_ma": 0.31,
                "vision_anomaly_mae": 4.4148,
                "vision_anomaly_mae_ma": 4.2012,
                "p_siren": 0.802,
                "p_siren_ma5": 0.755,
                "p_siren_delta": 0.047,
                "siren_score_template": 0.2145,
                "siren_detected_mae": True,
                "siren_strength_mae": "strong",
            },
            "hysteresis": {"state": "active"},
        },
        "vlm_payload": {},
        "plan": {},
        "meta_exec": {},
    }

    out = multiview.process_multi_view_frame(
        np.zeros((540, 960, 3), dtype=np.uint8), 0, state
    )

    assert out.shape == (1080, 1920, 3)
    assert "[Vision] RT-DETR:0.278 MA5:0.31" in rendered_text
    assert "MAE-Tiny distance: 4.4148" in rendered_text
    assert "MAE-Tiny MA5: 4.2012" in rendered_text
    assert "[Siren] p:0.802 AST:0.214 MAE:yes HYST:ACTIVE TRIG:False" in rendered_text
    assert "p_siren MA5: 0.755" in rendered_text
    assert "p_siren delta: 0.047" in rendered_text
    assert any("VISION risk 0.278" in text for text in rendered_text)
    assert any("DOA rate" in text and "TRIGGER raw=" in text for text in rendered_text)
    assert sum("FRAME 0 | DETECTIONS" in text for text in rendered_text) == 4
    assert sum(
        "MPC DEMO steer -1.0deg | brake 60% lamp ON | throttle 0%" in text
        for text in rendered_text
    ) == 4
    assert sum("MPC VIRTUAL inputs:" in text for text in rendered_text) == 4


def test_missing_scores_render_as_unavailable(monkeypatch):
    rendered_text = []
    original_put_text = multiview.cv2.putText

    def capture_text(image, text, *args, **kwargs):
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(multiview.cv2, "putText", capture_text)
    state = {
        "event_trigger": {"details": {}, "hysteresis": {}},
        "vlm_payload": {},
        "plan": {},
        "meta_exec": {},
    }

    multiview.process_multi_view_frame(
        np.zeros((540, 960, 3), dtype=np.uint8), 0, state
    )

    assert "[Vision] RT-DETR:n/a MA5:n/a" in rendered_text
    assert "MAE-Tiny distance: n/a" in rendered_text
    assert "p_siren MA5: n/a" in rendered_text
    assert any("VISION risk n/a" in text for text in rendered_text)


def test_custom_upload_does_not_render_mock_scores_as_perception(monkeypatch):
    from src.app.gradio_app import render_uploaded_video_frame

    rendered_text = []
    original_put_text = multiview.cv2.putText

    def capture_text(image, text, *args, **kwargs):
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(multiview.cv2, "putText", capture_text)
    state = {
        "frame_idx": 2,
        "event_trigger": {
            "details": {
                "vision_anomaly_score": 0.2,
                "p_siren": 0.0,
                "siren_score_template": 0.42,
                "siren_detected_mae": True,
                "siren_strength_mae": "strong",
            },
            "raw_trigger": False,
            "is_triggered": False,
            "hysteresis": {"state": "idle"},
        },
        "audio": {
            "siren": {"confidence": 0.0},
            "doa": {"dir_deg": 0.0, "confidence": 0.0},
        },
        "doa": {"dir_deg": 0.0, "confidence": 0.0},
        "percept": {"boxes": [], "has_lead": False},
        "meta_exec": {},
        "vlm_payload": {"llm_action_text": "Qwen(Local): KEEP", "llm_source": "stub"},
        "plan": {"steer_deg": 0, "speed_kmh": 0},
    }
    render_uploaded_video_frame(np.zeros((540, 960, 3), dtype=np.uint8), 2, state)

    assert any("VISION risk n/a" in text for text in rendered_text)
    assert any("AUDIO p_siren n/a | DOA n/a" in text for text in rendered_text)
    assert any(
        text.startswith("[Siren] p:n/a AST:n/a MAE:n/a")
        for text in rendered_text
    )
    assert "AST strength: n/a" in rendered_text
    assert any("TRIGGER n/a (no measured live perception input)" in text
               for text in rendered_text)
    assert state["event_trigger"]["details"]["vision_anomaly_score"] == 0.2
