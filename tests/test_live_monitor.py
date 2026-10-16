import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.app.live_monitor import (
    LiveObjectDetector,
    live_metrics,
    microphone_level,
    microphone_metrics,
    vision_risk_score,
)


def test_live_vision_risk_uses_configured_class_weights():
    detections = [
        {"label": "car", "confidence": 0.9},
        {"label": "person", "confidence": 0.5},
    ]
    weights = {
        "default_weight": 0.0,
        "min_conf": 0.25,
        "classes": {"car": 0.3, "person": 0.55},
    }

    assert vision_risk_score(detections, weights) == 0.275


def test_live_vision_risk_ignores_detections_below_min_confidence():
    weights = {"default_weight": 0.0, "min_conf": 0.5, "classes": {"car": 0.3}}

    assert vision_risk_score(
        [{"label": "car", "confidence": 0.4}], weights
    ) == 0.0


def test_live_detector_decodes_model_results_and_scores_them():
    class FakeBoxes:
        xyxy = np.array([[1, 2, 30, 40]])
        cls = np.array([0])
        conf = np.array([0.9])

    class FakeModel:
        def predict(self, **kwargs):
            assert kwargs["imgsz"] == 640
            return [type("Result", (), {"boxes": FakeBoxes(), "names": {0: "car"}})()]

    detector = LiveObjectDetector()
    detector._model = FakeModel()
    detector._device = "cpu"
    detector._weights = {
        "default_weight": 0.0,
        "min_conf": 0.25,
        "classes": {"car": 0.3},
    }

    detections, risk = detector.analyze(np.zeros((48, 64, 3), dtype=np.uint8))

    assert detections == [{"label": "car", "confidence": 0.9, "bbox": [1, 2, 30, 40]}]
    assert risk == 0.27


def test_microphone_level_reports_signal_metrics_without_classification():
    result = microphone_level((16000, np.array([0.0, 0.5, -0.5], dtype=np.float32)))

    assert "Audio level only (no siren classifier)" in result
    assert "RMS 0.4082" in result
    assert "peak 0.5000" in result
    assert "16000 Hz" in result
    metrics = microphone_metrics(
        (16000, np.array([0.0, 0.5, -0.5], dtype=np.float32))
    )
    assert metrics["rms"] == pytest.approx(0.408248)
    assert metrics["peak_dbfs"] == pytest.approx(-6.0206)
    assert metrics["duration_seconds"] == pytest.approx(3 / 16000)
    assert metrics["siren_classifier"] == "not_connected"


def test_microphone_level_rejects_non_finite_samples():
    with pytest.raises(ValueError, match="non-finite"):
        microphone_level((16000, np.array([float("nan")], dtype=np.float32)))


def test_live_metrics_keeps_unavailable_values_and_returns_numeric_measurements():
    values = live_metrics(
        {
            "frame_idx": 12,
            "event_trigger": {
                "details": {"vision_anomaly_ma": 0.3},
                "raw_trigger": True,
                "is_triggered": True,
                "hysteresis": {"state": "active"},
            },
            "meta_exec": {"lat_ms": 4.0, "score_status": {"siren": "not_connected"}},
            "audio": {"siren": {"confidence": None}},
            "plan": {"steer_deg": 1.2, "speed_kmh": 10},
            "vlm_payload": {"llm_action_text": "MONITOR", "llm_source": "qwen"},
        },
        [
            {"label": "car", "confidence": 0.8, "bbox": [0, 0, 10, 10]},
            {"label": "car", "confidence": 0.6, "bbox": [10, 0, 20, 10]},
        ],
        0.24,
        250.0,
        (480, 640, 3),
    )

    assert values["objects"]["count"] == 2
    assert values["objects"]["classes"] == {"car": 2}
    assert values["objects"]["confidence_mean"] == 0.7
    assert values["vision"]["risk_score"] == 0.24
    assert values["audio"]["siren_probability"] is None
    assert values["audio"]["doa_degrees"] is None
    assert values["audio"]["doa_status"] == "not_connected"
    assert values["runtime"]["inference_fps"] == 4.0
    assert values["qwen"]["confidence_status"] == "not_provided_by_model"
    assert values["objects"]["items"][0]["bbox_area_fraction"] == pytest.approx(
        100 / (640 * 480)
    )
    assert values["template_plan"]["status"] == "template_only_not_live_inference"
    assert values["mpc_simulation"]["status"] == "simulation_only"
    assert values["mpc_simulation"]["actuators_connected"] is False
