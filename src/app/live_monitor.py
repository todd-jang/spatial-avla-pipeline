"""Read-only live camera detection and microphone level helpers."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from src.app.mpc_simulation import compute_fixed_demo_mpc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "rtdetr-l.pt"
DEFAULT_RISK_WEIGHTS_PATH = ROOT / "configs" / "vision_risk_weights.json"
MIN_CONFIDENCE = 0.25

# Keep Ultralytics settings inside the workspace instead of its user-home default.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".gradio"))


def vision_risk_score(detections: list[dict], weights: dict) -> float:
    """Apply the pipeline's configured class risk weights to live detections."""
    classes = weights["classes"]
    default_weight = float(weights["default_weight"])
    min_confidence = float(weights.get("min_conf", MIN_CONFIDENCE))
    scores = []
    for detection in detections:
        confidence = float(detection["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Detection confidence must be finite and in [0, 1]")
        if confidence < min_confidence:
            continue
        weight = float(classes.get(detection["label"], default_weight))
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("Vision risk weights must be finite and in [0, 1]")
        scores.append(weight * confidence)
    return round(max(scores, default=0.0), 4)


class LiveObjectDetector:
    """Lazy-loaded RT-DETR inference using the workspace's local checkpoint."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        risk_weights_path: str | Path = DEFAULT_RISK_WEIGHTS_PATH,
    ):
        self.model_path = Path(model_path)
        self.risk_weights_path = Path(risk_weights_path)
        self._model = None
        self._device = None
        self._weights = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"RT-DETR checkpoint not found: {self.model_path}")
        if not self.risk_weights_path.is_file():
            raise FileNotFoundError(
                f"Vision risk weights not found: {self.risk_weights_path}"
            )

        try:
            import torch
            from ultralytics import RTDETR
        except ImportError as exc:
            raise RuntimeError(
                "Live camera detection requires the project's torch and "
                "ultralytics dependencies"
            ) from exc

        if torch.cuda.is_available():
            self._device = "cuda"
        elif (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            self._device = "mps"
        else:
            self._device = "cpu"

        self._weights = json.loads(self.risk_weights_path.read_text())
        if not isinstance(self._weights.get("classes"), dict):
            raise ValueError(f"Invalid vision risk weights: {self.risk_weights_path}")
        self._model = RTDETR(str(self.model_path))
        self._model.to(self._device)

    def analyze(self, rgb_frame: np.ndarray) -> tuple[list[dict], float]:
        if (
            not isinstance(rgb_frame, np.ndarray)
            or rgb_frame.ndim != 3
            or rgb_frame.shape[2] != 3
            or rgb_frame.size == 0
        ):
            raise ValueError("Expected a non-empty RGB camera frame")
        self._load()
        results = self._model.predict(
            source=rgb_frame,
            imgsz=640,
            conf=float(self._weights.get("min_conf", MIN_CONFIDENCE)),
            verbose=False,
            device=self._device,
        )
        if not results:
            raise RuntimeError("RT-DETR returned no result for the live frame")

        result = results[0]
        boxes = result.boxes
        detections = []
        if boxes is not None:
            names = result.names
            for coords, class_id, confidence in zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
            ):
                label = names[int(class_id)]
                detections.append(
                    {
                        "label": str(label),
                        "confidence": float(confidence),
                        "bbox": [float(value) for value in coords[:4]],
                    }
                )
        return detections, vision_risk_score(detections, self._weights)


def render_live_frame(
    rgb_frame: np.ndarray,
    detections: list[dict],
    risk_score: float,
    frame_idx: int,
    latency_ms: float,
    pipeline_triggered: bool,
) -> np.ndarray:
    """Draw actual camera detections and read-only pipeline status."""
    image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    scale = min(1.0, 1280.0 / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    out_h, out_w = image.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = (
            int(round(float(value) * scale)) for value in detection["bbox"]
        )
        x1, x2 = sorted((max(0, min(out_w - 1, x1)), max(0, min(out_w - 1, x2))))
        y1, y2 = sorted((max(0, min(out_h - 1, y1)), max(0, min(out_h - 1, y2))))
        color = (40, 40, 230) if "person" in detection["label"].lower() else (255, 160, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{detection['label']} {float(detection['confidence']):.2f}",
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    header = (
        f"READ-ONLY | FRAME {frame_idx} | vision risk {risk_score:.3f} | "
        f"pipeline trigger {'ON' if pipeline_triggered else 'off'} | "
        f"{latency_ms:.1f} ms"
    )
    cv2.rectangle(image, (0, 0), (out_w, 38), (20, 20, 20), -1)
    cv2.putText(
        image, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (0, 220, 255) if pipeline_triggered else (230, 230, 230), 2,
    )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def live_metrics(graph_state: dict, detections: list[dict], risk_score: float,
                 inference_latency_ms: float, frame_shape: tuple[int, ...]) -> dict:
    """Return numeric live perception values without inventing missing scores."""
    confidences = [float(item["confidence"]) for item in detections]
    labels: dict[str, int] = {}
    for item in detections:
        label = str(item["label"])
        labels[label] = labels.get(label, 0) + 1

    ev = graph_state.get("event_trigger") or {}
    details = ev.get("details") or {}
    doa = graph_state.get("doa") or (graph_state.get("audio") or {}).get("doa") or {}
    meta = graph_state.get("meta_exec") or {}
    audio = graph_state.get("audio") or {}
    siren = audio.get("siren") or {}
    siren_status = (meta.get("score_status") or {}).get("siren", "not_connected")
    latency = float(inference_latency_ms)
    mpc = graph_state.get("mpc_simulation") or compute_fixed_demo_mpc()
    doa_confidence = doa.get("confidence")
    doa_gt = doa.get("gt_dir_deg")
    doa_available = bool(
        (doa_confidence is not None and float(doa_confidence) > 0.0)
        or (doa_gt is not None and float(doa_gt) != -1.0)
    )
    return {
        "frame": int(graph_state.get("frame_idx", 0)),
        "image": {
            "width_px": int(frame_shape[1]),
            "height_px": int(frame_shape[0]),
        },
        "objects": {
            "count": len(detections),
            "classes": labels,
            "confidence_max": max(confidences) if confidences else None,
            "confidence_mean": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "items": [
                {
                    "label": str(item["label"]),
                    "confidence": float(item["confidence"]),
                    "bbox_xyxy_px": [float(value) for value in item["bbox"]],
                    "bbox_area_fraction": (
                        max(0.0, float(item["bbox"][2]) - float(item["bbox"][0]))
                        * max(0.0, float(item["bbox"][3]) - float(item["bbox"][1]))
                        / (int(frame_shape[1]) * int(frame_shape[0]))
                    ),
                }
                for item in detections
            ],
        },
        "vision": {
            "risk_score": float(risk_score),
            "risk_ma5": details.get("vision_anomaly_ma"),
            "mae_distance": details.get("vision_anomaly_mae"),
            "mae_ma5": details.get("vision_anomaly_mae_ma"),
        },
        "audio": {
            "siren_probability": (
                None
                if siren_status == "not_connected"
                else siren.get("confidence", details.get("p_siren"))
            ),
            "siren_classifier_status": siren_status,
            "doa_degrees": doa.get("dir_deg") if doa_available else None,
            "doa_confidence": doa_confidence if doa_available else None,
            "doa_ground_truth_degrees": doa_gt if doa_available else None,
            "doa_rate_degrees_per_second": (
                details.get("doa_change_rate_dps_ma") if doa_available else None
            ),
            "doa_status": "available" if doa_available else "not_connected",
        },
        "trigger": {
            "raw": bool(ev.get("raw_trigger", False)),
            "active": bool(ev.get("is_triggered", False)),
            "hysteresis_state": (ev.get("hysteresis") or {}).get("state"),
            "reasons": ev.get("trigger_reason", []),
            "confidence": None,
            "confidence_status": "not_provided_by_trigger_model",
        },
        "mpc_simulation": mpc,
        "template_plan": {
            "steer_degrees": (graph_state.get("plan") or {}).get("steer_deg"),
            "speed_kmh": (graph_state.get("plan") or {}).get("speed_kmh"),
            "status": "template_only_not_live_inference",
        },
        "runtime": {
            "inference_latency_ms": latency,
            "inference_fps": 1000.0 / latency if latency > 0 else None,
            "graph_latency_ms": meta.get("lat_ms"),
        },
        "qwen": {
            "advisory": (graph_state.get("vlm_payload") or {}).get(
                "llm_action_text"
            ),
            "source": (graph_state.get("vlm_payload") or {}).get("llm_source"),
            "confidence": (graph_state.get("vlm_payload") or {}).get("confidence"),
            "confidence_status": "not_provided_by_model",
        },
    }


def microphone_metrics(audio) -> dict:
    """Quantify microphone chunks without claiming siren classification."""
    if audio is None:
        return {
            "status": "waiting_for_audio",
            "rms": None,
            "peak": None,
            "rms_dbfs": None,
            "peak_dbfs": None,
            "sample_rate_hz": None,
            "channels": None,
            "duration_seconds": None,
            "siren_classifier": "not_connected",
        }
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        raise ValueError("Expected a (sample_rate, audio_samples) microphone chunk")
    sample_rate, samples = audio
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("Microphone sample rate must be positive")
    values = np.asarray(samples)
    if values.size == 0:
        return {
            "status": "waiting_for_audio",
            "rms": None,
            "peak": None,
            "rms_dbfs": None,
            "peak_dbfs": None,
            "sample_rate_hz": sample_rate,
            "channels": None,
            "duration_seconds": 0.0,
            "siren_classifier": "not_connected",
        }
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        values = values.astype(np.float64) / max(abs(info.min), abs(info.max))
    else:
        values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Microphone chunk contains non-finite samples")
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values))))
    channel_count = int(values.shape[1]) if values.ndim > 1 else 1
    sample_count = int(values.shape[0])
    return {
        "status": "available",
        "rms": rms,
        "peak": peak,
        "rms_dbfs": 20.0 * math.log10(rms) if rms > 0 else None,
        "peak_dbfs": 20.0 * math.log10(peak) if peak > 0 else None,
        "sample_rate_hz": sample_rate,
        "channels": channel_count,
        "duration_seconds": sample_count / sample_rate,
        "siren_classifier": "not_connected",
    }


def microphone_level(audio) -> str:
    """Format microphone values for text UI; this does not classify sirens."""
    metrics = microphone_metrics(audio)
    if metrics["status"] != "available":
        return "Microphone: waiting for audio"
    return (
        f"Audio level only (no siren classifier): RMS {metrics['rms']:.4f} "
        f"({metrics['rms_dbfs']:.1f} dBFS) | peak {metrics['peak']:.4f} "
        f"({metrics['peak_dbfs']:.1f} dBFS) | "
        f"{metrics['sample_rate_hz']} Hz | {metrics['channels']} ch"
    )
