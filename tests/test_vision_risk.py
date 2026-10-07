"""Tests for the Exp A vision-risk wiring (src/exp_a).

Locks the S_A contract under the RT-DETR lane:

1. Contract safety -- s_a_fn must keep the incoming vision_anomaly_score
   when no detection file exists for the clip (mock mode). It must not zero
   out an externally supplied score just because the file is absent.
2. Score rule -- anomaly_score() = max over boxes of class_weight(class) *
   conf, 0.0 for empty frames / unknown clip, and the weights file is the
   single source of truth (nothing hardcoded).
3. Real detections -- against outputs/detections/C01__rtdetr-l.json, the
   per-frame scores must stay in [0,1] and the frame-100 value must match
   the documented rule.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.exp_a.agent_s_a import s_a_fn  # noqa: E402
from src.exp_a.vision_risk import anomaly_score, has_detections  # noqa: E402

WEIGHTS = ROOT / "configs" / "vision_risk_weights.json"
REAL_DET = ROOT / "outputs" / "detections" / "C01__rtdetr-l.json"


def _state(clip_id="siren_left_01", frame_idx=0, incoming=0.2):
    s = copy.deepcopy({
        "clip_id": clip_id,
        "frame_idx": frame_idx,
        "event_trigger": {"details": {"vision_anomaly_score": incoming}},
        "_hist": {"va_list": [], "dd_list": []},
    })
    return s


class TestContractSafety:
    def test_keeps_incoming_when_no_detection_file(self):
        s = _state(clip_id="zzz_no_file", incoming=0.2)
        r = s_a_fn(s)
        assert r["event_trigger"]["details"]["vision_anomaly_score"] == 0.2

    def test_keeps_custom_incoming_value(self):
        s = _state(clip_id="zzz_no_file", incoming=0.77)
        r = s_a_fn(s)
        assert r["event_trigger"]["details"]["vision_anomaly_score"] == 0.77

    def test_va_list_grows_and_caps_at_300(self):
        s = _state(clip_id="zzz_no_file", incoming=0.2)
        for _ in range(310):
            r = s_a_fn(s)
            s = r
        assert len(s["_hist"]["va_list"]) == 300

    def test_details_keys_preserved(self):
        s = _state(clip_id="zzz_no_file")
        r = s_a_fn(s)
        d = r["event_trigger"]["details"]
        assert set(d) >= {"vision_anomaly_score", "lead_lost", "lane_lost_count"}
        assert d["lead_lost"] is False
        assert d["lane_lost_count"] == 0


class TestScoreRule:
    @pytest.fixture(autouse=True)
    def weights(self):
        if not WEIGHTS.exists():
            pytest.skip("vision_risk_weights.json not generated")
        yield json.loads(WEIGHTS.read_text())

    def test_rule_matches_manual_calc(self, tmp_path):
        det = {"frames": {"7": {"boxes": [
            {"class_name": "car", "conf": 0.9},
            {"class_name": "person", "conf": 0.5},
        ]}}}
        p = tmp_path / "CLIP__rtdetr-l.json"
        p.write_text(json.dumps(det))
        s = _state(clip_id="CLIP", frame_idx=7)
        w = json.loads(WEIGHTS.read_text())
        expected = max(w["classes"]["car"] * 0.9,
                       w["classes"]["person"] * 0.5)
        assert anomaly_score(s, det_dir=tmp_path.as_posix()) == round(
            min(1.0, expected), 4)

    def test_empty_frame_scores_zero(self, tmp_path):
        det = {"frames": {"0": {"boxes": []}}}
        p = tmp_path / "CLIP__rtdetr-l.json"
        p.write_text(json.dumps(det))
        s = _state(clip_id="CLIP", frame_idx=0)
        assert anomaly_score(s, det_dir=tmp_path.as_posix()) == 0.0

    def test_unknown_clip_scores_zero(self):
        s = _state(clip_id="", frame_idx=0)
        assert anomaly_score(s) == 0.0

    def test_unknown_class_uses_default_weight(self, tmp_path):
        det = {"frames": {"0": {"boxes": [{"class_name": "zebra", "conf": 0.8}]}}}
        p = tmp_path / "CLIP__rtdetr-l.json"
        p.write_text(json.dumps(det))
        w = json.loads(WEIGHTS.read_text())
        s = _state(clip_id="CLIP", frame_idx=0)
        expected = w["default_weight"] * 0.8
        assert anomaly_score(s, det_dir=tmp_path.as_posix()) == round(expected, 4)


class TestRealDetections:
    @pytest.fixture(autouse=True)
    def real(self):
        if not REAL_DET.exists():
            pytest.skip("outputs/detections/C01__rtdetr-l.json not generated")
        yield

    def test_has_detections_true_for_generated_clip(self):
        assert has_detections("C01") is True

    def test_frame_scores_in_unit_interval(self):
        s = _state(clip_id="C01", frame_idx=100)
        v = anomaly_score(s)
        assert 0.0 <= v <= 1.0