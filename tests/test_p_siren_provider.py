"""Tests for p_siren wiring: real per-frame measurement into the live pipeline.

todo: ship real p_siren (mean cosine 0.507) into the right position — the
bridge seeds event_trigger.details.p_siren from P_sirenProvider instead of the
hardwired 0.0.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.audio.p_siren_provider import P_sirenProvider  # noqa: E402
from src.core.bridge import GraphBridge  # noqa: E402

REAL_REPORT = ROOT / "outputs" / "p_siren_report.json"


def make_report(tmp_path, frame_cosines, pooled=0.5, frame_p_siren=None):
    report = {
        "real_sirens": {
            "G44-01-Sirens": {
                "cosine_vs_synth_centroid": pooled,
                "frame_cosine_mean": float(sum(frame_cosines) / len(frame_cosines)),
                "frame_cosine_std": 0.1,
                "n_frames": len(frame_cosines),
                "device_used": "cpu",
                "frame_cosines": [round(float(v), 4) for v in frame_cosines],
            }
        }
    }
    if frame_p_siren is not None:
        report["real_sirens"]["G44-01-Sirens"]["frame_p_siren"] = [round(float(v), 4) for v in frame_p_siren]
    path = tmp_path / "p_siren_report.json"
    path.write_text(json.dumps(report))
    return path


class TestP_sirenProvider:
    def test_unknown_clip_returns_zero(self, tmp_path):
        path = make_report(tmp_path, [0.5, 0.6, 0.4])
        p = P_sirenProvider(report_path=path)
        assert p.frame_value("C01", 0) == 0.0

    def test_missing_report_returns_zero(self, tmp_path):
        p = P_sirenProvider(report_path=tmp_path / "nope.json")
        assert p.frame_value("G44-01-Sirens", 0) == 0.0

    def test_real_clip_frame_lookup(self, tmp_path):
        cosines = [0.42, 0.55, 0.61, 0.38]
        path = make_report(tmp_path, cosines)
        p = P_sirenProvider(report_path=path)
        assert p.frame_value("G44-01-Sirens", 0) == pytest.approx(0.42)
        assert p.frame_value("G44-01-Sirens", 2) == pytest.approx(0.61)
        assert p.frame_value("G44-01-Sirens", 1) == pytest.approx(0.55)

    def test_out_of_range_frame_falls_back_to_pooled(self, tmp_path):
        path = make_report(tmp_path, [0.42, 0.55, 0.61, 0.38], pooled=0.5)
        p = P_sirenProvider(report_path=path)
        assert p.frame_value("G44-01-Sirens", 999) == pytest.approx(0.5)

    def test_frame_p_siren_preferred_over_cosine(self, tmp_path):
        path = make_report(tmp_path, [0.42, 0.55, 0.61, 0.38],
                           frame_p_siren=[0.99, 0.97, 0.98, 0.96])
        p = P_sirenProvider(report_path=path)
        assert p.frame_value("G44-01-Sirens", 0) == pytest.approx(0.99)
        assert p.frame_value("G44-01-Sirens", 3) == pytest.approx(0.96)

    def test_real_report_has_nonzero_signal(self):
        if not REAL_REPORT.exists():
            pytest.skip("p_siren_report.json not generated")
        report = json.loads(REAL_REPORT.read_text())
        real = report.get("real_sirens", {})
        assert len(real) >= 6, "expected 6 real siren clips scored"
        for clip, entry in real.items():
            assert entry["cosine_vs_synth_centroid"] > 0.0
            assert len(entry["frame_cosines"]) == entry["n_frames"]
        mean = sum(v["cosine_vs_synth_centroid"] for v in real.values()) / len(real)
        assert mean == pytest.approx(0.507, abs=0.02)

    def test_real_report_has_calibrated_head(self):
        if not REAL_REPORT.exists():
            pytest.skip("p_siren_report.json not generated")
        report = json.loads(REAL_REPORT.read_text())
        head = report.get("head", {})
        assert head.get("leave_one_clip_out_per_clip_accuracy")
        assert head["accuracy"] > 0.7, f"LOCO acc {head['accuracy']} should beat cosine coin-flip"
        assert head.get("operating_point_threshold") == 0.5
        for entry in report["real_sirens"].values():
            assert len(entry.get("frame_p_siren", [])) == entry["n_frames"]


class TestBridgeP_siren:
    def test_real_clip_seeds_measured_p_siren(self, tmp_path):
        cosines = [0.42, 0.55, 0.61, 0.38]
        report = make_report(tmp_path, cosines)
        b = GraphBridge(p_siren_report=report)
        s = b.invoke_frame(1, 33.33, clip_id="G44-01-Sirens")
        d = s["event_trigger"]["details"]
        assert d["p_siren"] == pytest.approx(0.55)
        assert d["audio_confidence"] == pytest.approx(0.55)
        assert s["audio"]["siren"]["confidence"] == pytest.approx(0.55)

    def test_unknown_clip_keeps_zero(self, tmp_path):
        report = make_report(tmp_path, [0.42, 0.55])
        b = GraphBridge(p_siren_report=report)
        s = b.invoke_frame(0, 0.0, clip_id="C01")
        assert s["event_trigger"]["details"]["p_siren"] == 0.0