"""Tests for the verified demo sequence infrastructure.

Covers the three contracts the demo scripts depend on:

1. outputs/demo_sequence.json schema (scripts/build_demo_sequence.py output)
   -- the manifest the renderers play, nothing is hand-picked.
2. assert_segment() -- the render-time guard that a played clip REPRODUCES its
   manifest-encoded expectation (must_be_quiet / must_fire_within_window).
3. invoke_frame_stateful() -- the state-threading bridge call that keeps S_B's
   hysteresis FSM alive across frames. The regression this locks: the old
   stateless invoke_frame cold-starts the FSM every frame, which turns every
   render into a per-frame crossing counter and can never reproduce the hold
   the manifest was derived from.

The gradio app itself is import-skipped when gradio is absent (it is a demo
dependency, not a pipeline dependency).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.app.demo_support import (  # noqa: E402
    assert_segment,
    load_manifest,
    segments_by_manifest,
)
from src.core.bridge import GraphBridge  # noqa: E402

MANIFEST = ROOT / "outputs" / "demo_sequence.json"
REPORT = ROOT / "outputs" / "p_siren_report.json"


class TestManifestSchema:
    @pytest.fixture(autouse=True)
    def manifest(self):
        if not MANIFEST.exists():
            pytest.skip("demo_sequence.json not generated")
        yield load_manifest(MANIFEST)

    def test_schema_marker(self, manifest):
        assert manifest["schema"] == "demo-sequence/v1"
        assert manifest["engine_params"]["siren_raw_threshold"] == 0.8
        assert manifest["engine_params"]["min_hold_frames"] == 35

    def test_segments_ordered_and_complete(self, manifest):
        segs = segments_by_manifest(manifest)
        assert segs, "manifest must carry at least one segment"
        assert [s["order"] for s in segs] == sorted(s["order"] for s in segs)
        for s in segs:
            assert {k: s.get(k) for k in
                    ("clip_id", "group", "expected", "has_video",
                     "demo_n_frames", "active_frames", "active_rate",
                     "first_raw_trigger_frame", "mean_p", "assert")}
            assert s["expected"] in ("quiet", "trigger")
            assert s["demo_n_frames"] > 0
            ar = s["assert"]
            assert set(ar) >= {"must_be_quiet", "must_fire_within_window"}

    def test_quiet_segments_have_zero_measured_active(self, manifest):
        """A quiet manifest entry must come from measured zero active frames."""
        for s in segments_by_manifest(manifest):
            if s["assert"]["must_be_quiet"]:
                assert s["active_frames"] == 0, s["clip_id"]
                assert s["first_raw_trigger_frame"] == -1, s["clip_id"]

    def test_trigger_segments_fired_within_their_window(self, manifest):
        """A trigger manifest entry must have measured a fire inside the window."""
        for s in segments_by_manifest(manifest):
            if s["assert"]["must_fire_within_window"]:
                assert s["first_raw_trigger_frame"] >= 0, s["clip_id"]
                assert s["first_raw_trigger_frame"] < s["demo_n_frames"], s["clip_id"]
                assert s["active_frames"] > 0, s["clip_id"]


class TestAssertSegment:
    def _seg(self, **over):
        base = {"clip_id": "X", "expected": "quiet", "demo_n_frames": 100,
                "assert": {"must_be_quiet": True, "must_fire_within_window": False}}
        base.update(over)
        return base

    def test_quiet_pass(self):
        ok, msg = assert_segment(self._seg(), {"active_frames": 0,
                                               "first_raw_trigger_frame": -1})
        assert ok and "PASS" in msg

    def test_quiet_fail_when_fires(self):
        ok, msg = assert_segment(self._seg(), {"active_frames": 3,
                                               "first_raw_trigger_frame": 10})
        assert not ok and "QUIET" in msg and "3/100" in msg

    def test_trigger_pass_with_hold(self):
        seg = self._seg(expected="trigger", **{
            "assert": {"must_be_quiet": False, "must_fire_within_window": True}})
        ok, msg = assert_segment(seg, {"active_frames": 60,
                                       "first_raw_trigger_frame": 2})
        assert ok and "PASS" in msg

    def test_trigger_fail_never_fired(self):
        seg = self._seg(expected="trigger", **{
            "assert": {"must_be_quiet": False, "must_fire_within_window": True}})
        ok, msg = assert_segment(seg, {"active_frames": 0,
                                       "first_raw_trigger_frame": -1})
        assert not ok and "never fired" in msg

    def test_trigger_fail_raw_but_zero_hold(self):
        """raw fired but hysteresis held nothing -> not a real trigger."""
        seg = self._seg(expected="trigger", **{
            "assert": {"must_be_quiet": False, "must_fire_within_window": True}})
        ok, msg = assert_segment(seg, {"active_frames": 0,
                                       "first_raw_trigger_frame": 5})
        assert not ok and "zero active" in msg

    def test_unknown_expected_rejected(self):
        ok, msg = assert_segment(self._seg(expected="maybe"),
                                 {"active_frames": 0, "first_raw_trigger_frame": -1})
        assert not ok and "unknown expected" in msg


class TestStatefulHysteresis:
    NF = 450

    def _counts(self, stateful):
        if not REPORT.exists():
            pytest.skip("p_siren_report.json not generated")
        bridge = GraphBridge(fast_mode=True)
        prev, active, first_raw = None, 0, -1
        for i in range(self.NF):
            if stateful:
                res = bridge.invoke_frame_stateful(i, i * (1000.0 / 30),
                                                   clip_id="C01", prev_state=prev)
                prev = res
            else:
                res = bridge.invoke_frame(i, i * (1000.0 / 30), clip_id="C01")
            ev = res.get("event_trigger", {}) or {}
            if ev.get("raw_trigger") and first_raw < 0:
                first_raw = i
            if ev.get("is_triggered"):
                active += 1
        return active, first_raw

    def test_stateful_holds_longer_than_stateless(self):
        stateful_active, _ = self._counts(stateful=True)
        stateless_active, _ = self._counts(stateful=False)
        # The regression: stateless cold-starts the FSM every frame, so it can
        # only count raw-crossing frames. Stateful threading carries the hold.
        assert stateful_active > stateless_active, (
            f"stateful={stateful_active} stateless={stateless_active}: "
            f"hysteresis must extend, never shrink")

    def test_stateful_reproduces_manifest_crossing(self):
        """C01's manifest-excluded profile (83 held frames) must reproduce."""
        if not REPORT.exists():
            pytest.skip("p_siren_report.json not generated")
        manifest = json.loads(MANIFEST.read_text())
        c01 = next(s for s in manifest["excluded"] if s["clip_id"] == "C01")
        stateful_active, first_raw = self._counts(stateful=True)
        expected_hold = c01["active_frames"]
        assert stateful_active == expected_hold, (
            f"stateful={stateful_active} manifest={expected_hold}: renderer and "
            f"sequence builder must count the same frames")
        assert first_raw == c01["first_raw_trigger_frame"]


class TestGradioImport:
    def test_app_builds_when_gradio_present(self):
        gradio = pytest.importorskip("gradio")
        import src.app.gradio_app as app  # noqa: F401

        assert app.FULL_SEQUENCE in app.PRESET_CHOICES
        app.build()