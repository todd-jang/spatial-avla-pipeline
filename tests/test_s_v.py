import pytest

from src.core.agents import s_v


class FakeMaeScores:
    available = True

    def frame_value(self, clip_id, frame_idx):
        return 2.5 if (clip_id, frame_idx) == ("C01", 4) else None


class FakePSiren:
    def has_frame_value(self, clip_id, frame_idx):
        return True

    def frame_value(self, clip_id, frame_idx):
        return 0.8


class FakeMaeVisionSiren:
    available = True

    def score_clip(self, clip_id):
        return 0.15 if clip_id == "C01" else None

    def score(self, state):
        return self.score_clip(state.get("clip_id"))


def test_s_v_returns_frame_scores_and_temporal_state(monkeypatch):
    monkeypatch.setattr(s_v, "_MAE_SCORES", FakeMaeScores())
    monkeypatch.setattr(s_v, "_MAE_SIREN", FakeMaeVisionSiren())
    monkeypatch.setattr(s_v, "_P_SIREN", FakePSiren())
    monkeypatch.setattr(s_v, "has_detections", lambda clip_id: True)
    monkeypatch.setattr(s_v, "anomaly_score", lambda state: 0.3)
    state = {
        "clip_id": "C01",
        "frame_idx": 4,
        "event_trigger": {"details": {"existing": "preserved"}},
        "_hist": {"mae_list": [2.0], "siren_mae_list": [0.6]},
    }

    result = s_v.s_v_fn(state)
    details = result["event_trigger"]["details"]

    assert state["event_trigger"]["details"]["existing"] == "preserved"
    assert details["existing"] == "preserved"
    assert details["vision_anomaly_score"] == 0.3
    assert details["vision_anomaly_mae"] == 2.5
    assert details["p_siren"] == 0.8
    assert details["siren_score_mae"] == details["siren_score_template"] == 0.15
    assert details["siren_detected_mae"] is True
    assert details["siren_strength_mae"] == "weak"
    assert result["meta_exec"]["score_status"]["siren_template"] == "available"
    assert details["vision_anomaly_mae_ma"] == 2.25
    assert details["p_siren_ma5"] == 0.7
    assert details["p_siren_delta"] == pytest.approx(0.2)
    assert result["_hist"]["mae_list"] == [2.0, 2.5]
    assert result["_hist"]["siren_mae_list"] == [0.6, 0.8]
    assert state["_hist"]["mae_list"] == [2.0]


def test_s_v_reports_missing_mae_frame_without_zero_fallback(monkeypatch):
    monkeypatch.setattr(s_v, "_MAE_SCORES", FakeMaeScores())
    monkeypatch.setattr(s_v, "_MAE_SIREN", FakeMaeVisionSiren())
    monkeypatch.setattr(s_v, "_P_SIREN", FakePSiren())
    monkeypatch.setattr(s_v, "has_detections", lambda clip_id: False)
    monkeypatch.setattr(s_v, "anomaly_score", lambda state: 0.3)

    result = s_v.s_v_fn({"clip_id": "C02", "frame_idx": 9})

    details = result["event_trigger"]["details"]
    assert details["vision_anomaly_mae"] is None
    assert details["vision_anomaly_score"] is None
    assert result["meta_exec"]["score_status"]["mae"] == "frame_unavailable"
    assert result["meta_exec"]["score_status"]["rtdetr"] == "detections_unavailable"
    assert details["siren_score_mae"] is None
    assert details["siren_score_template"] is None
    assert details["siren_detected_mae"] is False
    assert result["meta_exec"]["score_status"]["siren_template"] == "audio_unavailable"


@pytest.mark.parametrize(
    ("mode", "score", "expected"),
    [
        ("balanced", -0.20, False),
        ("balanced", -0.19, True),
        ("conservative", -0.10, False),
        ("conservative", -0.09, True),
        ("sensitive", -0.30, False),
        ("sensitive", -0.29, True),
    ],
)
def test_siren_template_threshold_modes(monkeypatch, mode, score, expected):
    class FixedMaeVisionSiren:
        available = True

        def score_clip(self, clip_id):
            return score

        def score(self, state):
            return self.score_clip(state.get("clip_id"))

    monkeypatch.setattr(s_v, "_MAE_SIREN", FixedMaeVisionSiren())
    monkeypatch.setattr(s_v, "_MAE_SCORES", FakeMaeScores())
    monkeypatch.setattr(s_v, "_P_SIREN", FakePSiren())
    monkeypatch.setattr(s_v, "has_detections", lambda clip_id: False)
    monkeypatch.setattr(s_v, "MAE_SIREN_THRESHOLD_MODE", mode)

    result = s_v.s_v_fn({"clip_id": "C01", "frame_idx": 4})

    details = result["event_trigger"]["details"]
    assert details["siren_score_mae"] == score
    assert details["siren_score_template"] == score
    assert details["siren_detected_mae"] is expected
    assert details["siren_threshold_mode"] == mode


def test_s_v_uses_frame_threshold_for_explicit_audio_path(monkeypatch):
    class FixedMaeVisionSiren(FakeMaeVisionSiren):
        def score(self, state):
            return -0.10

    monkeypatch.setattr(s_v, "_MAE_SIREN", FixedMaeVisionSiren())
    monkeypatch.setattr(s_v, "_MAE_SCORES", FakeMaeScores())
    monkeypatch.setattr(s_v, "_P_SIREN", FakePSiren())
    monkeypatch.setattr(s_v, "has_detections", lambda clip_id: False)

    result = s_v.s_v_fn({
        "clip_id": "C01",
        "frame_idx": 4,
        "audio_path": "/tmp/frame-window.wav",
    })

    details = result["event_trigger"]["details"]
    assert details["siren_score_mae"] == pytest.approx(-0.10)
    assert details["siren_detected_mae"] is False
    assert details["siren_threshold_scope"] == "frame"
    assert details["siren_threshold"] == pytest.approx(-0.04)


def test_t09_false_positives_do_not_pass_five_window_streak(monkeypatch):
    class FixedWindowSiren(FakeMaeVisionSiren):
        def score_frame(self, state, **kwargs):
            return T09_TWO_SECOND_SCORES[state["frame_idx"] // 30]

    monkeypatch.setattr(s_v, "_MAE_SIREN", FixedWindowSiren())
    monkeypatch.setattr(s_v, "_MAE_SCORES", FakeMaeScores())
    monkeypatch.setattr(s_v, "_P_SIREN", FakePSiren())
    monkeypatch.setattr(s_v, "has_detections", lambda clip_id: False)

    state = {"clip_id": "T09", "_hist": {}}
    for window_idx in range(len(T09_TWO_SECOND_SCORES)):
        state["frame_idx"] = window_idx * 30
        result = s_v.s_v_fn(state)
        assert result["event_trigger"]["details"]["siren_detected_mae"] is False
        state["_hist"] = result["_hist"]


def test_five_consecutive_siren_windows_trigger_after_warmup(monkeypatch):
    class FixedWindowSiren(FakeMaeVisionSiren):
        def score_frame(self, state, **kwargs):
            return 0.05

    monkeypatch.setattr(s_v, "_MAE_SIREN", FixedWindowSiren())
    monkeypatch.setattr(s_v, "_MAE_SCORES", FakeMaeScores())
    monkeypatch.setattr(s_v, "_P_SIREN", FakePSiren())
    monkeypatch.setattr(s_v, "has_detections", lambda clip_id: False)

    state = {"clip_id": "C01", "_hist": {}}
    for window_idx in range(s_v.SIREN_MIN_CONSECUTIVE_WINDOWS):
        state["frame_idx"] = window_idx * 30
        result = s_v.s_v_fn(state)
        detected = result["event_trigger"]["details"]["siren_detected_mae"]
        assert detected is (window_idx == 4)
        state["_hist"] = result["_hist"]


T09_TWO_SECOND_SCORES = [
    -0.16726547, -0.09093526, -0.05892760, -0.00608599,
    -0.01643461, -0.04052514, -0.09327370, -0.03578597,
    0.03551275, 0.04055399, 0.01156592, -0.07458514,
    -0.09140086, -0.08704579,
]
