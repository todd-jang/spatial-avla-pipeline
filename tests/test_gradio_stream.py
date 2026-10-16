import io
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.app import demo_support, gradio_app, video_stream  # noqa: E402


class FakeCapture:
    def __init__(self, path):
        self.frames = [np.zeros((540, 960, 3), dtype=np.uint8) for _ in range(6)]
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == video_stream.cv2.CAP_PROP_FPS:
            return 10.0
        if prop == video_stream.cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def grab(self):
        if self.index >= len(self.frames):
            return False
        self.index += 1
        return True

    def release(self):
        self.released = True


class FakeBridge:
    def __init__(self):
        self.calls = []

    def invoke_frame_stateful(self, frame_idx, **kwargs):
        self.calls.append(frame_idx)
        return {"meta_exec": {"lat_ms": 2.0}, "event_trigger": {"details": {}}}


def test_upload_renders_every_frame_but_runs_cpu_analysis_at_selected_rate(monkeypatch):
    capture = FakeCapture("video.mp4")
    bridge = FakeBridge()
    analyzed = []
    monkeypatch.setattr(video_stream.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(video_stream.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(video_stream.cv2, "putText", lambda image, *a, **kw: image)

    frames = list(video_stream.stream_uploaded_video(
        "video.mp4",
        bridge,
        lambda frame, frame_idx, result: np.zeros((1080, 1920, 3), dtype=np.uint8),
        analysis_fps=5,
        analyze_frame=lambda frame, frame_idx, video_time: (
            analyzed.append((frame_idx, video_time)) or {"custom_inference": True}
        ),
    ))

    assert len(frames) == 6
    assert all(frame.shape == (540, 960, 3) for frame in frames)
    assert bridge.calls == [0, 2, 4]
    assert analyzed == [(0, 0.0), (2, 0.2), (4, 0.4)]
    assert capture.released is True


def test_upload_releases_capture_when_stream_is_cancelled(monkeypatch):
    capture = FakeCapture("video.mp4")
    monkeypatch.setattr(video_stream.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(video_stream.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(video_stream.cv2, "putText", lambda image, *a, **kw: image)

    stream = video_stream.stream_uploaded_video(
        "video.mp4",
        FakeBridge(),
        lambda frame, frame_idx, result: np.zeros((1080, 1920, 3), dtype=np.uint8),
        analysis_fps=5,
    )
    next(stream)
    stream.close()

    assert capture.released is True


def test_upload_respects_maximum_realtime_duration(monkeypatch):
    capture = FakeCapture("video.mp4")
    bridge = FakeBridge()
    monkeypatch.setattr(video_stream.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(video_stream.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(video_stream.cv2, "putText", lambda image, *a, **kw: image)

    frames = list(video_stream.stream_uploaded_video(
        "video.mp4",
        bridge,
        lambda frame, frame_idx, result: np.zeros((1080, 1920, 3), dtype=np.uint8),
        analysis_fps=10,
        max_duration_seconds=0.3,
    ))

    assert len(frames) == 3
    assert bridge.calls == [0, 1, 2]
    assert capture.released is True


def test_upload_duration_cannot_exceed_five_minutes(monkeypatch):
    capture = FakeCapture("video.mp4")
    monkeypatch.setattr(video_stream.cv2, "VideoCapture", lambda path: capture)

    stream = video_stream.stream_uploaded_video(
        "video.mp4",
        FakeBridge(),
        lambda frame, frame_idx, result: np.zeros((1080, 1920, 3), dtype=np.uint8),
        max_duration_seconds=301,
    )
    try:
        next(stream)
    except ValueError as exc:
        assert "between 0 and 300 seconds" in str(exc)
    else:
        raise AssertionError("durations above five minutes must be rejected")


def test_custom_upload_defaults_to_five_minutes(monkeypatch):
    calls = []
    soundtrack = (16000, np.zeros(4, dtype=np.float32))

    def fake_stream_upload(*args, **kwargs):
        calls.append((args, kwargs))
        yield np.zeros((540, 960, 3), dtype=np.uint8)

    monkeypatch.setattr(gradio_app, "stream_upload", fake_stream_upload)
    monkeypatch.setattr(
        gradio_app,
        "uploaded_video_audio",
        lambda path, duration: soundtrack,
    )

    frames = list(gradio_app.run(
        gradio_app.CUSTOM_UPLOAD, "video.mp4", 0, 10, upload_duration_minutes=5
    ))

    assert calls[0][0] == ("video.mp4",)
    assert {
        key: value for key, value in calls[0][1].items() if key != "analyze_frame"
    } == {
        "max_frames": 0,
        "analysis_fps": 10,
        "max_duration_seconds": 300,
    }
    assert callable(calls[0][1]["analyze_frame"])
    assert len(frames) == 1
    assert frames[0][1][0] == soundtrack[0]
    assert np.array_equal(frames[0][1][1], soundtrack[1])


def test_upload_selection_automatically_selects_custom_probe():
    assert gradio_app.select_upload_preset("video.mp4") == gradio_app.CUSTOM_UPLOAD
    assert gradio_app.select_upload_preset(None) == gradio_app.FULL_SEQUENCE


def test_uploaded_video_audio_decodes_original_audio(monkeypatch):
    buffer = io.BytesIO()
    samples = np.array([1000, -1000, 2000, -2000], dtype="<i2")
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return gradio_app.subprocess.CompletedProcess(
                command, 0, stdout="0\n", stderr=""
            )
        return gradio_app.subprocess.CompletedProcess(
            command, 0, stdout=buffer.getvalue(), stderr=b""
        )

    monkeypatch.setattr(gradio_app.subprocess, "run", fake_run)

    sample_rate, audio = gradio_app.uploaded_video_audio("video.mp4", 60)

    assert sample_rate == 16000
    assert np.allclose(audio, samples.astype(np.float32) / 32768.0)
    assert calls[1][calls[1].index("-t") + 1] == "60"
    assert calls[1][calls[1].index("-ar") + 1] == "16000"
    assert calls[1][calls[1].index("-ac") + 1] == "2"


def test_uploaded_frame_analyzer_runs_real_modalities_and_reports_limits(monkeypatch):
    class FakeDetector:
        def analyze(self, frame):
            assert frame.shape == (24, 32, 3)
            return ([{
                "label": "car",
                "confidence": 0.9,
                "bbox": [1.0, 2.0, 20.0, 18.0],
            }], 0.27)

    class FakeAudioScorer:
        def __init__(self):
            self.windows = []

        def score_waveform(self, samples, sample_rate):
            self.windows.append((samples.copy(), sample_rate))
            return 0.1

    class FakeQwenAudio:
        def update(self, window_index, samples, sample_rate):
            return {
                "status": "ready",
                "source": "qwen_audio",
                "text": f"window {window_index}",
                "window_index": window_index,
            }

    monkeypatch.setattr(gradio_app, "live_detector", FakeDetector())
    audio_scorer = FakeAudioScorer()
    monkeypatch.setattr(gradio_app, "upload_audio_scorer", audio_scorer)
    monkeypatch.setattr(gradio_app, "upload_qwen_audio", FakeQwenAudio())
    audio = (16000, np.zeros(16000 * 8, dtype=np.float32))
    analyze = gradio_app.uploaded_frame_analyzer(audio)

    states = [
        analyze(np.zeros((24, 32, 3), dtype=np.uint8), idx * 30, float(idx))
        for idx in range(5)
    ]

    assert states[-1]["percept"]["boxes"][0]["label"] == "car"
    details = states[-1]["event_trigger"]["details"]
    assert details["vision_anomaly_score"] == 0.27
    assert details["siren_score_mae"] == 0.1
    assert details["siren_detected_mae"] is True
    assert details["siren_consecutive_windows"] == 5
    assert states[-1]["meta_exec"]["score_status"] == {
        "rtdetr": "available",
        "mae": "model_unavailable",
        "siren": "not_connected",
        "siren_template": "available",
        "trigger": "not_fused_for_custom_upload",
    }
    assert states[-1]["vlm_payload"] == {
        "llm_source": "qwen_audio",
        "llm_action_text": "window 4",
        "status": "ready",
        "audio_window_index": 4,
    }
    assert len(audio_scorer.windows) == 5


def test_custom_upload_renderer_preserves_inference_results(monkeypatch):
    rendered_text = []
    original_put_text = gradio_app.process_multi_view_frame.__globals__["cv2"].putText

    def capture_text(image, text, *args, **kwargs):
        rendered_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(
        gradio_app.process_multi_view_frame.__globals__["cv2"], "putText", capture_text
    )
    result = gradio_app.render_uploaded_video_frame(
        np.zeros((540, 960, 3), dtype=np.uint8),
        0,
        {
            "event_trigger": {
                "details": {
                    "vision_anomaly_score": 0.27,
                    "vision_anomaly_ma": 0.27,
                    "siren_score_mae": 0.1,
                    "siren_score_template": 0.1,
                    "siren_detected_mae": True,
                },
                "hysteresis": {"state": "not_fused"},
            },
            "percept": {
                "boxes": [{
                    "label": "car",
                    "confidence": 0.9,
                    "bbox": [10, 20, 200, 300],
                }],
                "has_lead": True,
            },
            "audio": {"siren": {"detected": True, "type": "ast_template"}},
            "meta_exec": {
                "lat_ms": 10.0,
                "custom_inference": {"rtdetr": "live"},
                "score_status": {
                    "rtdetr": "available",
                    "mae": "model_unavailable",
                    "siren": "not_connected",
                    "siren_template": "available",
                    "trigger": "not_fused_for_custom_upload",
                },
            },
            "vlm_payload": {"llm_source": "stub", "llm_action_text": "stub"},
            "plan": {},
        },
    )

    assert result.shape == (1080, 1920, 3)
    assert "[UPLOAD] Live RT-DETR" in rendered_text
    assert "[Vision] RT-DETR:0.270 MA5:0.27" in rendered_text
    assert "[Siren] p:n/a AST:0.1000 MAE:yes HYST:NOT_FUSED TRIG:n/a" in rendered_text
    assert any("fusion not calibrated" in text for text in rendered_text)


def test_verified_preset_forwards_audio_source(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gradio_app,
        "demo_audio",
        lambda choice, audio_source, max_frames: None,
    )
    monkeypatch.setattr(
        gradio_app,
        "stream_segment",
        lambda segment, max_frames, audio_source: (
            calls.append((segment["clip_id"], max_frames, audio_source)) or iter(())
        ),
    )

    list(gradio_app.run(
        gradio_app.SEG_ITEMS[0], None, 0, 10, 5, audio_source="untouched"
    ))

    expected_clip = gradio_app.SEG_MAP[gradio_app.SEG_ITEMS[0]]["clip_id"]
    assert calls == [(expected_clip, 0, "untouched")]


def test_demo_audio_uses_selected_video_audio_and_preserves_source_duration(monkeypatch):
    segment = gradio_app.SEG_MAP[gradio_app.SEG_ITEMS[0]]
    audio_path = Path("/audio/untouched.wav")
    audio = np.zeros((30 * 20, 2), dtype=np.float64)
    monkeypatch.setattr(
        gradio_app, "template_audio_path",
        lambda selected_segment, source: (
            audio_path if selected_segment is segment and source == "untouched" else None
        ),
    )
    monkeypatch.setattr(gradio_app, "read_wav", lambda path: (audio, 30))

    sample_rate, playback = gradio_app.demo_audio(
        gradio_app.SEG_ITEMS[0], "untouched", 0
    )

    assert sample_rate == 30
    assert playback.shape == audio.shape
    assert np.array_equal(playback, audio)

    _, limited_playback = gradio_app.demo_audio(
        gradio_app.SEG_ITEMS[0], "untouched", 5
    )
    assert np.array_equal(limited_playback, audio[:5])


def test_ast_audio_source_resolves_requested_variant_without_fallback(
    tmp_path, monkeypatch
):
    clip_dir = tmp_path / "exports" / "TL_A" / "train" / "C01"
    audio_dir = clip_dir / "audio"
    audio_dir.mkdir(parents=True)
    siren_path = audio_dir / "siren.wav"
    untouched_path = audio_dir / "untouched.wav"
    siren_path.touch()
    untouched_path.touch()
    (clip_dir / "clip_meta.json").write_text(
        '{"audio":{"with_siren":"audio/siren.wav",'
        '"untouched":"audio/untouched.wav"}}'
    )
    monkeypatch.setattr(demo_support, "ROOT", tmp_path)
    segment = {"clip_id": "C01", "has_video": True}

    assert demo_support.template_audio_path(segment, "with_siren") == siren_path
    assert demo_support.template_audio_path(segment, "untouched") == untouched_path

    (clip_dir / "clip_meta.json").write_text('{"audio":{"with_siren":null}}')
    assert demo_support.template_audio_path(segment, "with_siren") is None
