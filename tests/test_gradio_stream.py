import sys
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
    monkeypatch.setattr(video_stream.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(video_stream.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(video_stream.cv2, "putText", lambda image, *a, **kw: image)

    frames = list(video_stream.stream_uploaded_video(
        "video.mp4",
        bridge,
        lambda frame, frame_idx, result: np.zeros((1080, 1920, 3), dtype=np.uint8),
        analysis_fps=5,
    ))

    assert len(frames) == 6
    assert all(frame.shape == (540, 960, 3) for frame in frames)
    assert bridge.calls == [0, 2, 4]
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

    def fake_stream_upload(*args, **kwargs):
        calls.append((args, kwargs))
        yield np.zeros((540, 960, 3), dtype=np.uint8)

    monkeypatch.setattr(gradio_app, "stream_upload", fake_stream_upload)

    list(gradio_app.run(
        gradio_app.CUSTOM_UPLOAD, "video.mp4", 0, 10, upload_duration_minutes=5
    ))

    assert calls[0][0] == ("video.mp4",)
    assert calls[0][1] == {
        "max_frames": 0,
        "analysis_fps": 10,
        "max_duration_seconds": 300,
    }


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
