import json

import numpy as np
import pytest
import torch

from src.core import mae_vision
from src.core.mae_vision import (
    AST_MODEL_NAME,
    MaeVisionSirenScorer,
    save_template_file,
    siren_detected,
)


def _templates(path):
    save_template_file(
        path,
        AST_MODEL_NAME,
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 1.0])],
        ["G44-01-Sirens_16k.wav"],
        ["C01:audio/clean.wav"],
    )


def test_scorer_uses_clip_audio_and_caches_full_recording_feature(
    tmp_path, monkeypatch
):
    clips_root = tmp_path / "clips"
    clip_dir = clips_root / "train" / "C01"
    audio_dir = clip_dir / "audio"
    audio_dir.mkdir(parents=True)
    wav_path = audio_dir / "mix.wav"
    untouched_path = audio_dir / "clean.wav"
    wav_path.touch()
    untouched_path.touch()
    (clip_dir / "clip_meta.json").write_text(json.dumps({
        "clip_id": "C01",
        "audio": {
            "with_siren": "audio/mix.wav",
            "untouched": "audio/clean.wav",
        },
    }))
    template_path = tmp_path / "templates.pt"
    _templates(template_path)
    scorer = MaeVisionSirenScorer(template_path, clips_root, device="cpu")
    inferred_paths = []

    def fake_feature(model, path, device):
        inferred_paths.append((path, device))
        return torch.tensor([1.0, 0.0])

    monkeypatch.setattr(scorer, "_get_model", lambda: object())
    monkeypatch.setattr(mae_vision, "ast_feature_from_file", fake_feature)

    first_score = scorer.score_clip("C01")
    second_score = scorer.score({"clip_id": "C01", "frame_idx": 150})

    assert first_score == pytest.approx(1.0)
    assert second_score == pytest.approx(1.0)
    assert inferred_paths == [(untouched_path, "cpu")]


def test_scorer_aligns_two_second_windows_to_one_second_hops(
    tmp_path, monkeypatch
):
    clips_root = tmp_path / "clips"
    clip_dir = clips_root / "train" / "C01"
    audio_dir = clip_dir / "audio"
    audio_dir.mkdir(parents=True)
    audio_path = audio_dir / "clean.wav"
    audio_path.touch()
    (clip_dir / "clip_meta.json").write_text(json.dumps({
        "clip_id": "C01",
        "audio": {"untouched": "audio/clean.wav"},
    }))
    template_path = tmp_path / "templates.pt"
    _templates(template_path)
    scorer = MaeVisionSirenScorer(template_path, clips_root, device="cpu")
    reads = []

    import soundfile

    monkeypatch.setattr(
        soundfile,
        "info",
        lambda path: type("AudioInfo", (), {"samplerate": 16000})(),
    )
    monkeypatch.setattr(
        soundfile,
        "read",
        lambda path, start, frames, always_2d: (
            reads.append((path, start, frames))
            or (np.zeros((frames, 1), dtype=np.float32), 16000)
        ),
    )
    monkeypatch.setattr(scorer, "_get_model", lambda: object())
    monkeypatch.setattr(
        mae_vision,
        "ast_feature_from_waveform",
        lambda model, waveform, sample_rate, device: torch.tensor([1.0, 0.0]),
    )

    assert scorer.score_frame({"clip_id": "C01", "frame_idx": 0}) == pytest.approx(1.0)
    assert scorer.score_frame({"clip_id": "C01", "frame_idx": 29}) == pytest.approx(1.0)
    assert scorer.score_frame({"clip_id": "C01", "frame_idx": 30}) == pytest.approx(1.0)
    assert [(start, frames) for _, start, frames in reads] == [
        (0, 32000),
        (16000, 32000),
    ]
    assert scorer.score_frame({
        "audio_path": str(audio_path),
        "frame_idx": 60,
    }) == pytest.approx(1.0)
    assert [(start, frames) for _, start, frames in reads] == [
        (0, 32000),
        (16000, 32000),
        (32000, 32000),
    ]


def test_scorer_prefers_explicit_audio_path(tmp_path, monkeypatch):
    audio_path = tmp_path / "uploaded.wav"
    audio_path.touch()
    template_path = tmp_path / "templates.pt"
    _templates(template_path)
    scorer = MaeVisionSirenScorer(template_path, tmp_path / "clips", device="cpu")
    monkeypatch.setattr(scorer, "_get_model", lambda: object())
    monkeypatch.setattr(
        mae_vision,
        "ast_feature_from_file",
        lambda model, path, device: torch.tensor([0.0, 1.0]),
    )

    assert scorer.score({"audio_path": str(audio_path)}) == pytest.approx(-1.0)


def test_missing_explicit_audio_path_does_not_fall_back_to_clip(tmp_path):
    clips_root = tmp_path / "clips"
    clip_dir = clips_root / "train" / "C01"
    audio_dir = clip_dir / "audio"
    audio_dir.mkdir(parents=True)
    untouched_path = audio_dir / "clean.wav"
    untouched_path.touch()
    (clip_dir / "clip_meta.json").write_text(json.dumps({
        "clip_id": "C01",
        "audio": {"untouched": "audio/clean.wav"},
    }))
    template_path = tmp_path / "templates.pt"
    _templates(template_path)
    scorer = MaeVisionSirenScorer(template_path, clips_root, device="cpu")

    assert scorer.score({
        "clip_id": "C01",
        "audio_path": str(tmp_path / "missing.wav"),
    }) is None


def test_scorer_is_unavailable_without_precomputed_templates(tmp_path):
    scorer = MaeVisionSirenScorer(tmp_path / "missing.pt", tmp_path / "clips")

    assert scorer.available is False
    assert scorer.score_clip("C01") is None


@pytest.mark.parametrize(
    ("mode", "score", "expected"),
    [
        ("balanced", -0.20, False),
        ("balanced", -0.19, True),
        ("conservative", -0.10, False),
        ("conservative", -0.09, True),
        ("sensitive", -0.30, False),
        ("sensitive", -0.29, True),
        ("balanced", None, False),
    ],
)
def test_siren_threshold_modes(mode, score, expected):
    assert siren_detected(score, mode) is expected


def test_siren_threshold_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown siren threshold mode"):
        siren_detected(0.0, "unknown")


def test_siren_thresholds_are_separate_for_frame_and_clip_scores():
    assert siren_detected(-0.10, "balanced", scope="clip") is True
    assert siren_detected(-0.10, "balanced", scope="frame") is False
    assert siren_detected(-0.03, "balanced", scope="frame") is True
