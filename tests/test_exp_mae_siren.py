import numpy as np

from src.exp_a import mae_siren


def test_mae_siren_score_uses_explicit_audio_path(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio_siren_stereo_16k.wav"
    audio_path.touch()
    read_paths = []
    feature_inputs = []

    def fake_read(path, always_2d):
        read_paths.append((path, always_2d))
        return np.zeros((8, 2)), 16000

    monkeypatch.setattr(mae_siren.sf, "read", fake_read)
    monkeypatch.setattr(
        mae_siren,
        "_ast_feature",
        lambda wav, sr: feature_inputs.append((wav, sr)) or "audio-feature",
    )
    monkeypatch.setattr(
        mae_siren, "_score_from_feature", lambda feature: 0.42
    )
    monkeypatch.setattr(
        mae_siren,
        "_load_clip_audio",
        lambda clip_id, key: _unexpected_clip_lookup(),
    )

    score = mae_siren.mae_siren_score({
        "clip_id": "C01",
        "audio_path": str(audio_path),
    })

    assert score == 0.42
    assert read_paths == [(str(audio_path), True)]
    assert len(feature_inputs) == 1
    assert feature_inputs[0][1] == 16000


def test_mae_siren_score_missing_explicit_path_returns_none(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        mae_siren,
        "_load_clip_audio",
        lambda clip_id, key: _unexpected_clip_lookup(),
    )

    assert mae_siren.mae_siren_score({
        "clip_id": "C01",
        "audio_path": str(tmp_path / "missing.wav"),
    }) is None


def test_mae_siren_score_falls_back_to_with_siren_then_untouched(
    tmp_path, monkeypatch
):
    audio_path = tmp_path / "with_siren.wav"
    audio_path.touch()
    looked_up = []
    monkeypatch.setattr(
        mae_siren,
        "_load_clip_audio",
        lambda clip_id, key: looked_up.append((clip_id, key)) or (
            audio_path if key == "with_siren" else None
        ),
    )
    monkeypatch.setattr(
        mae_siren.sf,
        "read",
        lambda path, always_2d: (np.zeros((8, 2)), 16000),
    )
    monkeypatch.setattr(mae_siren, "_ast_feature", lambda wav, sr: "feature")
    monkeypatch.setattr(mae_siren, "_score_from_feature", lambda feature: 0.5)

    assert mae_siren.mae_siren_score({"clip_id": "C01"}) == 0.5
    assert looked_up == [("C01", "with_siren")]


def test_mae_siren_score_uses_untouched_when_with_siren_is_unavailable(
    tmp_path, monkeypatch
):
    untouched_path = tmp_path / "untouched.wav"
    untouched_path.touch()
    looked_up = []
    monkeypatch.setattr(
        mae_siren,
        "_load_clip_audio",
        lambda clip_id, key: looked_up.append((clip_id, key)) or (
            untouched_path if key == "untouched" else None
        ),
    )
    monkeypatch.setattr(
        mae_siren.sf,
        "read",
        lambda path, always_2d: (np.zeros((8, 2)), 16000),
    )
    monkeypatch.setattr(mae_siren, "_ast_feature", lambda wav, sr: "feature")
    monkeypatch.setattr(mae_siren, "_score_from_feature", lambda feature: 0.25)

    assert mae_siren.mae_siren_score({"clip_id": "C01"}) == 0.25
    assert looked_up == [("C01", "with_siren"), ("C01", "untouched")]


def _unexpected_clip_lookup():
    raise AssertionError("explicit audio_path must not fall back to clip audio")
