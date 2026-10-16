import json

import pytest

from src.core.mae_score_provider import MaeScoreProvider


def test_returns_raw_frame_score(tmp_path):
    report_path = tmp_path / "mae_frame_report.json"
    report_path.write_text(json.dumps({
        "schema_version": 1,
        "clips": {"C01": {"frames": {"12": {"mae_euclidean_raw": 3.25}}}},
    }))

    provider = MaeScoreProvider(report_path)

    assert provider.available is True
    assert provider.frame_value("C01", 12) == 3.25
    assert provider.frame_value("C01", 13) is None
    assert provider.frame_value("unknown", 12) is None


def test_missing_report_is_unavailable(tmp_path):
    provider = MaeScoreProvider(tmp_path / "missing.json")

    assert provider.available is False
    assert provider.frame_value("C01", 0) is None


def test_invalid_report_schema_is_explicit(tmp_path):
    report_path = tmp_path / "bad.json"
    report_path.write_text(json.dumps({"schema_version": 2, "clips": {}}))

    with pytest.raises(ValueError, match="Invalid MAE score report schema"):
        MaeScoreProvider(report_path).available


def test_non_finite_score_is_explicit(tmp_path):
    report_path = tmp_path / "bad_score.json"
    report_path.write_text(
        '{"schema_version":1,"clips":{"C01":{"frames":{"0":'
        '{"mae_euclidean_raw":NaN}}}}}'
    )

    with pytest.raises(ValueError, match="Non-finite MAE score"):
        MaeScoreProvider(report_path).frame_value("C01", 0)
