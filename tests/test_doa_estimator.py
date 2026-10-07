"""Acceptance tests for todo 12 (coherence-gated GCC-PHAT estimator)."""
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.audio.doa_gccphat import estimate_doa  # noqa: E402
from src.core.audio.spatial_geometry import HEAD_RADIUS, MAX_ITD_SECONDS, woodworth_tau  # noqa: E402

SIREN = ROOT / "exports/synthetic/siren"
ANGLES = list(range(-90, 91, 15))
MAX_ITD_SAMPLES = MAX_ITD_SECONDS * 16000.0


def load(theta):
    w = wave.open(str(SIREN / f"siren_theta{int(theta):03d}.wav"))
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return a.reshape(-1, 2).astype(np.float64) / 32768.0


def test_ideal_sweep_meets_gates():
    errs, ambiguous = [], 0
    for theta in ANGLES:
        a = load(theta)
        r = estimate_doa(a[:, 0], a[:, 1], 16000)
        ambiguous += int(r["ambiguous"])
        errs.append(abs(r["dir_deg"] - float(theta)))
    assert ambiguous == 0
    assert max(errs) <= 2.0
    assert sum(errs) / len(errs) <= 1.0


@pytest.mark.parametrize("theta", ANGLES)
def test_confidence_strictly_below_one(theta):
    a = load(theta)
    r = estimate_doa(a[:, 0], a[:, 1], 16000)
    assert 0.0 <= r["confidence"] < 1.0


def test_worst_case_error_is_sub_degree():
    worst = 0.0
    for theta in ANGLES:
        a = load(theta)
        est = estimate_doa(a[:, 0], a[:, 1], 16000)["dir_deg"]
        worst = max(worst, abs(est - float(theta)))
    assert worst <= 1.0


def _stereo_from_lag(lag_samples, n=16000):
    """Two ears carrying the same tone stack separated by lag_samples."""
    t = np.arange(n) / 16000.0
    mono = sum(np.sin(2.0 * np.pi * f * t) for f in (325.0, 430.0, 650.0))
    spec = np.fft.rfft(mono)
    k = np.arange(len(spec))
    shifted = np.fft.irfft(spec * np.exp(2j * np.pi * k * lag_samples / n), n)
    return mono, shifted


@pytest.mark.parametrize("theta,expected", [(-45, "left"), (45, "right"), (0, "center")])
def test_direction_mapping(theta, expected):
    a = load(theta)
    assert estimate_doa(a[:, 0], a[:, 1], 16000)["direction"] == expected


@pytest.mark.parametrize("deg", [0, 5, 8, -5, -8])
def test_within_deadzone_maps_to_center(deg):
    """The centre deadzone is +-10 deg, so only |dir_deg| <= 10 reports 'center'."""
    lag = woodworth_tau(math.radians(deg)) * 16000.0
    left, right = _stereo_from_lag(lag)
    assert estimate_doa(left, right, 16000)["direction"] == "center"


@pytest.mark.parametrize("deg,expected", [(15, "right"), (-15, "left")])
def test_just_outside_deadzone_does_not_report_center(deg, expected):
    lag = woodworth_tau(math.radians(deg)) * 16000.0
    left, right = _stereo_from_lag(lag)
    assert estimate_doa(left, right, 16000)["direction"] == expected


def test_extreme_90_degrees_is_not_ambiguous():
    for theta in (-90, 90):
        a = load(theta)
        assert estimate_doa(a[:, 0], a[:, 1], 16000)["ambiguous"] is False


def test_lag_beyond_maximum_itd_clamps_to_90():
    """A lag past the +-90 deg limit clamps rather than being discarded."""
    lag = MAX_ITD_SAMPLES + 3.5
    left, right = _stereo_from_lag(lag)
    r = estimate_doa(left, right, 16000)
    assert r["ambiguous"] is False
    assert r["dir_deg"] == pytest.approx(90.0)
    assert 0.0 <= r["confidence"] < 1.0


def test_120_degree_source_clamps_to_90():
    lag = woodworth_tau(math.radians(120.0)) * 16000.0
    assert lag > MAX_ITD_SAMPLES
    left, right = _stereo_from_lag(lag)
    r = estimate_doa(left, right, 16000)
    assert r["ambiguous"] is False
    assert r["dir_deg"] == pytest.approx(90.0)
    assert r["direction"] == "right"


def test_in_range_lag_is_not_ambiguous():
    left, right = _stereo_from_lag(MAX_ITD_SAMPLES - 2.0)
    assert estimate_doa(left, right, 16000)["ambiguous"] is False


def test_all_zero_input_returns_zero_confidence_without_raising():
    zeros = np.zeros(16000)
    r = estimate_doa(zeros, zeros, 16000)
    assert r["confidence"] == 0.0
    assert r["dir_deg"] == 0.0


def test_silence_in_one_channel_returns_zero_confidence():
    tone = np.zeros(16000)
    r = estimate_doa(tone, tone + np.sin(2.0 * np.pi * 325.0 * np.arange(16000) / 16000.0), 16000)
    assert r["confidence"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        estimate_doa(np.zeros(100), np.zeros(200), 16000)


def test_estimator_does_not_redeclare_geometry_constants():
    src = (ROOT / "src/core/audio/doa_gccphat.py").read_text()
    assert str(HEAD_RADIUS) not in src
    assert "HEAD_RADIUS =" not in src
    assert "SPEED_OF_SOUND =" not in src
    assert "def woodworth_tau" not in src
    assert "def woodworth_angle" not in src


def test_estimator_imports_geometry_from_single_source():
    src = (ROOT / "src/core/audio/doa_gccphat.py").read_text()
    assert "from src.core.audio.spatial_geometry import" in src
    assert "MAX_ITD_SECONDS" in src


def test_no_scipy_dependency():
    src = (ROOT / "src/core/audio/doa_gccphat.py").read_text()
    assert "scipy" not in src


def test_estimator_is_deterministic():
    a = load(45)
    first = estimate_doa(a[:, 0], a[:, 1], 16000)
    second = estimate_doa(a[:, 0], a[:, 1], 16000)
    assert first == second


def test_synthetic_fixtures_are_present():
    assert len(sorted(SIREN.glob("siren_theta*.wav"))) == 13
    idx = json.loads((SIREN / "index.json").read_text())
    assert len(idx["static"]) == 13


# Noise is rendered through synth_siren_stereo.py itself; reimplementing it here
# would test the estimator against a signal it was never measured on.
FIXTURE_SECONDS = 3


def _renderer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "synth_siren_stereo", ROOT / "scripts/synth_siren_stereo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _noisy(theta, snr_db, kind, seed):
    mod = _renderer()
    rng = np.random.default_rng(seed)
    n = FIXTURE_SECONDS * mod.SR
    t = np.arange(n) / mod.SR
    mono = mod.harmonic_stack(t, mod.CORE_HARMONICS) * mod.wail_envelope(t)
    left, right = mod.render_ears(mono, float(theta))
    return mod.add_noise(left, right, snr_db, kind, rng)


def _matrix_row(snr_db, kind, seed):
    errs, ambiguous = [], 0
    for i, theta in enumerate(ANGLES):
        left, right = _noisy(theta, snr_db, kind, seed + i)
        r = estimate_doa(left, right, 16000)
        ambiguous += int(r["ambiguous"])
        errs.append(abs(r["dir_deg"] - float(theta)))
    return ambiguous, max(errs), sum(errs) / len(errs)


@pytest.mark.parametrize("snr_db", [20, 15, 10, 5])
def test_white_noise_accuracy_survives(snr_db):
    ambiguous, _, mean = _matrix_row(snr_db, "white", 1000)
    assert ambiguous == 0
    assert mean <= 15.0


@pytest.mark.parametrize("snr_db,mean_cap,max_cap", [
    (20, 7.0, 11.0),
    (10, 11.0, 17.0),
    (5, 14.0, None),
])
def test_road_noise_accuracy_survives(snr_db, mean_cap, max_cap):
    ambiguous, worst, mean = _matrix_row(snr_db, "road", 2000)
    assert ambiguous == 0
    assert mean <= mean_cap
    if max_cap is not None:
        assert worst <= max_cap


def test_matrix_stays_inside_the_tested_angle_range():
    for theta in ANGLES:
        left, right = _noisy(theta, 20, "road", 3000 + ANGLES.index(theta))
        r = estimate_doa(left, right, 16000)
        assert -90.0 <= r["dir_deg"] <= 90.0