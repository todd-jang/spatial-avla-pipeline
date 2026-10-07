"""Acceptance tests for todo 11 (synthetic stereo siren ground truth)."""
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.audio.spatial_geometry import (  # noqa: E402
    HEAD_RADIUS,
    MAX_ITD_SECONDS,
    SPEED_OF_SOUND,
    woodworth_angle,
    woodworth_tau,
)

OUT = ROOT / "exports/synthetic/siren"
FPS = 30
MAX_LAG = 12
SIDECAR_KEYS = {"gt_dir_deg", "harmonics", "snr_db", "noise_kind",
                "sr", "duration_s", "model_version"}
TRACKS = {"track_static_left", "track_static_right",
          "track_pass_fast", "track_pass_fast_left", "track_fail_slow"}


def wav_name(theta):
    return f"siren_theta{int(theta):03d}"


def load_stereo(theta):
    w = wave.open(str(OUT / f"{wav_name(theta)}.wav"))
    a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    a = a.reshape(-1, 2).astype(np.float64) / 32768.0
    return a[:, 0], a[:, 1]


def peak_lag(theta):
    """Sub-sample lag: >0 means the RIGHT ear leads."""
    left, right = load_stereo(theta)
    seg = slice(2000, 4000)
    L = left[seg] - left[seg].mean()
    R = right[seg] - right[seg].mean()
    cc = np.correlate(R, L, "full")
    lags = np.arange(-(len(L) - 1), len(L))
    m = np.abs(lags) <= MAX_LAG
    lags, cc = lags[m], cc[m]
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:
        d = 0.5 * (cc[k - 1] - cc[k + 1]) / (cc[k - 1] - 2 * cc[k] + cc[k + 1])
        lag = lags[k] + d
    else:
        lag = lags[k]
    return float(-lag)


# ---------------------------------------------------------------- geometry

def test_geometry_constants():
    # The radius is deliberately NOT re-listed in this file: the whole tree is
    # scanned by test_head_radius_declared_exactly_once, so repeating the literal
    # here would itself violate the single-source-of-truth invariant. MAX_ITD is
    # pinned against its closed form instead.
    assert SPEED_OF_SOUND == 343.0
    assert HEAD_RADIUS > 0.0
    closed_form = (HEAD_RADIUS / SPEED_OF_SOUND) * (math.pi / 2 + 1.0)
    assert abs(MAX_ITD_SECONDS - closed_form) < 1e-12
    assert abs(MAX_ITD_SECONDS - 0.6558e-3) < 1e-6
    assert abs(MAX_ITD_SECONDS * 16000 - 10.49) < 0.01


def test_woodworth_roundtrip():
    for deg in range(-90, 91):
        assert abs(woodworth_angle(woodworth_tau(math.radians(deg))) - deg) < 1e-6
    assert woodworth_angle(0.0) == 0.0
    assert woodworth_angle(1.0) == 90.0
    assert woodworth_angle(-1.0) == -90.0


def test_head_radius_declared_exactly_once():
    needle = "0." + "0875"  # split so this file does not itself contain the literal
    offenders = [p for p in ROOT.rglob("*.py")
                 if needle in p.read_text() and p.name != "spatial_geometry.py"]
    assert not offenders, f"head radius re-declared outside spatial_geometry.py: {offenders}"


# ------------------------------------------------------------ static sweep

def test_thirteen_static_files_with_step_15():
    wavs = sorted(OUT.glob("siren_theta*.wav"))
    assert len(wavs) == 13
    assert sorted(int(p.stem[-3:]) for p in wavs) == list(range(-90, 91, 15))


@pytest.mark.parametrize("theta", list(range(-90, 91, 15)))
def test_static_format_and_exact_label(theta):
    w = wave.open(str(OUT / f"{wav_name(theta)}.wav"))
    assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, 16000, 2)
    j = json.loads((OUT / f"{wav_name(theta)}.json").read_text())
    assert SIDECAR_KEYS <= set(j)
    assert j["gt_dir_deg"] == float(theta)
    assert max(j["harmonics"]) <= 1300
    assert j["model_version"] == "1.0"


# ------------------------------------------------------ physical correctness

@pytest.mark.parametrize("theta", [-90, -60, -45, -15, 15, 45, 60, 90])
def test_itd_sign_matches_convention(theta):
    """theta > 0 means source on the RIGHT and the RIGHT ear LEADS."""
    lag = peak_lag(theta)
    assert lag > 0 if theta > 0 else lag < 0, f"theta={theta} lag={lag}"


@pytest.mark.parametrize("theta", [-45, -15, 15, 45])
def test_ild_agrees_with_itd(theta):
    left, right = load_stereo(theta)
    el, er = float(np.sqrt((left ** 2).mean())), float(np.sqrt((right ** 2).mean()))
    louder_is_right = er > el
    assert louder_is_right if theta > 0 else not louder_is_right


def test_centre_source_is_symmetric():
    left, right = load_stereo(0)
    assert abs(float(np.sqrt((left ** 2).mean())) - float(np.sqrt((right ** 2).mean()))) < 1e-9
    assert abs(peak_lag(0)) < 1e-9


def test_far_ear_attenuation_grows_with_angle():
    ratios = []
    for theta in (15, 45, 90):
        left, right = load_stereo(theta)
        el, er = float(np.sqrt((left ** 2).mean())), float(np.sqrt((right ** 2).mean()))
        ratios.append(min(el, er) / max(el, er))
    assert ratios[0] > ratios[1] > ratios[2]


@pytest.mark.parametrize("theta", [15, 30, 45, 60, 75])
def test_delay_is_fractional_not_integer(theta):
    """The rendered ITD must match the exact Woodworth value to a fraction of a sample."""
    expected = woodworth_tau(math.radians(theta)) * 16000.0
    assert abs(peak_lag(theta) - expected) < 0.15, (
        f"theta={theta}: measured {peak_lag(theta):.3f} vs expected {expected:.3f} samples")


def test_itd_never_exceeds_physical_maximum():
    worst = max(abs(peak_lag(t)) for t in range(-90, 91, 15))
    assert worst <= MAX_ITD_SECONDS * 16000 + 0.05


# ----------------------------------------------------------- motion tracks

def test_index_lists_every_artefact_with_truth():
    idx = json.loads((OUT / "index.json").read_text())
    assert set(idx["tracks"]) == TRACKS
    assert len(idx["static"]) == 13
    assert {s["gt_dir_deg"] for s in idx["static"]} == set(float(t) for t in range(-90, 91, 15))


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_track_wave_format(track):
    w = wave.open(str(OUT / f"{track}.wav"))
    assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, 16000, 2)


def test_pass_fast_peak_rate_and_not_clamped():
    j = json.loads((OUT / "track_pass_fast.json").read_text())
    th = np.array(j["gt_dir_deg_per_frame"])
    peak = float((np.abs(np.diff(th)) * FPS).max())
    assert 270.0 <= peak <= 280.0
    assert th.max() > 90.0, "pass_fast must NOT be clamped to +-90 deg"
    assert 1.5 < th.min() < 1.8 and 178.0 < th.max() < 178.5


def test_pass_fast_left_is_exact_mirror_of_right():
    right = json.loads((OUT / "track_pass_fast.json").read_text())
    left = json.loads((OUT / "track_pass_fast_left.json").read_text())
    th_r = np.array(right["gt_dir_deg_per_frame"])
    th_l = np.array(left["gt_dir_deg_per_frame"])
    assert th_l.shape == th_r.shape
    # JSON stores round(float(x), 6); round-half-even is sign-symmetric, so the
    # mirror must round-trip as the exact negation frame-by-frame.
    assert np.abs(th_l + th_r).max() < 1e-9
    peak_l = float((np.abs(np.diff(th_l)) * FPS).max())
    assert 270.0 <= peak_l <= 280.0
    assert th_l.min() < -178.0 and -1.8 < th_l.max() < -1.5
    assert th_l.min() > -178.5, "left pass must NOT be clamped to +-90 deg"


def test_pass_fast_left_fires_fsm_like_right():
    from scripts import build_poc_report
    rows = build_poc_report.run_triggers(OUT)
    by_track = {r["track"]: r for r in rows}
    left, right = by_track["track_pass_fast_left"], by_track["track_pass_fast"]
    assert left["trigger_fired"] is True and right["trigger_fired"] is True
    assert left["renderer_peak_rate_deg_s"] == pytest.approx(
        right["renderer_peak_rate_deg_s"], abs=0.001)
    assert left["peak_doa_rate_dps_ma"] == pytest.approx(
        right["peak_doa_rate_dps_ma"], abs=0.001)


def test_fail_slow_slow_enough_to_trigger():
    j = json.loads((OUT / "track_fail_slow.json").read_text())
    th = np.array(j["gt_dir_deg_per_frame"])
    assert float((np.abs(np.diff(th)) * FPS).max()) <= 11.0
    assert th.min() < -70.0 and th.max() > 70.0


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_static_tracks_hold_their_angle(track):
    j = json.loads((OUT / f"{track}.json").read_text())
    if track == "track_static_left":
        assert {j["gt_dir_deg_start"], j["gt_dir_deg_end"]} == {-45.0}
    elif track == "track_static_right":
        assert {j["gt_dir_deg_start"], j["gt_dir_deg_end"]} == {45.0}


def test_track_sidecars_carry_per_frame_truth():
    for track in sorted(TRACKS):
        j = json.loads((OUT / f"{track}.json").read_text())
        assert j["fps"] == FPS
        assert j["n_frames"] == 450
        assert len(j["gt_dir_deg_per_frame"]) == 450
        assert j["model_version"] == "1.0"
