"""Coherence-gated GCC-PHAT azimuth estimator (numpy only).

The band taper, the 257-tap magnitude gate and the +-17 lag window are fixed by
the measurements recorded in .omo/evidence/doa-math-grounding.md. Retuning them
regresses the recorded baselines (ideal 0.65 deg max / 0.27 mean / 0 ambiguous;
white noise 0 ambiguous).

Geometry comes exclusively from spatial_geometry: a radius mismatch between the
renderer and this estimator would scale the error by (r_est/r_syn - 1) * theta,
about 2.7 deg at 90 deg, which alone would break the 2.0 deg gate.
"""
import numpy as np

from src.core.audio.spatial_geometry import MAX_ITD_SECONDS, woodworth_angle

MAX_LAG = 17
MAG_TAPS = 257
BAND_LOW_IN = 150.0
BAND_LOW_FULL = 350.0
BAND_HIGH_FULL = 770.0
BAND_HIGH_OUT = 1070.0
CENTER_DEADZONE_DEG = 10.0


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _band_taper(freqs):
    low = _smoothstep((freqs - BAND_LOW_IN) / (BAND_LOW_FULL - BAND_LOW_IN))
    high = _smoothstep((BAND_HIGH_OUT - freqs) / (BAND_HIGH_OUT - BAND_HIGH_FULL))
    return low * high


def _moving_average(x, taps):
    if taps <= 1:
        return np.asarray(x, dtype=np.float64)
    pad = taps // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), pad, mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded)))
    return (cumulative[taps:] - cumulative[:-taps]) / taps


def _direction(dir_deg):
    if dir_deg < -CENTER_DEADZONE_DEG:
        return "left"
    if dir_deg > CENTER_DEADZONE_DEG:
        return "right"
    return "center"


def _result(dir_deg, confidence, ambiguous, direction):
    return {"dir_deg": float(dir_deg), "confidence": float(confidence),
            "ambiguous": bool(ambiguous), "direction": direction}


def estimate_doa(left, right, sr=16000):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"channel length mismatch: {left.shape} vs {right.shape}")
    if left.size == 0 or not np.any(left) or not np.any(right):
        return _result(0.0, 0.0, True, "center")

    window = left.size
    nfft = 1 << int(np.ceil(np.log2(max(2 * window, 2))))
    cross = np.fft.rfft(left, nfft) * np.conj(np.fft.rfft(right, nfft))
    magnitude = np.abs(cross)
    phat = cross / (magnitude + 1e-12)

    smoothed = _moving_average(magnitude, MAG_TAPS)
    gate = np.clip(smoothed / (smoothed.max() + 1e-12), 0.0, 1.0) ** 2
    band = _band_taper(np.fft.rfftfreq(nfft, 1.0 / sr))
    weight = phat * gate * band

    cc = np.abs(np.fft.irfft(weight, nfft))
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    window_cc = cc[lags % nfft]
    peak_at = int(np.argmax(window_cc))

    peak = float(window_cc[peak_at])
    mean_cc = float(window_cc.mean())
    confidence = min(peak / (peak + mean_cc + 1e-12), 1.0)

    offset = 0.0
    if 0 < peak_at < window_cc.size - 1:
        left_v, centre_v, right_v = window_cc[peak_at - 1:peak_at + 2]
        denominator = left_v - 2.0 * centre_v + right_v
        if denominator != 0.0:
            offset = float(np.clip(0.5 * (left_v - right_v) / denominator, -1.0, 1.0))

    lag = float(lags[peak_at] + offset)
    # MAX_LAG (17) searches wider than the geometric limit (MAX_ITD, 10.49
    # samples), so woodworth_angle clamps out-of-range peaks to +-90. Discarding
    # them instead fabricated a 90 deg error at every true +-90 deg source.
    # Sign contract: with cross = FFT(left) * conj(FFT(right)), a source on the
    # RIGHT (right ear leading, per todo 11) yields a POSITIVE peak lag, which is
    # the same sign woodworth_angle expects. Verified against all 13 fixtures.
    dir_deg = woodworth_angle(lag / sr)
    return _result(dir_deg, confidence, False, _direction(dir_deg))
