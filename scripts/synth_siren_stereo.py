"""Render stereo sirens at exact known azimuths. Ground truth by construction.

No real siren/emergency audio is downloaded or sampled: every sample here is
synthesised, so there is no licensing question and the angle label is exact.
"""
import argparse
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from src.core.audio.spatial_geometry import woodworth_tau  # noqa: E402

SR = 16000
FPS = 30
WAIL_HZ = 2.2
CORE_HARMONICS = (325.0, 430.0, 650.0)
OPTIONAL_HARMONICS = (975.0, 1300.0)
ROAD_PARTIALS = (90.0, 180.0, 360.0, 720.0, 1440.0)
FAR_EAR_ATTEN = 0.45
FRAME_SAMPLES = SR // FPS


def wail_envelope(t):
    return 0.6 + 0.4 * np.sin(2.0 * np.pi * WAIL_HZ * t)


def harmonic_stack(t, harmonics):
    out = np.zeros_like(t)
    for k, f in enumerate(harmonics, start=1):
        out += np.sin(2.0 * np.pi * f * t) / k
    return out


def fractional_delay(x, delay_samples):
    """Delay x by a possibly fractional number of samples, exactly."""
    n = len(x)
    spec = np.fft.rfft(x)
    k = np.arange(len(spec))
    return np.fft.irfft(spec * np.exp(-2j * np.pi * k * delay_samples / n), n)


def render_ears(mono, theta_deg):
    """theta > 0 means the source is on the RIGHT and the RIGHT ear LEADS.

    Both ears are the same source separated by the Woodworth ITD, with a
    broadband far-ear attenuation standing in for the head shadow.
    """
    tau = woodworth_tau(math.radians(theta_deg))
    half = 0.5 * tau * SR
    left = fractional_delay(mono, half)
    right = fractional_delay(mono, -half)
    far = 1.0 - FAR_EAR_ATTEN * abs(math.sin(math.radians(theta_deg)))
    if theta_deg < 0.0:
        left, right = left, right * far
    else:
        left, right = left * far, right
    return left, right


def _noise_sample(n, kind, rng):
    if kind == "white":
        return rng.standard_normal(n)
    t = np.arange(n) / SR
    sample = np.zeros(n)
    for f in ROAD_PARTIALS:
        sample += np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi))
    return sample


def add_noise(left, right, snr_db, kind, rng):
    # Independent draw per ear: identical noise in both channels is common-mode,
    # which peaks at zero lag and biases the estimate toward centre.
    if snr_db is None or kind == "none":
        return left, right
    n = len(left)
    rms = float(np.sqrt(np.mean(left**2))) or 1e-12
    target = rms * (10.0 ** (-snr_db / 20.0))
    out = []
    for channel in (left, right):
        sample = _noise_sample(n, kind, rng)
        scale = target / (float(np.sqrt(np.mean(sample**2))) or 1e-12)
        out.append(channel + scale * sample)
    return out[0], out[1]


def write_wav(path, left, right):
    stacked = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stacked))) or 1.0
    pcm = np.round(stacked / peak * 0.95 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as h:
        h.setnchannels(2)
        h.setsampwidth(2)
        h.setframerate(SR)
        h.writeframes(pcm.tobytes())


def render_static(out, angles, harmonics, snr_db, noise_kind, rng):
    made = []
    for theta in angles:
        n = int(SR * 3.0)
        t = np.arange(n) / SR
        mono = harmonic_stack(t, harmonics) * wail_envelope(t)
        mono /= float(np.max(np.abs(mono))) or 1.0
        left, right = render_ears(mono, float(theta))
        left, right = add_noise(left, right, snr_db, noise_kind, rng)
        name = f"siren_theta{int(theta):03d}"
        write_wav(out / f"{name}.wav", left, right)
        meta = {
            "clip": name,
            "kind": "static",
            "gt_dir_deg": float(theta),
            "harmonics": list(harmonics),
            "snr_db": snr_db,
            "noise_kind": noise_kind,
            "sr": SR,
            "duration_s": round(n / SR, 6),
            "model_version": "1.0",
        }
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        made.append(meta)
    return made


def render_track(out, track_id, thetas_deg, harmonics, snr_db, noise_kind, rng):
    n_frames = len(thetas_deg)
    left = np.zeros(n_frames * FRAME_SAMPLES)
    right = np.zeros(n_frames * FRAME_SAMPLES)
    block_t = np.arange(FRAME_SAMPLES) / SR
    mono_block = harmonic_stack(block_t, harmonics) * wail_envelope(block_t)
    for i, theta in enumerate(thetas_deg):
        l, r = render_ears(mono_block, float(theta))
        left[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] = l
        right[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] = r
    left, right = add_noise(left, right, snr_db, noise_kind, rng)
    write_wav(out / f"{track_id}.wav", left, right)
    rates = np.abs(np.diff(thetas_deg)) * FPS
    meta = {
        "clip": track_id,
        "kind": "track",
        "fps": FPS,
        "n_frames": n_frames,
        "duration_s": round(n_frames / FPS, 6),
        "gt_dir_deg_per_frame": [round(float(x), 6) for x in thetas_deg],
        "gt_dir_deg_start": round(float(thetas_deg[0]), 6),
        "gt_dir_deg_end": round(float(thetas_deg[-1]), 6),
        "peak_abs_rate_deg_s": round(float(rates.max()) if rates.size else 0.0, 4),
        "harmonics": list(harmonics),
        "snr_db": snr_db,
        "noise_kind": noise_kind,
        "sr": SR,
        "model_version": "1.0",
    }
    (out / f"{track_id}.json").write_text(json.dumps(meta, indent=2))
    return meta


def track_definitions(duration_s):
    n = int(duration_s * FPS)
    t = (np.arange(n) + 0.5) / FPS
    pass_fast = np.degrees(np.arctan2(2.5, 12.0 * (t - duration_s / 2.0)))
    # Mirror of pass_fast: same geometry on the LEFT (theta < 0), i.e. the LHD
    # twin of the right-side (RHD) pass; peak angular rate is identical by symmetry.
    pass_fast_left = -pass_fast
    fail_slow = -75.0 + 10.0 * t
    return {
        "track_static_left": np.full(n, -45.0),
        "track_static_right": np.full(n, 45.0),
        "track_pass_fast": pass_fast,
        "track_pass_fast_left": pass_fast_left,
        "track_fail_slow": fail_slow,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exports/synthetic/siren")
    ap.add_argument("--sweep-step", type=int, default=15)
    ap.add_argument("--duration-s", type=float, default=15.0)
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--noise-kind", default="none", choices=("none", "white", "road"))
    ap.add_argument("--bright-partials", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    harmonics = CORE_HARMONICS + (OPTIONAL_HARMONICS if args.bright_partials else ())
    if max(harmonics) > 1300.0:
        raise SystemExit(f"harmonic cap exceeded: {max(harmonics)} Hz > 1300 Hz")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    angles = list(range(-90, 91, args.sweep_step))
    static = render_static(out, angles, harmonics, args.snr_db, args.noise_kind, rng)
    tracks = {
        tid: render_track(out, tid, th, harmonics, args.snr_db, args.noise_kind, rng)
        for tid, th in track_definitions(args.duration_s).items()
    }

    index = {
        "model_version": "1.0",
        "sr": SR,
        "fps": FPS,
        "harmonics": list(harmonics),
        "snr_db": args.snr_db,
        "noise_kind": args.noise_kind,
        "static": [{"clip": m["clip"], "gt_dir_deg": m["gt_dir_deg"]} for m in static],
        "tracks": {tid: {"clip": m["clip"], "n_frames": m["n_frames"],
                         "peak_abs_rate_deg_s": m["peak_abs_rate_deg_s"]}
                   for tid, m in tracks.items()},
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"SIREN_SYNTH_OK {len(static)} static + {len(tracks)} motion tracks -> {out}")


if __name__ == "__main__":
    main()
