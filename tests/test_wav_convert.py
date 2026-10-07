"""Tests for src/core/audio/wav_convert.py: 24-bit/48 kHz ingest -> 16k int16.

Real-world siren recordings arrive in 48 kHz 24-bit; the pipeline loaders
refuse anything that is not 16 kHz int16 by design. These tests lock the
one-time ingest conversion so a downstream consumer never sees a raw format.
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.audio import wav_convert as wc  # noqa: E402
from src.core.audio.wav_io import load_wav_mono_16k, load_wav_stereo_16k  # noqa: E402


def _write_wav(path: Path, samples: np.ndarray, rate: int, width: int,
               channels: int = 1) -> Path:
    """Write int16/24-bit PCM WAV (24-bit stored little-endian 3 bytes)."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        if width == 2:
            pcm = np.clip(samples, -1, 1) * 32767
            payload = pcm.astype("<i2").tobytes()
        elif width == 3:
            pcm = np.clip(samples, -1, 1) * (1 << 23)
            i = pcm.round().astype(np.int32)
            u = (i & 0xFFFFFF).astype(np.uint32)
            payload = np.stack([u & 0xFF, (u >> 8) & 0xFF, u >> 16],
                               axis=-1).astype(np.uint8).tobytes()
        else:
            raise ValueError(width)
        handle.writeframes(payload)
    return path


def _sine(freq: float, rate: int, dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * dur)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_24bit_48k_converts_to_16k_int16(tmp_path):
    src = _write_wav(tmp_path / "s.wav", _sine(1200.0, 48000, 1.0), 48000, 3)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = wc.convert_to_pipeline(src, out_dir)
    assert out == out_dir / "s_16k.wav"
    a = load_wav_mono_16k(out)
    assert np.asarray(a).shape == (16000,)
    assert abs(float(np.abs(a).max()) - 0.5) < 0.02


def test_16k_16bit_passthrough_stays_16k(tmp_path):
    src = _write_wav(tmp_path / "s.wav", _sine(440.0, 16000, 0.5), 16000, 2)
    out = wc.convert_to_pipeline(src, tmp_path / "out.wav")
    a = load_wav_mono_16k(out)
    assert np.asarray(a).shape == (8000,)


def test_stereo_downmixed_to_mono(tmp_path):
    left = _sine(300.0, 16000, 0.5).astype(np.float32)
    right = np.zeros_like(left)
    stereo = np.stack([left, right], axis=1)
    src = _write_wav(tmp_path / "s.wav", stereo, 16000, 2, channels=2)
    out = wc.convert_to_pipeline(src, tmp_path / "out.wav")
    a = np.asarray(load_wav_mono_16k(out))
    assert a.shape == (8000,)
    # Downmix of (0.5, 0) is 0.25; conversion halves again via low energy
    assert abs(float(np.abs(a).max()) - 0.25) < 0.02


def test_high_frequency_energy_does_not_alias_into_8k_plus(tmp_path):
    # A 15 kHz tone at 48 kHz would alias badly under naive interpolation;
    # the anti-alias low-pass must keep the 8-16k band quiet after resample.
    src = _write_wav(tmp_path / "s.wav", _sine(15000.0, 48000, 1.0, 0.9), 48000, 3)
    a = np.asarray(load_wav_mono_16k(wc.convert_to_pipeline(src, tmp_path / "out.wav")))
    win = np.hanning(len(a))
    spec = np.abs(np.fft.rfft(a * win)) ** 2
    freqs = np.fft.rfftfreq(len(a), 1 / 16000)
    above_8k = spec[freqs > 8000].sum()
    total = spec.sum()
    assert above_8k / total < 0.05


def test_48000_to_16000_duration_ratio(tmp_path):
    src = _write_wav(tmp_path / "s.wav", _sine(800.0, 48000, 2.0), 48000, 3)
    out = wc.convert_to_pipeline(src, tmp_path / "out.wav")
    with wave.open(str(out)) as h:
        frames = h.getnframes()
        rate = h.getframerate()
    assert rate == 16000
    assert abs(frames / rate - 2.0) < 2 / 16000


def test_stereo_silent_right_keeps_left_pure(tmp_path):
    src = _write_wav(tmp_path / "s.wav", _sine(600.0, 16000, 0.5), 16000, 2)
    mono, rate = wc.read_wav_float_mono(src)
    wc.write_wav_int16_stereo_16k(tmp_path / "out.wav", mono)
    with wave.open(str(tmp_path / "out.wav")) as h:
        assert h.getnchannels() == 2
        assert h.getsampwidth() == 2
        assert h.getframerate() == 16000
        frames = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").reshape(-1, 2)
    assert abs(frames[:, 0].max() / 32767.0 - 0.5) < 0.02
    assert (frames[:, 1] == 0).all()


def test_stereo_silent_right_loads_through_strict_loader(tmp_path):
    src = _write_wav(tmp_path / "s.wav", _sine(600.0, 16000, 0.5), 16000, 2)
    mono, _ = wc.read_wav_float_mono(src)
    wc.write_wav_int16_stereo_16k(tmp_path / "out.wav", mono)
    stereo = np.asarray(load_wav_stereo_16k(tmp_path / "out.wav"))
    assert stereo.shape[1] == 2
    assert (stereo[:, 1] == 0).all()