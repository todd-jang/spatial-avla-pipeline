"""Convert arbitrary WAV files into the pipeline's on-disk format.

The pipeline's loaders (``wav_io.load_wav_mono_16k`` / ``load_wav_stereo_16k``)
refuse anything that is not 16 kHz int16 PCM, by design: silently resampling at
load time degrades the signal before WavLM ever sees it. Real-world recordings
(CC0 siren libraries, phone captures) arrive in 48 kHz 24-bit or stereo 44.1 kHz,
so they must be converted **once, at ingest**, by this module.

Conversion path: unpack any sane WAV to float32 mono in [-1, 1], apply a
windowed-sinc low-pass at 0.9 * (Nyquist of the *target* 16 kHz), then resample
by linear interpolation onto the 16 kHz grid, and finally requantise to int16.

Resampling is deliberately simple (numpy-only, no scipy/torchaudio): the
project's Python 3.14 environment has neither, and WavLM's own front-end is a
learned 16 kHz representation — a textbook polyphase filter would be invisible
to it. The low-pass exists to keep energy out of the 8–16 kHz band that a naive
interpolator would alias into.

The output is mono. DoA needs stereo, but the DoA path in this repo never
consumes raw recordings: it consumes ``audio_siren_stereo_16k.wav``, which the
slicer renders from a mono source panned to a known bearing. So ingest keeps the
mono master and lets the renderer own stereo.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

PIPELINE_RATE = 16000
MAX_INT24 = 1 << 23


def read_wav_float_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any WAV (8/16/24-bit, mono/stereo, any rate) as float32 mono in [-1, 1].

    Returns ``(mono, sample_rate)``. 24-bit and 8-bit are handled explicitly;
    16-bit is the common path. Multi-channel input is downmixed by mean.
    """
    with wave.open(str(path)) as handle:
        n_ch = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
        n_frames = handle.getnframes()

    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2")
    elif width == 3:
        # Little-endian 24-bit: three bytes per sample, sign in the top bit.
        u = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        samples = u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)
        samples = np.where(samples >= MAX_INT24, samples - (1 << 24), samples)
    elif width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128
    else:
        raise SystemExit(f"UNSUPPORTED_SAMPWIDTH {width} in {path}")

    frames = samples.reshape(n_frames, n_ch)
    mono = frames.astype(np.float32) / (1 << (8 * width - 1))
    return mono.mean(axis=1) if n_ch > 1 else mono[:, 0], rate


def lowpass_taps(cutoff: float, rate: int, n_taps: int = 255) -> np.ndarray:
    """Windowed-sinc low-pass filter taps for the given cutoff (Hz).

    A Hamming window keeps the transition band honest; 255 taps gives ~60 dB
    stopband attenuation, far more than any consumer of a siren recording needs.
    """
    if cutoff >= 0.5 * rate:
        return np.array([1.0])
    half = n_taps // 2
    n = np.arange(-half, half + 1, dtype=np.float64)
    fc = cutoff / rate
    taps = np.sinc(2 * fc * n) * 2 * fc
    taps *= 0.54 - 0.46 * np.cos(2 * np.pi * (n + half) / n_taps)
    return (taps / taps.sum()).astype(np.float32)


def to_pipeline_mono_16k(mono: np.ndarray, rate: int) -> np.ndarray:
    """Resample float32 mono at any ``rate`` to 16 kHz int16 mono.

    Low-passes at 0.9 * 8 kHz to protect the 8–16 kHz band from aliasing, then
    interpolates onto the target grid. An already-16 kHz input is returned
    unchanged (the high frequencies were already gone or never present).
    """
    if rate == PIPELINE_RATE:
        return np.clip(mono, -1.0, 1.0)
    if rate <= 0 or PIPELINE_RATE <= 0:
        raise ValueError(f"invalid rates: {rate} -> {PIPELINE_RATE}")
    cutoff = 0.9 * (PIPELINE_RATE / 2)
    taps = lowpass_taps(min(cutoff, 0.49 * rate), rate)
    if len(taps) > 1:
        mono = np.convolve(mono, taps, mode="same")
    n_out = round(len(mono) * PIPELINE_RATE / rate)
    out = np.interp(
        np.linspace(0.0, len(mono) - 1, num=n_out, dtype=np.float64),
        np.arange(len(mono), dtype=np.float64),
        mono.astype(np.float64),
    )
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def write_wav_int16_mono_16k(path: str | Path, mono: np.ndarray) -> None:
    """Persist float32 mono as 16 kHz int16 PCM (the pipeline format)."""
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(PIPELINE_RATE)
        handle.writeframes(pcm.tobytes())


def write_wav_int16_stereo_16k(path: str | Path, mono: np.ndarray,
                               silent_right: bool = True) -> None:
    """Persist float32 mono as 16 kHz int16 stereo, right channel silenced.

    Precache_wavlm and the DoA path both insist on two channels. For a mono
    source, duplicating the signal into both channels would invent an inter-
    channel time difference of zero and let GCC-PHAT read that silence as a
    centred source. Zeroing the right channel instead makes the "no ITD" fact
    explicit: any channel pair with a silent side carries no bearing estimate.
    """
    left = (np.clip(mono, -1.0, 1.0) * 32767.0).round().astype("<i2")
    right = np.zeros_like(left) if silent_right else left
    frame = np.stack([left, right], axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(PIPELINE_RATE)
        handle.writeframes(frame.tobytes())


def convert_to_pipeline(path: str | Path, dest: str | Path) -> Path:
    """One-shot: read ``path``, write 16 kHz int16 mono WAV to ``dest``.

    ``dest`` may be a directory (file keeps its stem, ``_16k`` suffix added) or
    a full target path. Returns the written path.
    """
    src = Path(path)
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True) if d.is_dir() or not d.suffix else None
    target = d / f"{src.stem}_16k.wav" if d.is_dir() or not d.suffix else d
    mono, rate = read_wav_float_mono(src)
    write_wav_int16_mono_16k(target, to_pipeline_mono_16k(mono, rate))
    return target