"""WAV reading for the audio pipeline.

Deliberately stdlib ``wave`` + numpy rather than ``torchaudio``: the newest
torchaudio build is older than the installed torch, so adding it silently
downgrades torch. Every clip this project produces is 16 kHz int16 stereo PCM,
so a loader that validates and refuses anything else is preferable to one that
silently resamples and degrades the signal before WavLM ever sees it.

Mono is derived here purely for the WavLM front end. Slicers keep the stereo
channels intact on disk, because the DoA path needs the inter-channel time
difference that a mono downmix destroys.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2


def load_wav_mono_16k(path: str | Path) -> np.ndarray:
    """Return mono float32 in [-1, 1] at 16 kHz.

    Scaling matches torchaudio's int16 convention: divide by ``1 << (8*width - 1)``.
    """
    with wave.open(str(path)) as handle:
        n_ch = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != SAMPLE_WIDTH_BYTES:
        raise SystemExit(f"UNSUPPORTED_SAMPWIDTH {width} in {path} (expected 2 = int16)")
    if rate != SAMPLE_RATE:
        raise SystemExit(f"UNSUPPORTED_SAMPLE_RATE {rate} in {path} (expected {SAMPLE_RATE})")
    if n_ch < 1:
        raise SystemExit(f"UNSUPPORTED_CHANNELS {n_ch} in {path}")
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch)
    mono = samples.astype(np.float32) / 32768.0
    return mono.mean(axis=1) if n_ch > 1 else mono[:, 0]


def load_wav_stereo_16k(path: str | Path) -> np.ndarray:
    """Return (n, 2) float32 stereo in [-1, 1] at 16 kHz, for the DoA path."""
    with wave.open(str(path)) as handle:
        n_ch = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != SAMPLE_WIDTH_BYTES:
        raise SystemExit(f"UNSUPPORTED_SAMPWIDTH {width} in {path} (expected 2 = int16)")
    if rate != SAMPLE_RATE:
        raise SystemExit(f"UNSUPPORTED_SAMPLE_RATE {rate} in {path} (expected {SAMPLE_RATE})")
    if n_ch != 2:
        raise SystemExit(f"NOT_STEREO: {path} has {n_ch} channel(s); DoA needs 2.")
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0
