"""Single source of truth for the frame<->audio-sample timebase.

Video frames are CFR-extracted at FPS=30 from the same ffmpeg window as the
16 kHz audio (see scripts/slice_clips.py --fps/--sr), so frame i plays at
t = i/30 s and its aligned audio starts at sample round(i * SR/FPS) =
round(i * 533.333...).

A legacy integer hop (HOP = SR // FPS = 533) truncated the true 533.333...
boundary, accumulating a 0.333-sample-per-frame error -- 150 samples (9.4 ms)
over a 450-frame clip. Every per-frame audio consumer (DoA windows, GT
comparison, training windows) must derive boundaries from frame_start_sample()
so they share one exact, drift-free timebase. The AV-sync gate
(scripts/verify_av_sync.py) enforces that the underlying assets actually
satisfy this clock: jpg count == decoded frame count within encoder CFR
tolerance, and audio length == frames * SR/FPS within one frame.
"""
from __future__ import annotations

SR = 16000
FPS = 30
# Exact sample rate-per-frame ratio. NOT an integer: 533.3333...
SAMPLES_PER_FRAME = SR / FPS


def frame_start_sample(frame_idx: int) -> int:
    """First audio sample belonging to frame frame_idx (exact, drift-free)."""
    return round(frame_idx * SAMPLES_PER_FRAME)


def frame_count_from_audio(n_samples: int) -> int:
    """Whole frames a buffer of n_samples can cover at the reference clock."""
    return int(n_samples * FPS / SR)


def expected_audio_samples(n_frames: int) -> int:
    """Samples a clean clip of n_frames should hold at the reference clock."""
    return round(n_frames * SAMPLES_PER_FRAME)