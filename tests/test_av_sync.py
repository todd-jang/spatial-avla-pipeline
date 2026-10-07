"""Tests for the frame<->audio timebase (frame_clock) and the AV-sync gate.

1. frame_start_sample() has zero cumulative drift: the reference clock maps
   frame i to round(i*SR/FPS) samples, so a 450-frame clip ends exactly at
   240000 samples. The legacy integer hop (SR//FPS = 533) drifted 150 samples
   (9.4 ms) over the same clip -- the regression this suite locks.
2. scripts/verify_av_sync.py must PASS on every sliced clip under exports/TL_A
   (audio length within one frame of n_frames*SR/FPS, n_frames within one
   frame of seconds*fps, plus jpg-count equality wherever frames are shipped).
   Skips when no clips have been sliced yet (bundle keeps the gate for Colab,
   where slicing happens).
"""
import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.audio.frame_clock import (  # noqa: E402
    SAMPLES_PER_FRAME,
    expected_audio_samples,
    frame_count_from_audio,
    frame_start_sample,
)


def test_frame_start_sample_has_zero_cumulative_drift():
    assert frame_start_sample(450) == 240000
    assert frame_start_sample(449) == round(449 * SAMPLES_PER_FRAME)


def test_legacy_hop_drift_is_150_samples_over_450_frames():
    legacy = 450 * (16000 // 30)
    assert 240000 - legacy == 150


def test_audio_window_matches_reference_clock():
    win = 8000
    start = frame_start_sample(100)
    assert start == round(100 * SAMPLES_PER_FRAME)
    assert start + win >= 0


def test_expected_audio_samples_roundtrip():
    n_frames = 450
    assert frame_count_from_audio(expected_audio_samples(n_frames)) == n_frames


_clip_dirs = sorted(
    [d for split in ("train", "test")
     for d in (ROOT / "exports" / "TL_A" / split).glob("*")
     if (d / "clip_meta.json").exists()]
)
_clips_exist = bool(_clip_dirs)


@pytest.mark.skipif(not _clips_exist, reason="no sliced clips yet")
def test_av_sync_gate_passes_on_sliced_clips():
    sys.path.insert(0, (ROOT / "scripts").as_posix())
    from verify_av_sync import check_clip, main as gate_main

    roster = json.loads((ROOT / "configs" / "clip_roster.json").read_text())
    failed = [check_clip(d, roster) for d in _clip_dirs
              if check_clip(d, roster)["verdict"] != "PASS"]
    assert not failed, f"AV-sync gate failed: {failed}"


def test_gate_writes_report_and_exit_code(tmp_path):
    sys.path.insert(0, (ROOT / "scripts").as_posix())
    import verify_av_sync

    rep = tmp_path / "av_sync.json"
    rc = verify_av_sync.main(["--root", str(ROOT / "exports/TL_A"),
                              "--roster", str(ROOT / "configs/clip_roster.json"),
                              "--out", str(rep)])
    data = json.loads(rep.read_text())
    if _clips_exist:
        assert rc == 0
        assert data["verdict"] == "PASS"
    else:
        assert rc == 2
        assert data["verdict"] == "FAIL"
        assert data["n_clips"] == 0