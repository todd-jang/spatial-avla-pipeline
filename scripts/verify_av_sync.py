#!/usr/bin/env python3
"""AV-sync gate: prove every sliced clip satisfies the reference frame clock.

Checks, per clip under --root (default exports/TL_A, train+test):

1. The audio sample count matches round(n_frames * SAMPLES_PER_FRAME)
   within one frame (533.33 samples at 16 kHz / 30 fps); a larger gap means
   the wav and the video do not cover the same wall-clock window.
2. n_frames matches round(seconds * fps) from the roster within one frame
   (ffmpeg CFR rounding at the tail routinely yields 450 vs 451).
3. When frames/ is present (slicing machine), the exported jpg count equals
   the n_frames recorded in clip_meta.json. The audio bundle does not ship
   video frames, so there this check is recorded as trusted-from-slice-time
   rather than re-verified.

Verdict is PASS only when every clip passes every applicable check. Any clip
failing the audio-length check means frame-aligned consumers (DoA windows, GT
comparison, the training windows in colab_train_smoke) would silently map
audio onto the wrong frame -- the exact regression frame_clock.py exists to
prevent.

Writes --out (default outputs/av_sync_report.json) with per-clip rows; exit
code 0 only on PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

import numpy as np

from src.core.audio.frame_clock import (  # noqa: E402
    SAMPLES_PER_FRAME,
    expected_audio_samples,
)


def audio_samples(wav: Path) -> int:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes()


def check_clip(clip_dir: Path, roster: dict) -> dict:
    meta = json.loads((clip_dir / "clip_meta.json").read_text())
    frames = sorted(clip_dir.glob("frames/frame_*.jpg"))
    jpg_count = len(frames)
    n_frames = int(meta["n_frames"])
    fps = int(meta.get("fps", roster.get("fps", 30)))
    seconds = float(roster.get("seconds", 15.0))
    wav = clip_dir / "audio" / "audio_stereo_16k.wav"
    if not wav.exists():
        return {"clip_id": clip_dir.parent.name + "/" + clip_dir.name, "verdict": "FAIL",
                "checks": [{"check": "wav present", "ok": False, "detail": str(wav)}]}
    n_audio = audio_samples(wav)
    expected_frames = round(seconds * fps)
    expected_audio_for_frames = expected_audio_samples(n_frames)

    checks = [
        {"check": f"audio samples ~= n_frames*SR/FPS ({expected_audio_for_frames} +-{int(SAMPLES_PER_FRAME)})",
         "ok": abs(n_audio - expected_audio_for_frames) <= int(SAMPLES_PER_FRAME),
         "detail": f"{n_audio} samples, delta {n_audio - expected_audio_for_frames}"},
        {"check": f"n_frames ~= roster seconds*fps ({expected_frames} +-1)",
         "ok": abs(n_frames - expected_frames) <= 1,
         "detail": f"delta {n_frames - expected_frames}"},
    ]
    if jpg_count:
        checks.insert(0,
            {"check": f"jpg count == clip_meta.n_frames ({n_frames})",
             "ok": jpg_count == n_frames,
             "detail": f"{jpg_count} jpgs"})
    else:
        checks.insert(0,
            {"check": "jpg count == clip_meta.n_frames",
             "ok": True,
             "detail": "frames/ not shipped here; n_frames trusted from slice time"})
    return {"clip_id": clip_dir.parent.name + "/" + clip_dir.name,
            "n_frames": n_frames, "jpg_count": jpg_count,
            "audio_samples": n_audio, "expected_audio_samples": expected_audio_for_frames,
            "verdict": "PASS" if all(c["ok"] for c in checks) else "FAIL",
            "checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT / "exports/TL_A")
    ap.add_argument("--roster", default=ROOT / "configs/clip_roster.json")
    ap.add_argument("--out", default=ROOT / "outputs/av_sync_report.json")
    args = ap.parse_args(argv)

    roster = json.loads(Path(args.roster).read_text())
    root = Path(args.root)
    clip_dirs = sorted(
        [d for split in ("train", "test") for d in (root / split).glob("*")
         if (d / "clip_meta.json").exists()])

    rows = [check_clip(d, roster) for d in clip_dirs]

    rows = [check_clip(d, roster) for d in clip_dirs]
    for r in rows:
        tag = "PASS" if r["verdict"] == "PASS" else "FAIL"
        print(f"{tag} {r['clip_id']}: frames {r['jpg_count']}/{r['n_frames']} "
              f"audio {r['audio_samples']}/{r['expected_audio_samples']}")
        for c in r["checks"]:
            if not c["ok"]:
                print(f"    {c['check']}: {c['detail']}")

    report = {
        "source": "measured",
        "measured_on": [r["clip_id"] for r in rows],
        "clock": {"sr": 16000, "fps": 30, "samples_per_frame": SAMPLES_PER_FRAME},
        "verdict": "PASS" if rows and all(r["verdict"] == "PASS" for r in rows) else "FAIL",
        "n_clips": len(rows),
        "clips": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not rows:
        print(f"NO_CLIPS: no clip_meta.json under {root}", file=sys.stderr)
        return 2
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())