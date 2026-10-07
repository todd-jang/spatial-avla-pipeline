#!/usr/bin/env python3
"""Score real siren recordings against the synthetic Woodworth baseline.

This was the first experiment that gave ``p_siren`` a non-trivial value, and it
is now the archive of the REJECTED cosine axis: cosine-to-centroid cannot
separate siren from non-siren (balanced accuracy ~0.52, a coin flip), so
``scripts/fit_p_siren_head.py`` fits an L2-logistic head over the same cached
frames instead, and the pipeline consumes its out-of-sample ``frame_p_siren``
probabilities. This script documents why the cosine axis was abandoned.

The comparison is embedding-space: every clip was embedded by the same
WavLM-base-plus pass in ``scripts/precache_wavlm.py`` (one model, one device,
one manifest), so the only difference between a real and a synthetic entry is
the audio waveform itself.

What "score" means here: for each real siren, take its 50 Hz frame sequence,
mean-pool to (1, 768), and measure

    cosine(real_mean, synthetic_centroid)

where the synthetic centroid is the mean of all 13 ``siren_theta*`` baseline
frames. Also reported: text (non-theta) synthetic clips, plus the siren-free
dashcam negatives (C01/C02/C04/T09 clean) and siren-mixed C*_siren positives
that the calibration sweep uses.

Fails loudly on a missing manifest, an empty baseline, or zero real entries.
Writes ``outputs/p_siren_report.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def load_frames(manifest: dict, clip: str) -> np.ndarray:
    """Return (1, T, 768) embeddings for one clip as float32."""
    path = ROOT / manifest["entries"][clip]["emb_path"]
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t.detach().cpu().float().numpy()


def mean_pool(frames: np.ndarray) -> np.ndarray:
    """(1, T, D) -> (1, D); WavLM 50 Hz frames collapse to one vector per clip."""
    return frames.mean(axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="cache/emb/manifest.json")
    ap.add_argument("--out", default="outputs/p_siren_report.json")
    args = ap.parse_args(argv)

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"NO_MANIFEST: {manifest_path} is missing. Run scripts/precache_wavlm.py first.")
    manifest = json.loads(manifest_path.read_text())

    entries = manifest.get("entries", {})
    baseline = {k: v for k, v in entries.items() if k.startswith("siren_theta")}
    real = {k: v for k, v in entries.items() if k.startswith("G44")}
    text = {k: v for k, v in entries.items() if k.startswith("track")}
    clean = {k: v for k, v in entries.items() if k in ("C01", "C02", "C04", "T09")}
    noisy = {k: v for k, v in entries.items() if k.endswith("_siren")}
    if not baseline:
        raise SystemExit("EMPTY_BASELINE: no siren_theta* entries in the manifest.")
    if not real:
        raise SystemExit("EMPTY_REAL: no G44* real-siren entries in the manifest.")

    baseline_frames = np.concatenate([load_frames(manifest, k) for k in baseline])
    centroid = mean_pool(baseline_frames)[0]
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)


    def frame_cosines(manifest: dict, clip: str) -> np.ndarray:
        frames = load_frames(manifest, clip)
        return np.array([cosine(p, centroid_norm) for p in frames.reshape(-1, 768)])


    def clip_stats(clip: str) -> dict:
        per = frame_cosines(manifest, clip)
        frames = load_frames(manifest, clip)
        return {
            "cosine_vs_synth_centroid": round(cosine(mean_pool(frames)[0], centroid_norm), 4),
            "frame_cosine_mean": round(float(per.mean()), 4),
            "frame_cosine_std": round(float(per.std()), 4),
            "n_frames": int(frames.shape[1]),
            "device_used": manifest.get("device_used", "unknown"),
            "frame_cosines": [round(float(v), 4) for v in per],
        }

    real_scores = {clip: clip_stats(clip) for clip in sorted(real)}
    clean_scores = {clip: clip_stats(clip) for clip in sorted(clean)}
    noisy_scores = {clip: clip_stats(clip) for clip in sorted(noisy)}
    text_scores = {
        clip: round(cosine(mean_pool(load_frames(manifest, clip))[0], centroid_norm), 4)
        for clip in sorted(text)
    } if text else {}

    mean_real = float(np.mean([v["cosine_vs_synth_centroid"] for v in real_scores.values()]))

    neg_frames = np.concatenate([np.array(clean_scores[c]["frame_cosines"]) for c in clean_scores])
    pos_frames = np.concatenate(
        [np.array(real_scores[c]["frame_cosines"]) for c in real_scores]
        + [np.array(noisy_scores[c]["frame_cosines"]) for c in noisy_scores]
    )
    thresholds = np.arange(0.10, 0.95, 0.05)
    accs = []
    for t in thresholds:
        tp = float((pos_frames >= t).mean())
        tn = float((neg_frames < t).mean())
        accs.append({"threshold": round(float(t), 2), "true_positive_rate": round(tp, 4),
                     "true_negative_rate": round(tn, 4), "balanced_accuracy": round((tp + tn) / 2, 4)})
    best = max(accs, key=lambda a: a["balanced_accuracy"])
    report = {
        "experiment": "p_siren: real siren vs synthetic Woodworth baseline",
        "embedding_model": "microsoft/wavlm-base-plus",
        "device_used": manifest.get("device_used", "unknown"),
        "synthetic_centroid": "mean of all siren_theta* pooled frames",
        "n_baseline_clips": len(baseline),
        "n_baseline_frames": int(baseline_frames.shape[0]),
        "mean_real_cosine": round(mean_real, 4),
        "real_sirens": real_scores,
        "synthetic_text_clips": text_scores,
        "negative_control_clips": clean_scores,
        "noisy_positive_clips": noisy_scores,
        "calibration": {
            "negative": "clean dashcam audio (no siren mixed): C01/C02/C04/T09",
            "positive": "real sirens G44* plus siren-mixed dashcam C*_siren",
            "n_neg_frames": int(neg_frames.shape[0]),
            "n_pos_frames": int(pos_frames.shape[0]),
            "neg_frame_cosine_mean": round(float(neg_frames.mean()), 4),
            "pos_frame_cosine_mean": round(float(pos_frames.mean()), 4),
            "sweeps": accs,
            "best_threshold": best["threshold"],
            "best_balanced_accuracy": best["balanced_accuracy"],
            "recommended_p_siren_threshold": best["threshold"],
        },
        "interpretation": (
            "cos ~1: real siren occupies the synthetic siren's WavLM neighbourhood. "
            "cos ~0: the synthetic baseline is not representative of real sirens. "
            "The calibration sweep picks the threshold maximising balanced accuracy "
            "between negative (siren-free) and positive (real/mixed siren) frames."
        ),
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())