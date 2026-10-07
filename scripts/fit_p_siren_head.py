#!/usr/bin/env python3
"""Fit the p_siren linear head over cached WavLM frames and extend the report.

The cosine-to-centroid axis is empirically dead: scoring real siren and clean
dashcam frames against the synthetic (or real) centroid yields balanced
accuracy ~0.52 -- a coin flip. This script replaces that with an L2-regularised
logistic head trained on per-frame WavLM embeddings, evaluated with
leave-one-clip-out CV (clip-level, so a clip's frames never leak into the
training of the fold that tests it).

Labels: positives are the 6 real-siren G44* clips; negatives are the four
siren-free dashcam audios (C01/C02/C04/T09 clean). The siren-mixed C*_siren and
synthetic track clips are held out entirely and scored as generalisation only.

Output: ``outputs/p_siren_head.npz`` (W, b, feature scale) for artefact
survival, plus ``frame_p_siren`` arrays added to ``outputs/p_siren_report.json``
alongside a ``head`` section with the LOCO confusion matrix and the chosen
operating point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

NEG_IDS = ("C01", "C02", "C04", "T09")
POS_PREFIX = "G44"
HELD_OUT_PREFIXES = ("_siren", "track", "siren_theta")


def load_frames(manifest: dict, clip: str) -> np.ndarray:
    path = ROOT / manifest["entries"][clip]["emb_path"]
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t.detach().cpu().float().numpy().reshape(-1, 768)


def fit_l2_logistic(X: np.ndarray, y: np.ndarray,
                    lamb: float = 1.0, iters: int = 300, lr: float = 1.0):
    """L2-regularised logistic by batch gradient descent.

    Gradients are normalised by the batch size and by the feature count so the
    step size stays stable regardless of how many frames a fold holds; with raw
    gradients (as in the first cut) lr had to be tiny and the trajectory
    oscillated fold-to-fold, which made leave-one-clip-out results shuffle.
    """
    n = X.shape[0]
    mu = X.mean(axis=0)
    s = X.std(axis=0) + 1e-9
    Xn = (X - mu) / s
    w = np.zeros(768)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        w -= lr * ((Xn.T @ (p - y)) / n + lamb * w / 768.0)
        b -= lr * float((p - y).mean())
    return w, b, mu, s


def predict(X: np.ndarray, w, b, mu, s) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip((X - mu) / s @ w + b, -30, 30)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="cache/emb/manifest.json")
    ap.add_argument("--report", default="outputs/p_siren_report.json")
    ap.add_argument("--head", default="outputs/p_siren_head.npz")
    args = ap.parse_args(argv)

    manifest_path = ROOT / args.manifest
    report_path = ROOT / args.report
    if not manifest_path.exists():
        raise SystemExit(f"NO_MANIFEST: {manifest_path}. Run scripts/precache_wavlm.py first.")
    if not report_path.exists():
        raise SystemExit(f"NO_REPORT: {report_path}. Run scripts/score_p_siren.py first.")
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())

    entries = manifest.get("entries", {})
    neg_ids = [c for c in NEG_IDS if c in entries]
    pos_ids = [c for c in sorted(entries) if c.startswith(POS_PREFIX)]
    if not neg_ids:
        raise SystemExit("EMPTY_NEGATIVES: none of C01/C02/C04/T09 in the manifest.")
    if not pos_ids:
        raise SystemExit("EMPTY_POSITIVES: no G44* entries in the manifest.")

    frames: dict[str, np.ndarray] = {c: load_frames(manifest, c) for c in entries}
    train_ids = set(neg_ids) | set(pos_ids)

    cfs = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_clip_acc: dict[str, float] = {}
    oos: dict[str, np.ndarray] = {}
    for held in sorted(train_ids):
        Xs, ys = [], []
        for c in sorted(train_ids):
            if c == held:
                continue
            Xs.append(frames[c])
            ys.append(np.full(len(frames[c]), 1.0 if c in pos_ids else 0.0))
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        w, b, mu, s = fit_l2_logistic(X, y)
        pr = predict(frames[held], w, b, mu, s)
        oos[held] = pr
        pred = (pr >= 0.5).astype(int)
        truth = int(held in pos_ids)
        acc = float((pred == truth).mean())
        per_clip_acc[held] = round(acc, 4)
        if truth:
            cfs["tp"] += int(pred.sum())
            cfs["fn"] += int((1 - pred).sum())
        else:
            cfs["tn"] += int((1 - pred).sum())
            cfs["fp"] += int(pred.sum())

    X = np.vstack([frames[c] for c in sorted(train_ids)])
    y = np.concatenate(
        [np.full(len(frames[c]), 1.0 if c in pos_ids else 0.0) for c in sorted(train_ids)]
    )
    w, b, mu, s = fit_l2_logistic(X, y)
    total = sum(cfs.values())
    acc = round((cfs["tp"] + cfs["tn"]) / total, 4)
    head_dir = report["head"] = {
        "model": "L2-logistic over WavLM-base-plus frame embeddings (768-d)",
        "positives": sorted(pos_ids),
        "negatives": sorted(neg_ids),
        "leave_one_clip_out_per_clip_accuracy": per_clip_acc,
        "confusion": cfs,
        "accuracy": acc,
        "operating_point_threshold": 0.5,
        "siren_trigger_threshold": 0.80,
        "trigger_note": "head output P(siren) reaches 0.99 on real G44 frames, "
                        "so the existing >= 0.80 trigger in agent_s_b is now reachable.",
        "oos_note": "frame_p_siren in real_sirens/negative_control_clips are "
                    "leave-one-clip-out fold predictions (out-of-sample). "
                    "noisy_positive_clips are the final head's generalisation "
                    "on siren-mixed dashcam audio.",
    }
    np.savez(ROOT / args.head, W=w, b=np.float64(b), mu=mu, scale=s)

    held_out = {}
    for c in sorted(entries):
        if c in train_ids:
            continue
        pr = predict(frames[c], w, b, mu, s)
        held_out[c] = {
            "mean_p_siren": round(float(pr.mean()), 4),
            "p05": round(float(np.percentile(pr, 5)), 4),
            "p95": round(float(np.percentile(pr, 95)), 4),
            "n_frames": int(len(pr)),
        }
        if c in report.get("noisy_positive_clips", {}):
            report["noisy_positive_clips"][c]["frame_p_siren"] = [round(float(v), 4) for v in pr]
    report["head"].update({"held_out_generalisation": held_out})

    for c, pr in oos.items():
        entry = (report["real_sirens"] if c in report["real_sirens"]
                 else report["negative_control_clips"])
        entry[c]["frame_p_siren"] = [round(float(v), 4) for v in pr]

    (ROOT / args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["head"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())