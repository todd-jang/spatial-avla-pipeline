#!/usr/bin/env python3
"""Colab T4 smoke training for the tiny AVLA head (plan todo 9).

Cache-only by contract: this script NEVER loads WavLM into VRAM. It consumes
embeddings already computed by ``scripts/precache_wavlm.py`` from
``cache/emb/<clip>.pt``. That split is deliberate -- the training step then
fits on a T4 without a ~1.2 GB model resident, and the train/infer paths stay
WavLM-free as the plan requires.

What this proves: the harness wires up, the head has trainable parameters, and
the loss genuinely decreases over >=3 epochs, with artifacts on disk as proof.
What this does NOT prove: siren-detection accuracy. The supervision target is a
deterministic loudness task derived from the clip's own audio (window RMS above
the median window), NOT a labelled emergency-event dataset. Any accuracy claim
built on this run would be unfounded.

Fails loudly, by design. Missing CUDA, a non-positive learning rate, a frozen
head, a single-class target, or a loss that does not decrease are hard errors
that abort before any "passing" artifact is written.

Usage (Colab, GPU runtime):
    python3 scripts/colab_train_smoke.py \
        --manifest cache/emb/manifest.json \
        --embedding cache/emb/C01.pt \
        --audio exports/TL_A/train/C01/audio/audio_stereo_16k.wav \
        --frames 16 --epochs 3 --lr 1e-3

Local structural check (no GPU, no real embeddings, no artifacts):
    python3 scripts/colab_train_smoke.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from src.core.audio.frame_clock import frame_count_from_audio, frame_start_sample  # noqa: E402
from src.core.audio.wav_io import load_wav_mono_16k  # noqa: E402

WAVLM_HIDDEN = 768  # wavlm-base-plus hidden size; only used for the dry-run stub
WAVLM_ROWS_15S = 750  # 50 Hz WavLM frame rate over a 15 s clip
MIN_EPOCHS = 3


def require_cuda(dry_run: bool) -> str:
    """Return the device to train on. Aborts if CUDA is absent outside dry-run."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if dry_run:
        return "cpu (dry-run)"
    raise SystemExit(
        "CUDA_REQUIRED: this entrypoint is Colab-T4-only by decision, and the "
        "torch.cuda.is_available() gate is deliberately left intact. "
        "Re-run on a GPU runtime, or pass --dry-run for a structural check only."
    )


def assert_trainable(head) -> int:
    """Abort unless the head has trainable parameters. Returns their count."""
    count = sum(p.numel() for p in head.parameters() if p.requires_grad)
    if count == 0:
        raise SystemExit("FROZEN_HEAD: no head parameter has requires_grad=True.")
    return count


def build_head(feature_dim: int, lr: float):
    """Tiny AVLA head. Refuses a non-positive lr or a head with no trainable params."""
    import torch.nn as nn

    if lr <= 0:
        raise SystemExit(f"LR_NOT_POSITIVE: lr={lr}; refusing to report a flat loss as progress.")
    head = nn.Linear(feature_dim, 1)
    assert_trainable(head)
    return head


def load_cached_embedding(path: Path, dry_run: bool):
    """Load the cached WavLM tensor (T, D). Dry-run substitutes a deterministic stub."""
    import torch

    if dry_run:
        gen = torch.Generator().manual_seed(0)
        return torch.randn(WAVLM_ROWS_15S, WAVLM_HIDDEN, generator=gen), True
    if not path.exists():
        raise SystemExit(
            f"NO_EMBEDDING: {path} is missing. Run scripts/precache_wavlm.py on the "
            "GPU runtime first -- it is the only place allowed to hold WavLM."
        )
    emb = torch.load(path, map_location="cpu")
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    if emb.ndim != 2:
        raise SystemExit(f"BAD_EMBEDDING_SHAPE: {path} has {emb.ndim} dims, expected 2")
    return emb, False


def build_frame_labels(audio_path: Path, n_frames: int):
    """Deterministic binary target: 1 where the window RMS exceeds the median window.

    Real signal from the actual clip, so the head has something learnable and the
    run is reproducible. Not an event label -- see the module docstring.
    """
    mono = load_wav_mono_16k(audio_path)
    total_frames = frame_count_from_audio(mono.shape[0])
    idx = np.linspace(0, total_frames - 1, n_frames).round().astype(int)
    edges = np.append([frame_start_sample(i) for i in idx], mono.shape[0])
    rms = np.array(
        [float(np.sqrt(np.mean(mono[a:b] ** 2))) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])],
        dtype=np.float64,
    )
    median = float(np.median(rms))
    labels = (rms > median).astype(np.float32)
    if labels.min() == labels.max():
        raise SystemExit(
            f"SINGLE_CLASS_TARGET: all {n_frames} windows fell on one side of the "
            f"median RMS ({median:.6f}); the loudness task is degenerate for this clip."
        )
    return labels, rms, median


def select_frames(emb, n_frames: int):
    """Take n_frames evenly spaced embedding rows so 8-16 frames cover the clip."""
    total = emb.shape[0]
    if n_frames > total:
        raise SystemExit(f"TOO_MANY_FRAMES: asked for {n_frames} but the cache holds {total} rows.")
    idx = np.linspace(0, total - 1, n_frames).round().astype(int)
    return emb[idx]


def assert_decreased(initial: float, final: float) -> None:
    """Abort unless the loss actually fell. A flat loss is not progress."""
    if not final < initial:
        raise SystemExit(
            f"LOSS_NOT_DECREASING: initial={initial:.6f} final={final:.6f}. "
            "No pass artifact written -- a flat loss means the head is not learning."
        )


def train_loop(head, x, y, epochs: int, lr: float, device: str, best_path: Path, patience: int = 5):
    """Full-batch Adam loop; checkpoints the best epoch to best_path. Returns losses with early stopping."""
    import torch
    import torch.nn as nn

    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    xt, yt = x.to(device), y.to(device).reshape(-1, 1)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    best = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        opt.zero_grad()
        loss = lossf(head(xt), yt)
        loss.backward()
        opt.step()
        value = float(loss.item())
        losses.append(value)
        if value < best:
            best = value
            torch.save(head.state_dict(), best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Load best weights
    if best_path.exists():
        head.load_state_dict(torch.load(best_path))
    return losses


def write_loss_curve(losses, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "NO_MATPLOTLIB: loss_curve.png is a required artifact. "
            "Install matplotlib (preinstalled on Colab) and re-run."
        ) from exc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("AVLA head smoke training (cache-only WavLM)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def selfcheck_loss_gate_rejects_flat_and_rising() -> None:
    for initial, final in ((0.5, 0.5), (0.5, 0.6)):
        try:
            assert_decreased(initial, final)
        except SystemExit as exc:
            got = str(exc).split(":", 1)[0]
            if got != "LOSS_NOT_DECREASING":
                raise SystemExit(
                    f"GUARD_MISMATCH: expected LOSS_NOT_DECREASING, got {got}"
                ) from exc
            continue
        raise SystemExit(f"GUARD_SILENT: LOSS_NOT_DECREASING accepted {initial} -> {final}")
    print("  guard fires: LOSS_NOT_DECREASING (flat and rising loss both rejected)")


def selfcheck_guards() -> None:
    """Prove each fail-loud guard fires, by calling the real production functions."""
    import torch.nn as nn

    for expected, fn in (
        ("LR_NOT_POSITIVE", lambda: build_head(8, 0.0)),
        ("FROZEN_HEAD", lambda: assert_trainable(_freeze(nn.Linear(8, 1)))),
    ):
        try:
            fn()
        except SystemExit as exc:
            got = str(exc).split(":", 1)[0]
            if got != expected:
                raise SystemExit(f"GUARD_MISMATCH: expected {expected}, got {got}") from exc
            print(f"  guard fires: {got}")
            continue
        raise SystemExit(f"GUARD_SILENT: {expected} did not fire")

    selfcheck_loss_gate_rejects_flat_and_rising()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cache/emb/manifest.json")
    ap.add_argument("--embedding", default="cache/emb/C01.pt")
    ap.add_argument("--audio", default="exports/TL_A/train/C01/audio/audio_stereo_16k.wav")
    ap.add_argument("--out", default="models/checkpoints/smoke")
    ap.add_argument("--mirror", default="outputs")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", default="C01")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch

    if args.frames < 8 or args.frames > 16:
        raise SystemExit(f"FRAMES_OUT_OF_RANGE: {args.frames}; plan todo 9 specifies 8-16.")
    if args.epochs < MIN_EPOCHS:
        raise SystemExit(f"TOO_FEW_EPOCHS: {args.epochs}; plan todo 9 requires >= {MIN_EPOCHS}.")

    if args.dry_run:
        print("DRY RUN -- structural check only. No CUDA, stub embeddings, no artifacts.")
        print("Fail-loud guard self-check:")
        selfcheck_guards()

    device = require_cuda(args.dry_run)
    torch.manual_seed(0)
    emb, stubbed = load_cached_embedding(Path(args.embedding), args.dry_run)
    labels, rms, median = build_frame_labels(Path(args.audio), args.frames)
    x = select_frames(emb, args.frames).float()
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    head = build_head(x.shape[1], args.lr)
    trainable = assert_trainable(head)

    print(f"clip={args.clip} frames={args.frames} feature_dim={x.shape[1]} trainable={trainable}")
    print(
        f"labels: {int(labels.sum())} high / {int((1 - labels).sum())} low  median_rms={median:.6f}"
    )
    print(f"embedding: {tuple(emb.shape)}{' (stub)' if stubbed else ''}  device={device}")

    if args.dry_run:
        rehearsal = train_loop(head, x, torch.from_numpy(labels), args.epochs, args.lr,
                               "cpu", Path("/dev/null"))
        assert_decreased(rehearsal[0], rehearsal[-1])
        print(f"  cpu rehearsal loss {rehearsal[0]:.6f} -> {rehearsal[-1]:.6f} (gate satisfied)")
        print("Dry run complete. CUDA step and artifacts intentionally skipped.")
        return

    if device != "cuda":
        raise SystemExit(f"UNEXPECTED_DEVICE: {device}")

    out = Path(args.out)
    losses = train_loop(head, x, torch.from_numpy(labels), args.epochs, args.lr, device,
                        out / "best.pt")
    for epoch, value in enumerate(losses, start=1):
        print(f"  epoch {epoch}/{args.epochs} loss={value:.6f}")

    initial, final = losses[0], losses[-1]
    assert_decreased(initial, final)

    write_loss_curve(losses, out / "loss_curve.png")
    log = {
        "clip": args.clip,
        "device_used": device,
        "wavlm_loaded_in_vram": False,
        "epochs": args.epochs,
        "lr": args.lr,
        "frames": args.frames,
        "feature_dim": int(x.shape[1]),
        "trainable_params": trainable,
        "embedding": str(args.embedding),
        "label_rule": "window_rms_above_median",
        "labels_high": int(labels.sum()),
        "labels_low": int((1 - labels).sum()),
        "median_rms": median,
        "window_rms": [round(float(v), 8) for v in rms],
        "initial_loss": initial,
        "final_loss": final,
        "losses": losses,
        "loss_decreased": True,
        "task_note": "harness smoke test only; not a siren-detection accuracy claim",
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    mirror = Path(args.mirror) / "smoke"
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "train_log.json").write_text(json.dumps(log, indent=2))
    (mirror / "loss_curve.png").write_bytes((out / "loss_curve.png").read_bytes())
    print(f"ARTIFACTS_OK initial_loss={initial:.6f} final_loss={final:.6f} -> {out}")


if __name__ == "__main__":
    main()
