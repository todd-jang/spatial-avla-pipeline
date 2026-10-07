#!/usr/bin/env python3
"""RT-DETR batch detection on sliced clip frames -> per-frame detections JSON.

Runs an Ultralytics RT-DETR checkpoint (default rtdetr-l.pt, COCO-pretrained)
over every jpg under exports/TL_A/{train,test}/<clip>/frames/ for the selected
clips and writes one JSON document per clip to --out-dir:

    {
      "clip_id": "C01",
      "condition": "day",
      "model": "rtdetr-l",
      "device": "mps",
      "n_frames": 450,
      "min_conf": 0.25,
      "frames": {
        "0":  {"frame_idx": 0, "file": "frame_000001.jpg",
               "boxes": [{"class_name": "car", "conf": 0.92, "bbox": [x1,y1,x2,y2]}]},
        ...
      }
    }

Design intent:
- Deterministic: clip list and frame list are sorted; boxes are written in model
  output order (confidence-descending) and the file is only written on success.
- Resume-safe: a clip whose output already exists is skipped unless --force.
- Device auto-select: cuda (Linux/Colab) > mps (Apple Silicon) > cpu, or pinned
  with --device. A --limit N option exists for smoke tests (N frames per clip).
- The output feeds src.exp_a (S_A): each detection is turned into an anomaly
  contribution via configs/vision_risk_weights.json; S_A never re-runs the model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

import torch  # noqa: E402


FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def pick_device(pin: str | None) -> str:
    if pin:
        return pin
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_clip(model, frames: list[Path], device: str, min_conf: float,
                clip_id: str, condition: str, model_name: str) -> dict:
    frames_out = {}
    call_times = []
    for i, f in enumerate(sorted(frames, key=lambda p: int(FRAME_RE.search(p.name).group(1)))):  # noqa: E501
        t0 = time.perf_counter()
        res = model(f, verbose=False)[0]
        call_times.append((time.perf_counter() - t0) * 1000)
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for cls, conf, xyxy in zip(res.boxes.cls.tolist(),
                                       res.boxes.conf.tolist(),
                                       res.boxes.xyxy.tolist()):
                if float(conf) < min_conf:
                    continue
                dets.append({
                    "class_name": res.names[int(cls)],
                    "class_id": int(cls),
                    "conf": round(float(conf), 4),
                    "bbox": [round(float(v), 2) for v in xyxy],
                })
        frame_idx = int(FRAME_RE.search(f.name).group(1)) - 1
        frames_out[str(frame_idx)] = {
            "frame_idx": frame_idx,
            "file": f.name,
            "boxes": dets,
        }
        if (i + 1) % 100 == 0:
            print(f"  {clip_id}: {i+1}/{len(frames)} frames "
                  f"({sum(call_times)/(i+1):.1f} ms/frame)")
    return {
        "clip_id": clip_id,
        "condition": condition,
        "model": model_name,
        "device": device,
        "n_frames": len(frames),
        "min_conf": min_conf,
        "ms_per_frame": round(sum(call_times) / len(call_times), 2),
        "ms_p50": round(sorted(call_times)[len(call_times) // 2], 2),
        "frames": frames_out,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT / "exports/TL_A")
    ap.add_argument("--out-dir", default=ROOT / "outputs/detections")
    ap.add_argument("--clips", nargs="+", default=None,
                    help="clip ids to detect; default: every clip under train+test")
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu; default auto")
    ap.add_argument("--min-conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke mode: only process first N frames per clip")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args(argv)

    try:
        from ultralytics import RTDETR
    except Exception as e:  # noqa: BLE001
        print(f"ultralytics import failed: {e}", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    print(f"device={device}")
    model = RTDETR(args.model)
    model.to(device)
    model_name = Path(args.model).stem

    root = Path(args.root)
    clip_dirs = sorted(
        [d for split in ("train", "test") for d in (root / split).glob("*")
         if (d / "clip_meta.json").exists()])
    if args.clips:
        clip_dirs = [d for d in clip_dirs if d.name in args.clips]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for d in clip_dirs:
        meta = json.loads((d / "clip_meta.json").read_text())
        frames = sorted(d.glob("frames/frame_*.jpg"))
        if not frames:
            print(f"SKIP {d.name}: no frames/")
            continue
        if args.limit:
            frames = frames[: args.limit]
        out = out_dir / f"{d.name}__{model_name}.json"
        if out.exists() and not args.force and args.limit == 0:
            print(f"SKIP {d.name}: {out.name} exists (use --force)")
            results[d.name] = json.loads(out.read_text())
            continue
        print(f"DETECT {d.name} ({meta.get('condition')}), {len(frames)} frames")
        doc = detect_clip(model, frames, device, args.min_conf,
                          d.name, meta.get("condition", "unknown"), model_name)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        results[d.name] = doc
        print(f"DONE {d.name}: {len(frames)} frames, {doc['ms_per_frame']} ms/frame "
              f"(p50 {doc['ms_p50']}) -> {out}")

    summary = {"model": model_name, "device": device,
               "min_conf": args.min_conf, "clips": list(results)}
    (out_dir / "_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())