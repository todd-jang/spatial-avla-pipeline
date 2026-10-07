#!/usr/bin/env python3
"""Exp A ablation: baseline (mock constant S_A) vs RT-DETR-driven S_A.

Consumes the per-clip detection JSONs under outputs/detections/
(<clip>__rtdetr-l.json, written by scripts/run_rtdetr_detect.py) plus
configs/vision_risk_weights.json and produces outputs/abl_report_s_a.json.

For every frame of every detected clip it computes the vision_anomaly_score
the way src.exp_a does:

  baseline variant -> the mock constant (default 0.2, --baseline-constant)
  rtdetr variant   -> src.exp_a.vision_risk.anomaly_score (max over boxes of
                      class_weight * conf; every figure comes from the weights
                      file, nothing hardcoded)

Then it threads each variant through src.exp_b's MA_w5 rule (vam >= 0.75) to
measure the fraction of frames that would drive the S_B vision trigger — the
quantity the ablation actually cares about (does the real detector change when
the system would call a vision anomaly?). Per-clip stats and per-condition
aggregation (day/rain/night) are emitted, along with the measured per-frame
latency already recorded in the detection JSONs.

Output schema: {"schema": "abl-report-sa/v1", ...}.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.exp_a.vision_risk import anomaly_score  # noqa: E402

DET_DIR = ROOT / "outputs" / "detections"
TRIGGER_VA_MA = 0.75


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
                "max": 0.0, "n": 0, "ma5_trigger_rate": 0.0}
    vs = sorted(values)
    trig = 0
    buf = []
    for v in values:
        buf.append(v)
        if len(buf) > 5:
            buf.pop(0)
        if sum(buf) / len(buf) >= TRIGGER_VA_MA:
            trig += 1
    return {
        "mean": round(statistics.fmean(vs), 4),
        "sd": round(statistics.pstdev(vs), 4),
        "p50": round(vs[len(vs) // 2], 4),
        "p90": round(vs[int(len(vs) * 0.90)], 4),
        "p99": round(vs[int(len(vs) * 0.99)], 4),
        "max": round(vs[-1], 4) if vs else 0.0,
        "n": len(vs),
        "ma5_trigger_rate": round(trig / len(vs), 4),
    }


def _series(model_name: str, clip: str, n_frames: int, constant: float) -> tuple[list, list]:
    base = [constant] * n_frames
    det = []
    for fi in range(n_frames):
        det.append(anomaly_score({"clip_id": clip, "frame_idx": fi},
                                 model=model_name))
    return base, det


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default=DET_DIR.as_posix())
    ap.add_argument("--model", default="rtdetr-l")
    ap.add_argument("--baseline-constant", type=float, default=0.2,
                    help="mock S_A constant (the value used when no detections)")
    ap.add_argument("--out", default=(ROOT / "outputs" / "abl_report_s_a.json").as_posix())
    args = ap.parse_args(argv)

    det_dir = Path(args.det_dir)
    paths = sorted(det_dir.glob(f"*__{args.model}.json"))
    if not paths:
        print(f"no detection files under {det_dir} (run run_rtdetr_detect.py first)",
              file=sys.stderr)
        return 2

    clips = {}
    for p in paths:
        doc = json.loads(p.read_text())
        n = int(doc["n_frames"])
        cond = doc.get("condition", "unknown")
        base, det = _series(args.model, doc["clip_id"], n, args.baseline_constant)
        clips[doc["clip_id"]] = {
            "condition": cond,
            "n_frames": n,
            "ms_per_frame": doc.get("ms_per_frame", 0.0),
            "ms_p50": doc.get("ms_p50", 0.0),
            "baseline": _stats(base),
            "rtdetr": _stats(det),
        }

    conditions = {}
    for cid, c in clips.items():
        agg = conditions.setdefault(c["condition"], {"clips": [], "baseline": [], "rtdetr": []})
        agg["clips"].append(cid)
        agg["baseline"].append(c["baseline"]["mean"])
        agg["rtdetr"].append(c["rtdetr"]["mean"])
    for cond, agg in conditions.items():
        agg["baseline_mean"] = round(statistics.fmean(agg.pop("baseline")), 4)
        agg["rtdetr_mean"] = round(statistics.fmean(agg.pop("rtdetr")), 4)

    report = {
        "schema": "abl-report-sa/v1",
        "method": f"{args.model} + configs/vision_risk_weights.json",
        "baseline": {"name": "mock_constant", "constant": args.baseline_constant},
        "trigger_rule": {"vision_anomaly_ma_threshold": TRIGGER_VA_MA, "ma_window": 5},
        "clips": clips,
        "conditions": conditions,
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"{'clip':5s} {'cond':6s} {'base_ma_trig':>12s} {'rt_ma_trig':>10s} "
          f"{'rt_mean':>8s} {'rt_p90':>7s} {'ms/frame':>8s}")
    for cid, c in clips.items():
        print(f"{cid:5s} {c['condition']:6s} "
              f"{c['baseline']['ma5_trigger_rate']:12.3f} "
              f"{c['rtdetr']['ma5_trigger_rate']:10.3f} "
              f"{c['rtdetr']['mean']:8.4f} {c['rtdetr']['p90']:7.4f} "
              f"{c['ms_per_frame']:8.2f}")
    print("condition aggregates:")
    for cond, agg in conditions.items():
        print(f"  {cond:6s} baseline_mean {agg['baseline_mean']:.4f}  "
              f"rtdetr_mean {agg['rtdetr_mean']:.4f}  clips {agg['clips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())