#!/usr/bin/env python3
"""Exp B audio ablation: baseline (mock constant p_siren) vs measured p_siren.

Consumes outputs/p_siren_report.json (per-frame calibrated P(siren), written by
scripts/fit_p_siren_head.py from precached WavLM embeddings) and produces
outputs/abl_report_s_b.json.

For every measured frame of every scored clip it computes the S_B audio input
the way src.exp_b consumes it:

  baseline variant -> the mock constant (default 0.0, --baseline-constant:
                      the pre-MVP pipeline carried p_siren=0.0 on every frame)
  real variant     -> the report's per-frame ``frame_p_siren``

Then it instantiates the two quantities the ablation cares about -- does real
audio change when the system would call a siren?

  siren_trigger_rate : fraction of frames with p_siren >= siren_trigger_threshold
                       (the S_B raw-trigger audio rule, p >= 0.80; the threshold
                       is READ from the report's head.siren_trigger_threshold,
                       never hardcoded)
  is_triggered_rate  : fraction of frames where the actual S_B hysteresis FSM
                       (35-frame hold, threaded statefully across frames) is
                       active -- the quantity the demo actually displays

Per-clip stats (mean/sd/p50/p90/p99/max/n + both rates) and per-condition
aggregation are emitted. Conditions are derived the same way the vision report
derives them: roster weather for the dashcam clips (day/rain/night), plus a
``real_siren`` condition for the G44* recordings and ``siren_mixed`` for the
12 dB SNR synthetic mix (C*_siren). The measured head accuracy and per-clip
leave-one-clip-out accuracy are carried into the report header so the trigger
rates read against the model's own measured quality.

Output schema: {"schema": "abl-report-sb/v1", ...}.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.audio.p_siren_provider import P_sirenProvider  # noqa: E402
from src.exp_b.agent_s_b import s_b_fn  # noqa: E402

DEFAULT_REPORT = ROOT / "outputs" / "p_siren_report.json"
HOLD_FRAMES = 35

# Roster weather per dashcam clip — same convention as run_abl_report.py for
# the vision channel (day/rain/night). Everything else is derived at runtime.
CONDITION_BY_CLIP = {"C01": "day", "C02": "rain", "C04": "night", "T09": "day"}


def _stats(values: list[float], *, siren_thr: float) -> dict:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
                "max": 0.0, "n": 0, "siren_trigger_rate": 0.0,
                "is_triggered_rate": 0.0}
    vs = sorted(values)
    siren_hits = sum(1 for v in values if v >= siren_thr)
    active = _fsm_active_frames(values, siren_thr)
    return {
        "mean": round(statistics.fmean(vs), 4),
        "sd": round(statistics.pstdev(vs), 4),
        "p50": round(vs[len(vs) // 2], 4),
        "p90": round(vs[int(len(vs) * 0.90)], 4),
        "p99": round(vs[int(len(vs) * 0.99)], 4),
        "max": round(vs[-1], 4) if vs else 0.0,
        "n": len(vs),
        "siren_trigger_rate": round(siren_hits / len(vs), 4),
        "is_triggered_rate": round(active / len(vs), 4),
    }


def _fsm_active_frames(values: list[float], siren_thr: float) -> int:
    """Thread the real S_B hysteresis FSM over a p_siren series.

    Mirrors scripts/build_demo_sequence.py::simulate_fsm: s_b_fn carries the
    FSM across frames via the returned state, so a 35-frame hold behaves like
    the live graph. Only the audio rule is exercised (no vision/DoA inputs) —
    the vision channel is the S_A report's job.
    """
    state: dict = {
        "frame_idx": 0,
        "event_trigger": {
            "hysteresis": {"state": "idle", "active_until_frame_idx": -1},
            "details": {"p_siren": 0.0, "audio_confidence": 0.0},
        },
        "_hist": {},
        "meta_exec": {},
        "audio": {},
        "doa": {},
    }
    active = 0
    for fi, p in enumerate(values):
        s = dict(state)
        s["frame_idx"] = fi
        s["event_trigger"] = {
            "hysteresis": dict(state["event_trigger"]["hysteresis"]),
            "details": dict(state["event_trigger"]["details"]),
        }
        s["event_trigger"]["details"]["p_siren"] = p
        s["event_trigger"]["details"]["audio_confidence"] = p
        s["_hist"] = {
            "va_list": list(state["_hist"].get("va_list", [])),
            "dd_list": list(state["_hist"].get("dd_list", [])),
            "dps_list": list(state["_hist"].get("dps_list", [])),
        }
        out = s_b_fn(s)
        if out["event_trigger"].get("is_triggered"):
            active += 1
        state = out
    return active


def _series(provider: P_sirenProvider, clip_id: str, n_frames: int,
            constant: float) -> tuple[list[float], list[float]]:
    base = [constant] * n_frames
    real = [provider.frame_value(clip_id, fi) for fi in range(n_frames)]
    return base, real


def _condition(clip_id: str, report_groups: dict[str, dict]) -> str:
    if clip_id in report_groups.get("real_sirens", {}):
        return "real_siren"
    if clip_id in report_groups.get("noisy_positive_clips", {}):
        return "siren_mixed"
    return CONDITION_BY_CLIP.get(clip_id, "unknown")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    ap.add_argument("--baseline-constant", type=float, default=0.0,
                    help="mock p_siren constant (the pre-MVP pipeline value)")
    ap.add_argument("--out", default=(ROOT / "outputs" / "abl_report_s_b.json").as_posix())
    ap.add_argument("--hold-frames", type=int, default=HOLD_FRAMES)
    args = ap.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"no p_siren report at {report_path} (run fit_p_siren_head.py first)",
              file=sys.stderr)
        return 2
    doc = json.loads(report_path.read_text())
    head = doc.get("head", {}) or {}
    siren_thr = float(head.get("siren_trigger_threshold", 0.80))
    groups = {g: doc.get(g, {}) for g in
              ("real_sirens", "negative_control_clips", "noisy_positive_clips")}

    provider = P_sirenProvider(report_path=report_path)
    clips: dict[str, dict] = {}
    for grp in ("real_sirens", "noisy_positive_clips", "negative_control_clips"):
        for clip_id, entry in groups[grp].items():
            n = int(entry.get("n_frames", 0))
            base, real = _series(provider, clip_id, n, args.baseline_constant)
            cond = _condition(clip_id, groups)
            clips[clip_id] = {
                "condition": cond,
                "group": grp,
                "n_frames": n,
                "baseline": _stats(base, siren_thr=siren_thr),
                "measured": _stats(real, siren_thr=siren_thr),
                "head_lloc_accuracy": head.get("leave_one_clip_out_per_clip_accuracy",
                                               {}).get(clip_id),
            }

    conditions: dict[str, dict] = {}
    for cid, c in clips.items():
        agg = conditions.setdefault(c["condition"],
                                    {"clips": [], "baseline": [], "measured": []})
        agg["clips"].append(cid)
        agg["baseline"].append(c["baseline"]["siren_trigger_rate"])
        agg["measured"].append(c["measured"]["siren_trigger_rate"])
    for cond, agg in conditions.items():
        agg["baseline_trigger_rate"] = round(statistics.fmean(agg.pop("baseline")), 4)
        agg["measured_trigger_rate"] = round(statistics.fmean(agg.pop("measured")), 4)
        agg["measured_is_triggered_rate"] = round(
            statistics.fmean([clips[cid]["measured"]["is_triggered_rate"]
                              for cid in agg["clips"]]), 4)

    report = {
        "schema": "abl-report-sb/v1",
        "method": "WavLM-base-plus L2-logistic head (measured p_siren per frame)",
        "baseline": {"name": "mock_constant", "constant": args.baseline_constant},
        "trigger_rule": {
            "siren_trigger_threshold": siren_thr,
            "hysteresis_hold_frames": args.hold_frames,
            "head_accuracy": head.get("accuracy"),
            "head_confusion": head.get("confusion"),
        },
        "clips": clips,
        "conditions": conditions,
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"siren trigger >= {siren_thr:.2f}, FSM hold {args.hold_frames}f, "
          f"head acc {head.get('accuracy')}")
    print(f"{'clip':24s} {'cond':12s} {'grp':22s} base_trig meas_trig  active  "
          f"p90  max   LLOC")
    for cid, c in clips.items():
        m = c["measured"]
        lloc = c.get("head_lloc_accuracy")
        print(f"{cid:24s} {c['condition']:12s} {c['group']:22s} "
              f"{c['baseline']['siren_trigger_rate']:9.3f} "
              f"{m['siren_trigger_rate']:9.3f} {m['is_triggered_rate']:7.3f} "
              f"{m['p90']:5.3f} {m['max']:5.3f} "
              f"{('-' if lloc is None else f'{lloc:.3f}')}")
    print("condition aggregates (siren trigger rate):")
    for cond, agg in conditions.items():
        print(f"  {cond:12s} baseline {agg['baseline_trigger_rate']:.4f}  "
              f"measured {agg['measured_trigger_rate']:.4f}  "
              f"active {agg['measured_is_triggered_rate']:.4f}  clips {agg['clips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())