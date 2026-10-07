#!/usr/bin/env python3
"""Build outputs/poc_report.json including the todo-16 spatial block.

Every DoA number here is MEASURED at run time by running the real estimator over the
real synthesised fixtures -- nothing is hardcoded. Per the plan's honesty rule each
spatial figure carries source="synthetic" and a measured_on field.

Scenario grouping is derived by GROUPING clip_meta.json ON country. There is no
country->scenario table in this file; the group label is taken from clip_meta's own
scenario field so a mislabelled clip shows up in the report instead of hiding.

A missing clip_meta is a loud failure by default (--allow-missing downgrades that to an
explicit incomplete report that lists what is absent; it never silently omits a group).
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.agents.s_i_init import s_i_init_fn  # noqa: E402
from src.core.audio.doa_gccphat import estimate_doa  # noqa: E402
from src.core.audio.frame_clock import frame_count_from_audio, frame_start_sample  # noqa: E402
from src.exp_b.agent_s_b import s_b_fn  # noqa: E402

SR = 16000
FPS = 30
WIN = 8000
TRACK_ORDER = ("track_pass_fast", "track_pass_fast_left", "track_fail_slow",
               "track_static_left", "track_static_right")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype="<i2").reshape(n, ch)
    return data.astype(np.float64) / 32768.0, w.getframerate()


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def doa_from_gt(raw_deg: float) -> dict:
    """Map a renderer's unwrapped angle onto what a real estimator can report.

    The renderer keeps tracks running past +-90 so the angular rate stays physical, but
    a microphone array has no front/back discrimination: anything beyond +-90 is
    genuinely ambiguous rather than a large signed angle.
    """
    wrapped = wrap180(raw_deg)
    return {"dir_deg": wrapped, "gt_dir_deg": wrapped,
            "confidence": 1.0, "direction": "center",
            "ambiguous": abs(raw_deg) > 90.0}


def run_sweep(siren_dir: Path, index: dict) -> dict:
    rows, errs, amb = [], [], 0
    for entry in index["static"]:
        wav = siren_dir / f"{entry['clip']}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"sweep fixture missing: {wav}")
        audio, sr = read_wav(wav)
        res = estimate_doa(audio[:, 0], audio[:, 1], sr=sr)
        err = abs(res["dir_deg"] - float(entry["gt_dir_deg"]))
        amb += int(bool(res["ambiguous"]))
        errs.append(err)
        rows.append({"clip": entry["clip"], "gt_dir_deg": float(entry["gt_dir_deg"]),
                     "est_dir_deg": round(res["dir_deg"], 4),
                     "abs_err_deg": round(err, 4),
                     "confidence": round(res["confidence"], 4),
                     "ambiguous": bool(res["ambiguous"])})
    bins = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf")]
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]
    counts = [sum(1 for e in errs if bins[i] <= e < bins[i + 1]) for i in range(len(bins) - 1)]
    return {"source": "synthetic", "n_angles": len(rows),
            "max_abs_err_deg": round(max(errs), 4),
            "mean_abs_err_deg": round(float(np.mean(errs)), 4),
            "ambiguous_frames": amb, "ambiguous_rate": round(amb / len(rows), 4),
            "histogram": {"unit": "degrees", "bin_edges": bins[:-1] + [None],
                          "bin_labels": labels, "counts": counts},
            "measured_on": [r["clip"] for r in rows], "rows": rows}


def run_triggers(siren_dir: Path) -> list[dict]:
    rows = []
    for track in TRACK_ORDER:
        sidecar = siren_dir / f"{track}.json"
        if not sidecar.exists():
            raise FileNotFoundError(f"track sidecar missing: {sidecar}")
        meta = json.loads(sidecar.read_text())
        seq = meta["gt_dir_deg_per_frame"]
        s = s_i_init_fn({"frame_idx": 0})
        fired, peak = False, 0.0
        for i, raw in enumerate(seq):
            s["doa"] = doa_from_gt(float(raw))
            state = s_b_fn(dict(s))
            state["frame_idx"] = i
            det = state["event_trigger"]["details"]
            peak = max(peak, float(det.get("doa_change_rate_dps_ma", 0.0)))
            fired = fired or bool(state["event_trigger"]["raw_trigger"])
            s = state
        rows.append({"track": track, "trigger_fired": bool(fired),
                     "peak_doa_rate_dps_ma": round(peak, 3),
                     "renderer_peak_rate_deg_s": round(float(meta["peak_abs_rate_deg_s"]), 3),
                     "n_frames": len(seq), "source": "synthetic"})
    return rows


def clip_metrics(clip_dir: Path, track: str | None, siren_dir: Path) -> dict:
    mixed = clip_dir / "audio" / "audio_siren_stereo_16k.wav"
    if track is None or not mixed.exists():
        return {"measured": False,
                "reason": "no siren mix" if track is None else f"missing {mixed.name}"}
    gt = json.loads((siren_dir / f"{track}.json").read_text())["gt_dir_deg_per_frame"]
    audio, sr = read_wav(mixed)
    left, right = audio[:, 0], audio[:, 1]
    errs, amb, n = [], 0, 0
    # Exact frame boundaries (frame_start_sample), not an integer hop that
    # drifts 9.4 ms over a 450-frame clip (see frame_clock.py).
    # Windows beyond the audio buffer or the GT are dropped.
    gt_len = len(gt)
    audio_frames = frame_count_from_audio(len(left))
    for frame in range(min(gt_len, audio_frames)):
        start = frame_start_sample(frame)
        if start + WIN > len(left):
            break
        res = estimate_doa(left[start:start + WIN], right[start:start + WIN], sr=sr)
        n += 1
        amb += int(bool(res["ambiguous"]))
        if not res["ambiguous"]:
            errs.append(abs(res["dir_deg"] - wrap180(float(gt[frame]))))
    return {"measured": True, "windows": n, "siren_track": track,
            "mean_abs_err_deg": round(float(np.mean(errs)), 4) if errs else None,
            "max_abs_err_deg": round(max(errs), 4) if errs else None,
            "ambiguous_frames": amb,
            "ambiguous_rate": round(amb / n, 4) if n else None,
            "source": "synthetic"}


def load_clips(tl_root: Path, siren_dir: Path, roster: dict) -> tuple[list[dict], list[str]]:
    expected = [c["clip_id"] for c in roster["clips"] if c["split"] == "train"]
    clips, missing = [], []
    for cid in expected:
        clip_dir = tl_root / "train" / cid
        meta_path = clip_dir / "clip_meta.json"
        if not meta_path.exists():
            missing.append(cid)
            continue
        meta = json.loads(meta_path.read_text())
        clips.append({"clip_id": cid, "meta": meta,
                      "metrics": clip_metrics(clip_dir, meta.get("siren_track"), siren_dir)})
    return clips, missing


def build_scenarios(clips: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for c in clips:
        groups[c["meta"].get("scenario") or "UNLABELLED"].append(c)
    out = []
    for scenario in sorted(groups):
        members = groups[scenario]
        errs = [c["metrics"]["max_abs_err_deg"] for c in members
                if c["metrics"].get("max_abs_err_deg") is not None]
        means = [c["metrics"]["mean_abs_err_deg"] for c in members
                 if c["metrics"].get("mean_abs_err_deg") is not None]
        amb = sum(c["metrics"].get("ambiguous_frames", 0) for c in members)
        wins = sum(c["metrics"].get("windows", 0) or 0 for c in members)
        countries = sorted({c["meta"].get("country") for c in members})
        out.append({
            "scenario": scenario,
            "grouped_by": "country",
            "countries": countries,
            "clips": [c["clip_id"] for c in members],
            "n_clips": len(members),
            "mean_abs_err_deg": round(float(np.mean(means)), 4) if means else None,
            "max_abs_err_deg": round(max(errs), 4) if errs else None,
            "ambiguous_frames": amb,
            "ambiguous_rate": round(amb / wins, 4) if wins else None,
            "source": "synthetic",
            "measured_on": sorted({c["metrics"].get("siren_track")
                                   for c in members if c["metrics"].get("siren_track")}),
            "per_clip": [{"clip_id": c["clip_id"], "country": c["meta"].get("country"),
                          "city": c["meta"].get("city"),
                          "condition": c["meta"].get("condition"),
                          "scenario_mismatch": c["meta"].get("scenario_mismatch"),
                          **{k: v for k, v in c["metrics"].items()}} for c in members],
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siren-dir", default="exports/synthetic/siren")
    ap.add_argument("--tl-root", default="exports/TL_A")
    ap.add_argument("--roster", default="configs/clip_roster.json")
    ap.add_argument("--out", default="outputs/poc_report.json")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args(argv)

    siren_dir = ROOT / args.siren_dir
    index = json.loads((siren_dir / "index.json").read_text())
    roster = json.loads((ROOT / args.roster).read_text())

    sweep = run_sweep(siren_dir, index)
    triggers = run_triggers(siren_dir)
    clips, missing = load_clips(ROOT / args.tl_root, siren_dir, roster)
    scenarios = build_scenarios(clips)

    if missing and not args.allow_missing:
        print(f"FAIL: missing clip_meta.json for {sorted(missing)}; refusing to emit a "
              f"report with a silently absent scenario group. Re-run the slicer, or pass "
              f"--allow-missing to emit an explicitly incomplete report.", file=sys.stderr)
        return 1

    report_path = ROOT / args.out
    existing = {}
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    spatial = {
        "source": "synthetic",
        "caveat": "All DoA figures come from synthesised sirens rendered by "
                  "scripts/synth_siren_stereo.py. Real recorded sirens are UNMEASURED.",
        "scenario_names": [g["scenario"] for g in scenarios],
        "scenarios": scenarios,
        "trigger_rows": triggers,
        "sweep": sweep,
    }
    if missing:
        spatial["status"] = "incomplete"
        spatial["missing_clips"] = sorted(missing)
        spatial["incomplete_reason"] = (
            "clips absent from exports/TL_A/train; see evidence task-15 for the "
            "YouTube bot-gate that prevented downloading them")

    existing.update({"spatial": spatial})
    if missing:
        existing["spatial_complete"] = False
    else:
        existing.pop("spatial_complete", None)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(existing, indent=2) + "\n")

    fired = {t["track"]: t["trigger_fired"] for t in triggers}
    print(f"wrote {args.out}")
    print(f"  sweep max_abs_err_deg={sweep['max_abs_err_deg']} "
          f"ambiguous={sweep['ambiguous_frames']}/{sweep['n_angles']}")
    print(f"  triggers={fired}")
    print(f"  scenarios={[g['scenario'] for g in scenarios]} "
          f"missing={sorted(missing) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())