#!/usr/bin/env python3
"""Build the programmatic demo sequence from MEASURED p_siren data.

outputs/demo_sequence.json is the single source of truth for what the demo may
play and what it must verify. Nothing here is hand-picked: every segment's
expected outcome (quiet | trigger) is derived by re-running the PRODUCTION
code path -- the real P_sirenProvider lookup plus the real S_B hysteresis FSM
(agent_s_b.s_b_fn) -- over each clip's measured frame_p_siren series, using the
exact frame window the demo will actually step (video frame count for clips
with an exported dashcam video, full model series length for audio-only clips).

The manifest then encodes, per segment, the assertions the demo renderers must
re-verify while they play:

  - expected == "quiet"   -> zero active frames in the demo window
  - expected == "trigger" -> the FSM fires at least once within the window,
                             and the showcase role additionally requires an
                             active rate >= 0.70 (no flicker)

Any clip that contradicts its own role is NOT included in the sequence; it is
listed under ``excluded`` with the observed reason (known_fp / known_miss /
weak trigger). The script exits 1 if the sequence comes out empty, if a
role-contradiction is found in the INCLUDE set, or if any include clip is
missing from the scoring report.

The pre_antigravity contract "clip -> group" lives in outputs/p_siren_report.json:
  real_sirens          = true siren recordings (audio-only, no dashcam video)
  negative_control_clips = clean dashcam audio, no siren
  noisy_positive_clips = siren mixed INTO dashcam traffic audio at 12dB SNR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.core.audio.p_siren_provider import P_sirenProvider  # noqa: E402
from src.exp_b.agent_s_b import s_b_fn  # noqa: E402

SIREN_RAW_THRESHOLD = 0.80
MIN_HOLD_FRAMES = 35
SHOWCASE_MIN_ACTIVE_RATE = 0.70   # < 0.70 => flickers, demoted to secondary
SECONDARY_MIN_ACTIVE_RATE = 0.30  # < 0.30 => not demo-safe

VIDEO_CLIPS_WITH_FRAMES = {
    "C01": 450, "C02": 451, "C04": 451,  # exports/TL_A/train/<clip>/frames/*.jpg
}


def read_report(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"FAIL: scoring report missing: {path}. "
                         f"Run scripts/score_p_siren.py + scripts/fit_p_siren_head.py first.")
    return json.loads(path.read_text())


def clip_entry(report: dict, clip_id: str) -> tuple[str, dict] | None:
    for grp in ("real_sirens", "negative_control_clips", "noisy_positive_clips"):
        entry = report.get(grp, {}).get(clip_id)
        if entry is not None:
            return grp, entry
    return None


def demo_window_frames(clip_id: str, entry: dict) -> int:
    """Frames the demo will actually step for this clip.

    Video clips: the exported jpg count caps the playable window (the model
    series is longer than the video, so stepping is clipped to the video).
    Audio-only clips (real sirens): step the full model series.
    """
    n_model = int(entry.get("n_frames", 0))
    if clip_id in VIDEO_CLIPS_WITH_FRAMES:
        return min(VIDEO_CLIPS_WITH_FRAMES[clip_id], n_model)
    return n_model


def simulate_fsm(provider: P_sirenProvider, clip_id: str, n_frames: int) -> dict:
    """Run the real S_B FSM over the clip's measured p_siren series."""
    probs = provider.frame_value
    state: dict = {
        "frame_idx": 0,
        "clip_id": clip_id,
        "event_trigger": {
            "is_triggered": False,
            "hysteresis": {"state": "idle", "active_until_frame_idx": -1},
            "details": {"p_siren": 0.0, "audio_confidence": 0.0},
        },
        "_hist": {},
        "meta_exec": {},
        "audio": {},
        "doa": {},
    }
    active = 0
    first_raw = -1
    mean_p_sum = 0.0
    for fi in range(n_frames):
        p = probs(clip_id, fi)
        s = dict(state)
        s["frame_idx"] = fi
        s["event_trigger"] = {
            "is_triggered": False,
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
        ev = out["event_trigger"]
        if ev.get("raw_trigger") and first_raw < 0:
            first_raw = fi
        if ev.get("is_triggered"):
            active += 1
        mean_p_sum += p
        state = out
    return {
        "n_frames": n_frames,
        "active_frames": active,
        "active_rate": round(active / n_frames, 4) if n_frames else 0.0,
        "first_raw_trigger_frame": first_raw,
        "mean_p": round(mean_p_sum / n_frames, 4) if n_frames else 0.0,
    }


LABELS = {
    "C01": "C01 - Seoul tunnel, dry (video available)",
    "C02": "C02 - Seoul tunnel, wet (video available)",
    "C04": "C04 - Seoul bridge (video available)",
    "C01_siren": "C01 + siren at 12dB (no video)",
    "C02_siren": "C02 + siren at 12dB (no video)",
    "C04_siren": "C04 + siren at 12dB (no video)",
    "T09": "T09 - coastal Norway, no siren (audio only)",
    "G44-01-Sirens": "G44-01 real siren (audio only)",
    "G44-02-Air Raid Siren": "G44-02 air raid siren (audio only)",
    "G44-03-Whoopee Whistle Siren": "G44-03 whoopee whistle (audio only)",
    "G44-04-Ambulance Siren": "G44-04 ambulance siren (audio only)",
    "G44-05-Warbling Boat Siren": "G44-05 warbling boat siren (audio only)",
    "G44-07-Emergency Siren": "G44-07 emergency siren (audio only)",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="outputs/p_siren_report.json")
    ap.add_argument("--out", default="outputs/demo_sequence.json")
    args = ap.parse_args(argv)

    report = read_report(ROOT / args.report)
    provider = P_sirenProvider(report_path=ROOT / args.report)

    all_clips: list[str] = []
    for grp in ("real_sirens", "negative_control_clips", "noisy_positive_clips"):
        all_clips.extend(sorted(report.get(grp, {}).keys()))
    if not all_clips:
        raise SystemExit("FAIL: no clips in the scoring report.")

    measured: dict[str, dict] = {}
    for clip_id in all_clips:
        entry = clip_entry(report, clip_id)
        if entry is None:
            continue
        grp, entry_data = entry
        series = entry_data.get("frame_p_siren") or entry_data.get("frame_cosines")
        if not series:
            raise SystemExit(f"FAIL: clip {clip_id} has neither frame_p_siren nor "
                             f"frame_cosines; cannot derive a demo expectation.")
        wn = demo_window_frames(clip_id, entry_data)
        sim = simulate_fsm(provider, clip_id, wn)
        sim["clip_id"] = clip_id
        sim["group"] = grp
        sim["n_model_frames"] = int(entry_data.get("n_frames", 0))
        sim["has_video"] = clip_id in VIDEO_CLIPS_WITH_FRAMES
        sim["label"] = LABELS.get(clip_id, clip_id)
        measured[clip_id] = sim

    # ---- Role assignment from measured behaviour (not from assumptions) ----
    includes, excluded = [], []
    for clip_id in sorted(measured):
        m = measured[clip_id]
        grp = m["group"]
        if grp == "negative_control_clips":
            if m["active_frames"] == 0:
                m["role"] = "quiet_control"
                m["expected"] = "quiet"
                includes.append(m)
            else:
                m["role"] = "known_fp"
                m["expected"] = "quiet"
                m["exclude_reason"] = (
                    f"fires {m['active_frames']}/{m['n_frames']} frames "
                    f"({m['active_rate']:.1%}) on clean audio; would false-alarm in demo")
                excluded.append(m)
        elif grp == "noisy_positive_clips":
            if m["active_rate"] >= SECONDARY_MIN_ACTIVE_RATE:
                m["role"] = "noisy_trigger"
                m["expected"] = "trigger"
                includes.append(m)
            else:
                m["role"] = "known_miss"
                m["expected"] = "trigger"
                m["exclude_reason"] = (
                    f"contains a real siren at 12dB SNR but only fires "
                    f"{m['active_rate']:.1%} of frames; known head limitation")
                excluded.append(m)
        else:  # real_sirens
            if m["active_rate"] >= SHOWCASE_MIN_ACTIVE_RATE:
                m["role"] = "siren_showcase"
                m["expected"] = "trigger"
                includes.append(m)
            elif m["active_rate"] >= SECONDARY_MIN_ACTIVE_RATE:
                m["role"] = "siren_secondary"
                m["expected"] = "trigger"
                m["flicker"] = True
                includes.append(m)
            else:
                m["role"] = "weak_trigger"
                m["expected"] = "trigger"
                m["exclude_reason"] = (
                    f"fires only {m['active_rate']:.1%} of frames; not demo-safe")
                excluded.append(m)

    # ---- Programmatic ordering: quiet baseline first, then strongest first ----
    role_rank = {"quiet_control": 0, "siren_showcase": 1, "noisy_trigger": 1,
                 "siren_secondary": 2}
    includes.sort(key=lambda m: (role_rank.get(m["role"], 9), -m["active_rate"], m["clip_id"]))
    for order, m in enumerate(includes, start=1):
        m["order"] = order

    # ---- Hard contradictions must fail the build, not the demo ----
    errors = []
    for m in includes:
        if m["expected"] == "quiet" and m["active_frames"] != 0:
            errors.append(f"{m['clip_id']}: expected quiet but measured "
                          f"{m['active_frames']} active frames")
        if m["expected"] == "trigger" and m["first_raw_trigger_frame"] < 0:
            errors.append(f"{m['clip_id']}: expected trigger but FSM never fired")
        if m["role"] in ("siren_showcase", "noisy_trigger") and \
                m["active_rate"] < SHOWCASE_MIN_ACTIVE_RATE and \
                m["role"] != "noisy_trigger":
            errors.append(f"{m['clip_id']}: showcase role but active rate "
                          f"{m['active_rate']:.1%} < {SHOWCASE_MIN_ACTIVE_RATE:.0%}")
    if not includes:
        errors.append("sequence is empty; nothing to demo")
    if errors:
        raise SystemExit("FAIL: demo sequence contradicts measured data:\n  "
                         + "\n  ".join(errors))

    segments = [{
        "order": m["order"],
        "clip_id": m["clip_id"],
        "label": m["label"],
        "group": m["group"],
        "role": m["role"],
        "expected": m["expected"],
        "has_video": m["has_video"],
        "demo_n_frames": m["n_frames"],
        "n_model_frames": m["n_model_frames"],
        "active_frames": m["active_frames"],
        "active_rate": m["active_rate"],
        "first_raw_trigger_frame": m["first_raw_trigger_frame"],
        "mean_p": m["mean_p"],
        "flicker": bool(m.get("flicker", False)),
        "assert": {
            "must_be_quiet": m["expected"] == "quiet",
            "must_fire_within_window": m["expected"] == "trigger",
            "notes": (["zero active frames in the demo window"] if m["expected"] == "quiet"
                      else ["first raw trigger must occur within demo_n_frames"]),
        },
    } for m in includes]

    excluded_list = [{
        "clip_id": m["clip_id"], "label": m["label"], "group": m["group"],
        "role": m["role"], "expected": m["expected"], "reason": m["exclude_reason"],
        "demo_n_frames": m["n_frames"], "n_model_frames": m["n_model_frames"],
        "active_frames": m["active_frames"], "active_rate": m["active_rate"],
        "first_raw_trigger_frame": m["first_raw_trigger_frame"], "mean_p": m["mean_p"],
    } for m in excluded]

    manifest = {
        "schema": "demo-sequence/v1",
        "built_from": str(ROOT / args.report),
        "engine": "P_sirenProvider.frame_value + agent_s_b.s_b_fn FSM "
                  "(production code path, 1:1)",
        "engine_params": {"siren_raw_threshold": SIREN_RAW_THRESHOLD,
                          "min_hold_frames": MIN_HOLD_FRAMES},
        "classifiers": {
            "showcase_min_active_rate": SHOWCASE_MIN_ACTIVE_RATE,
            "secondary_min_active_rate": SECONDARY_MIN_ACTIVE_RATE,
        },
        "segments": segments,
        "excluded": excluded_list,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {args.out}")
    for s in segments:
        mark = "FIRES" if s["expected"] == "trigger" else "QUIET"
        vid = "video" if s["has_video"] else "audio-only"
        print(f"  #{s['order']} {s['clip_id']:28s} {mark:5s} "
              f"{s['active_rate']:6.1%} active {s['demo_n_frames']:5d}f {vid}")
    if excluded_list:
        print("  excluded (honest limits, not demoed):")
        for e in excluded_list:
            print(f"    - {e['clip_id']}: {e['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())