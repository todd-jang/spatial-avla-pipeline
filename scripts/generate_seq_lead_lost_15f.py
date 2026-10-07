import json
from pathlib import Path
from datetime import datetime, timezone


def iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    src = Path("exports/TL_A/train/C01/frames")
    if not src.exists() or not list(src.glob("frame_*.jpg")):
        raise FileNotFoundError("Need Priority Slice C01 frames")
    out = Path("golden/fixtures/seq_lead_lost_15f")
    out.mkdir(parents=True, exist_ok=True)
    n = 15
    fps = 30
    dt = 1000.0 / fps
    base = 1759500000000.0
    (out / "meta.json").write_text(json.dumps({"sequence_id": "seq_lead_lost_15f", "n_frames": n, "fps": fps, "delta_ts_ms": dt, "monotonic": True, "generated_at": iso()}, indent=2))
    for i in range(n):
        ts = base + i * dt
        pkt = {
            "version": "1.0",
            "clip_id": "C01",
            "frame_id": i + 1,
            "frame_idx": i,
            "ts_ms": ts,
            "timestamp_ms": ts,
            # contract-required at top level; shapes from mocks/mock_base.json
            "audio": {
                "siren": {"detected": False, "confidence": 0, "type": "none"},
                # gt_dir_deg -1 = unknown sentinel (see contract); this fixture carries no audio.
                "doa": {"dir_deg": 0.0, "gt_dir_deg": -1.0, "confidence": 0.0, "direction": "center"},
                "audio_features_b64": "",
            },
            "vlm_payload": {
                "trigger": False,
                "mode": "symbolic",
                "prompt_context": "",
            },
            "event_trigger": {
                "details": {
                    "vision_anomaly_score": 0.2 + (0.05 * i if i < 8 else 0.0),
                    "lane_lost_count": i if 3 <= i < 6 else 0,
                    "lead_lost": i >= 12,
                },
                "hysteresis": {"min_hold_frames": 35, "state": "idle"},
            },
            "meta_exec": {"errors": []},
        }
        (out / f"frame_{i:03d}.json").write_text(json.dumps(pkt, indent=2))
    print("OK")


if __name__ == "__main__":
    main()
