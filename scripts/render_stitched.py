#!/usr/bin/env python3
"""Headless render of the stitched 2x2 comparison view (todo 8 / demo MVP).

Drives the real LangGraph pipeline once per frame through GraphBridge and
renders each result with src.app.multiview.process_multi_view_frame.

SEQUENCE MODE (default): reads outputs/demo_sequence.json (derived from MEASURED
p_siren data by scripts/build_demo_sequence.py) and plays every segment in its
programmatic order. Each rendered clip is then checked with
assert_segment() against the manifest-encoded expectation -- a clip that was
measured QUIET must produce zero active frames, a clip measured TRIGGER must
fire the FSM -- and the script exits 1 if any segment contradicts its own
manifest entry. Nothing is hand-picked: the sequence is the measured envelope.

Frames are threaded statefully via GraphBridge.invoke_frame_stateful, because
S_B's hysteresis is THE behaviour under test: a per-frame cold start (the old
stateless invoke_frame) resets the FSM every frame and cannot reproduce the
trigger hold that demo_sequence.json was derived from.

  outputs/rendered_demos/stitched_<clip_id>.mp4   1920x1080 @ 30fps
  outputs/poc_report.json                          rendered_demos aggregated

The DoA overlay comes from the clip's measured audio (video clips:
audio_siren_stereo_16k.wav; audio-only real sirens: exports/real_sirens/
<clip>_stereo_16k.wav) through the real estimator per frame, so the HUD shows
measured directions, never invented ones. Missing audio or a missing manifest
is a loud failure, never a silent empty mp4.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, ROOT.as_posix())

from src.app.demo_support import (  # noqa: E402
    FPS,
    assert_segment,
    doa_series,
    load_manifest,
    placeholder_frame,
    resolve_frames,
    segment_audio,
    segments_by_manifest,
)
from src.app.multiview import TILE_H, TILE_W, process_multi_view_frame  # noqa: E402
from src.core.bridge import GraphBridge  # noqa: E402

OUT_W, OUT_H = TILE_W * 2, TILE_H * 2


def mux_audio(out_path: Path, wav_path: Path) -> str:
    """Add the clip's measured audio track to the rendered mp4, in place.

    OpenCV's mp4v writer emits video-only files, so the stitched demo plays
    silent. Muxing the same wav the DoA/p_siren overlays were measured from
    keeps the sound exactly aligned with the HUD (same sample 0 clock). If
    the wav is shorter than the video (audio-only real sirens), the video
    keeps its full length and the track ends naturally.

    Returns "ok" | "no-ffmpeg" | "failed:<msg>". Never raises: a silent
    video is logged, not a crash -- the frame content is the deliverable.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "no-ffmpeg"
    tmp = out_path.with_suffix(".mux.mp4")
    cmd = [ffmpeg, "-y", "-i", str(out_path), "-i", str(wav_path),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.exists():
        return f"failed:{proc.stderr.strip()[-120:]}"
    tmp.replace(out_path)
    return "ok"


def render_segment(bridge: GraphBridge, seg: dict, out_dir: Path,
                   max_frames: int, siren_dir: Path) -> dict:
    clip_id = seg["clip_id"]
    n_target = int(seg["demo_n_frames"])
    if max_frames:
        n_target = min(n_target, max_frames)

    frames = resolve_frames(seg, n_target)
    n = len(frames) if frames is not None else n_target
    if n <= 0:
        raise RuntimeError(f"segment {clip_id}: zero frames to render")

    wav_path, gt_track = segment_audio(seg)
    doa = doa_series(wav_path, gt_track, n)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stitched_{clip_id}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {out_path}")

    latencies, t_last, all_ok, n_err = [], 0.0, True, 0
    active_frames, first_raw = 0, -1
    prev = None
    for i in range(n):
        img = cv2.imread(str(frames[i])) if frames is not None else placeholder_frame(seg["label"])
        if img is None:
            raise RuntimeError(f"unreadable frame {frames[i] if frames else clip_id}")
        res = bridge.invoke_frame_stateful(i, ts_ms=i * (1000.0 / FPS),
                                           clip_id=clip_id, prev_state=prev)
        prev = res
        me = res.get("meta_exec", {}) or {}
        lat = me.get("lat_ms")
        if lat is not None:
            latencies.append(float(lat))
        t_last = float(me.get("poc_completed_at_ms") or t_last)
        if me.get("poc_all_ok") is not True:
            all_ok = False
        n_err += len(me.get("errors", []) or [])

        ev = res.get("event_trigger", {}) or {}
        if ev.get("raw_trigger") and first_raw < 0:
            first_raw = i
        if ev.get("is_triggered"):
            active_frames += 1

        state = {"event_trigger": ev, "vlm_payload": res.get("vlm_payload", {}) or {},
                 "plan": res.get("plan", {}) or {}, "meta_exec": me, "doa": doa[i],
                 "percept": res.get("percept", {}) or {}}
        tile = process_multi_view_frame(img, i, state)
        if tile.shape[:2] != (OUT_H, OUT_W):
            raise RuntimeError(f"tile geometry drifted to {tile.shape}")
        writer.write(tile)
    writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"{out_path} is empty; refusing to report success")

    audio_track = mux_audio(out_path, wav_path)

    latencies.sort()
    p95 = latencies[max(0, int(round(0.95 * (len(latencies) - 1))))] if latencies else None
    ok, verdict_msg = assert_segment(
        seg, {"active_frames": active_frames,
              "first_raw_trigger_frame": first_raw})

    return {
        "clip_id": clip_id, "label": seg.get("label", clip_id),
        "mp4": str(out_path.relative_to(ROOT)),
        "frames": n, "demanded_frames": n_target, "width": OUT_W, "height": OUT_H,
        "fps": FPS, "bytes": out_path.stat().st_size,
        "latency_p50_ms": round(latencies[len(latencies) // 2], 3) if latencies else None,
        "latency_p95_ms": round(p95, 3) if p95 is not None else None,
        "t_last_ms": round(t_last, 3), "all_ok": all_ok, "errors": n_err,
        "expected": seg.get("expected"), "active_frames": active_frames,
        "first_raw_trigger_frame": first_raw,
        "verdict": "PASS" if ok else "FAIL", "verdict_msg": verdict_msg,
        "has_video": bool(seg.get("has_video")),
        "doa_source": f"measured ({wav_path.name})",
        "audio_track": audio_track}


def render_sequence(bridge: GraphBridge, manifest: dict, out_dir: Path,
                    max_frames: int, siren_dir: Path) -> list[dict]:
    infos = []
    for seg in segments_by_manifest(manifest):
        info = render_segment(bridge, seg, out_dir, max_frames, siren_dir)
        infos.append(info)
        print(f"  {info['verdict']} {info['clip_id']:28s} active={info['active_frames']}"
              f"/{info['frames']} first_raw=f{info['first_raw_trigger_frame']} "
              f"-> {info['mp4']}")
    return infos


def render_single(bridge: GraphBridge, clip_id: str, clip_dir: Path, out_dir: Path,
                  max_frames: int, siren_dir: Path) -> dict:
    """Backwards-compatible single-clip render (still stateful, no assertion)."""
    frames = sorted(clip_dir.glob("frames/*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames under {clip_dir / 'frames'}")
    if max_frames:
        frames = frames[:max_frames]
    meta_path = clip_dir / "clip_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    track = meta.get("siren_track")
    gt_track = siren_dir / f"{track}.json" if track else None
    wav = clip_dir / "audio" / "audio_siren_stereo_16k.wav"
    doa = doa_series(wav, gt_track, len(frames))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stitched_{clip_id}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {out_path}")

    latencies, t_last, all_ok, n_err = [], 0.0, True, 0
    prev = None
    for i, fpath in enumerate(frames):
        img = cv2.imread(str(fpath))
        if img is None:
            raise RuntimeError(f"unreadable frame {fpath}")
        res = bridge.invoke_frame_stateful(i, ts_ms=i * (1000.0 / FPS),
                                           clip_id=clip_id, prev_state=prev)
        prev = res
        me = res.get("meta_exec", {}) or {}
        lat = me.get("lat_ms")
        if lat is not None:
            latencies.append(float(lat))
        t_last = float(me.get("poc_completed_at_ms") or t_last)
        if me.get("poc_all_ok") is not True:
            all_ok = False
        n_err += len(me.get("errors", []) or [])

        state = {"event_trigger": res.get("event_trigger", {}) or {},
                 "vlm_payload": res.get("vlm_payload", {}) or {},
                 "plan": res.get("plan", {}) or {}, "meta_exec": me, "doa": doa[i],
                 "percept": res.get("percept", {}) or {}}
        tile = process_multi_view_frame(img, i, state)
        if tile.shape[:2] != (OUT_H, OUT_W):
            raise RuntimeError(f"tile geometry drifted to {tile.shape}")
        writer.write(tile)
    writer.release()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"{out_path} is empty; refusing to report success")

    audio_track = mux_audio(out_path, wav)

    latencies.sort()
    p95 = latencies[max(0, int(round(0.95 * (len(latencies) - 1))))] if latencies else None
    return {"clip_id": clip_id, "mp4": str(out_path.relative_to(ROOT)),
            "frames": len(frames), "width": OUT_W, "height": OUT_H, "fps": FPS,
            "bytes": out_path.stat().st_size,
            "latency_p50_ms": round(latencies[len(latencies) // 2], 3) if latencies else None,
            "latency_p95_ms": round(p95, 3) if p95 is not None else None,
            "t_last_ms": t_last, "all_ok": all_ok, "errors": n_err,
            "siren_track": track, "doa_source": f"measured ({wav.name})",
            "audio_track": audio_track}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", nargs="?", const=str(ROOT / "outputs/demo_sequence.json"),
                    default=None,
                    help="path to demo_sequence.json; plays every segment in order")
    ap.add_argument("--clip", default="C01")
    ap.add_argument("--clip-dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="outputs/rendered_demos")
    ap.add_argument("--report", default="outputs/poc_report.json")
    ap.add_argument("--siren-dir", default="exports/synthetic/siren")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap per-segment frame count (0 = manifest window)")
    args = ap.parse_args(argv)

    bridge = GraphBridge(fast_mode=True)
    report_path = ROOT / args.report
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = {}

    if args.sequence is not None:
        manifest = load_manifest(Path(args.sequence))
        infos = render_sequence(bridge, manifest, ROOT / args.out_dir,
                                args.max_frames, ROOT / args.siren_dir)
        report["rendered_demos"] = infos
        fails = [i for i in infos if i["verdict"] != "PASS"]
        report["demo_render_verdict"] = "PASS" if not fails else "FAIL"
    else:
        clip_dir = Path(args.clip_dir) if args.clip_dir else ROOT / "exports/TL_A" / args.split / args.clip
        info = render_single(bridge, args.clip, clip_dir, ROOT / args.out_dir,
                             args.max_frames, ROOT / args.siren_dir)
        report.update({
            "mAP": report.get("mAP", 0.0),
            "latency": {"p50_ms": info["latency_p50_ms"], "p95_ms": info["latency_p95_ms"]},
            "T_last": info["t_last_ms"],
            "All_OK": info["all_ok"],
            "Errors_Count": info["errors"],
            "rendered_demos": [info],
            "render_source": "synthesised siren audio; downloaded dashcam video",
        })
        infos = [info]
        fails = []

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    if fails:
        print(f"DEMO_SEQUENCE_FAIL {len(fails)}/{len(infos)} segments contradicted "
              f"their manifest expectation:", file=sys.stderr)
        for f in fails:
            print(f"  - {f['clip_id']}: {f['verdict_msg']}", file=sys.stderr)
        return 1
    print(f"STITCHED_OK sequence={len(infos)} clips -> {ROOT / args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())