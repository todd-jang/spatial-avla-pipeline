"""Gradio demo app for the verified siren-detection sequence.

The app is a viewer over outputs/demo_sequence.json (derived from MEASURED
p_siren data by scripts/build_demo_sequence.py), NOT an open-ended live
pipeline demo. Every preset segment is:

  * driven statefully through GraphBridge.invoke_frame_stateful (S_B's
    hysteresis FSM is the behaviour under test; the old stateless
    invoke_frame cold-starts the FSM every frame and cannot reproduce the
    manifest envelope),
  * given its real measured clip_id (audio-only real sirens / video clips),
  * asserted with assert_segment() against its manifest-encoded expectation
    (must_be_quiet / must_fire_within_window) -- the verdict is drawn onto
    the final streamed frame, PASS or FAIL,
  * shown with the overlay HUD from src.app.multiview (p_siren badge, DoA
    from the clip's measured audio, S_B hysteresis state).

Choosing "Full verified sequence" replays every segment in programmatic
order; each segment is independently asserted. Custom video uploads are
supported as a manual probe but carry NO assertion -- they are not part of
the verified sequence and the UI says so.
"""
from __future__ import annotations

import time

import cv2
import gradio as gr

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
from src.app.multiview import process_multi_view_frame  # noqa: E402
from src.core.bridge import GraphBridge  # noqa: E402

bridge = GraphBridge(fast_mode=True)
MANIFEST = load_manifest()
SEGMENTS = segments_by_manifest(MANIFEST)

FULL_SEQUENCE = "Full verified sequence"
CUSTOM_UPLOAD = "Custom video upload (no assertion)"
SEG_ITEMS = [f"{s['order']:02d} | {s['clip_id']} | {s['expected']} | {s['label']}"
             for s in SEGMENTS]
SEG_MAP = dict(zip(SEG_ITEMS, SEGMENTS))
PRESET_CHOICES = [FULL_SEQUENCE, CUSTOM_UPLOAD] + SEG_ITEMS


def _overlay_verdict(tile, ok, msg):
    cv2.rectangle(tile, (0, tile.shape[0] - 44), (tile.shape[1], tile.shape[0]),
                  (20, 120, 20) if ok else (40, 20, 20), -1)
    cv2.putText(tile, f"VERDICT: {'PASS' if ok else 'FAIL'}  {msg}",
                (12, tile.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    return tile


def _stream_state(img, frame_idx, res, doa):
    return {"event_trigger": res.get("event_trigger", {}) or {},
            "vlm_payload": res.get("vlm_payload", {}) or {},
            "plan": res.get("plan", {}) or {},
            "meta_exec": res.get("meta_exec", {}) or {},
            "percept": res.get("percept", {}) or {},
            "doa": doa}


def stream_segment(seg: dict, max_frames: int = 0):
    """Yield RGB tiles for one manifest segment, asserting its expectation."""
    clip_id = seg["clip_id"]
    n_target = int(seg["demo_n_frames"])
    if max_frames:
        n_target = min(n_target, max_frames)
    frames = resolve_frames(seg, n_target)
    n = len(frames) if frames is not None else n_target
    if n <= 0:
        raise gr.Error(f"segment {clip_id}: zero frames to stream")

    wav_path, gt_track = segment_audio(seg)
    doa = doa_series(wav_path, gt_track, n)

    prev, active_frames, first_raw = None, 0, -1
    last_tile = None
    for i in range(n):
        img = cv2.imread(str(frames[i])) if frames is not None else placeholder_frame(seg["label"])
        if img is None:
            raise gr.Error(f"unreadable frame for {clip_id}")
        res = bridge.invoke_frame_stateful(i, ts_ms=i * (1000.0 / FPS),
                                           clip_id=clip_id, prev_state=prev)
        prev = res
        ev = res.get("event_trigger", {}) or {}
        if ev.get("raw_trigger") and first_raw < 0:
            first_raw = i
        if ev.get("is_triggered"):
            active_frames += 1
        tile = process_multi_view_frame(img, i, _stream_state(img, i, res, doa[i]))
        lat = float((res.get("meta_exec") or {}).get("lat_ms") or 0.0)
        cv2.putText(tile, f"LAT:{lat:.1f}ms", (tile.shape[1] - 170, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        last_tile = tile
        yield cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        time.sleep(0.005)

    ok, msg = assert_segment(
        seg, {"active_frames": active_frames,
              "first_raw_trigger_frame": first_raw})
    yield cv2.cvtColor(_overlay_verdict(last_tile.copy(), ok, msg),
                       cv2.COLOR_BGR2RGB)


def stream_upload(vp, max_frames: int = 0):
    """Manual probe over an uploaded video; NOT asserted (not in the manifest)."""
    if vp is None:
        raise gr.Error("Upload a video to probe, or pick a verified preset")
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise gr.Error("Could not open uploaded video")
    frame_idx, start = 0, time.time() * 1000.0
    prev = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
        fr = cv2.resize(frame, (480, 270))
        res = bridge.invoke_frame_stateful(frame_idx, ts_ms=start + frame_idx * 33.333,
                                           clip_id="custom_upload", prev_state=prev)
        prev = res
        tile = process_multi_view_frame(fr, frame_idx, res)
        lat = float((res.get("meta_exec") or {}).get("lat_ms") or 0.0)
        cv2.putText(tile, f"LAT:{lat:.1f}ms", (tile.shape[1] - 170, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        yield cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        frame_idx += 1
        time.sleep(0.005)
    cap.release()


def run(choice, vp, max_frames):
    if choice == FULL_SEQUENCE:
        for seg in SEGMENTS:
            yield from stream_segment(seg, max_frames)
    elif choice == CUSTOM_UPLOAD:
        yield from stream_upload(vp, max_frames)
    elif choice in SEG_MAP:
        yield from stream_segment(SEG_MAP[choice], max_frames)
    else:
        raise gr.Error(f"unknown preset: {choice!r}")


def build():
    with gr.Blocks(title="Spatial-AVLA Verified Siren Demo") as demo:
        gr.Markdown("## Verified siren-detection sequence (outputs/demo_sequence.json)")
        with gr.Row():
            preset = gr.Dropdown(PRESET_CHOICES, value=FULL_SEQUENCE,
                                 label="Preset (each segment is asserted)")
            vp = gr.Video(label="Custom probe (no assertion)", sources=["upload"])
            mf = gr.Slider(0, 1500, value=0, step=1,
                           label="Max frames per segment (0 = full manifest window)")
        oi = gr.Image(label="Stream", streaming=True, type="numpy")
        gr.Button("Run").click(run, [preset, vp, mf], oi, show_progress=False)
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()