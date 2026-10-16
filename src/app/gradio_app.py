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

import copy
import time

import cv2
import gradio as gr

from src.app.demo_support import (  # noqa: E402
    FPS,
    assert_segment,
    doa_series,
    load_manifest,
    placeholder_frame,
    read_wav,
    resolve_frames,
    segment_audio,
    segments_by_manifest,
    template_audio_path,
)
from src.app.multiview import process_multi_view_frame  # noqa: E402
from src.app.video_stream import (  # noqa: E402
    DEFAULT_ANALYSIS_FPS,
    stream_uploaded_video,
)
from src.app.live_monitor import (  # noqa: E402
    LiveObjectDetector,
    live_metrics,
    microphone_level,
    microphone_metrics,
    render_live_frame,
)
from src.app.mpc_simulation import compute_fixed_demo_mpc  # noqa: E402
from src.app.qwen_action import QwenActionAdvisor  # noqa: E402
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
live_detector = LiveObjectDetector()


def process_live_frame(rgb_frame, session, analysis_fps, max_frames, advisor):
    if rgb_frame is None:
        return gr.skip(), session, advisor.status()
    session = dict(session or {})
    session.setdefault("analyzed_frames", 0)
    session.setdefault("last_analysis_at", 0.0)
    session.setdefault("frame_idx", 0)
    session.setdefault("prev_state", None)
    session.setdefault("pipeline_triggered", False)
    session.setdefault("risk", 0.0)
    session.setdefault("labels", [])

    status = (
        advisor.update(
            session["pipeline_triggered"], session["risk"], session["labels"]
        )
        if session["analyzed_frames"]
        else advisor.status()
    )
    if max_frames and session["analyzed_frames"] >= int(max_frames):
        return gr.skip(), session, f"{status} | frame limit reached"
    now = time.monotonic()
    if now - session["last_analysis_at"] < 1.0 / float(analysis_fps):
        return gr.skip(), session, status

    session["last_analysis_at"] = now
    start = time.perf_counter()
    try:
        detections, risk = live_detector.analyze(rgb_frame)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise gr.Error(f"Live camera detection failed: {exc}") from exc
    elapsed = now - session.setdefault("started_at", now)
    frame_idx = max(int(session["frame_idx"]) + 1, round(elapsed * 30))
    res = bridge.invoke_frame_stateful(
        frame_idx,
        ts_ms=time.time() * 1000.0,
        clip_id="live_camera",
        prev_state=session["prev_state"],
        vision_anomaly_score=risk,
    )
    res = dict(res)
    res["mpc_simulation"] = compute_fixed_demo_mpc()
    meta = res.get("meta_exec", {}) or {}
    if meta.get("errors"):
        raise gr.Error(f"Live graph pipeline failed: {meta['errors']}")
    details = (res.get("event_trigger", {}) or {}).get("details", {})
    # No live siren classifier is connected; keep its score visibly unavailable.
    details["p_siren"] = None
    details["siren_score_mae"] = None
    meta.setdefault("score_status", {})["siren"] = "not_connected"
    triggered = bool((res.get("event_trigger") or {}).get("is_triggered"))
    labels = [detection["label"] for detection in detections]
    status = advisor.update(triggered, risk, labels)
    latency_ms = (time.perf_counter() - start) * 1000.0
    rendered = render_live_frame(
        rgb_frame,
        detections,
        risk,
        frame_idx,
        latency_ms,
        triggered,
    )
    session.update(
        analyzed_frames=session["analyzed_frames"] + 1,
        frame_idx=frame_idx,
        prev_state=res,
        pipeline_triggered=triggered,
        risk=risk,
        labels=labels,
    )
    return rendered, session, status, live_metrics(
        res, detections, risk, latency_ms, rgb_frame.shape
    )


def process_microphone_chunk(audio):
    try:
        return microphone_level(audio), microphone_metrics(audio)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"Invalid microphone chunk: {exc}") from exc


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


def render_uploaded_video_frame(frame, frame_idx, graph_res):
    """Render custom video with unavailable non-inferred modalities marked honestly."""
    display_state = copy.deepcopy(graph_res)
    meta = display_state.setdefault("meta_exec", {})
    meta.setdefault("score_status", {}).update(
        rtdetr="not_connected",
        mae="not_connected",
        siren="not_connected",
        siren_template="not_connected",
    )
    details = display_state.setdefault("event_trigger", {}).setdefault("details", {})
    for key in (
        "vision_anomaly_score",
        "vision_anomaly_ma",
        "vision_anomaly_mae",
        "vision_anomaly_mae_ma",
        "p_siren",
        "p_siren_ma5",
        "p_siren_delta",
        "siren_score_mae",
        "siren_score_template",
        "siren_detected_mae",
        "siren_strength_mae",
        "audio_confidence",
        "doa_change_rate_dps_ma",
        "lead_lost",
        "lane_lost_count",
    ):
        details[key] = None
    display_state["doa"] = {}
    audio = display_state.setdefault("audio", {})
    audio["siren"] = {"detected": None, "confidence": None, "type": "not_connected"}
    audio["doa"] = {}
    percept = display_state.setdefault("percept", {})
    percept["boxes"] = None
    percept["has_lead"] = None
    return process_multi_view_frame(frame, frame_idx, display_state)


def stream_segment(
    seg: dict, max_frames: int = 0, audio_source: str = "with_siren"
):
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
    ast_audio_path = template_audio_path(seg, audio_source)
    doa = doa_series(wav_path, gt_track, n)

    prev, active_frames, first_raw = None, 0, -1
    last_tile = None
    target_dt = 1.0 / FPS
    for i in range(n):
        t_start = time.perf_counter()
        img = cv2.imread(str(frames[i])) if frames is not None else placeholder_frame(seg["label"])
        if img is None:
            raise gr.Error(f"unreadable frame for {clip_id}")
        res = bridge.invoke_frame_stateful(i, ts_ms=i * (1000.0 / FPS),
                                           clip_id=clip_id, prev_state=prev,
                                           audio_path=(
                                               str(ast_audio_path)
                                               if ast_audio_path is not None
                                               else ""
                                           ))
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
        cv2.putText(tile, f"FRAME: {i + 1}/{n} ({(i + 1) / FPS:.1f}s)", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        last_tile = tile
        # 1280x720 해상도로 리사이즈하여 브라우저 버퍼링/프리징 방지 및 부드러운 30 FPS 스트리밍
        disp_tile = cv2.resize(tile, (1280, 720), interpolation=cv2.INTER_LINEAR)
        yield cv2.cvtColor(disp_tile, cv2.COLOR_BGR2RGB)
        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.005, target_dt - elapsed))

    ok, msg = assert_segment(
        seg, {"active_frames": active_frames,
              "first_raw_trigger_frame": first_raw})
    verdict_tile = _overlay_verdict(last_tile.copy(), ok, msg)
    disp_verdict = cv2.resize(verdict_tile, (1280, 720), interpolation=cv2.INTER_LINEAR)
    yield cv2.cvtColor(disp_verdict, cv2.COLOR_BGR2RGB)
    time.sleep(1.0)


def stream_upload(vp, max_frames: int = 0,
                  analysis_fps: float = DEFAULT_ANALYSIS_FPS,
                  max_duration_seconds: float = 5 * 60):
    """Manual probe over an uploaded video; NOT asserted (not in the manifest)."""
    if vp is None:
        raise gr.Error("Upload a video to probe, or pick a verified preset")
    yield from stream_uploaded_video(
        vp,
        bridge,
        render_uploaded_video_frame,
        max_frames,
        analysis_fps,
        error_type=gr.Error,
        max_duration_seconds=max_duration_seconds,
    )


def demo_audio(choice, audio_source: str, max_frames: int):
    """Return the selected demo soundtrack as a Gradio (sample rate, audio) value."""
    import numpy as np

    if choice == CUSTOM_UPLOAD:
        return None
    if choice == FULL_SEQUENCE:
        segments = SEGMENTS
    elif choice in SEG_MAP:
        segments = [SEG_MAP[choice]]
    else:
        raise gr.Error(f"unknown preset: {choice!r}")

    chunks = []
    sample_rate = None
    for seg in segments:
        path = (
            template_audio_path(seg, audio_source)
            if seg.get("has_video")
            else segment_audio(seg)[0]
        )
        if path is None:
            raise gr.Error(
                f"no {audio_source} playback audio available for {seg['clip_id']}"
            )
        audio, current_rate = read_wav(path)
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise gr.Error(
                f"demo audio sample rates differ: {sample_rate} and {current_rate}"
            )
        if max_frames:
            sample_count = round(int(max_frames) * current_rate / FPS)
            audio = audio[:sample_count]
        chunks.append(audio)

    try:
        combined = np.concatenate(chunks, axis=0)
    except ValueError as exc:
        raise gr.Error("demo audio channel layouts do not match") from exc
    return sample_rate, combined


def run(
    choice,
    vp,
    max_frames,
    analysis_fps,
    upload_duration_minutes,
    audio_source="with_siren",
):
    if choice == FULL_SEQUENCE:
        segments = SEGMENTS
    elif choice == CUSTOM_UPLOAD:
        for frame in stream_upload(
            vp,
            max_frames=0,
            analysis_fps=analysis_fps,
            max_duration_seconds=float(upload_duration_minutes) * 60,
        ):
            yield frame, None
        return
    elif choice in SEG_MAP:
        segments = [SEG_MAP[choice]]
    else:
        raise gr.Error(f"unknown preset: {choice!r}")

    soundtrack = demo_audio(choice, audio_source, max_frames)
    first_frame = True
    for seg in segments:
        for frame in stream_segment(seg, max_frames, audio_source):
            yield frame, soundtrack if first_frame else gr.skip()
            first_frame = False


def build():
    with gr.Blocks(title="Spatial-AVLA Verified Siren Demo") as demo:
        with gr.Tab("Verified playback"):
            gr.Markdown("## Verified siren-detection sequence (outputs/demo_sequence.json)")
            with gr.Row():
                preset = gr.Dropdown(PRESET_CHOICES, value=FULL_SEQUENCE,
                                     label="Preset (each segment is asserted)")
                vp = gr.Video(label="Custom probe (no assertion)", sources=["upload"])
                mf = gr.Slider(
                    0, 1500, value=0, step=30,
                    label="Max frames per segment (0 = full verified window)",
                )
                analysis_fps = gr.Slider(1, 24, value=DEFAULT_ANALYSIS_FPS, step=1,
                                         label="Upload analysis rate (FPS)")
                upload_duration = gr.Slider(
                    1, 5, value=5, step=1,
                    label="Custom upload maximum realtime inference duration (minutes)",
                )
            audio_source = gr.Dropdown(
                choices=["with_siren", "untouched"],
                value="with_siren",
                label="AST and playback audio source (verified clips only)",
            )
            oi = gr.Image(label="Stream", type="numpy", streaming=True)
            demo_sound = gr.Audio(
                label="Original audio (autoplay)",
                autoplay=True,
                type="numpy",
            )
            gr.Button("Run").click(
                run,
                [preset, vp, mf, analysis_fps, upload_duration, audio_source],
                [oi, demo_sound],
                show_progress=False,
            )

        with gr.Tab("Live read-only monitor"):
            gr.Markdown(
                "## Live sensor monitor\n"
                "**Read-only:** this app has no vehicle or actuator connection. "
                "Camera frames run through the local RT-DETR checkpoint; microphone "
                "input reports signal level only and does not detect sirens. "
                "The MPC steering/braking display is a fixed virtual demo scenario, "
                "not a result derived from camera perception and not an actuator command."
            )
            with gr.Row():
                camera = gr.Image(
                    label="Camera input (allow browser camera access)",
                    sources=["webcam"],
                    type="numpy",
                    streaming=True,
                )
                live_output = gr.Image(label="Live RT-DETR detections", type="numpy")
            with gr.Row():
                live_analysis_fps = gr.Slider(
                    0.5, 1.25, value=1.0, step=0.25,
                    label="Live CPU analysis rate (FPS; measured RT-DETR ~1.5 FPS)",
                )
                live_max_frames = gr.Slider(
                    0, 1800, value=300, step=30,
                    label="Max analyzed frames (0 = unlimited; 300 = about 5 min at 1 FPS)",
                )
            camera_state = gr.State({})
            qwen_advisor = gr.State(QwenActionAdvisor)
            qwen_status = gr.Textbox(
                label="Qwen action inference (advisory only, no vehicle control)",
                value="Qwen read-only advisory: waiting for camera analysis",
                interactive=False,
            )
            live_metrics_output = gr.JSON(label="Quantified live perception")
            camera.stream(
                process_live_frame,
                inputs=[
                    camera, camera_state, live_analysis_fps, live_max_frames,
                    qwen_advisor,
                ],
                outputs=[
                    live_output, camera_state, qwen_status, live_metrics_output,
                ],
                stream_every=0.75,
                concurrency_limit=1,
            )

            microphone = gr.Audio(
                label="Microphone input (signal level only)",
                sources=["microphone"],
                type="numpy",
                streaming=True,
            )
            microphone_status = gr.Textbox(
                label="Microphone status",
                value="Microphone: waiting for audio",
                interactive=False,
            )
            microphone_metrics_output = gr.JSON(label="Quantified microphone input")
            microphone.stream(
                process_microphone_chunk,
                inputs=microphone,
                outputs=[microphone_status, microphone_metrics_output],
                stream_every=0.5,
                concurrency_limit=1,
            )
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()