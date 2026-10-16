"""Realtime-paced playback with lower-rate CPU graph analysis."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np

MAX_STREAM_FPS = 24.0
DEFAULT_ANALYSIS_FPS = 10.0
STREAM_SIZE = (960, 540)
FALLBACK_SOURCE_FPS = 30.0
MAX_UPLOAD_DURATION_SECONDS = 5 * 60


def stream_uploaded_video(
    video_path: str,
    bridge,
    render_frame: Callable,
    max_frames: int = 0,
    analysis_fps: float = DEFAULT_ANALYSIS_FPS,
    error_type: type[Exception] = ValueError,
    max_duration_seconds: float = MAX_UPLOAD_DURATION_SECONDS,
    analyze_frame: Callable | None = None,
) -> Iterator[np.ndarray]:
    """Render smooth video frames while refreshing graph state at analysis_fps."""
    if analysis_fps <= 0:
        raise error_type("Analysis FPS must be greater than zero")
    if not 0 < max_duration_seconds <= MAX_UPLOAD_DURATION_SECONDS:
        raise error_type(
            f"Maximum upload duration must be between 0 and "
            f"{MAX_UPLOAD_DURATION_SECONDS} seconds"
        )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise error_type("Could not open uploaded video")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not 1.0 <= source_fps <= 120.0:
        source_fps = FALLBACK_SOURCE_FPS
    display_fps = min(source_fps, MAX_STREAM_FPS)
    display_interval = 1.0 / display_fps
    analysis_interval = 1.0 / min(float(analysis_fps), display_fps)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_wall_ms = time.time() * 1000.0
    started_at = time.monotonic()
    deadline = started_at
    frame_idx = 0
    analyzed_idx = -1
    next_analysis_at = 0.0
    prev = None
    latest_res = None

    try:
        while cap.isOpened():
            if max_frames and frame_idx >= max_frames:
                break
            if frame_idx / source_fps >= max_duration_seconds:
                break
            if time.monotonic() - started_at >= max_duration_seconds:
                break
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / source_fps
            if latest_res is None or video_time >= next_analysis_at:
                inference = (
                    analyze_frame(frame, frame_idx, video_time)
                    if analyze_frame is not None
                    else None
                )
                latest_res = bridge.invoke_frame_stateful(
                    frame_idx,
                    ts_ms=start_wall_ms + video_time * 1000.0,
                    clip_id="custom_upload",
                    prev_state=prev,
                )
                if inference is not None:
                    for key, value in inference.items():
                        if key == "meta_exec":
                            latest_res[key] = {
                                **latest_res.get(key, {}),
                                **value,
                            }
                        else:
                            latest_res[key] = value
                prev = latest_res
                analyzed_idx = frame_idx
                next_analysis_at = video_time + analysis_interval

            source_frame = cv2.resize(
                frame, (480, 270), interpolation=cv2.INTER_AREA
            )
            tile = render_frame(source_frame, frame_idx, latest_res)
            latency_ms = float((latest_res.get("meta_exec") or {}).get("lat_ms") or 0.0)
            analysis_age = (frame_idx - analyzed_idx) / source_fps
            cv2.putText(
                tile,
                f"LAT:{latency_ms:.1f}ms  ANALYSIS AGE:{analysis_age:.2f}s",
                (tile.shape[1] - 390, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            duration = (
                min(source_frame_count / source_fps, max_duration_seconds)
                if source_frame_count
                else max_duration_seconds
            )
            elapsed_video = frame_idx / source_fps
            duration_label = f"/{duration:.1f}s" if duration else ""
            cv2.putText(
                tile,
                f"FRAME:{frame_idx + 1}  {elapsed_video:.1f}s{duration_label}  "
                f"ANALYSIS:{min(float(analysis_fps), display_fps):.0f}FPS",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            display_frame = cv2.resize(
                tile, STREAM_SIZE, interpolation=cv2.INTER_AREA
            )
            yield cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            deadline += display_interval
            now = time.monotonic()
            remaining = deadline - now
            if remaining > 0:
                time.sleep(remaining)
            now = time.monotonic()
            target_frame = max(
                frame_idx + 1,
                round(max(deadline - started_at, now - started_at) * source_fps),
            )
            can_continue = True
            while frame_idx + 1 < target_frame:
                if not cap.grab():
                    can_continue = False
                    break
                frame_idx += 1
            if not can_continue:
                break
            frame_idx += 1
    finally:
        cap.release()
