from typing import TypedDict, Dict, Any


class FrameAudioPacketState(TypedDict):
    version: str
    clip_id: str
    frame_id: int
    frame_idx: int
    ts_ms: float
    timestamp_ms: float
    video_path: str
    wav_path: str
    img: Dict[str, Any]
    audio_win: Dict[str, Any]
    doa: Dict[str, Any]
    percept: Dict[str, Any]
    geom: Dict[str, Any]
    plan: Dict[str, Any]
    weather: Dict[str, Any]
    vlm: Dict[str, Any]
    avla: Dict[str, Any]
    audio: Dict[str, Any]
    vlm_payload: Dict[str, Any]
    event_trigger: Dict[str, Any]
    meta_exec: Dict[str, Any]
    _hist: Dict[str, Any]
    _sync: Dict[str, Any]
