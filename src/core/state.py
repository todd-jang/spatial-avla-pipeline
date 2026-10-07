import operator
from typing import Annotated, Dict, Any


def _last(a, b):
    """스칼라 / 문자열 필드: 마지막 쓰기 값 채택 (last-write-wins)."""
    return b


def _merge_dict(a, b):
    """
    dict 필드: deep merge (b 우선).
    - dict 값 → 재귀 merge
    - list 값 → a + b (중복 제거 없이 누적; va_list 등 순서 보존)
    - 기타 스칼라 → b (후기 쓰기 우선)
    병렬 노드(S_A ‖ S_T ‖ S_V)의 동시 쓰기를 안전하게 병합.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    merged = dict(a)
    for k, bv in b.items():
        av = merged.get(k)
        if isinstance(av, dict) and isinstance(bv, dict):
            merged[k] = _merge_dict(av, bv)
        elif isinstance(av, list) and isinstance(bv, list):
            # 리스트는 두 값 중 더 긴 쪽을 채택 (누적 시나리오)
            # 단, 빈 리스트는 기존 값 보존
            merged[k] = bv if bv else av
        else:
            merged[k] = bv
    return merged


# ─── FrameAudioPacketState ────────────────────────────────────────────────────
# LangGraph 1.x 병렬 노드(S_A ‖ S_T ‖ S_V)가 동일 키를 동시에 업데이트할 때
# Annotated reducer 없이는 InvalidUpdateError 가 발생한다.
# - 스칼라 → _last  (마지막 값 채택)
# - dict   → _merge_dict  (shallow merge, 후기 값 우선)
# ─────────────────────────────────────────────────────────────────────────────
class FrameAudioPacketState(dict):
    """
    LangGraph StateGraph 호환 상태 컨테이너.
    TypedDict 대신 dict 서브클래스를 사용하여 Annotated reducer 를 우회.
    모든 에이전트는 dict 를 반환하며, LangGraph 가 채널별 reducer 로 병합한다.
    """
    pass


# ─── 채널 정의 (Annotated TypedDict 방식) ─────────────────────────────────────
from typing import TypedDict


class FrameAudioPacketStateTyped(TypedDict):
    version:       Annotated[str,            _last]
    clip_id:       Annotated[str,            _last]
    frame_id:      Annotated[int,            _last]
    frame_idx:     Annotated[int,            _last]
    ts_ms:         Annotated[float,          _last]
    timestamp_ms:  Annotated[float,          _last]
    video_path:    Annotated[str,            _last]
    wav_path:      Annotated[str,            _last]
    img:           Annotated[Dict[str, Any], _merge_dict]
    audio_win:     Annotated[Dict[str, Any], _merge_dict]
    doa:           Annotated[Dict[str, Any], _merge_dict]
    percept:       Annotated[Dict[str, Any], _merge_dict]
    geom:          Annotated[Dict[str, Any], _merge_dict]
    plan:          Annotated[Dict[str, Any], _merge_dict]
    weather:       Annotated[Dict[str, Any], _merge_dict]
    vlm:           Annotated[Dict[str, Any], _merge_dict]
    avla:          Annotated[Dict[str, Any], _merge_dict]
    audio:         Annotated[Dict[str, Any], _merge_dict]
    vlm_payload:   Annotated[Dict[str, Any], _merge_dict]
    event_trigger: Annotated[Dict[str, Any], _merge_dict]
    meta_exec:     Annotated[Dict[str, Any], _merge_dict]
    _hist:         Annotated[Dict[str, Any], _merge_dict]
    _sync:         Annotated[Dict[str, Any], _merge_dict]
