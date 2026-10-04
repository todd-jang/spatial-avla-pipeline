# Agents — spatial-avla-pipeline
> **버전**: baseline / `pre-antigravity`

---

## 에이전트 맵

```
                  ┌──────────────────────────────────────┐
                  │          LangGraph DAG               │
                  │                                      │
  START → [pre_frame_init] → [pre_validator_sync]        │
                                    │                    │
                               [validator_pre]           │
                                    │                    │
                              ┌─[S_I_init]─┐            │
                              │            │             │
                   ┌──────────┴──┐  ┌──────┴──┐  ┌─────┴──┐
                   │    S_A      │  │   S_T   │  │  S_V   │
                   │ (Exp A)     │  │ (core)  │  │ (core) │
                   └──────────┬──┘  └──────┬──┘  └─────┬──┘
                              │            │             │
                              └──────┬─────┘             │
                                     │←─────────────────┘
                              [aggregator_event] (S_B, Exp B)
                                     │
                                   [S_C] (Exp C)
                                     │
                              [team_barrier]
                                     │
                            [validator_final]
                                     │
                                [metrics]
                                     │
                        [pre_validator_sync_post]
                                     │
                                    END
```

---

## 에이전트 상세

### S_I — State Initializer
**파일**: `src/core/agents/s_i_init.py`  
**노드**: `S_I_init`  
**역할**: `FrameAudioPacketState` 전체 필드 기본값 보장.

| 출력 필드 | 기본값 |
|-----------|--------|
| `version` | `"1.0"` |
| `clip_id` | `"siren_left_01"` |
| `frame_idx` | `0` |
| `ts_ms` / `timestamp_ms` | `0.0` |
| `event_trigger` | `{is_triggered: false, hysteresis: {state: idle, min_hold: 35}}` |
| `meta_exec` | `{status: idle, errors: [], poc_all_ok: false}` |
| `_hist` | `{va_list: [], dd_list: []}` |

**Wrapper**: `wrap_agent("S_I", s_i_init_fn)` — 예외 캐치, 타임스탬프 기록.

---

### S_A — Vision Anomaly Agent
**파일**: `src/exp_a/agent_s_a.py`  
**노드**: `S_A`  
**실험**: Exp A (비전 이상 감지)

**처리 흐름**:
1. `event_trigger.details.vision_anomaly_score` 읽기 (기본 0.2)
2. `_hist.va_list` 에 추가 (최대 300개 슬라이딩 윈도우)
3. `lead_lost`, `lane_lost_count` 정규화

**출력**: `_hist.va_list` 누적 (S_B가 MA 계산에 사용)

---

### S_T — Training Harness Agent
**파일**: `src/core/agents/s_t.py`  
**노드**: `S_T`  
**역할**: 훈련 파이프라인 연결 상태 메타 기록

| 모드 | `meta_exec.train_harness.status` |
|------|----------------------------------|
| `UI_FAST=True` | `built_verified_ui_skip_check` |
| `UI_FAST=False` | `built_verified` |

**WavLM 캐시**: `avla.wavlm_emb_path` = `cache/emb/wavlm.pt`

---

### S_V — Validation Fixture Agent
**파일**: `src/core/agents/s_v.py`  
**노드**: `S_V`  
**역할**: 검증 fixture 메타데이터 및 기본 메트릭 초기화

| 출력 | 내용 |
|------|------|
| `meta_exec.golden_fixtures.sequences` | `["seq_lead_lost_15f"]` |
| `meta_exec.metrics` | jitter_rms, steering_smoothness, path_curvature, event_alignment_ms, direction_match_rate, direction_mention_rate |

---

### S_B — Event Aggregator (Hysteresis)
**파일**: `src/exp_b/agent_s_b.py`  
**노드**: `aggregator_event`  
**실험**: Exp B (멀티모달 이벤트 집계)

**알고리즘**:
```
MA_w5(va_list) → vision_anomaly_ma
MA_w5(dd_list) → doa_change_rate_ma

raw_trigger:
  p_siren >= 0.80          → 사이렌 고확률
  doa_change_rate_ma >= 15 → 음원 방향 급변
  lead_lost == True        → 선행차 소실
  lane_lost_count >= 3     → 차선 3회 연속 손실
  vision_anomaly_ma >= 0.75→ 비전 이상치 누적

Hysteresis FSM:
  [idle] --(raw)--→ [active] (active_until = fi + 35)
  [active] --(raw)--→ extend active_until
  [active] --(~raw & fi >= active_until)--→ [idle]

is_triggered = (state == "active")
```

---

### S_C — Context / VLM Payload Builder
**파일**: `src/exp_c/agent_s_c.py`  
**노드**: `S_C`  
**실험**: Exp C (VLM 컨텍스트 빌더)

**처리 흐름**:
1. `event_trigger.is_triggered` 확인
2. trigger_reason 리스트 → 문자열 합산
3. `vlm_payload.prompt_context` = `"[AUDIO/VISION ALERT] {reasons}"`
4. `vlm_payload.trigger` = bool(is_triggered)

---

## 인프라 컴포넌트

### pre_frame_init
**파일**: `src/core/integrator/pre_validator.py`  
- trigger_reason / trigger_reason_raw 초기화 (프레임마다 리셋)

### pre_validator_sync
**파일**: `src/core/integrator/pre_validator.py`  
- `frame_idx`, `timestamp_ms` 단조 증가 검증
- 위반 시 `meta_exec.errors` 에 `NON_MONOTONIC_*` 추가

### validator_pre / validator_final
**파일**: `src/core/validator.py`  
- `contracts/frame_audio_packet_v1.0.json` (Draft-7) 적용
- 위반 시 `SCHEMA_VALIDATION_ERROR` 기록

### team_barrier
**파일**: `src/core/integrator/team_barrier.py`  
- 모든 에이전트 완료 타임스탬프 중 최대값 → `poc_completed_at_ms`
- `_agent_status` 전체 `"done"` + errors 없음 → `poc_all_ok = True`
- `status` = `"ok"` | `"partial_or_error"`

### GraphBridge
**파일**: `src/core/bridge.py`  
- Gradio UI에서 `graph_ui.invoke(s)` 호출 래퍼
- `lat_ms` 측정 (`time.perf_counter`)
- 예외 시 `bridge.GRAPH_INVOKE_EXCEPTION` 기록

### agent_wrapper / agent_wrapper_fast
**파일**: `src/core/tools/agent_wrapper.py`  
- 모든 에이전트를 `wrap_agent` 로 감싸 실행 시간/상태 자동 기록
- `_agent_status[name]` = `"running"` → `"done"` | `"error"`

---

## 상태 타입 참조
**파일**: `src/core/state.py` — `FrameAudioPacketState(TypedDict)`

## Mock 데이터
**파일**: `mocks/mock_base.json` / `mocks/mock_base.py` — MOCK_STATE (단일 프레임 기본값)
