# MVP Specification — spatial-avla-pipeline
> **태그**: `pre-antigravity` | **단계**: PoC Baseline  
> **최종 작성**: 2026-10-04

---

## 1. 목적 (Purpose)

자율주행 차량의 **공간 오디오-비전-언어 통합(Spatial Audio-Vision-Language Alignment, AVLA)** 파이프라인의 PoC Baseline을 검증한다.  
핵심 질문: *단일 프레임 패킷이 6개 에이전트(S_I → S_A/S_T/S_V → S_B → S_C)를 LangGraph DAG 로 순회하여 계약(Contract) 검증까지 통과하는가?*

---

## 2. Scope — Baseline (pre-antigravity)

| 범주 | IN (포함) | OUT (제외) |
|------|-----------|-----------|
| 파이프라인 | LangGraph DAG 실행, pre/post JSON-Schema 검증 | 실제 모델 추론 (WavLM, VLM) |
| 에이전트 | S_I, S_A, S_B, S_C, S_T, S_V (mock 로직) | 실 GPU 연산 |
| UI | Gradio 2x2 Multi-View streaming | 실제 영상 분류 결과 overlay |
| 계약 | FrameAudioPacketState v1.0 JSON-Schema | 세부 필드 enum 검증 |
| 테스트 | mock_base 단일 프레임 smoke test | 멀티 프레임 시계열 회귀 |
| 인프라 | 로컬 CPU 실행 | GPU 클러스터, 분산 학습 |

---

## 3. 아키텍처 개요

```
START
  |
  v
pre_frame_init --> pre_validator_sync --> validator_pre (JSON-Schema)
                                                |
                                                v
                                           S_I_init
                                        /     |     \
                                       v      v      v
                                      S_A    S_T    S_V
                                        \     |     /
                                         v    v    v
                                      aggregator_event (S_B)
                                                |
                                                v
                                              S_C
                                                |
                                                v
                                         team_barrier
                                                |
                                                v
                              validator_final --> metrics --> pre_validator_sync_post
                                                |
                                                v
                                              END
```

**UI Fast Path** (`graph_ui`): `validator_pre/final`, `metrics`, `pre_validator_sync_post` 노드 제거 — 레이턴시 목표 < 33 ms/frame.

---

## 4. 에이전트 역할

| 에이전트 | 노드명 | 책임 |
|----------|--------|------|
| S_I | `S_I_init` | 상태 초기화, 누락 필드 기본값 설정 |
| S_A | `S_A` | 비전 이상 스코어(`vision_anomaly_score`) 기록, `_hist.va_list` 누적 |
| S_T | `S_T` | 훈련 하네스 메타데이터 기록 (`meta_exec.train_harness`) |
| S_V | `S_V` | 검증 fixture 메타데이터 기록 (`meta_exec.golden_fixtures`, `metrics`) |
| S_B | `aggregator_event` | 이동 평균(`va_ma`, `dd_ma`) 계산, raw trigger 판정, 히스테리시스 상태 머신 |
| S_C | `S_C` | VLM 페이로드 구성 (`vlm_payload`), 이벤트 이유 문자열화 |

---

## 5. 상태 계약 (FrameAudioPacketState v1.0)

계약 파일: `contracts/frame_audio_packet_v1.0.json`  
필수 필드: `frame_idx`, `timestamp_ms`, `audio`, `vlm_payload`, `event_trigger`, `meta_exec`

### 핵심 서브 스키마

```
event_trigger.details:
  p_siren              float  [0, 1]   -- 사이렌 확률
  audio_confidence     float  [0, 1]
  doa_change_rate_deg  float           -- DOA 변화율 (deg/frame)
  vision_anomaly_score float  [0, 1]
  lane_lost_count      int    >= 0
  lead_lost            bool
  vision_anomaly_ma    float           -- 5-frame MA
  doa_change_rate_ma   float           -- 5-frame MA

event_trigger.hysteresis:
  state                enum  idle | armed | active
  min_hold_frames      int   = 35
  active_since_frame_idx   int
  active_until_frame_idx   int
  last_trigger_frame_idx   int

meta_exec:
  status               enum  idle | ok | partial_or_error | error
  poc_all_ok           bool
  poc_completed_at_ms  int   -- Unix ms
  errors               list[{path, code, msg}]
```

---

## 6. 트리거 로직 (S_B 기준)

```python
raw_trigger = (
    p_siren        >= 0.80  OR
    doa_change_ma  >= 15.0  OR
    lead_lost      == True  OR
    lane_lost_cnt  >= 3     OR
    vision_anom_ma >= 0.75
)

hysteresis:
  idle/armed + raw  --> active (hold: frame_idx + 35)
  active + raw      --> extend active_until
  active + !raw + fi >= active_until --> idle
```

---

## 7. 검증 기준 (Baseline Accept Criteria)

| 항목 | 기준 |
|------|------|
| `meta_exec.poc_all_ok` | `True` |
| `meta_exec.errors` | `[]` (빈 리스트) |
| `meta_exec.status` | `"ok"` |
| JSON-Schema 검증 | pre / final 모두 통과 |
| UI Fast Path 레이턴시 | < 33 ms/frame (목표) |
| Fallback 발동 없음 | `meta_exec.fallback` 키 없음 |

---

## 8. 다음 단계 (post-baseline)

1. **Exp A**: 실제 OpenCV 프레임 → 이상 스코어 모델 연결
2. **Exp B**: 실 WavLM 추론 → `p_siren`, `doa_change_rate_deg` 실측
3. **Exp C**: VLM (LLaVA / GPT-4V) 실 호출 연결
4. 멀티 프레임 시계열 golden fixture 회귀 테스트 (`golden/fixtures/seq_lead_lost_15f`)
5. GPU 서버 배포 → 레이턴시 < 33 ms 실증
