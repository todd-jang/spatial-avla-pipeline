# spatial-avla-pipeline

**Spatial Audio-Vision-Language Alignment Pipeline — PoC Baseline**  
태그: `pre-antigravity` | 상태: Baseline 검증 완료

---

## 개요

자율주행 차량에서 **사이렌 감지 → DOA 추적 → 비전 이상 탐지 → VLM 컨텍스트 생성**까지의  
멀티모달 이벤트 트리거 파이프라인을 LangGraph DAG로 구현한 PoC.

```
S_I → [S_A ‖ S_T ‖ S_V] → S_B (Hysteresis) → S_C (VLM Payload)
```

- **6개 에이전트** (mock 로직, GPU 추론 없음)  
- **JSON-Schema 계약** 검증 (pre/final)  
- **Gradio 2×2 Multi-View** 스트리밍 UI  
- **레이턴시 목표** < 33 ms/frame (UI fast path)

---

## 문서

| 파일 | 설명 |
|------|------|
| [mvp_spec.md](./mvp_spec.md) | MVP 사양, 스코프, 검증 기준 |
| [agents.md](./agents.md) | 에이전트 역할, 알고리즘, 상태 계약 상세 |
| [.env.example](./.env.example) | 환경변수 템플릿 |

---

## 디렉토리 구조

```
spatial-avla-pipeline/
├── contracts/
│   └── frame_audio_packet_v1.0.json   # JSON-Schema 계약 (Draft-7)
├── mocks/
│   ├── mock_base.py                   # MOCK_STATE (Python dict)
│   └── mock_base.json                 # MOCK_STATE (JSON, Gradio bridge용)
├── scripts/
│   ├── download_raw_videos.py         # 원시 영상 다운로드
│   ├── generate_seq_lead_lost_15f.py  # golden fixture 생성
│   └── precache_wavlm.py              # WavLM 임베딩 사전 캐시
├── src/
│   ├── app/
│   │   ├── gradio_app.py              # Gradio UI 빌더
│   │   └── multiview.py              # 2x2 프레임 오버레이 렌더러
│   ├── core/
│   │   ├── state.py                   # FrameAudioPacketState TypedDict
│   │   ├── pipeline.py                # LangGraph DAG 정의 (graph, graph_ui)
│   │   ├── bridge.py                  # GraphBridge (Gradio <-> graph_ui)
│   │   ├── validator.py               # JSON-Schema 계약 검증
│   │   ├── agents/
│   │   │   ├── s_i_init.py            # S_I: 상태 초기화
│   │   │   ├── s_t.py                 # S_T: 훈련 하네스 메타
│   │   │   └── s_v.py                 # S_V: 검증 fixture 메타
│   │   ├── integrator/
│   │   │   ├── pre_validator.py       # pre_frame_init, pre_validator_sync
│   │   │   └── team_barrier.py        # team_barrier (POC 완료 집계)
│   │   └── tools/
│   │       └── agent_wrapper.py       # wrap_agent / wrap_agent_fast
│   ├── exp_a/
│   │   └── agent_s_a.py               # S_A: 비전 이상 감지
│   ├── exp_b/
│   │   └── agent_s_b.py               # S_B: 멀티모달 집계 + 히스테리시스
│   └── exp_c/
│       └── agent_s_c.py               # S_C: VLM 페이로드 구성
├── main.py                            # CLI 진입점 (단일 프레임 smoke test)
├── run_gradio.py                      # Gradio UI 진입점
├── requirements.txt                   # Python 의존성
├── .env.example                       # 환경변수 템플릿
├── mvp_spec.md                        # MVP 사양서
└── agents.md                          # 에이전트 설계 문서
```

---

## 빠른 시작

### 1. 환경 준비

```bash
# Python 3.10+ 권장
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
```

### 2. CLI 실행 (단일 프레임 smoke test)

```bash
python main.py
```

**기대 출력**:
```
POC.POC_Completed_At_ms_T_last: 1759500XXXXXXXXX
POC.All_OK: True
Status: ok
Errors_Count: 0
```

`poc_all_ok: True`, `Errors_Count: 0` 이면 Baseline 검증 통과.

### 3. Gradio UI 실행

```bash
python run_gradio.py
```

브라우저에서 `http://localhost:7860` 접속.  
(또는 터미널에 출력된 share URL 사용)

UI에서 mp4 동영상 업로드 → **Run** 버튼 → 2×2 실시간 스트리밍 확인.

---

## 테스트 방법

### Smoke Test (CLI — 의존성 없음)

```bash
python main.py
```

| 확인 항목 | 판정 기준 |
|-----------|-----------|
| `POC.All_OK: True` | PASS |
| `Status: ok` | PASS |
| `Errors_Count: 0` | PASS |
| `Fallback:` 출력 없음 | PASS |

### Mock State 직접 검증

```python
from copy import deepcopy
from mocks.mock_base import MOCK_STATE
from src.core.pipeline import graph

res = graph.invoke(deepcopy(MOCK_STATE))
me = res["meta_exec"]
assert me["poc_all_ok"] is True,   f"poc_all_ok={me['poc_all_ok']}"
assert me["errors"] == [],          f"errors={me['errors']}"
assert me["status"] == "ok",        f"status={me['status']}"
print("ALL ASSERTIONS PASSED")
```

### 트리거 로직 단위 검증

```python
from mocks.mock_base import MOCK_STATE
from copy import deepcopy
from src.core.pipeline import graph

# 사이렌 p >= 0.80 → is_triggered 기대
s = deepcopy(MOCK_STATE)
s["event_trigger"]["details"]["p_siren"] = 0.85
res = graph.invoke(s)
ev = res["event_trigger"]
print("is_triggered:", ev["is_triggered"])   # True 기대
print("hysteresis state:", ev["hysteresis"]["state"])  # active 기대
```

### JSON-Schema 계약 직접 검증

```bash
python -c "
from mocks.mock_base import MOCK_STATE
from src.core.validator import contract_validate
res = contract_validate(MOCK_STATE, 'manual_test')
errs = res.get('meta_exec', {}).get('errors', [])
print('Schema errors:', errs)
"
```

### WavLM 사전 캐시 (GPU 필요, 선택)

```bash
# exports/ 디렉토리에 audio_stereo_16k.wav 파일이 있어야 함
python scripts/precache_wavlm.py
# 결과: cache/emb/<clip>.pt, cache/emb/manifest.json
```

### Golden Fixture 생성 (선택)

```bash
# exports/TL_A/train/C01/frames/frame_*.jpg 필요
python scripts/generate_seq_lead_lost_15f.py
# 결과: golden/fixtures/seq_lead_lost_15f/frame_000.json ... frame_014.json
```

---

## 의존성

```
langgraph>=0.2.0       # DAG 오케스트레이션
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
gradio>=4.44.0         # 스트리밍 UI
torch>=2.0.0           # WavLM 추론 (선택)
transformers>=4.40.0
torchaudio>=2.0.0
opencv-python          # 영상 처리
numpy
scipy
jsonschema             # 계약 검증
Pillow
yt-dlp                 # 원시 영상 다운로드
```

> **최소 실행 (mock only)**: `langgraph`, `jsonschema` 만 있으면 `python main.py` 동작.  
> Gradio UI: `gradio`, `opencv-python` 추가 필요.

---

## 환경변수 주요 항목

`.env.example` 참조. 주요 항목:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `WAVLM_EMB_PATH` | `cache/emb/wavlm.pt` | WavLM 임베딩 캐시 경로 |
| `GRADIO_SERVER_PORT` | `7860` | UI 포트 |
| `GRADIO_SHARE` | `true` | 공개 URL 생성 여부 |
| `HYSTERESIS_MIN_HOLD_FRAMES` | `35` | 트리거 최소 지속 프레임 |
| `P_SIREN_THRESHOLD` | `0.80` | 사이렌 확률 임계값 |

---

## Git 태그

```bash
# pre-antigravity 태그 확인
git log --oneline
git tag
```

이 태그(`pre-antigravity`)는 Antigravity SDK 연동 전 순수 LangGraph baseline 상태를 표시한다.

---

## 라이선스

Internal PoC — 미공개. 배포 전 라이선스 검토 필요.
