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

### 실시간 센서 모니터 (읽기 전용)

`Live read-only monitor` 탭에서 브라우저 카메라 프레임을 로컬 `rtdetr-l.pt`
체크포인트로 분석하고, 마이크 입력은 RMS/peak 음량만 표시합니다. 카메라 접근은
`localhost` 또는 HTTPS에서 허용해야 합니다. 실시간 사이렌 분류기는 연결되어 있지
않으므로 마이크 지표를 사이렌 인식 결과로 해석하지 마세요. 이 앱은 차량·조향·가감속
액추에이터와 연결되지 않으며, 주행 제어 출력을 만들지 않습니다. 기본 실행은 로컬 전용이며
카메라·마이크 사용 중 공개 Gradio 링크(`GRADIO_SHARE=true`)를 켜면 센서 데이터가
원격 서버로 전송되므로 신뢰할 수 있는 환경에서만 활성화하세요.

실시간 카메라 분석률은 기본 1 FPS이며 0.5–1.25 FPS로 조절할 수 있습니다. 워크스페이스의
RT-DETR 체크포인트를 실제로 측정한 결과(워밍업 후 약 0.56–0.69초/프레임)에 맞춘 값입니다.
Ultralytics 설정은 기본적으로 워크스페이스의 `.gradio/` 아래에 저장해 사용자 홈 경로의
설정 파일 권한을 요구하지 않습니다.
Qwen은 `QWEN_ACTION_MODEL` (기본 `Qwen/Qwen2.5-0.5B-Instruct`)을 별도 백그라운드
작업으로 실행하며, 화면에는 `MONITOR`/`CAUTION`/`ALERT_OPERATOR` 권고만 표시합니다.
첫 사용 시 모델을 내려받을 수 있고, Qwen 오류는 상태 칸에 표시됩니다. 권고는 차량 제어
명령이 아니며, 동작 클래스/입력 변경 후 추론 빈도는 최대 1회/5초로 제한됩니다.
검증 재생은 전체 시퀀스 검증을 보존하기 위해 프레임 한도 기본값을 `0`(전체 구간)으로
유지합니다. 사용자 영상 업로드에는 별도의 실시간 추론 시간 설정(기본 5분, 최대 5분)이
적용됩니다. 업로드 영상이 먼저 끝나면 재생도 종료되며, 영상을 반복 재생하지는 않습니다.
읽기 전용 카메라 모니터의 한도 기본값은 분석 프레임 300개(1 FPS 기준 약 5분)입니다.
모든 영상 타일과 라이브 인지 JSON에는 고정 가상 입력을 사용하는 MPC 데모의 조향각,
가감속/브레이크등 상태·스로틀 비율, 예측 횡오차를 덧붙입니다. 이는 카메라 추정값이 아니며
시뮬레이션 시각화만을 위한 값으로, 어떤 차량 액추에이터에도 전달되지 않습니다.

### AST 사이렌 템플릿 점수 (실험용)

오프라인 클립 재생에서 AST 전체 녹음 feature와 실사이렌/도로 템플릿을 비교하려면 먼저
템플릿을 생성합니다.

```bash
python scripts/build_mae_siren_templates.py --device mps
```

기본 출력은 `outputs/mae_siren_templates_ast.pt`입니다. 사이렌 템플릿은 G44-01/02/03,
도로 템플릿은 C01/C02의 untouched 녹음에서 각각 전체 녹음 단위 AST feature 평균으로
생성합니다. S_V는 `siren_score_mae = cos(clip_audio, siren_template) -
cos(clip_audio, road_template)`을 계산합니다. AST 입력은 첫 1024 log-mel 프레임을 사용하며,
긴 녹음은 약 16.4초 이후가 잘립니다.

클립 단위 판정 모드는 `balanced` (`score > -0.20`, 기본), `conservative`
(`score > -0.10`), `sensitive` (`score > -0.30`)이며
`AVLA_SIREN_THRESHOLD_MODE` 환경 변수로 선택합니다. 프레임 입력에는 2초 윈도우를
1초 hop으로 정렬하고 `score > -0.04`인 윈도우가 **5회 연속**일 때만
`siren_detected_mae`를 올립니다. 중복 비디오 프레임은 같은 윈도우로 처리하여 streak에
중복 집계하지 않습니다. Gradio의 `AST and playback audio source`에서 검증 클립의
`with_siren`/`untouched` 입력을 바꿀 수 있으며, 선택한 오디오는 검증 재생에서 자동
재생됩니다. 전체 시퀀스는 클립별 원본 오디오를 이어 붙이고, 프레임 제한을 설정하면 오디오도
해당 구간만 재생합니다. `audio_path`는 프레임 시점에 해당하는
전체 녹음 경로로 사용하며, 프레임 번호로부터 같은 2초 윈도우를 잘라냅니다. 템플릿이나
해당 오디오가 없으면 점수는 `null`이고 `score_status`에 사용할 수 없는 이유를 남깁니다.
결과의 `siren_threshold_scope` 필드는 `frame` 또는 `clip`으로 적용 범위를 나타냅니다.

2초 윈도우 실험의 T09 untouched는 14개 중 6개가 단일 임계값을 넘었지만 최장 연속 길이는
4개였습니다. `with_siren` 양성 C01/C02/C04는 각각 14개 윈도우 모두 임계값을 넘었습니다.
따라서 5개 연속 기준은 이 실험 시퀀스에서 T09 오탐을 제거하면서 해당 양성 구간을 유지합니다.
이는 제한된 데이터에 대한 실험 결과이며 독립적인 일반화 성능을 보장하지 않습니다.
12개 양성·5개 음성 녹음 실험에서 각각 balanced는
TPR/FPR 1.00/0.00, conservative는 0.917/0.00, sensitive는 1.00/0.40을 보였습니다.
단, 이 평가는 템플릿을 만든 G44-01/02/03, C01/C02와 같은 원본에서 나온 점수도 포함하므로
독립 검증이 아닙니다. 이 수치는 잠정 operating point이며 독립 테스트 성능으로 해석하면
안 됩니다. S_V의 clip 단위 점수는 해당 clip 전체 오디오를 참조하므로 실시간/미래 프레임
없는 추론이 아닙니다. 미연결 라이브 마이크에는 AST 점수가 계산되지 않습니다.

S_B는 사이렌 검출과 mini-MAE의 인과적 robust-z 이상 점수(`mae_anomaly_z_ma >= 5.0`)가
동시에 참일 때만 `siren_vision_trigger`를 올리고 `trigger_reason`에
`siren_detected`, `vision_anomaly`를 기록합니다. 이는 원시 MAE 거리 비교가 아니라 기존
MAE 트리거의 정규화 기준을 사용합니다.

검증/업로드의 2×2 각 화면에는 프레임, 탐지 수·confidence, bbox 면적, RT-DETR 위험도와
MA5, MAE 거리, 사이렌 확률, DoA/신뢰도, 트리거·히스테리시스, 시뮬레이션 계획,
Qwen 결과, 파이프라인 지연/FPS를 표시합니다. 라이브 탭의 `Quantified live perception`과
`Quantified microphone input`은 같은 값을 JSON 숫자로 제공합니다. 미연결 모달리티는
0으로 꾸미지 않고 `n/a`/`null` 및 상태로 표시하며, Qwen confidence는 모델이 제공하지
않으므로 unavailable로 표시합니다. 계획값은 template-only이며 실시간 인지/차량 제어값이
아닙니다.

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
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Gradio listen 주소 |
| `GRADIO_SHARE` | `false` | 공개 URL 생성 여부 (실시간 센서 데이터 보호를 위해 기본 비활성) |
| `QWEN_ACTION_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | 읽기 전용 액션 권고용 Qwen 모델 |
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
