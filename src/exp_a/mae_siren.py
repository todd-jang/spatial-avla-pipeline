"""MAE + AST 기반 siren score.

window=2.0s, hop=1.0s로 프레임별 siren score 계산.
- 클립 오디오를 슬라이싱
- AST feature -> siren_template/road_template 코사인 차이
- score > -0.04 -> siren detected
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from transformers import ASTModel

# -- 경로 --
PIPELINE_ROOT = Path("/Users/robotjang/spatial-avla-pipeline")
TL_A = PIPELINE_ROOT / "exports" / "TL_A"
TEMPLATE_PATH = PIPELINE_ROOT / "configs" / "siren_template.pt"

# -- 설정 --
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
AUDIO_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 256
TARGET_LEN = 1024

# window=2.0s (12m 결과)
WINDOW_SEC = 2.0
HOP_SEC = 1.0

# 임계값 (window=2.0s)
SIREN_THRESHOLD = -0.04
SIREN_THRESHOLD_CONSERVATIVE = 0.00
SIREN_THRESHOLD_SENSITIVE = -0.10

# -- lazy load --
_AST = None
_SIREN_TEMPLATE = None
_ROAD_TEMPLATE = None


def _load():
    global _AST, _SIREN_TEMPLATE, _ROAD_TEMPLATE
    if _AST is None:
        _AST = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(DEVICE).eval()

    if _SIREN_TEMPLATE is None:
        if not TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"템플릿 없음: {TEMPLATE_PATH}\n"
                f"먼저 scripts/13a_build_siren_templates.py 실행"
            )
        ckpt = torch.load(TEMPLATE_PATH, map_location="cpu", weights_only=False)
        _SIREN_TEMPLATE = F.normalize(
            ckpt["siren_template"].to(DEVICE), dim=-1
        )
        _ROAD_TEMPLATE = F.normalize(
            ckpt["road_template"].to(DEVICE), dim=-1
        )

    return _AST, _SIREN_TEMPLATE, _ROAD_TEMPLATE


# -- 오디오 로드 --
def _load_clip_audio(clip_id: str, key: str = "untouched") -> Optional[Path]:
    """클립 오디오 경로."""
    for split in ["train", "test"]:
        clip_dir = TL_A / split / clip_id
        if not clip_dir.exists():
            continue
        meta_path = clip_dir / "clip_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        rel = meta.get("audio", {}).get(key)
        if isinstance(rel, str):
            p = clip_dir / rel
            if p.exists():
                return p
    return None


def _load_logmel_from_array(wav: np.ndarray, sr: int,
                             n_mels=N_MELS, n_fft=N_FFT,
                             hop=HOP, target_len=TARGET_LEN):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    if sr != AUDIO_SR:
        wav_t = torchaudio.functional.resample(wav_t, sr, AUDIO_SR)
    mel = T.MelSpectrogram(
        sample_rate=AUDIO_SR, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop
    )(wav_t)
    lm = torch.log(mel + 1e-6).squeeze(0)
    Tc = lm.shape[-1]
    if Tc < target_len:
        lm = F.pad(lm, (0, target_len - Tc))
    else:
        lm = lm[..., :target_len]
    return lm


def _ast_feature(wav: np.ndarray, sr: int) -> torch.Tensor:
    ast, _, _ = _load()
    lm = _load_logmel_from_array(wav, sr).unsqueeze(0)
    x = lm.transpose(1, 2)
    x = (x - x.mean()) / (x.std() + 1e-6)
    with torch.no_grad():
        out = ast(x.to(DEVICE))
    return F.normalize(out.pooler_output.squeeze(0), dim=-1)


# -- score --
def _score_from_feature(feat: torch.Tensor) -> float:
    _, siren_t, road_t = _load()
    return ((feat @ siren_t) - (feat @ road_t)).item()


# -- 프레임 단위 score --
def compute_frame_siren_scores(clip_id: str, key: str = "untouched") -> np.ndarray:
    """클립 오디오 -> window=2.0s, hop=1.0s 프레임별 score."""
    audio_path = _load_clip_audio(clip_id, key)
    if audio_path is None:
        return np.array([])

    wav, sr = sf.read(str(audio_path), always_2d=True)
    win_len = int(WINDOW_SEC * sr)
    hop_len = int(HOP_SEC * sr)

    scores = []
    for start in range(0, len(wav) - win_len + 1, max(1, hop_len)):
        s = wav[start:start + win_len]
        feat = _ast_feature(s, sr)
        scores.append(_score_from_feature(feat))

    return np.array(scores)


# -- 클립 단위 score --
@lru_cache(maxsize=16)
def compute_clip_siren_score(clip_id: str, key: str = "untouched") -> float:
    """클립 전체 오디오 -> score 1개."""
    audio_path = _load_clip_audio(clip_id, key)
    if audio_path is None:
        return 0.0

    wav, sr = sf.read(str(audio_path), always_2d=True)
    feat = _ast_feature(wav, sr)
    return _score_from_feature(feat)


# -- 판정 --
def siren_detected(score: float, mode: str = "balanced",
                   level: str = "clip") -> bool:
    """score -> siren 유무.

    mode:  "conservative" | "balanced" | "sensitive"
    level: "clip" (클립 단위) | "frame" (프레임 단위)

    클립 단위 임계값: -0.20
    프레임 단위 임계값: -0.04
    """
    if score is None:
        return False

    if level == "frame":
        # 프레임 단위 (window=2.0s, 12m 결과)
        thresholds = {
            "conservative": 0.00,
            "balanced": -0.04,
            "sensitive": -0.10,
        }
    else:
        # 클립 단위 (12k 결과)
        thresholds = {
            "conservative": -0.10,
            "balanced": -0.20,
            "sensitive": -0.30,
        }

    thr = thresholds.get(mode, thresholds["balanced"])
    return score > thr


# -- 통합 함수 --
def mae_siren_score(state: dict) -> Optional[float]:
    """state -> siren score (None이면 계산 불가).

    우선순위:
      1. state["audio_path"]가 있으면 그 파일 사용
      2. state["clip_id"]가 있으면 클립 with_siren/untouched 사용
      3. 둘 다 없으면 None
    """
    # 1) 오디오 경로 직접 지정
    audio_path = state.get("audio_path")
    if audio_path:
        p = Path(audio_path)
        if p.exists():
            wav, sr = sf.read(str(p), always_2d=True)
            feat = _ast_feature(wav, sr)
            return _score_from_feature(feat)
        return None

    # 2) 클립 ID로 오디오 찾기
    clip_id = state.get("clip_id")
    if not clip_id:
        return None

    # with_siren 우선, 없으면 untouched
    for key in ["with_siren", "untouched"]:
        path = _load_clip_audio(clip_id, key)
        if path is not None:
            wav, sr = sf.read(str(path), always_2d=True)
            feat = _ast_feature(wav, sr)
            return _score_from_feature(feat)

    return None
