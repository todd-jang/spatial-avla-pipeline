"""AST audio-template scoring for siren presence in a clip."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = ROOT / "outputs" / "mae_siren_templates_ast.pt"
CLIPS_ROOT = ROOT / "exports" / "TL_A"
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
VIDEO_FPS = 30
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
AST_INPUT_FRAMES = 1024
SIREN_WINDOW_SECONDS = 2.0
SIREN_HOP_SECONDS = 1.0
SIREN_THRESHOLD_FRAME = -0.04
SIREN_THRESHOLD_CLIP = -0.20
SIREN_THRESHOLD_BALANCED = SIREN_THRESHOLD_CLIP
SIREN_THRESHOLD_CONSERVATIVE = -0.10
SIREN_THRESHOLD_SENSITIVE = -0.30
SIREN_THRESHOLD_MODES = {
    "balanced": SIREN_THRESHOLD_BALANCED,
    "conservative": SIREN_THRESHOLD_CONSERVATIVE,
    "sensitive": SIREN_THRESHOLD_SENSITIVE,
}


def siren_detected(
    score: float | None, mode: str = "balanced", scope: str = "clip"
) -> bool:
    """Classify a clip or supplied audio-window score."""
    if mode not in SIREN_THRESHOLD_MODES:
        raise ValueError(
            f"Unknown siren threshold mode {mode!r}; "
            f"expected one of {tuple(SIREN_THRESHOLD_MODES)}"
        )
    if score is None:
        return False
    if scope == "frame":
        threshold = SIREN_THRESHOLD_FRAME
    elif scope == "clip":
        threshold = SIREN_THRESHOLD_MODES[mode]
    else:
        raise ValueError(f"Unknown siren score scope {scope!r}; expected 'clip' or 'frame'")
    return score > threshold


def ast_feature_from_waveform(
    model: Any,
    waveform: np.ndarray,
    sample_rate: int,
    device: str,
):
    """Run AST on mono audio using the same log-mel preprocessing as templates."""
    import torch
    import torch.nn.functional as functional
    import torchaudio
    import torchaudio.transforms as transforms

    audio = torch.as_tensor(np.asarray(waveform), dtype=torch.float32)
    if audio.ndim == 2:
        audio = audio.mean(dim=1)
    if audio.ndim != 1 or audio.numel() == 0:
        raise ValueError("AST input must be a non-empty mono waveform")
    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate: {sample_rate}")
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)

    mel = transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )(audio.unsqueeze(0))
    log_mel = torch.log(mel + 1e-6)
    current_frames = log_mel.shape[-1]
    if current_frames < AST_INPUT_FRAMES:
        log_mel = functional.pad(log_mel, (0, AST_INPUT_FRAMES - current_frames))
    else:
        log_mel = log_mel[..., :AST_INPUT_FRAMES]

    inputs = log_mel.squeeze(0).transpose(0, 1).unsqueeze(0)
    inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-6)
    with torch.inference_mode():
        output = model(inputs.to(device))
    return functional.normalize(output.pooler_output[0].float(), dim=-1).cpu()


def ast_feature_from_file(model: Any, path: str | Path, device: str):
    from src.core.audio.wav_io import load_wav_mono_16k

    waveform = load_wav_mono_16k(path)
    return ast_feature_from_waveform(model, waveform, SAMPLE_RATE, device)


def save_template_file(
    output_path: str | Path,
    model_name: str,
    siren_features: list,
    road_features: list,
    siren_sources: list[str],
    road_sources: list[str],
) -> dict:
    """Normalize feature means and save a validated, weights-only-loadable file."""
    import torch
    import torch.nn.functional as functional

    if not siren_features:
        raise ValueError("Cannot build a siren template without siren features")
    if not road_features:
        raise ValueError("Cannot build a road template without road features")
    siren_template = functional.normalize(torch.stack(siren_features).mean(dim=0), dim=-1)
    road_template = functional.normalize(torch.stack(road_features).mean(dim=0), dim=-1)
    if siren_template.shape != road_template.shape:
        raise ValueError("Siren and road template feature dimensions do not match")

    payload = {
        "schema_version": 3,
        "model_name": model_name,
        "feature_scope": "full_clip_first_1024_log_mel_frames",
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "target_frames": AST_INPUT_FRAMES,
        "siren_template": siren_template.cpu(),
        "road_template": road_template.cpu(),
        "siren_sources": list(siren_sources),
        "road_sources": list(road_sources),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)
    return payload


class MaeVisionSirenScorer:
    """Lazy AST model/template provider for frame-aligned input audio."""

    def __init__(
        self,
        template_path: str | Path | None = None,
        clips_root: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self._template_path = Path(template_path or DEFAULT_TEMPLATE_PATH)
        self._clips_root = Path(clips_root or CLIPS_ROOT)
        self._device = self._pick_device(device)
        self._templates: dict | None = None
        self._model = None
        self._clip_meta: dict[str, dict] | None = None
        self._score_cache: dict[str, float] = {}
        self._window_score_cache: dict[tuple[str, int, int], float] = {}

    @staticmethod
    def _pick_device(device: str) -> str:
        if device != "auto":
            return device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def available(self) -> bool:
        return self._template_path.is_file()

    def _load_templates(self) -> dict | None:
        if self._templates is not None:
            return self._templates
        if not self._template_path.is_file():
            return None

        import torch

        templates = torch.load(
            self._template_path, map_location="cpu", weights_only=True
        )
        if (
            not isinstance(templates, dict)
            or templates.get("schema_version") != 3
            or templates.get("model_name") != AST_MODEL_NAME
            or templates.get("feature_scope") != "full_clip_first_1024_log_mel_frames"
        ):
            raise ValueError(f"Invalid AST siren template file: {self._template_path}")
        siren = templates.get("siren_template")
        road = templates.get("road_template")
        if (
            not isinstance(siren, torch.Tensor)
            or not isinstance(road, torch.Tensor)
            or siren.ndim != 1
            or road.shape != siren.shape
            or not torch.isfinite(siren).all()
            or not torch.isfinite(road).all()
        ):
            raise ValueError(f"Invalid template tensors in {self._template_path}")
        self._templates = templates
        return templates

    def _get_model(self):
        if self._model is None:
            from transformers import ASTModel

            self._model = (
                ASTModel.from_pretrained(AST_MODEL_NAME)
                .to(self._device)
                .eval()
            )
        return self._model

    def _load_clip_meta(self) -> dict[str, dict]:
        if self._clip_meta is None:
            metas = {}
            for meta_path in self._clips_root.rglob("clip_meta.json"):
                with meta_path.open(encoding="utf-8") as stream:
                    meta = json.load(stream)
                meta["_meta_path"] = meta_path
                metas[meta["clip_id"]] = meta
            self._clip_meta = metas
        return self._clip_meta

    def _resolve_clip_audio(
        self, clip_id: str, audio_key: str = "untouched"
    ) -> Path | None:
        meta = self._load_clip_meta().get(clip_id)
        if meta is not None:
            audio = meta.get("audio", {})
            relative_path = audio.get(audio_key)
            if not isinstance(relative_path, str):
                return None
            path = meta["_meta_path"].parent / relative_path
            if not path.is_file():
                return None
            return path
        return None

    def _resolve_audio(self, state: dict) -> Path | None:
        if "audio_path" in state and state["audio_path"] is not None:
            raw_path = state["audio_path"]
            if not isinstance(raw_path, (str, Path)):
                return None
            path = Path(raw_path)
            return path if path.is_file() else None

        clip_id = str(state.get("clip_id", ""))
        if clip_id:
            return self._resolve_clip_audio(clip_id, "untouched")

        raw_path = state.get("wav_path")
        if not isinstance(raw_path, str) or raw_path == "raw_videos/sample.wav":
            return None
        path = Path(raw_path)
        if not path.is_file():
            return None
        return path

    def _score_audio_path(self, path: Path) -> float | None:
        templates = self._load_templates()
        if templates is None:
            return None
        if not path.is_file():
            return None
        cache_key = str(path.resolve())
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        feature = ast_feature_from_file(self._get_model(), path, self._device)
        score = self._score_feature(feature, templates)
        self._score_cache[cache_key] = score
        return score

    @staticmethod
    def _score_feature(feature, templates: dict) -> float:
        import torch.nn.functional as functional

        siren_template = templates["siren_template"].float()
        road_template = templates["road_template"].float()
        if feature.shape != siren_template.shape:
            raise ValueError(
                f"AST feature/template dimensions differ: {feature.shape} != "
                f"{siren_template.shape}"
            )
        return float(
            functional.cosine_similarity(
                feature[None, :], siren_template[None, :]
            ).item()
            - functional.cosine_similarity(
                feature[None, :], road_template[None, :]
            ).item()
        )

    def score_frame(
        self,
        state: dict,
        window_seconds: float = SIREN_WINDOW_SECONDS,
        hop_seconds: float = SIREN_HOP_SECONDS,
        fps: int = VIDEO_FPS,
    ) -> float | None:
        """Score a two-second audio window aligned to this video frame.

        Frame windows advance once per hop, not once per video frame. An
        explicit ``audio_path`` is treated as an already-sliced audio window.
        """
        templates = self._load_templates()
        if templates is None:
            return None

        explicit_path = state.get("audio_path")
        if explicit_path is not None:
            if not isinstance(explicit_path, (str, Path)):
                return None
            if not str(explicit_path):
                return None
            path = Path(explicit_path)
            if state.get("audio_windowed") is True:
                return self._score_audio_path(path)
        else:
            clip_id = str(state.get("clip_id", ""))
            if not clip_id:
                return None
            audio_key = state.get("audio_source", "untouched")
            if audio_key not in ("untouched", "with_siren"):
                raise ValueError(f"Unknown audio source {audio_key!r}")
            path = self._resolve_clip_audio(clip_id, audio_key)
            if path is None:
                return None

        frame_idx = int(state.get("frame_idx", 0))
        if frame_idx < 0 or fps <= 0 or window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("Frame index, FPS, window, and hop must be positive")

        window_index = int(frame_idx / (fps * hop_seconds))
        start_seconds = window_index * hop_seconds
        import soundfile as sf

        info = sf.info(path)
        start_sample = round(start_seconds * info.samplerate)
        window_samples = round(window_seconds * info.samplerate)
        cache_key = (str(path.resolve()), start_sample, window_samples)
        if cache_key in self._window_score_cache:
            return self._window_score_cache[cache_key]

        waveform, sample_rate = sf.read(
            path, start=start_sample, frames=window_samples, always_2d=True
        )
        if waveform.size == 0:
            return None
        feature = ast_feature_from_waveform(
            self._get_model(), waveform, sample_rate, self._device
        )
        score = self._score_feature(feature, templates)
        self._window_score_cache[cache_key] = score
        if len(self._window_score_cache) > 4096:
            self._window_score_cache.pop(next(iter(self._window_score_cache)))
        return score

    def score_clip(self, clip_id: str) -> float | None:
        """Score the metadata's untouched audio recording for debugging."""
        if self._load_templates() is None:
            return None
        path = self._resolve_clip_audio(clip_id, "untouched")
        return None if path is None else self._score_audio_path(path)

    def score(self, state: dict) -> float | None:
        """Score audio_path when supplied, otherwise the clip's untouched audio."""
        if self._load_templates() is None:
            return None
        path = self._resolve_audio(state)
        return None if path is None else self._score_audio_path(path)


def mae_siren_score(state: dict) -> float | None:
    """Score the complete recording identified by a pipeline state."""
    return _DEFAULT_SCORER.score(state)


_DEFAULT_SCORER = MaeVisionSirenScorer()
