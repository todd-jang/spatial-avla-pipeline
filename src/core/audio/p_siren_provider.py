"""Per-frame p_siren lookup for the live pipeline.

The pipeline is deliberately WavLM-free (see wav_io and colab_train_smoke's
contracts): embeddings are computed offline by scripts/precache_wavlm.py and
scored by scripts/fit_p_siren_head.py, which fits an L2-logistic head over the
cached frames and adds per-frame ``frame_p_siren`` probabilities to the scoring
report (its measured leave-one-clip-out accuracy lives in the report's ``head``
section). This module is the bridge between that offline model and the realtime
graph: given a clip_id and frame_idx, it returns the calibrated P(siren) that
p_siren should carry, instead of the hardwired 0.0.

``frame_p_siren`` is preferred; ``frame_cosines`` (the rejected cosine axis)
is kept as a fallback so older reports still resolve. Unknown clips and
missing reports fall back to 0.0, preserving the pre-MVP behaviour for the
synthetic/track corpus where no measurement exists.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_REPORT = ROOT / "outputs" / "p_siren_report.json"


class P_sirenProvider:
    def __init__(self, report_path: str | Path | None = None):
        self._path = Path(report_path) if report_path is not None else DEFAULT_REPORT
        self._report: dict | None = None
        self._n_frames: dict[str, int] = {}

    def _load(self) -> dict | None:
        if self._report is None and self._path.exists():
            self._report = json.loads(self._path.read_text())
            groups = ("real_sirens", "negative_control_clips", "noisy_positive_clips")
            for grp in groups:
                for clip, entry in self._report.get(grp, {}).items():
                    self._n_frames[clip] = int(entry.get("n_frames", 0))
        return self._report

    def _entry(self, clip_id: str) -> dict | None:
        report = self._load()
        if report is None:
            return None
        for grp in ("real_sirens", "noisy_positive_clips", "negative_control_clips"):
            entry = report.get(grp, {}).get(clip_id)
            if entry:
                return entry
        return None

    def frame_value(self, clip_id: str, frame_idx: int) -> float:
        """Calibrated P(siren) for this frame; 0.0 if unknown."""
        entry = self._entry(clip_id)
        if not entry:
            return 0.0
        probs = entry.get("frame_p_siren")
        if probs and 0 <= frame_idx < len(probs):
            return float(probs[frame_idx])
        cosines = entry.get("frame_cosines")
        if cosines and 0 <= frame_idx < len(cosines):
            return float(cosines[frame_idx])
        return float(entry.get("cosine_vs_synth_centroid", 0.0))