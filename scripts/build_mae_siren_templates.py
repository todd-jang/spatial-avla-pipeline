"""Build full-recording AST templates for siren and road audio."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import ASTModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.mae_vision import (
    AST_MODEL_NAME,
    CLIPS_ROOT,
    DEFAULT_TEMPLATE_PATH,
    ast_feature_from_file,
    save_template_file,
)

REAL_SIRENS_ROOT = ROOT / "exports" / "real_sirens"
SIREN_TRAIN_STEMS = (
    "G44-01-Sirens_16k",
    "G44-02-Air Raid Siren_16k",
    "G44-03-Whoopee Whistle Siren_16k",
)
ROAD_TRAIN_CLIPS = ("C01", "C02")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--clips-root", type=Path, default=CLIPS_ROOT)
    parser.add_argument("--sirens-root", type=Path, default=REAL_SIRENS_ROOT)
    args = parser.parse_args()
    device = pick_device(args.device)

    siren_files_by_stem = {
        path.stem: path for path in args.sirens_root.glob("G44-*_16k.wav")
    }
    missing_sirens = [
        stem for stem in SIREN_TRAIN_STEMS if stem not in siren_files_by_stem
    ]
    if missing_sirens:
        raise FileNotFoundError(
            f"Missing siren training samples in {args.sirens_root}: {missing_sirens}"
        )
    siren_files = [siren_files_by_stem[stem] for stem in SIREN_TRAIN_STEMS]

    clip_metas = {}
    for meta_path in args.clips_root.rglob("clip_meta.json"):
        with meta_path.open(encoding="utf-8") as stream:
            meta = json.load(stream)
        clip_metas[meta["clip_id"]] = (meta_path, meta)

    road_files = []
    for clip in ROAD_TRAIN_CLIPS:
        if clip not in clip_metas:
            raise FileNotFoundError(f"Missing clip metadata for road template clip {clip}")
        meta_path, meta = clip_metas[clip]
        relative_path = meta.get("audio", {}).get("untouched")
        if not isinstance(relative_path, str):
            raise ValueError(f"{clip} has no untouched audio in {meta_path}")
        audio_path = meta_path.parent / relative_path
        if not audio_path.is_file():
            raise FileNotFoundError(f"Untouched audio missing for {clip}: {audio_path}")
        road_files.append((clip, audio_path))

    print(f"[device] {device}")
    print(f"[model] loading {AST_MODEL_NAME}")
    model = ASTModel.from_pretrained(AST_MODEL_NAME).to(device).eval()

    print(f"[templates] extracting full-recording features from {len(siren_files)} siren samples")
    siren_features = []
    siren_sources = []
    for path in siren_files:
        siren_features.append(ast_feature_from_file(model, path, device))
        siren_sources.append(path.name)
        print(f"  [siren] {path.name}")

    print(f"[templates] extracting full-recording features from {len(road_files)} untouched road samples")
    road_features = []
    road_sources = []
    for clip, path in road_files:
        road_features.append(ast_feature_from_file(model, path, device))
        road_sources.append(f"{clip}:{path.name}")
        print(f"  [road] {clip}: {path.name}")

    payload = save_template_file(
        args.output,
        AST_MODEL_NAME,
        siren_features,
        road_features,
        siren_sources,
        road_sources,
    )
    print(
        f"[saved] {args.output} "
        f"siren={len(payload['siren_sources'])} road={len(payload['road_sources'])} "
        f"dim={payload['siren_template'].numel()}"
    )


if __name__ == "__main__":
    main()
