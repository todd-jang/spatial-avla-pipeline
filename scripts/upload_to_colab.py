#!/usr/bin/env python3
"""Bundle project source for Colab upload. Excludes heavy/ephemeral artifacts.

Two modes, because the two Colab jobs need different payloads:

  (default)        code only -> colab_bundle.zip (~74 KB).
                   Enough to read the code and run unit tests on Colab.

  --with-audio     code + every 16 kHz stereo wav under exports/ and the
                   synthetic siren sweep -> colab_audio_bundle.zip (~13 MB).
                   This is what the actual experiments need: WavLM precache
                   (todo 5), smoke training (todo 9), the DoA sweep and the
                   notebook 01-05 spatial blocks all read wav files.

Frames are excluded in both modes. The 272 MB of extracted jpg frames are only
used by the stitched-render demo; keeping them out means the audio bundle can be
uploaded through the Colab file picker in seconds instead of going through
Drive. Pull frames over separately (or re-slice from raw video on the Colab box)
only when you actually want outputs/rendered_demos/stitched_*.mp4.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE_DIRS = ["src", "scripts", "tests", "mocks", "contracts", "configs", "notebooks", "golden", "docs"]
INCLUDE_FILES = ["main.py", "run_gradio.py", "requirements.txt"]
EXCLUDE_SUFFIX = {".pt", ".ckpt", ".onnx", ".mp4", ".wav", ".zip"}
EXCLUDE_DIR_NAMES = {"__pycache__", "cache", "raw_videos", "exports", "outputs", ".pytest_cache", ".ipynb_checkpoints"}

AUDIO_INCLUDE_DIRS = ["exports"]
AUDIO_SUFFIXES = {".wav", ".json"}
AUDIO_EXCLUDE_PARTS = {"frames", "raw", "raw_videos", "outputs", "cache", "__pycache__"}


def add_tree(z: zipfile.ZipFile, rel_dir: str, count: int, *, audio_only: bool) -> int:
    root = ROOT / rel_dir
    if not root.exists():
        return count
    excluded = AUDIO_EXCLUDE_PARTS if audio_only else EXCLUDE_DIR_NAMES
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if excluded & set(f.relative_to(ROOT).parts):
            continue
        if f.name.endswith(".ipynb_checkpoints"):
            continue
        if audio_only:
            if f.suffix not in AUDIO_SUFFIXES:
                continue
        elif f.suffix in EXCLUDE_SUFFIX:
            continue
        z.write(f, f.relative_to(ROOT))
        count += 1
    return count


def build(out: Path, *, with_audio: bool) -> Path:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for d in INCLUDE_DIRS:
            n = add_tree(z, d, n, audio_only=False)
        if with_audio:
            for d in AUDIO_INCLUDE_DIRS:
                n = add_tree(z, d, n, audio_only=True)
        for name in INCLUDE_FILES:
            f = ROOT / name
            if f.exists():
                z.write(f, name)
                n += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--with-audio",
        action="store_true",
        help="also bundle exports/**/*.wav (the DoA/WavLM/training payload)",
    )
    args = ap.parse_args()

    out = ROOT / ("colab_audio_bundle.zip" if args.with_audio else "colab_bundle.zip")
    build(out, with_audio=args.with_audio)
    with zipfile.ZipFile(out) as z:
        n = len(z.namelist())
    print(f"{out} ({out.stat().st_size / 1024:.0f} KB, {n} files, audio={args.with_audio})")


if __name__ == "__main__":
    main()
