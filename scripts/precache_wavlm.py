"""Pre-cache WavLM embeddings for every audio clip under an exports directory.

WavLM lives only here and in colab_train_smoke's dry-run stub. Once embeddings
exist, every downstream consumer (training, p_siren scoring) runs without the
~1.2 GB model resident. This script is deliberately GPU-optional: it picks cuda
when available and falls back to cpu, so the real-siren/e2-p_siren track can
pre-cache on a laptop in parallel with the T4 track.

CLI:
    # default: every exports/**/audio_stereo_16k.wav (legacy layout)
    python3 scripts/precache_wavlm.py

    # flat layout: every bonuses_16k.wav under exports/<dir>
    python3 scripts/precache_wavlm.py --dir real_sirens
    python3 scripts/precache_wavlm.py --dir synthetic/siren
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from src.core.audio.wav_io import load_wav_stereo_16k  # noqa: E402


def iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def discover(exp: Path, subdir: str | None) -> list[tuple[Path, str]]:
    """Return (wav_path, clip_name) pairs for the target layout.

    Legacy layout nests clips as exports/<grp>/<clip>/audio/audio_stereo_16k.wav
    and names the clip after the third path part (C01, T09, ...). The flat
    --dir layout keeps files at exports/<subdir>/<name>*_16k.wav and names the
    clip after the file stem, so real-siren and synthetic entries get their own
    readable ids instead of every flat file collapsing to "C01".
    """
    if subdir is None:
        pairs = []
        for w in sorted(exp.rglob("audio_stereo_16k.wav")):
            parts = w.relative_to(exp).parts
            clip = parts[2] if len(parts) > 2 else w.stem
            pairs.append((w, clip))
        for w in sorted(exp.rglob("audio_siren_stereo_16k.wav")):
            parts = w.relative_to(exp).parts
            clip = parts[2] if len(parts) > 2 else w.stem
            pairs.append((w, f"{clip}_siren"))
        return pairs

    root = exp / subdir
    if not root.is_dir():
        raise SystemExit(f"NO_DIR: {root} is not a directory under exports/")
    # Prefer the *_stereo_16k.wav variant when one exists next to the mono
    # companion, so a flat bucket can hold both without double entries.
    # Anything under a raw/ subdir is unnormalised provenance, not pipeline
    # audio, and is skipped regardless of naming.
    by_stem: dict[str, Path] = {}
    for w in sorted(root.rglob("*.wav")):
        if "raw" in w.relative_to(root).parts:
            continue
        stem = w.stem
        if stem.endswith("_stereo_16k"):
            by_stem[stem[:-len("_stereo_16k")]] = w
        elif stem.endswith("_16k"):
            by_stem.setdefault(stem[:-len("_16k")], w)
        else:
            by_stem.setdefault(stem, w)
    return [(w, stem) for stem, w in by_stem.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None,
                    help="exports-relative subdir with flat *_16k.wav files "
                         "(default: legacy nested layout)")
    args = ap.parse_args()

    exp = Path("exports")
    emb = Path("cache/emb")
    emb.mkdir(parents=True, exist_ok=True)
    wavs = discover(exp, args.dir)
    import torch
    from transformers import WavLMModel, AutoFeatureExtractor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev)
    m.eval()
    man = {"version": "1.0", "generated_at": iso(), "device_used": dev, "entries": {}}
    if (emb / "manifest.json").exists():
        prior = json.loads((emb / "manifest.json").read_text())
        man["entries"] = {k: v for k, v in prior.get("entries", {}).items()
                          if (Path(v["emb_path"])).exists()}
    for w, clip in wavs:
        stereo = load_wav_stereo_16k(w)
        assert stereo.shape[1] == 2, f"NOT_STEREO {w}: shape {stereo.shape}"
        mono = stereo.mean(axis=1)
        inp = ext(mono, sampling_rate=16000, return_tensors="pt")["input_values"].to(dev)
        with torch.no_grad():
            out = m(inp)
            embt = out.last_hidden_state
        p = emb / f"{clip}.pt"
        torch.save(embt.detach().cpu(), p)
        man["entries"][clip] = {"emb_path": str(p), "shape": list(embt.shape)}
    (emb / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"OK {len(wavs)} clips -> cache/emb on {dev}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
