import sys
import json
from pathlib import Path
from datetime import datetime, timezone


def iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main():
    exp = Path("exports")
    emb = Path("cache/emb")
    emb.mkdir(parents=True, exist_ok=True)
    wavs = sorted(exp.rglob("audio_stereo_16k.wav"))
    if not wavs:
        print("NO WAV")
        sys.exit(2)
    import torch
    import torchaudio
    from transformers import WavLMModel, WavLMFeatureExtractor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = WavLMFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev)
    m.eval()
    man = {"version": "1.0", "generated_at": iso(), "device_used": dev, "entries": {}}
    for w in wavs:
        try:
            parts = w.relative_to(exp).parts
        except Exception:
            parts = w.parts
        clip = parts[2] if len(parts) > 2 else "C01"
        wav, sr = torchaudio.load(w)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        inp = ext(wav.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt")["input_values"].to(dev)
        with torch.no_grad():
            out = m(inp)
            embt = out.last_hidden_state
        p = emb / f"{clip}.pt"
        torch.save(embt.detach().cpu(), p)
        man["entries"][clip] = {"emb_path": str(p), "shape": list(embt.shape)}
    (emb / "manifest.json").write_text(json.dumps(man, indent=2))
    print("OK")


if __name__ == "__main__":
    main()
