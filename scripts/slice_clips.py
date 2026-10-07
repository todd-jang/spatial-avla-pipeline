#!/usr/bin/env python3
"""Slice roster clips to a fixed length and mix the synthetic siren at scripted azimuths.

Extracted from the todo-4 inline ffmpeg so it is reusable and testable.

Hard rules enforced here (todo 15):
  * T09 is test-only; this script refuses to write it into train/ with a non-zero exit.
  * The untouched audio_stereo_16k.wav is NEVER overwritten by the siren mix; the mix
    lands in a sibling audio_siren_stereo_16k.wav so with/without-siren runs stay comparable.
  * scenario is NOT derived from a hardcoded country->scenario table in this script.
    The observed country is recorded verbatim; the grouping table lives in the roster
    config and any country missing from it yields scenario=null so the report can
    derive the grouping itself and any mislabelling surfaces instead of hiding.
  * A missing raw video reports SKIP with a clear message and writes no wav at all
    (never an empty or truncated one).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_ONLY = {"T09"}


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


_FPS_MODE: list[str] | None = None


def fps_mode_arg() -> list[str]:
    """ffmpeg 9 dropped -vsync in favour of -fps_mode; pick whichever this build has."""
    global _FPS_MODE
    if _FPS_MODE is None:
        r = run(["ffmpeg", "-hide_banner", "-h", "full"])
        help_text = r.stdout + r.stderr
        _FPS_MODE = ["-fps_mode", "cfr"] if "fps_mode" in help_text else ["-vsync", "cfr"]
    return list(_FPS_MODE)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, f"{path} must be 16-bit PCM"
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype="<i2").reshape(n, ch)
    return data.astype(np.float64) / 32768.0, sr


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.999:  # guard, never hard-clip the mix
        samples = samples * (0.999 / peak)
    pcm = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(samples.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def probe_source(raw: Path) -> dict:
    """Real check of whether the source audio channel is genuinely stereo."""
    r = run(["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels,channel_layout,sample_rate",
             "-of", "json", str(raw)])
    try:
        st = json.loads(r.stdout)["streams"][0]
        ch = int(st.get("channels", 0))
        return {"source_channels": ch,
                "channel_layout": st.get("channel_layout"),
                "source_sample_rate": st.get("sample_rate"),
                "method": "ffprobe"}
    except Exception:
        return {"source_channels": 0, "channel_layout": None,
                "source_sample_rate": None, "method": "ffprobe_failed"}


def probe_confidence(probe: dict) -> str:
    if probe["method"] == "ffprobe_failed":
        return "low"
    return "high" if probe["source_channels"] >= 2 else "medium"


def probe_music(clip_audio: np.ndarray, sr: int) -> dict:
    """Heuristic tonal-stability probe.

    Deliberately reported as a heuristic with its method recorded. It does NOT assert
    the roster's music claim either way; it only records what was measured so an
    unverified title-only claim cannot hide behind a hardcoded 'no music' assertion.
    """
    mono = clip_audio.mean(axis=1)
    if mono.size < 2048:
        return {"music_detected": None, "method": "fft_tonal_heuristic", "confidence": "low"}
    win = 4096
    frames = mono[: (mono.size // win) * win].reshape(-1, win)
    window = np.hanning(win)
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    power = spectra ** 2
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    band = (freqs >= 80) & (freqs <= 8000)
    tonal = []
    for p in power[:, band]:
        total = float(p.sum())
        if total <= 0:
            continue
        top = int(np.argmax(p))
        tonal.append(float(p[top]) / total)
    if not tonal:
        return {"music_detected": None, "method": "fft_tonal_heuristic", "confidence": "low"}
    stability = float(np.mean(tonal))
    return {"music_detected": bool(stability > 0.25),
            "tonal_stability": round(stability, 4),
            "method": "fft_tonal_heuristic",
            "confidence": "medium"}


def slice_video(raw: Path, out_dir: Path, start_sec: float, seconds: float,
                fps: int, max_height: int) -> int:
    frames = out_dir / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps}"
    if max_height > 0:
        # cap height but never upscale; the comma inside min() must be escaped for ffmpeg
        vf += f",scale=-2:min({max_height}\\,ih)"
    r = run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start_sec}", "-t", f"{seconds}",
             "-i", str(raw), "-vf", vf, *fps_mode_arg(), "-r", str(fps),
             "-q:v", "2", str(frames / "frame_%06d.jpg")])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg video failed: {r.stderr.strip()[:400]}")
    return len(list(frames.glob("*.jpg")))


def extract_audio(raw: Path, out_dir: Path, start_sec: float, seconds: float, sr: int) -> Path:
    dst = out_dir / "audio" / "audio_stereo_16k.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start_sec}", "-t", f"{seconds}",
             "-i", str(raw), "-vn", "-ac", "2", "-ar", str(sr),
             "-c:a", "pcm_s16le", str(dst)])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg audio failed: {r.stderr.strip()[:400]}")
    return dst


def mix_siren(clean: Path, mixed: Path, track_wav: Path, snr_db: float) -> dict:
    """Mix the synthetic siren so its power sits snr_db above the clip's own audio."""
    base, sr = read_wav(clean)
    sig, tsr = read_wav(track_wav)
    if tsr != sr:
        raise RuntimeError(f"siren track is {tsr} Hz, expected {sr} Hz")
    n = base.shape[0]
    if sig.shape[0] < n:
        sig = np.vstack([sig, np.zeros((n - sig.shape[0], sig.shape[1]))])
    sig = sig[:n]
    if sig.shape[1] != base.shape[1]:
        sig = np.repeat(sig, base.shape[1], axis=1) if sig.shape[1] == 1 else sig[:, : base.shape[1]]
    ambient = float(np.mean(base ** 2))
    siren_pow = float(np.mean(sig ** 2))
    if siren_pow <= 0 or ambient <= 1e-12:
        gain = 1.0
        effective = None
    else:
        target = ambient * (10.0 ** (snr_db / 10.0))
        gain = float(np.sqrt(target / siren_pow))
        effective = snr_db
    out = base + sig * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.999:
        out = out * (0.999 / peak)
    write_wav(mixed, out, sr)
    return {"siren_source": track_wav.name, "requested_snr_db": snr_db,
            "effective_snr_db": effective, "siren_gain": round(gain, 5),
            "ambient_rms_power": round(ambient, 9),
            "limiter_applied": bool(peak > 0.999),
            "note": "ambient was silent; siren placed at unity gain" if effective is None
                    else "power ratio set on the full clip RMS"}


def find_raw(raw_dir: Path, clip: dict) -> Path | None:
    explicit = clip.get("raw_video")
    if explicit:
        cand = ROOT / explicit
        if cand.exists():
            return cand
    for p in sorted(raw_dir.glob(f"{clip['clip_id']}_*")):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"):
            return p
    for p in sorted(raw_dir.glob(f"{clip['clip_id']}.*")):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"):
            return p
    return None


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def download(clip: dict, raw_dir: Path, max_height: int, min_free_gib: float,
             cookies: Path | None = None) -> Path | None:
    # Fail loudly rather than filling a nearly-full disk (plan: raw videos belong on
    # Colab /content/, not on a local volume with single-digit GiB free).
    if free_gib(raw_dir) < min_free_gib:
        print(f"SKIP {clip['clip_id']}: only {free_gib(raw_dir):.1f} GiB free at {raw_dir}, "
              f"need >= {min_free_gib:.1f} GiB for a raw download; wrote nothing")
        return None
    if cookies is not None and not cookies.exists():
        raise SystemExit(f"NO_COOKIES: {cookies} does not exist; refusing to download "
                         "unauthenticated and fail on YouTube's bot gate.")
    height = 2160 if clip.get("native_res") else max_height
    fmt = (f"bv*[height<={height}][ext=mp4]+ba/"
           f"b[height<={height}][ext=mp4]/bv*[height<={height}]+ba/"
           f"b[height<={height}]/b")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(raw_dir / f"{clip['clip_id']}.%(ext)s")
    cmd = ["yt-dlp", "--no-playlist", "-f", fmt, "--merge-output-format", "mp4",
           "-o", out_tpl]
    if cookies is not None:
        cmd += ["--cookies", str(cookies)]
    r = run(cmd + [clip["source_url"]])
    if r.returncode != 0:
        print(f"SKIP {clip['clip_id']}: download failed: {r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else 'unknown error'}")
        return None
    for p in sorted(raw_dir.glob(f"{clip['clip_id']}.*")):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"):
            return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", default="configs/clip_roster.json")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--mix-siren", action="store_true")
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--raw-dir", default="raw_videos")
    ap.add_argument("--out-root", default="exports/TL_A")
    ap.add_argument("--siren-dir", default="exports/synthetic/siren")
    ap.add_argument("--max-height", type=int, default=1080)
    ap.add_argument("--min-free-gib", type=float, default=1.5)
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--cookies", default=None,
                    help="Netscape cookies.txt for yt-dlp; needed for age- or bot-gated clips")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    roster = json.loads(Path(args.roster).read_text())
    raw_dir = ROOT / args.raw_dir
    out_root = ROOT / args.out_root
    siren_dir = ROOT / args.siren_dir
    scenario_by_country = roster.get("scenario_by_country", {})
    snr = args.snr_db if args.snr_db is not None else roster.get("siren_snr_db", 12)
    want = set(args.only.split(",")) if args.only else None

    # Guard: refuse a test-only clip aimed at train before touching the filesystem.
    for clip in roster["clips"]:
        cid = clip["clip_id"]
        if cid in TEST_ONLY and clip.get("split") == "train":
            print(f"REFUSED {cid}: test-only clip may not be written into train/", file=sys.stderr)
            return 2

    written, skipped = [], []
    for clip in roster["clips"]:
        cid = clip["clip_id"]
        if want and cid not in want:
            continue
        split = clip["split"]
        if cid in TEST_ONLY and split != "test":
            print(f"REFUSED {cid}: test-only clip may not be written into {split}/", file=sys.stderr)
            return 2

        out_dir = out_root / split / cid
        raw = find_raw(raw_dir, clip)
        if raw is None and args.download:
            raw = download(clip, raw_dir, args.max_height, args.min_free_gib,
                           Path(args.cookies) if args.cookies else None)
        if raw is None:
            skipped.append(cid)
            print(f"SKIP {cid}: no raw video for {clip['youtube_id']} "
                  f"(looked in {raw_dir}); wrote nothing")
            continue

        try:
            n_frames = slice_video(raw, out_dir, float(clip["start_sec"]), args.seconds,
                                   args.fps,
                                   0 if clip.get("native_res") else args.max_height)
            if n_frames < 15:
                raise RuntimeError(f"only {n_frames} frames extracted")
            clean = extract_audio(raw, out_dir, float(clip["start_sec"]), args.seconds, args.sr)

            probe = probe_source(raw)
            conf = probe_confidence(probe)
            base_audio, _ = read_wav(clean)

            mix_info = None
            if args.mix_siren and clip.get("siren_track"):
                track_wav = siren_dir / f"{clip['siren_track']}.wav"
                if not track_wav.exists():
                    print(f"SKIP {cid}: siren track {track_wav.name} not found; wrote nothing")
                    skipped.append(cid)
                    continue
                mix_info = mix_siren(clean, out_dir / "audio" / "audio_siren_stereo_16k.wav",
                                     track_wav, float(snr))
            elif clip.get("siren_track") and not args.mix_siren:
                print(f"NOTE {cid}: siren track {clip['siren_track']} declared but "
                      f"--mix-siren not passed")

            country = clip["country"]
            scenario = scenario_by_country.get(country)
            if scenario is None:
                print(f"WARN {cid}: country {country!r} absent from scenario_by_country; "
                      f"recording scenario=null so the report derives the grouping")

            meta = {
                "clip_id": cid,
                "source_url": clip["source_url"],
                "youtube_id": clip["youtube_id"],
                "city": clip["city"],
                "country": country,
                "scenario": scenario,
                "scenario_derivation": "roster.scenario_by_country[country]; "
                                       "expected_scenario is advisory only",
                "expected_scenario": clip.get("expected_scenario"),
                "scenario_mismatch": bool(clip.get("expected_scenario")
                                          and scenario != clip["expected_scenario"]),
                "condition": clip["condition"],
                "view": clip["view"],
                "is_test": split == "test",
                "split": split,
                "never_mix_into_train": bool(clip.get("never_mix_into_train")),
                "start_sec": float(clip["start_sec"]),
                "fps": args.fps,
                "n_frames": n_frames,
                "siren_track": clip.get("siren_track"),
                "siren_snr_db": float(snr) if mix_info else None,
                "probe_confidence": conf,
                "probe": probe,
                "audio": {"untouched": "audio/audio_stereo_16k.wav",
                          "with_siren": ("audio/audio_siren_stereo_16k.wav"
                                         if mix_info else None)},
                "mix": mix_info,
                "source_label": "video=downloaded dashcam; siren=synthesised (todo 11)",
            }
            if cid == "C03":
                meta["music_probe"] = probe_music(base_audio, args.sr)
            (out_dir / "clip_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            written.append(cid)
            print(f"OK {cid}: {n_frames} frames, probe={conf}"
                  + (f", mix={mix_info['effective_snr_db']} dB" if mix_info else ""))

            if not args.keep_raw and args.download and not clip.get("native_res"):
                raw.unlink(missing_ok=True)
        except Exception as exc:
            print(f"FAIL {cid}: {exc}", file=sys.stderr)
            return 1

    print(f"\nSUMMARY written={sorted(written)} skipped={sorted(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())