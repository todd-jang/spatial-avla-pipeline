"""Shared helpers for both demo renderers (render_stitched.py, gradio_app.py).

Everything here exists to enforce ONE property: the demo can only play what
outputs/demo_sequence.json has PROVEN from measured p_siren data, and every
played segment must REPRODUCE its manifest-encoded expectation at render time
(must_be_quiet / must_fire_within_window). A clip that contradicts its own
manifest entry is a loud failure, never a silent blank tile.

The FSM the renderers drive must carry state across frames, otherwise S_B's
hysteresis never actually holds: use bridge.invoke_frame_stateful(prev_state=...),
not the stateless invoke_frame() (which fresh-copies the mock every call).
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import cv2
import numpy as np

from src.core.audio.doa_gccphat import estimate_doa
from src.core.audio.frame_clock import frame_start_sample

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST = ROOT / "outputs" / "demo_sequence.json"

# Render constants shared by both demo renderers (headless + gradio).
SR = 16000
FPS = 30
WIN = 8000
REAL_SIREN_AUDIO_DIR = ROOT / "exports" / "real_sirens"


def load_manifest(path: Path | str | None = None) -> dict:
    """Load the demo manifest, refusing loudly when it is absent/invalid."""
    p = Path(path) if path is not None else DEFAULT_MANIFEST
    if not p.exists():
        raise SystemExit(
            f"FAIL: demo manifest missing: {p}\n"
            f"  Run: python3 scripts/build_demo_sequence.py   (derives the verified "
            f"sequence from outputs/p_siren_report.json)"
        )
    try:
        manifest = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"FAIL: demo manifest {p} is not valid JSON: {e}")
    if manifest.get("schema") != "demo-sequence/v1":
        raise SystemExit(f"FAIL: {p} is not a demo-sequence/v1 manifest")
    if not manifest.get("segments"):
        raise SystemExit(f"FAIL: {p} has zero segments; nothing to demo")
    return manifest


def placeholder_frame(label: str, w: int = 1280, h: int = 720) -> np.ndarray:
    """BGR canvas for audio-only clips (real sirens have no dashcam video).

    process_multi_view_frame fits any size down to the 960x540 tile, so the
    placeholder only needs to be visually distinct and labelled -- the HUD
    overlay (p_siren, hysteresis, trigger state, DoA) is drawn on top by the
    tile renderer and that is what changes per frame.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (14, 16, 26)
    # faint diagonal sheen so the tile does not read as pure black
    ramp = np.linspace(0, 30, h, dtype=np.uint8)
    img += ramp[:, None, None]
    cv2.putText(img, "AUDIO-ONLY SEGMENT", (48, h // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 200, 255), 2)
    cv2.putText(img, label, (48, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    cv2.putText(img, "no dashcam video; HUD shows measured pipeline state",
                (48, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)
    return img


def assert_segment(seg: dict, observed: dict) -> tuple[bool, str]:
    """Check a played segment against its manifest-encoded expectation.

    ``observed`` must carry:
      active_frames           -- frames where event_trigger.is_triggered
      first_raw_trigger_frame -- first frame with raw_trigger, or -1

    Returns (ok, message). ``ok`` False means the demo MUST stop: the clip
    behaved differently from what the measured manifest promised.
    """
    clip_id = seg.get("clip_id", "?")
    expected = seg.get("expected")
    ar = seg.get("assert", {})
    af = int(observed.get("active_frames", 0))
    first_raw = int(observed.get("first_raw_trigger_frame", -1))
    win = int(seg.get("demo_n_frames", 0))

    problems = []
    if ar.get("must_be_quiet") and af != 0:
        problems.append(f"expected QUIET but fired {af}/{win} frames "
                        f"(first raw at f{first_raw})")
    if ar.get("must_fire_within_window"):
        if first_raw < 0:
            problems.append(f"expected TRIGGER but the FSM never fired in {win} frames")
        else:
            mark = "hold" if af > 0 else "no-hold"
            if af <= 0:
                problems.append(f"expected TRIGGER (first raw at f{first_raw}) but "
                                f"hysteresis produced zero active frames")
    if expected not in ("quiet", "trigger"):
        problems.append(f"manifest carries unknown expected={expected!r}")

    if problems:
        return False, f"[{clip_id}] {" | ".join(problems)}"
    return True, f"[{clip_id}] PASS expected={expected} active={af}/{win} " \
                 f"first_raw=f{first_raw}"


def segments_by_manifest(manifest: dict) -> list[dict]:
    """Return manifest segments in their programmatic order."""
    return sorted(manifest["segments"], key=lambda s: int(s.get("order", 0)))


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype="<i2").reshape(n, ch)
    return data.astype(np.float64) / 32768.0, w.getframerate()


def doa_series(wav_path: Path, gt_track: Path | None, n_frames: int) -> list[dict]:
    """Measured DoA per frame from a wav, with GT when a siren track exists."""
    if not wav_path.exists():
        raise FileNotFoundError(
            f"{wav_path} is absent; refusing to render a DoA overlay from "
            f"invented values. Run scripts/slice_clips.py --mix-siren for video "
            f"clips, or check exports/real_sirens for real-siren audio.")
    audio, sr = read_wav(wav_path)
    gt = None
    if gt_track and gt_track.exists():
        raw = json.loads(gt_track.read_text())["gt_dir_deg_per_frame"]
        gt = [(float(x) + 180.0) % 360.0 - 180.0 for x in raw]

    out = []
    for i in range(n_frames):
        start = min(frame_start_sample(i), max(0, len(audio) - WIN))
        res = estimate_doa(audio[start:start + WIN, 0], audio[start:start + WIN, 1], sr=sr)
        frame = {"dir_deg": res["dir_deg"], "gt_dir_deg": -1.0,
                 "confidence": res["confidence"], "direction": res["direction"],
                 "ambiguous": res["ambiguous"]}
        if gt and i < len(gt):
            frame["gt_dir_deg"] = gt[i]
        out.append(frame)
    return out


def segment_audio(seg: dict) -> tuple[Path, Path | None]:
    """(wav_path, gt_track) for a manifest segment.

    Video clips carry the siren mix (and an optional synthetic GT track) under
    exports/TL_A/train/<clip>/. Audio-only real sirens live in
    exports/real_sirens/<clip>_stereo_16k.wav with no GT.
    """
    clip_id = seg["clip_id"]
    if seg.get("has_video"):
        clip_dir = ROOT / "exports" / "TL_A" / "train" / clip_id
        wav = clip_dir / "audio" / "audio_siren_stereo_16k.wav"
        meta_path = clip_dir / "clip_meta.json"
        track = None
        if meta_path.exists():
            track = json.loads(meta_path.read_text()).get("siren_track")
        gt = ROOT / "exports/synthetic/siren" / f"{track}.json" if track else None
        return wav, gt
    wav = REAL_SIREN_AUDIO_DIR / f"{clip_id}_stereo_16k.wav"
    return wav, None


def resolve_frames(seg: dict, n: int) -> list[Path] | None:
    """Video frame paths for video segments, None for audio-only segments."""
    if not seg.get("has_video"):
        return None
    frames = sorted((ROOT / "exports/TL_A/train" / seg["clip_id"] / "frames").glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames under exports/TL_A/train/{seg['clip_id']}/frames")
    return frames[:n]