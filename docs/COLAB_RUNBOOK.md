# Colab T4 Runbook — spatial-avla-pipeline

Written 2026-10-06. Supersedes `~/.omo/evidence/COLAB-RUNBOOK.txt`, which contained a
wrong command (it told you to pass `--clips C01 --out cache/emb` to `precache_wavlm.py`;
that script has no CLI flags at all — see Step 5).

## Why Colab and not the local box

- The local host has no CUDA (`torch.cuda.is_available()` is `False`; MPS only). Both
  GPU-gated steps refuse to run without it, by design:
  - **todo 5** — WavLM pre-cache. This is the *only* place allowed to hold WavLM in VRAM.
  - **todo 9** — smoke training. Its acceptance criterion literally asserts
    `torch.cuda.is_available()`.
- Local disk is at 77% with 3.6 GiB free (measured). Raw 4K dashcam video does not fit; it belongs
  on Colab `/content/`, which is ephemeral and large.

## What to upload

`colab_audio_bundle.zip` — **20.0 MB, 145 files**. Rebuild it any time with:

```
python3 scripts/upload_to_colab.py --with-audio
```

It carries the full spatial chain (which the older 75 KB code-only bundle was missing):
`src/core/audio/` (spatial geometry, GCC-PHAT estimator, wav io), the siren synth, the
clip slicer, the report builder, notebooks 00–05, `configs/clip_roster.json`, the 15
`golden/fixtures` frames, and all 37 wavs:

| payload | count | used by |
|---|---|---|
| `exports/TL_A/{train,test}/*/audio/audio_stereo_16k.wav` | 4 | WavLM pre-cache, smoke training |
| `exports/TL_A/train/*/audio/audio_siren_stereo_16k.wav` | 3 | the with-siren vs without-siren comparison |
| `exports/synthetic/siren/siren_theta*.wav` | 13 | the ±90° DoA sweep (notebook 02, 05) |
| `exports/synthetic/siren/track_*.wav` | 5 | the trigger rows (pass_fast + LHD mirror / fail_slow / static L+R) |
| matching `.json` sidecars + `index.json` | — | ground-truth labels the tests assert against |

**Not included: the 272 MB of extracted `frames/*.jpg`.** Only the stitched-render demo
(notebook 03) needs those. Verified: extracting the bundle to a clean directory and running
`pytest -q` gives **179 passed, 9 skipped**, so the bundle is self-sufficient for every audio experiment.

## Steps

### 0. Runtime
`Runtime → Change runtime type → GPU (T4)`. Then:

```python
!nvidia-smi
import torch; assert torch.cuda.is_available(), 'GPU required'
print('CUDA OK:', torch.cuda.get_device_name(0))
```

### 1. Upload + unpack
Drag `colab_audio_bundle.zip` into the Colab file picker, or:

```python
from google.colab import files
files.upload()
```

```bash
!mkdir -p /content/avla && cd /content/avla && unzip -q ../colab_audio_bundle.zip && ls
```

Keep it on `/content/`, not MyDrive — the audio bundle is small and ephemeral is faster.
Only the artifacts from Step 7 are worth persisting.

### 2. Dependencies
```bash
%cd /content/avla
!pip install -q langgraph pydantic jsonschema numpy scipy opencv-python \
    torch transformers torchaudio matplotlib pytest
```
(`torch` is preinstalled on Colab and already CUDA-enabled. Installing it again is a ~2 GB
download for nothing — if `import torch` already reports `cuda: True`, skip it.)

### 3. Prove the code arrived intact
```bash
!python -m pytest -q
```
Expect `179 passed, 9 skipped`. Anything less means the bundle is stale — re-run
`python3 scripts/upload_to_colab.py --with-audio` locally and re-upload.

### 4. Prove the harness is wired up (no GPU work)
```bash
!python scripts/colab_train_smoke.py --dry-run
```
Fires all three fail-loud guards and rehearses the loss on CPU: `1.041227 -> 0.868845`.

### 5. WavLM pre-cache — todo 5
```bash
!python scripts/precache_wavlm.py
!python scripts/precache_wavlm.py --dir real_sirens
!python scripts/precache_wavlm.py --dir synthetic/siren
```
**All three calls are required.** The default invocation discovers only the nested
layout (`exports/TL_A/*/audio/audio_stereo_16k.wav` → C01/C02/C04/T09 + `*_siren`).
The real-siren recordings (`G44*`) and the synthetic sweep (`siren_theta*`, `track_*`)
live in flat buckets, which only the `--dir` calls pick up. Any other invocation order
is fine — each call merges into `cache/emb/manifest.json` without re-embedding clips it
already has. Expect the third call to report `OK 18 clips` (13 sweep angles + 5 tracks)
so the manifest ends with **31 entries**: 7 dashcam/siren-mixed (C01/C02/C04/T09
+ `*_siren`), 6 real-siren G44*, 13 sweep `siren_theta*`, and 5 track clips
(track_fail_slow, track_pass_fast, track_pass_fast_left, track_static_left,
track_static_right). Now every wav under `exports/` has an embedding — 37 wav files
fold to 31 unique clip ids because the stereo and mono companions of each G44*
recording share one stem.

```python
import json; d = json.load(open('cache/emb/manifest.json'))
print(d['device_used'], len(d['entries']))          # expect: cuda 31
assert d['device_used'] == 'cuda', d
```

Two rules this script enforces for you: it asserts `stereo.shape[1] == 2` *before* deriving
the mono copy WavLM needs, and it never writes that mono copy to disk.

Keep `cache/emb/*.pt` on `/content/` — embeddings are large. Copy **only** `manifest.json`
to Drive if the team needs to see which clips were cached.

### 6. Smoke training — todo 9
```bash
!python scripts/colab_train_smoke.py \
    --embedding cache/emb/C01.pt --clip C01 --epochs 3 --frames 16
```
Look for `ARTIFACTS_OK initial_loss=... final_loss=...`. It writes
`models/checkpoints/smoke/{best.pt,loss_curve.png,train_log.json}` and mirrors the log and
curve into `outputs/smoke/`.

```python
import json; d = json.load(open('models/checkpoints/smoke/train_log.json'))
assert d['initial_loss'] > d['final_loss'] and d['epochs'] >= 3, d
assert d['device_used'] == 'cuda' and d['wavlm_loaded_in_vram'] is False, d
```

The target is window-RMS-above-median on the clip's own audio. It proves the harness learns;
it is **not** a siren-detection accuracy claim, and `train_log.json` says so in
`task_note`. Do not quote it as one.

### 6.5 — MVP core: the train/validate/test loop (p_siren head)

The smoke run above only proves the harness learns. The actual MVP loop is the p_siren
head: an L2-regularised logistic over the cached WavLM frames, evaluated with
leave-one-clip-out CV. It runs in seconds on CPU — no GPU needed:

```bash
!python scripts/score_p_siren.py     # writes outputs/p_siren_report.json (cosine-rejection archive)
!python scripts/fit_p_siren_head.py  # adds the head section + frame_p_siren to the report
```

**Order matters.** `score_p_siren.py` rebuilds `outputs/p_siren_report.json` from scratch,
so running it *after* `fit_p_siren_head.py` silently wipes the `head` section and the
`frame_p_siren` arrays. Always run score first, fit last.

Labels (fixed in `fit_p_siren_head.py`):
- **positives**: the 6 real-siren G44* recordings
- **negatives (LOCO)**: the 4 siren-free dashcam audios C01/C02/C04/T09 — **T09 is a
  negative control here**, distinct from its test-only role in the spatial roster
- **held out entirely** (generalisation only): siren-mixed C*_siren, synthetic track_*
  and siren_theta* clips

Measured on the same 4-clip + G44 corpus (deterministic — re-running fit rebuilds the
demo manifest IDENTICAL):
- LOCO overall accuracy **0.8272**; per clip: C01 0.964, C02 1.000, C04 0.976, **T09 0.9052**,
  G44-04/05/07 ≥ 0.91, G44-01/02 weaker (0.617/0.667)
- held-out: the head does **not** fire on synthetic-siren mixes (C*_siren mean_p ≤ 0.075,
  siren_theta*/track_* ≈ 0.000) — it separates *real* sirens from clean dashcam, so quote
  it as that, nothing more.

Consumers of the same report: the programmatic demo `scripts/build_demo_sequence.py`
(re-runs IDENTICAL on the same report) and the live graph's `P_sirenProvider`.

### 7. Notebooks
Open `notebooks/01…05` and run them in order from `/content/avla`. Each one locates the
project root by walking up to `configs/clip_roster.json`, so they work from any directory.

| notebook | runs on the audio bundle? | what it gives you |
|---|---|---|
| `01_Setup_and_Roster` | yes | per-clip frame count, channel/rate of each wav, which clips are missing |
| `02_DoA_Sweep` | yes | estimated vs ground-truth angle over the 13-angle sweep |
| `05_Spatial_Audio_DoA` | yes | the sweep plot, the ±2° band, playable motion-track audio, trigger rows |
| `04_Evaluation_and_Export` | yes, with `--allow-missing` | `outputs/poc_report.json` spatial block |
| `03_MultiView_Render` | **no** | needs `frames/*.jpg`, which the audio bundle omits |

`00_Setup.ipynb` is the older Drive-based bootstrap. You do not need it for this flow.

### 8. Persist results
```python
from google.colab import drive; drive.mount('/content/drive')
!cp -r models/checkpoints/smoke /content/drive/MyDrive/ADAS_MultiView_POC/
!cp outputs/poc_report.json /content/drive/MyDrive/ADAS_MultiView_POC/
```

## Getting the 4 blocked clips — todo 15

`C03`, `C05`, `C06`, `C07` are blocked locally by YouTube's bot gate. **Worth trying from
Colab first**: the gate is largely IP/ASN-based, so a different egress IP often gets through,
and it costs two minutes. With cookies it is near-certain:

```python
!pip install -q yt-dlp cookiejar
from google.colab import files
files.upload()   # upload cookies.txt exported from a signed-in browser
```

```bash
!python scripts/slice_clips.py --roster configs/clip_roster.json \
    --seconds 15 --mix-siren --snr-db 12 --cookies cookies.txt --download
```

Or skip YouTube entirely: drop the mp4s into `raw_videos/<CLIP_ID>.mp4` and run the slicer
**without** `--download`; it picks up pre-placed files as-is.

Then `!python scripts/build_poc_report.py` (no `--allow-missing` — it is supposed to fail
loudly until every `clip_meta.json` exists). Expect an `S2_global` group beside `S1_korea`
and `ROSTER_OK` to pass. That closes todos 15 and 16 and unblocks the report you actually
want.

## Gotchas

- **`precache_wavlm.py` understands only `--dir`.** Any other flag (`--clips`, `--out`) is
  silently ignored. The default call covers the nested layout; the flat buckets need the
  two `--dir` calls in Step 5. All three are required before score/fit (Step 6.5).
- **Never load WavLM into the train or infer path.** That is why pre-caching is a separate
  script; it is what keeps todo 9 inside T4 memory.
- **`raw_videos/C05.f398.mp4.part` was deleted on 2026-10-07** — it was 720p AV1,
  4868 s, with **no audio stream at all** (verified with ffprobe before deletion), so it
  could not be sliced for a DoA experiment. Deleting it freed 826 MB; local disk now
  reports 4.6 GiB free (73%). If a new partial download appears, delete it the same way.
- **Notebook 05's last cell** reads `~/.omo/evidence/task-13-spatial-avla-poc.txt`, which
  does not exist on Colab. It prints `evidence file not found at …` and stops. Harmless —
  that cell only echoes a table; every measured number in the notebook is computed above it.
- **A clipped clip still reports honestly.** `clip_meta.json` records `probe_confidence`
  from a real channel check, and the report labels every DoA figure `source: "synthetic"`.
  Keep it that way when you add clips.