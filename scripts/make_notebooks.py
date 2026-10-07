#!/usr/bin/env python3
"""Generate notebooks 01-05 so they stay reproducible and diffable.

Each notebook is thin on prose and delegates to the real, tested scripts in scripts/,
so a reader running them exercises the same code path the evidence was collected from.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"

SETUP = [
    "import json, subprocess, sys\n"
    "from pathlib import Path\n\n"
    "def _find_root(start):\n"
    "    for p in [start, *start.parents]:\n"
    "        if (p / 'configs' / 'clip_roster.json').exists():\n"
    "            return p\n"
    "    raise RuntimeError('repo root not found: no configs/clip_roster.json above ' + str(start))\n\n"
    "ROOT = _find_root(Path.cwd().resolve())\n"
    "sys.path.insert(0, str(ROOT))\n"
    "print('root', ROOT)",
]

nb = lambda cells: {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
               "outputs": [], "source": c} for c in cells],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}


N01 = nb([
    "# 01 - Setup and clip roster\n"
    "# Confirms the sliced roster and that every train clip carries stereo 16 kHz audio.\n"
    + SETUP[0],
    "import json, glob, os, wave\n"
    "roster = json.loads((ROOT / 'configs/clip_roster.json').read_text())\n"
    "print(f\"{len(roster['clips'])} roster entries\")\n"
    "for c in roster['clips']:\n"
    "    d = ROOT / 'exports/TL_A' / c['split'] / c['clip_id']\n"
    "    meta = d / 'clip_meta.json'\n"
    "    wav = d / 'audio/audio_stereo_16k.wav'\n"
    "    n = len(glob.glob(str(d / 'frames/*.jpg')))\n"
    "    if wav.exists():\n"
    "        w = wave.open(str(wav)); ch, sr = w.getnchannels(), w.getframerate()\n"
    "    else:\n"
    "        ch = sr = None\n"
    "    print(c['clip_id'], c['split'], 'frames=', n, 'audio=', (ch, sr), 'meta=', meta.exists())",
    "missing = [c['clip_id'] for c in roster['clips']\n"
    "           if not (ROOT / 'exports/TL_A' / c['split'] / c['clip_id'] / 'clip_meta.json').exists()]\n"
    "print('MISSING (raw download blocked):', missing)",
])

N02 = nb([
    "# 02 - DoA estimator on the synthetic sweep\n"
    "# Runs the real estimator over the 13-angle sweep and reports error against ground truth.\n"
    + SETUP[0],
    "import json, wave\n"
    "import numpy as np\n"
    "from src.core.audio.doa_gccphat import estimate_doa\n\n"
    "def read_wav(p):\n"
    "    with wave.open(str(p), 'rb') as w:\n"
    "        d = np.frombuffer(w.readframes(w.getnframes()), '<i2')\n"
    "        return d.reshape(-1, w.getnchannels()).astype(np.float64)/32768.0, w.getframerate()\n\n"
    "idx = json.loads((ROOT / 'exports/synthetic/siren/index.json').read_text())\n"
    "errs = []\n"
    "for e in idx['static']:\n"
    "    a, sr = read_wav(ROOT / 'exports/synthetic/siren' / f\"{e['clip']}.wav\")\n"
    "    r = estimate_doa(a[:, 0], a[:, 1], sr=sr)\n"
    "    err = abs(r['dir_deg'] - e['gt_dir_deg'])\n"
    "    errs.append(err)\n"
    "    print(f\"{e['clip']:>18} gt={e['gt_dir_deg']:>6.1f} est={r['dir_deg']:>7.2f} err={err:>5.2f} amb={r['ambiguous']}\")\n"
    "print('max abs err', round(max(errs), 4), 'deg')",
])

N03 = nb([
    "# 03 - Multi-view render\n"
    "# Renders the stitched 2x2 comparison view through the real graph and writes the mp4.\n"
    + SETUP[0],
    "import subprocess, sys\n"
    "r = subprocess.run([sys.executable, 'scripts/render_stitched.py',\n"
    "                    '--clip', 'C01', '--max-frames', '60'],\n"
    "                   cwd=str(ROOT), capture_output=True, text=True)\n"
    "print(r.stdout or r.stderr)\n"
    "assert r.returncode == 0, 'render failed'",
])

N04 = nb([
    "# 04 - Evaluation and export\n"
    "# Writes outputs/poc_report.json: base keys plus the measured spatial block.\n"
    + SETUP[0],
    "import subprocess, sys, json\n"
    "r = subprocess.run([sys.executable, 'scripts/build_poc_report.py', '--allow-missing'],\n"
    "                   cwd=str(ROOT), capture_output=True, text=True)\n"
    "print(r.stdout or r.stderr)\n"
    "report = json.loads((ROOT / 'outputs/poc_report.json').read_text())\n"
    "print('base keys:', sorted(k for k in report if k != 'spatial'))\n"
    "sp = report['spatial']\n"
    "print('spatial source:', sp['source'], '| status:', sp.get('status'))\n"
    "print('sweep max abs err:', sp['sweep']['max_abs_err_deg'])\n"
    "print('triggers:', {t['track']: t['trigger_fired'] for t in sp['trigger_rows']})",
])

CAVEAT = (
    "**Caveat - read this before quoting any number below.** Every DoA figure in this "
    "notebook comes from a *synthesised* siren rendered by `scripts/synth_siren_stereo.py` "
    "and mixed into downloaded dashcam video. **Real recorded sirens are UNMEASURED.** "
    "Nothing here is a claim about real-world siren localisation accuracy."
)

N05 = nb([
    "# 05 - Spatial audio / DoA\n" + CAVEAT + "\n" + SETUP[0],
    "import json, wave\n"
    "import numpy as np\n"
    "from src.core.audio.doa_gccphat import estimate_doa\n\n"
    "def read_wav(p):\n"
    "    with wave.open(str(p), 'rb') as w:\n"
    "        d = np.frombuffer(w.readframes(w.getnframes()), '<i2')\n"
    "        return d.reshape(-1, w.getnchannels()).astype(np.float64)/32768.0, w.getframerate()\n\n"
    "idx = json.loads((ROOT / 'exports/synthetic/siren/index.json').read_text())\n"
    "gt, est, conf = [], [], []\n"
    "for e in idx['static']:\n"
    "    a, sr = read_wav(ROOT / 'exports/synthetic/siren' / f\"{e['clip']}.wav\")\n"
    "    r = estimate_doa(a[:, 0], a[:, 1], sr=sr)\n"
    "    gt.append(e['gt_dir_deg']); est.append(r['dir_deg']); conf.append(r['confidence'])\n"
    "err = np.abs(np.array(est) - np.array(gt))\n"
    "for g, s, c, x in zip(gt, est, conf, err):\n"
    "    print(f'gt={g:>6.1f} est={s:>7.2f} err={x:>5.2f} conf={c:.3f}')\n"
    "print('max', round(err.max(), 4), 'mean', round(err.mean(), 4))",
    "try:\n"
    "    import matplotlib\n"
    "    matplotlib.use('Agg')\n"
    "    import matplotlib.pyplot as plt\n"
    "    fig, ax = plt.subplots(figsize=(7, 5))\n"
    "    ax.plot(gt, est, 'o-')\n"
    "    ax.plot([-90, 90], [-90, 90], 'k--', lw=1, label='perfect')\n"
    "    ax.fill_between([-90, 90], [-90 + 2, 90 + 2], [-90 - 2, 90 - 2],\n"
    "                    color='green', alpha=0.15, label='+-2 deg ideal band')\n"
    "    ax.set_xlabel('ground truth (deg)'); ax.set_ylabel('estimated (deg)')\n"
    "    ax.set_title('DoA sweep - synthetic siren (source=synthetic)')\n"
    "    ax.legend(); ax.grid(alpha=0.3)\n"
    "    fig.savefig('doa_sweep.png', dpi=120, bbox_inches='tight')\n"
    "    print('wrote doa_sweep.png')\n"
    "except ImportError:\n"
    "    print('matplotlib unavailable; skipping the plot (the table above is the measurement)')",
    "import glob, shutil, base64\n"
    "from IPython.display import Audio, display\n"
    "for t in ('track_pass_fast', 'track_pass_fast_left', 'track_fail_slow', 'track_static_left', 'track_static_right'):\n"
    "    src = ROOT / 'exports/synthetic/siren' / f'{t}.wav'\n"
    "    if not src.exists():\n"
    "        print('missing', src); continue\n"
    "    print(f'{t}: 2ch 16 kHz synthetic motion track')\n"
    "    display(Audio(str(src)))",
    "ev = Path.home() / '.omo/evidence/task-13-spatial-avla-poc.txt'\n"
    "print('measured table from task-13 evidence:')\n"
    "print(ev.read_text() if ev.exists() else 'evidence file not found at', ev)",
])


def main() -> int:
    NB.mkdir(parents=True, exist_ok=True)
    out = {
        "01_Setup_and_Roster.ipynb": N01,
        "02_DoA_Sweep.ipynb": N02,
        "03_MultiView_Render.ipynb": N03,
        "04_Evaluation_and_Export.ipynb": N04,
        "05_Spatial_Audio_DoA.ipynb": N05,
    }
    for name, doc in out.items():
        (NB / name).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote notebooks/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())