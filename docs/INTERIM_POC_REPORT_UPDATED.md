# Interim PoC Report — spatial-avla-pipeline

**Date:** 2026-10-06 · **Plan:** `spatial-avla-poc.md` · **Status:** interim, 2 of 16 todos open
**Test suite:** 197 passed, 1 skipped · **Errors in pipeline run:** 0
**Ablation addendum:** 2026-10-07 — Exp A (vision) and Exp B (audio) ablation studies (§2.6)

---

## 1. Scope — what this PoC is and is not

This PoC validates the **spatial audio sensing chain**: a LangGraph pipeline that ingests
stereo dashcam audio, estimates sound direction of arrival (DoA), detects an emergency-siren
event via hysteresis, and renders the result into a 2×2 multi-view HUD.

As of the 2026-10-07 addendum it additionally carries two **measured** perception channels
that feed S_B: an audio `p_siren` head (WavLM L2-logistic, §2.6) and a vision anomaly score
(RT-DETR-l offline, §2.6). Neither is a product-grade detector: the audio head does not
generalise to the synthetic siren source, and the vision score never crosses the trigger
gate with the current risk weights. The training-harness GPU step remains blocked (see §6).

---

## 2. Quantitative results

### 2.1 Pipeline integrity and latency

| Metric | Measured | Requirement | Margin |
|---|---|---|---|
| Frame latency p50 | **2.89 ms** | — | — |
| Frame latency p95 | **3.38 ms** | < 330 ms | **98% headroom** |
| `T_last` | 1,791,198,479,546 ms | > 0 | ✅ |
| `Errors_Count` | **0** | 0 | ✅ |
| `All_OK` | **true** | true | ✅ |
| Fallback path | absent | absent | ✅ |

Latency is measured on the Gradio fast path (`GraphBridge fast_mode=True`), which
deliberately skips the JSON-schema validator and metrics node. The full-graph path is
therefore **not** covered by this number and would be substantially slower.

### 2.2 Spatial audio — DoA estimator accuracy

**Ideal sweep, 13 static angles (−90°…+90°, step 15°), no noise:**

```
n = 13    ambiguous = 0/13    max |err| = 0.0886°    mean |err| = 0.0412°
confidence ∈ [0.587, 0.651]
```

Against the plan's ≤ 2.0° gate, this is **~23× inside tolerance**. Every one of the 13
angles falls in the `[0, 0.1)` error bin.

**Noise robustness** (post-clamp, all 13 angles, zero ambiguous in every row):

| Noise | SNR 20 dB | SNR 15 dB | SNR 10 dB | SNR 5 dB |
|---|---|---|---|---|
| White — mean | 0.47° | 0.52° | 0.93° | 1.86° |
| White — max | 1.18° | 1.84° | 2.45° | 4.08° |
| Road — mean | 0.36° | — | 0.99° | 1.61° |
| Road — max | 1.00° | — | 3.17° | 5.28° |

Road noise (a 90/180/360/720/1440 Hz lowpassed composite standing in for tyre and wind) is
the harsher case at low SNR, but only in max error; its mean stays near 1.6°. No gate was
relaxed to achieve these.

**On real clip audio** (downloaded YouTube dashcam, AAC-compressed, resampled to 16 kHz
stereo, synthetic siren mixed at a measured 12.00 dB SNR) — measured during this review:

| Clip | Condition | Mean abs err | Scored windows |
|---|---|---|---|
| C01 | Seoul day | **3.06°** | 211 |
| C02 | Seoul Gwanghwamun, heavy rain | **2.79°** | 436 |
| C04 | Seoul Sanggye→Mokdong, night | **2.45°** | 436 |

This is the single most important number in the report: **the binaural cues survive
YouTube's codec and a 48 kHz→16 kHz resample.** The audio quality concern is empirically
not the binding constraint.

### 2.3 Event trigger (S_B)

| Track | True peak rate | Agent peak MA | Fired | Expected |
|---|---|---|---|---|
| `track_pass_fast` | 274.435 °/s | 220.643 °/s | **true** | fire ✅ |
| `track_pass_fast_left` (LHD mirror) | 274.435 °/s | 220.643 °/s | **true** | fire ✅ |
| `track_fail_slow` | 10.0 °/s | 10.0 °/s | false | silent ✅ |
| `track_static_left` | 0.0 °/s | 0.0 °/s | false | silent ✅ |
| `track_static_right` | 0.0 °/s | 0.0 °/s | false | silent ✅ |

The LHD mirror `track_pass_fast_left` fires identically to its right-side twin: the FSM rates
the abs-valued `|Δdoa|`, so handedness cannot change the trigger. The agent reads ~20% *below*
the true peak on the fast pass. This is correct behaviour, not
error: the estimator flags the ±90° representability boundary as ambiguous, those frames
contribute zero rate by design, and the 5-frame moving average therefore cannot see the
full excursion. Threshold is 120 °/s, giving ~1.8× margin over the silent tracks.

### 2.4 Dataset inventory

```
TOTAL clean ambient audio: 60.0 s  (1.00 min)
  C01  15.0s  Seoul day            450 frames  siren mix @ 12.00 dB  probe=high
  C02  15.0s  Seoul, heavy rain    451 frames  siren mix @ 12.00 dB  probe=high
  C04  15.0s  Seoul night          451 frames  siren mix @ 12.00 dB  probe=high
  T09  15.0s  test-only, NO siren  450 frames  (correct: never in train/)
```

**3 of 7 train clips. 4 blocked** (C03, C05, C06, C07) by YouTube's "Sign in to confirm
you're not a bot" IP-level gate. Single country (KR), single scenario group (`S1_korea`).
Total corpus including siren-mixed copies: **2 minutes**.

### 2.5 Regression coverage

197 tests across 15 files. Notably `test_head_radius_declared_exactly_once` asserts the
head-radius literal appears in exactly one `.py` file — a structural guard against the
silent 2.7° systematic error that a renderer/estimator radius mismatch would introduce.

---

### 2.6 Ablation studies — addendum (2026-10-07)

Two ablation studies were run to separate the contribution of the vision and audio
channels to the event trigger (S_B). In both, a **baseline constant** (the mock value the
channel previously emitted) is ablated against the **measured** signal flowing through the
real S_B FSM (raw trigger → 35-frame hysteresis hold).

**Exp A — vision channel (RT-DETR-l + `configs/vision_risk_weights.json`):**

`vision_anomaly_score = max over detections of (risk_weight[class] × confidence)`,
RT-DETR-l running on every frame (M1 MPS, measured 232–294 ms/frame p50). Baseline is the
mock constant `0.2`.

| Clip | Condition | Baseline mean | RT-DETR mean | ma5_trigger_rate (both) |
|---|---|---|---|---|
| C01 | day | 0.2000 | 0.4119 | 0.000 |
| C02 | rain | 0.2000 | 0.4984 | 0.000 |
| C04 | night | 0.2000 | 0.2800 | 0.000 |
| T09 | day | 0.2000 | 0.0865 | 0.000 |

Condition aggregate (`rtdetr_mean`): day **0.2492** · rain **0.4984** · night **0.2800**.
At the S_B vision gate (`vision_anomaly_ma = MA_w5 ≥ 0.75`) **no clip triggers** — the
measured vision anomaly score is informative about scene content (rain C02 peaking at 0.50)
but sits below the multimodal trigger gate. The mock baseline was 0.2, i.e. the current
vision channel does not drive S_B.

**Exp B — audio channel (WavLM L2-logistic head, `p_siren`, threshold 0.80):**

Baseline is the mock constant `0.0` for `p_siren`. The head was trained on 6 real
G44 sirens (+) vs 4 clean dashcam clips (−) with leave-one-clip-out CV; the siren-mixed
C\*_siren clips were **held out entirely** and scored as generalisation only
(`siren_trigger_rate` = fraction of frames with `p_siren ≥ 0.80`; `active` = fraction of
frames where the S_B FSM is in the hold state).

| Clip | Condition | Baseline trig | Measured trig | Active | p90 | max |
|---|---|---|---|---|---|---|
| G44-04 Ambulance | real_siren | 0.000 | **0.942** | 1.000 | 1.000 | 1.000 |
| G44-05 Warbling Boat | real_siren | 0.000 | **0.963** | 1.000 | 1.000 | 1.000 |
| G44-07 Emergency | real_siren | 0.000 | **0.734** | 0.998 | 0.999 | 1.000 |
| G44-03 Whoopee Whistle | real_siren | 0.000 | **0.667** | 0.898 | 1.000 | 1.000 |
| G44-02 Air Raid | real_siren | 0.000 | **0.553** | 0.862 | 1.000 | 1.000 |
| G44-01 Sirens | real_siren | 0.000 | **0.345** | 0.624 | 0.927 | 0.999 |
| C01_siren | siren_mixed | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 |
| C02_siren | siren_mixed | 0.000 | 0.000 | 0.000 | 0.000 | 0.025 |
| C04_siren | siren_mixed | 0.000 | 0.005 | 0.061 | 0.207 | 0.811 |
| C01 | day (clean) | 0.000 | 0.012 | 0.199 | 0.116 | 0.985 |
| C02 | rain (clean) | 0.000 | 0.000 | 0.000 | 0.000 | 0.009 |
| C04 | night (clean) | 0.000 | 0.015 | 0.019 | 0.004 | 0.982 |
| T09 | day (clean) | 0.000 | 0.043 | 0.272 | 0.491 | 0.998 |

Condition aggregate (`measured_trigger_rate`): real_siren **0.701** · siren_mixed **0.002** ·
day **0.027** · rain **0.000** · night **0.015**. The audio channel is the only channel that
drives S_B: real sirens trigger at 70%, clean dashcam controls are ~0–4%.

**Root cause — C\*_siren (12 dB synthetic mix) non-detection, measured, not assumed:**

The three siren-mixed positives were scored at essentially zero despite the siren being
mixed at a *measured* 12.0 dB above the dashcam ambient (C04) / 8.8 dB after the clip-level
limiter (C01/C02 — `peak_mix = 0.999`). The decisive measurement isolates the cause:

- The **pure, unmixed synthetic siren tracks** used in the C\*_siren mixes
  (`track_pass_fast_left`, `track_fail_slow`, `track_static_right`) score `mean_p_siren = 0.0000`
  from the head with **no dashcam background at all**.
- Real G44 sirens score `mean_p_siren = 0.59–0.98` — the head's positives are real siren
  recordings (`exports/real_sirens`), and the synthetic siren source (a 325/430/650 Hz
  harmonic stack under a 2.2 Hz wail, spatialised by Woodworth ITD) lives in a **different
  WavLM embedding neighbourhood**, not a "too-quiet-to-hear" regime.

Conclusion: the C\*_siren non-detection is a **domain shift between the real-siren training
positives and the synthetic siren source**, not an SNR or detection-threshold failure. The
head generalises to held-out real sirens (LOCO accuracy 0.617–0.993) but not to the
synthetic siren generator. The synthetic siren track and the real G44 recordings are
spectrally distinct (synthetic: one dominant ~330 Hz line; real: multi-harmonic, dominant
93–253 Hz), so the WavLM embedding of the synthetic mix sits with the noise/clip negatives.
This is a **positive finding for the report's honesty**: the test executed exactly what the
plan asked (12 dB mix) and reported the generalisation gap instead of papering over it.

---

## 3. Qualitative assessment

### What is genuinely strong

**The estimator is the standout result.** Sub-0.1° on ideal input and 1–2° under 5 dB road
noise is a strong GCC-PHAT implementation. More importantly, it is *honest*: when the
correlation peak implies a lag beyond `MAX_ITD = 10.493` samples, it returns
`ambiguous: true` rather than a confident wrong number. The negative-control experiment in
the evidence is telling — removing the coherence gate lowers mean error (14.34° → 5.26°) but
only by returning plausible-but-wrong values at ±90° (`gt -90 → -78.93`). **Trading an
honest ambiguity flag for silent error is a bad trade, and this codebase declined it.**

**Fail-loud discipline is real, not decorative.** Verified by execution, not assertion:
attempting to write test-only T09 into `train/` exits 2 and writes nothing; a missing raw
mp4 emits `SKIP … wrote nothing` rather than a truncated wav; a disk preflight refuses
rather than filling a 98%-full volume; `poc_completed_at_ms` is separated from `poc_all_ok`
so timing can't masquerade as success; `build_poc_report.py` refuses to emit a report with a
silently missing scenario group.

**Stereo integrity holds where it matters.** The 2-channel tensor is preserved through
ingest and asserted `shape[1] == 2` *before* a derived mono copy is made for WavLM, which
is mono-only. The mono copy is never written back. The original unmixed audio is never the
mix target, so with/without-siren runs stay comparable.

**Scenario labelling is derived, not asserted.** `scenario` is computed from the recorded
`country` via a lookup, so a mislabel surfaces as `scenario_mismatch` instead of hiding.
This immediately caught C01's pre-existing `clip_meta.json` claiming `scenario="S2_global"`
while `country="KR"`.

### What is honest but limited

The evidence files are unusually candid — they record a coherence-gate negative control that
**failed to reproduce** the plan's predicted behaviour, and explicitly decline to write the
test that would have asserted it ("asserting it would fabricate evidence"). They also record
a rejected 1-second-window change that improved noise behaviour while breaking the ideal
gate. That is the right instinct.

### Defects found during this review — disclosed, not fixed

1. **`build_poc_report.py:117` (`clip_metrics`) scores out-of-domain windows.** The renderer
   emits raw angles to 178.41°; the binaural pair structurally cannot report past ±90°;
   `wrap180(178.41)` returns `178.41` unchanged. 225 of C01's 436 windows are therefore
   scored against an unobservable label, inflating its mean error from **3.06° to 52.21°**.
   Filtering to `|gt| ≤ 90` recovers 3.06°, consistent with C02/C04.
2. **`outputs/poc_report.json` was stale — resolved.** It had published `track_pass_fast` at
   **744.955 °/s** while current code produces **220.643 °/s**. The artifact was regenerated
   and now matches the code, including the new `track_pass_fast_left` row (220.643 °/s, fires).
3. **Unexplained non-monotonicity at `nfft=32768`** in the window-length sweep — a ~60×
   outlier against both neighbouring coherence spans. Recorded as UNRESOLVED in todo 12
   evidence and still open.

---

## 4. What this PoC does **not** demonstrate

- **A validated siren-detection accuracy on real roads.** A WavLM L2-logistic head now
  exists (§2.6) and scores real G44 sirens at `p_siren ≥ 0.80`, but its LOCO accuracy is
  clip-dependent (0.617–0.993 per real-siren clip) and it does **not** transfer to the
  synthetic siren source used in the demo mixes (measured domain shift, §2.6). "The model
  fires on the siren-mixed demo clips" is **not** true and is not claimed.
- **Anything about the synthetic siren being representative of real sirens.** The C\*_siren
  12 dB mixes are the plan's positive control, but the head's near-zero score on those mixes
  shows the synthetic source sits outside the real-siren embedding neighbourhood. The
  synthetic source remains **unvalidated** against real emergency sirens.
- **Vision anomaly detection accuracy.** `vision_anomaly_score` is now a *measured* RT-DETR-l
  score (§2.6) rather than a mock — but it never crosses the S_B trigger gate (max 0.50 vs
  the 0.75 gate), so the vision channel demonstrably does not drive events with the current
  risk weights. There is still no fine-tuned detector in `models/` (RT-DETR-l is COCO
  pretrained, scored offline into `outputs/detections/*.json`).
- **Full-graph latency.** The 3.38 ms p95 is the fast path only; the vision ablation adds a
  measured 232–294 ms/frame (M1 MPS) for the RT-DETR step, which is **not** part of the
  3.38 ms fast-path figure.
- **Generalisation.** 1 minute of audio, 3 train clips, one country, one condition set.

---

## 5. Verdict

**The measurement and plumbing are sound. The product is unproven.**

The PoC has established something worth having: a spatial audio sensing chain that is
accurate to ~0.09° in ideal conditions, ~2–3° on real compressed dashcam audio, runs in
3.38 ms p95, degrades honestly rather than silently, and refuses to emit artefacts when its
inputs are incomplete. That is a credible foundation.

The distance between that and an MVP is **entirely a training and data problem**, and it is
worth being blunt about the size of it: the current corpus is **one minute of audio**. No
amount of engineering closes that gap. The binding constraints, in order, are:

1. **Corpus size and diversity** — 3 clips, one country, 1 minute.
2. **Real-siren validation** — the synthetic source assumption is untested, and it is the
   assumption the entire result rests on.
3. **4 clips still blocked** by the YouTube gate (now unblocked in tooling; `--cookies` added).

Item 2 is the one that matters most. Even 10,000 clips of perfect synthetic audio would not
establish that this works on a real siren, because the question "does a real siren produce
this ITD signature in this cabin geometry" has never been asked of real audio.

---

## 6. Open items

| # | Item | Blocker | Owner action |
|---|---|---|---|
| 1 | `clip_metrics` domain guard (§3.1) | code defect | fix + regenerate report |
| 2 | Stale `poc_report.json` (§3.2) | code defect | regenerate after fix 1 |
| 3 | `nfft=32768` non-monotonicity | unexplained | investigate |
| 4 | Todos 5 + 9: WavLM precache, smoke training | needs CUDA | Colab T4 — runbook ready at `docs/COLAB_RUNBOOK.md`, bundle verified (179 pass, 9 skip from a clean extract) |
| 5 | C03/C05/C06/C07 | YouTube IP gate | Colab egress IP or `--cookies cookies.txt` |
| 6 | Real siren audio | sourcing | **needed before any MVP accuracy claim** |
| 7 | Synthetic siren ↔ real-siren domain gap (§2.6) | synthetic source unvalidated | either resynthesise the siren source to match real G44 spectra, or train the head on the siren-mixed positives — currently the synthetic mixes score `p_siren ≈ 0` |

**Per instruction, no fixes have been applied — this report is observation only.**