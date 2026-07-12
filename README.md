# NeoGMA — automated General Movements Assessment for early CP detection

Pose-based automated GMA at the fidgety-movements stage (9–20 weeks corrected
age), targeted at the **HIE / therapeutic-hypothermia term infant** — a
population where the evidence base is currently three studies and 118 infants
(Seesahai 2020).

Read [`EVIDENCE.md`](EVIDENCE.md) first. It is the literature review that
determined this architecture, and it explains why several obvious choices are
wrong.

> **Research tool — not a medical device.** No output may drive clinical
> decisions. If ever deployed clinically this is Software as a Medical Device.

---

## What changed from Nmotion, and why

Nmotion (the neonatal seizure/HIE movement tool) used **RAFT dense optical flow**
over the whole frame. **That is the wrong front-end for GMA**, and the literature
is unambiguous about it:

- Fidgety movements are **small-amplitude, distal** (wrist/ankle) movements.
  Whole-frame flow measures **gross body motion** and averages them away.
- Flow is sensitive to background clutter and camera motion; pose is not.
- Segado's feature importance is dominated by **ankle (41%) and knee (39%)**.

**So: pose-first.** Optical flow is retained only as an optional second fusion
channel, because multi-modal fusion demonstrably helps (Kulvicius 2024: 94.5% vs
any single modality).

## What carries over from Nmotion — and it is the most valuable part

Segado et al. re-ran a published model (reported ROC-AUC **0.86**) with
subject-level splits enforced and got **0.60**. Record-wise splitting had been
leaking the same infant across train and test.

That is exactly the failure the Nmotion evaluation machinery was built to
prevent. It transfers wholesale:

| Nmotion component | Why it matters here |
|---|---|
| `StratifiedGroupKFold` grouped by subject + refusal to score when a class has <2 subjects | The dominant methodological error in this literature |
| SHA-256 content hashing | Catches the same infant re-uploaded under a new ID — grouping by ID cannot |
| `data_quality.csv` auto-audit | Constant features, all-NaN features, leak columns, duplicates, mixed frame rates, too-few subjects |
| Frame-rate standardisation | Segado: "All pose-data processing was normalized to each video's frame rate" |
| Leak-column flagging (`n_frames`, `fps`…) | Deterministic functions of the clip that a model reads the label off |
| Labelling loop + persistent store + Parquet ML export | Clinician-in-the-loop, analysis-ready corpus |

---

## Architecture

```
video (phone, supine, top-down, 60–120 s)
  │
  ├─ PROTOCOL GATE ──────── reject if outside 9–20 weeks corrected age,
  │                          not supine, <60 s, infant crying/drowsy
  │
  ├─ ViTPose-H (MMPose, pretrained, no fine-tuning)   ← best infant pose
  │      └─ 2D keypoints + per-joint confidence
  │
  ├─ NORMALISE
  │      ├─ resample to a common frame rate (30 fps)
  │      ├─ rotate to head-up
  │      └─ scale so torso length = 1  ← removes camera distance/zoom
  │
  ├─ WINDOW (5 s, 50% overlap)  ← never average over the whole video
  │
  ├─ FEATURES
  │      ├─ kinematic (Segado's 38): position/velocity/acceleration IQR,
  │      │    entropy, cross-correlation, joint angles — wrists, ankles,
  │      │    elbows, knees
  │      ├─ fidgety-specific (Morais): movement-direction variability of
  │      │    distal joints at small amplitude, in short segments
  │      └─ [optional] optical-flow battery from Nmotion (fusion channel)
  │
  ├─ MODEL — gradient boosting first; ST-GCN later if n supports it
  │      └─ ABSTAIN when uncertain
  │
  └─ EVALUATION
         ├─ StratifiedGroupKFold by infant  (NEVER record-wise)
         ├─ preregistered LOCK-BOX, opened exactly once
         └─ PR-AUC + sensitivity + specificity + PPV + NPV
              (ROC-AUC alone is misleading at ~10% prevalence)
```

## Target label

**Phase 1: fidgety movements ABSENT vs PRESENT** (the GMA score). This is the
achievable label and it is what Segado, and most of the field, actually predicts.

**Phase 2: CP at 12–24 months corrected age.** This is the true endpoint and it
requires follow-up data we do not yet have. Sporadic FM did *not* predict CP
(Støen 2019), so the contrast is absent vs present, not a graded scale.

We will state which target a given model was trained on, every time.

## Benchmarks we are measured against

| | sens | spec | PPV | NPV | AUC |
|---|---|---|---|---|---|
| Expert GMA (real-world, Støen) | 76% | 82% | 33% | 97% | — |
| Groos 2022 (external, 4 countries) | 71% | 94% | 68% | 95% | — |
| Segado 2026 (single site, open) | 53% | 90% | — | — | 0.77 |
| Gao 2023 | — | — | — | — | 0.967 |

**We will not beat these with a single-centre pilot.** Phase 1 success is a
validated, leak-proof pipeline and a clean, preregistered HIE cohort — not a
headline AUC.

## Status

Scaffolding. Nothing is trained. See `EVIDENCE.md` §7 for the non-negotiable
design constraints and §9 for why sample size is the binding constraint.

## Running it locally (identical to Nmotion)

Requires Docker Desktop with the NVIDIA runtime. GPU is CUDA 12.8 (Blackwell /
RTX 50-series); `cu121` wheels will not run on sm_120.

```bash
docker compose up --build -d      # first build pulls ViTPose-H weights
# open http://localhost:8000
docker compose logs -f neogma     # watch a clip being processed
```

Upload a clip with `subject_id`, `corrected_age_weeks`, `site` and `risk_group`.
The protocol gate refuses the recording (HTTP 422) if the infant is outside
9–20 weeks corrected age or the clip is under 60 s — because a GMA scored outside
the fidgety window is not a GMA, and a model trained on out-of-window clips is
learning something else.

Score it (`fm_present` / `fm_abnormal` / `fm_absent`) and the model retrains
after every label, exactly as Nmotion does. It will **refuse to report a metric**
until at least 2 infants exist in the smaller class, and CV is grouped by
`subject_id` throughout. CP status is joined in later via `/outcome`, whenever
the 12–24-month follow-up arrives.

Everything lives in `./runs/` on the host — video (L0), raw pose (L1),
normalised pose (L2), features (L3). None of it is in git; `.gitignore` enforces
that.

## Known limitation: unsupervised representations are not yet site-safe

`pipeline/probes.py` exists because an unsupervised model has no label to leak,
so leakage cannot be detected by the usual means. It asks the decisive question:
**holding the outcome constant, can the embedding still tell you which site the
infant was recorded at?**

On our own synthetic multi-site check the answer was yes. The outcome probe
scored a perfect 1.00 balanced accuracy — and the conditional nuisance probes
simultaneously flagged `site` (0.208 above chance) and `source_fps` / `torso_px`
(0.423). Verdict: **DO NOT USE**.

The residual leak is site-specific *tracker-jitter amplitude*. Normalisation
divides out frame rate, scale and rotation, and the 8 Hz low-pass attenuates
jitter — but it does not **equalise** it across rigs, and an encoder will use
whatever is left as a site fingerprint. So:

- Unsupervised motif/SSL representations are **not safe to use across sites**
  until domain-invariance training (adversarial site-confusion, or camera-transform
  contrastive augmentation per `CAMERA_AUGMENTATIONS`) is added and the
  conditional probe comes back clean.
- The supervised, hand-engineered feature path is unaffected — its features are
  defined in torso units on normalised pose and are individually inspectable.
- A perfect outcome probe means nothing while a nuisance probe is red. That is
  precisely the failure mode this file was written to catch, and it caught it on
  the first run.

This is recorded here rather than fixed-and-forgotten because it is the single
most likely way this project produces a beautiful, irreproducible result.
