# AI-GMA for early cerebral palsy detection — evidence brief

Prepared 11 July 2026, **before any code was written**, so that we build what the
field actually needs rather than repeating what already exists.

---

## 1. The clinical target is settled

The **General Movements Assessment (GMA)** at the **fidgety movements (FM) stage,
9–20 weeks corrected age**, is the strongest early predictor of cerebral palsy
(CP). Absent FMs predict CP.

Meta-analysis of 8 studies (Wang 2025): fidgety movements **sensitivity 0.95,
specificity 0.87, AUC 0.97, DOR 144** — significantly better than writhing
movements across every subgroup.

But **real-world performance is much lower than the classic figures.** In a
prospective multi-centre cohort of 405 high-risk infants (Støen 2019), absent or
sporadic FM gave **sensitivity 76.2%, specificity 82.4%, PPV 33.3%, NPV 96.8%**.

Three consequences we must design around:

- **PPV is low (~33%)** because CP prevalence in high-risk cohorts is ~10%. Any
  model we build inherits this. **Report PR-AUC, PPV and NPV — not ROC-AUC
  alone**, or we will fool ourselves.
- **Sporadic FM did not predict CP.** The meaningful contrast is **absent vs
  present FM**, not a graded scale.
- Accuracy rose to **95.3%** when absent FM was combined with abnormal neonatal
  imaging. **Multimodal beats video alone.**

---

## 2. What already exists (we are not first)

| Study | Approach | n | Result |
|---|---|---|---|
| **Groos 2022**, JAMA Netw Open | Deep learning on pose; 13 hospitals, 4 countries | 557 (84 CP) | **External** validation: sens 71.4%, spec 94.1%, PPV 68.2%, NPV 94.9%. Beat conventional ML (90.6% vs 72.7% accuracy, p<.001) but **not** expert GMA (85.9%, p=.11) |
| **Gao 2023**, Nature Communications | Deep learning motor assessment model | — | **AUC 0.967** external validation; quantitative GMA AUC 0.956 |
| **Ihlen 2019**, J Clin Med (CIMA) | Time–frequency decomposition of body-part trajectories | 377 | sens 92.7%, spec 81.6% |
| **Segado 2026**, GigaScience | **Open, preregistered**: ViTPose-H → 38 kinematic features → AutoML | 925 (single site) | Lock-box **ROC-AUC 0.77, PR-AUC 0.41**; sens 53%, spec 90% |
| **Kulvicius 2024**, Comms Medicine | **Sensor fusion** (pressure + inertial + video) | 51 | **94.5%** — significantly higher than any single modality |
| **Nguyen-Thai 2021** | ST-GCN on pose + attention | — | Reported ROC-AUC 0.82 — **but see §3** |

**A single-centre effort will not beat Groos or Gao.** Our value must come from
elsewhere (§6).

---

## 3. The single most important finding for us

Segado et al. (2026) re-benchmarked the published STAM model on their own
preregistered splits:

> "it initially appeared to show excellent performance (ROC-AUC = 0.86). However,
> we observed that the available implementation used **record-wise splitting**.
> This practice is a well-documented source of overfitting in medical data. After
> adjusting the split to avoid overlap, performance was substantially lower with
> an **ROC-AUC of 0.60**."

**A peer-reviewed AUC of 0.86 collapsed to 0.60 once clips from the same infant
stopped straddling the train/test boundary.**

This is precisely the failure we spent the Nmotion build defending against, and
we measured the identical collapse on synthetic data (0.975 ungrouped → 0.125
grouped). It is the dominant methodological error in this literature. **The
leak-proof evaluation machinery we already built is our most transferable
asset.**

---

## 4. Which pose estimator (also settled)

Systematic comparison of 7 methods on infants in supine position (2024):

> "state-of-the-art human pose estimation methods work well to estimate infant
> poses **without the need for additional training or finetuning**. **ViTPose has
> the best accuracy**, followed by HRNet (top-down)... **DeepLabCut... as well as
> MediaPipe with BlazePose does not provide competitive results at all.**"

Segado independently selected **ViTPose-H** over HRNet, PVTv2 and a fine-tuned
OpenPose.

**Decision: ViTPose-H via MMPose, pretrained, no fine-tuning.** Note that
**MediaPipe — the obvious "easy" choice — is a poor choice for infants.** Docker
containers are published (hub.docker.com/u/humanoidsctu; osf.io/x465b).

---

## 5. Optical flow is the WRONG primary front-end for GMA

This contradicts the Nmotion architecture, and it is the decision that most
changes what we build.

Fidgety movements are **small-amplitude, distal, continuous** movements of the
wrists, ankles, neck and trunk. Whole-frame optical flow measures **gross
whole-body motion** and averages them away. Nguyen-Thai state it directly:
appearance-based features "are sensitive to strong but irrelevant signals caused
by background clutter or a moving camera... they measure gross whole body
movements rather than specific joint/limb motion."

Segado, using pose but averaging over the whole video, hit the same wall:

> "infrequent, small amplitude rolls of the wrists and ankles carry significant
> clinical meaning, but are infrequent and **may be smoothed out when averaged
> over an entire video**."

Their permutation importance is dominated by **ankle (41%) and knee (39%)** —
distal joints, exactly as GMA theory predicts.

**Decision: pose-first, windowed, distal-joint-specific.** Optical flow is kept
only as a *secondary* fusion channel, because fusion demonstrably helps
(Kulvicius: 94.5% vs any single modality). It is not the backbone.

---

## 6. Where a genuine gap exists — and it is ours

Seesahai 2020 (Systematic Reviews) scoping review of GMA in **neonatal
encephalopathy** (term / late-preterm) found:

> Only **three studies** met inclusion criteria. **Total participants: 118.**
> None included late-preterm neonates.

The published AI-GMA work is overwhelmingly trained on **preterm/NICU** cohorts
(Segado: 70% preterm, 45% VLBW). **HIE / therapeutic-hypothermia term infants are
barely represented** — despite being a population where early CP prediction
matters enormously, and where this group has clinical expertise and a cohort.

**Positioning: AI-GMA in the HIE / cooled term infant.** Not "a better CP model
than Groos" — that is not winnable single-centre. Rather: the first rigorously
validated, leak-proof AI-GMA pipeline characterised in the encephalopathic term
population, where the evidence base is currently 3 studies and 118 infants.

---

## 7. Non-negotiable design constraints

1. **Subject-level splitting, always.** One infant never appears in both train and
   test. Grouped CV plus a preregistered lock-box opened exactly once (Segado's
   design is the template; their STAM re-analysis is the warning).
2. **Report PR-AUC, PPV, NPV** — not ROC-AUC alone. Prevalence ~10%.
3. **Frame-rate normalisation.** Segado: "All pose-data processing was normalized
   to each video's frame rate." We learned this the hard way in Nmotion.
4. **Scale/rotation normalisation of pose:** rotate head-up, torso length = 1
   unit. Removes camera distance and orientation as confounds — the pose-domain
   analogue of our frame-rate fix.
5. **Windowed, distal-joint features.** Never average over the whole video.
6. **Abstention / uncertainty.** Models that decline to score when unsure
   (Morais 2025; UDF-GMA 2025). Essential at 10% prevalence.
7. **Target = FM absent vs present** initially. CP at 12–24 months is the true
   endpoint and requires follow-up we do not have. Segado made the same
   compromise and said so; we will too.
8. **Age gate: 9–20 weeks corrected age.** Videos outside the window are rejected,
   not scored.
9. Align with the **Spittle 2025 AI-GMA roadmap** (consortium standards for
   validation, datasets, software, regulatory, implementation).
10. **Software as a Medical Device** territory if ever used clinically.
    Research-only labelling until a regulatory pathway is chosen.

---

## 8. Recording protocol (Segado; matches Prechtl standard)

- Infant **supine**, filmed **top-down**
- **Minimal attire** (nappy only)
- **No pacifier, no toys, no interaction** during recording
- **60–120 s** of usable video
- **9–20 weeks corrected age** (Segado mean 14.6 ± 2.1 weeks)
- Ordinary **phone/tablet camera is sufficient** — no specialised rig
- Infant awake and active; not crying, not drowsy

---

## 9. Sample size — the uncomfortable truth

Segado's scaling analysis (50 → 800 infants) found a power-law relationship
between training-set size and (1 − AUC). Their **925 infants at a single site
reached only ROC-AUC 0.77**. Groos needed **557 infants across 13 hospitals** for
external validity.

A pilot of 20–50 infants will not produce a usable model. It can produce a
**validated pipeline, a clean dataset, and a preregistered protocol** — which is
the correct goal for phase 1.

---

## References

1. Gao Q et al. Automating General Movements Assessment with quantitative deep learning to facilitate early screening of cerebral palsy. *Nature Communications* 2023. https://consensus.app/papers/details/65806b2338ec5ff2b6b4542d4ecfb6cc/
2. Groos D et al. Development and Validation of a Deep Learning Method to Predict Cerebral Palsy From Spontaneous Movements in Infants at High Risk. *JAMA Network Open* 2022. https://consensus.app/papers/details/848fddd5454a591b9401c86483cdbb54/
3. Ihlen EAF et al. Machine Learning of Infant Spontaneous Movements for the Early Prediction of Cerebral Palsy: A Multi-Site Cohort Study. *J Clin Med* 2019. https://consensus.app/papers/details/51297adcecee5c6f8491f3a0450105ea/
4. Segado M, Prosser LA, Duncan AF, Johnson MJ, Kording KP. A preregistered, open pipeline for early cerebral palsy risk assessment from infant videos. *GigaScience* 2026;15:giag003. https://doi.org/10.1093/gigascience/giag003
5. Kulvicius T et al. Deep learning empowered sensor fusion boosts infant movement classification. *Communications Medicine* 2024. https://consensus.app/papers/details/7b3dc0baabbd53d6aa0bbbb75b3c52b9/
6. Nguyen-Thai B et al. A Spatio-Temporal Attention-Based Model for Infant Movement Assessment From Videos. *IEEE JBHI* 2021. https://consensus.app/papers/details/f7f23e0624405247b04d80db5dc9f793/
7. Støen R et al. The Predictive Accuracy of the General Movement Assessment for Cerebral Palsy. *J Clin Med* 2019. https://consensus.app/papers/details/83e4d14b8fd95a68949a4d8d86329a26/
8. Seesahai J et al. The assessment of general movements in term and late-preterm infants diagnosed with neonatal encephalopathy, as a predictive tool of cerebral palsy by 2 years of age — a scoping review. *Systematic Reviews* 2020. https://consensus.app/papers/details/8cf53230368355b88ab2b543a9104476/
9. Spittle AJ et al. Towards universal early screening for cerebral palsy: a roadmap for automated General Movements Assessment. *eClinicalMedicine* 2025. https://consensus.app/papers/details/828229a39d4b5541a752b231d99b74dd/
10. Wang T et al. Predictive value and ranking of writhing and fidgety movements for cerebral palsy: a meta-analysis based on the Superiority Index. *Medicine* 2025. https://consensus.app/papers/details/bc6d3a18b83f5357921ddb9d94bfc5c7/
11. Morais R et al. Robust and Interpretable General Movement Assessment Using Fidgety Movement Detection. *IEEE JBHI* 2023. https://consensus.app/papers/details/a8b29be2d95d520c91f6de0224157b64/
12. Morais R et al. Confident and Trustworthy Model for Fidgety Movement Classification. *IEEE JBHI* 2025. https://consensus.app/papers/details/5f3767337c2850199e1a12c2a78ddc01/
13. Automatic infant 2D pose estimation from videos: comparing seven deep neural network methods. arXiv 2024. https://arxiv.org/html/2406.17382v1
14. Wahle CF et al. Video and Wearable Sensor Technologies for Early Detection of Cerebral Palsy in Infants: A Scoping Review. *J Clin Med* 2026. https://consensus.app/papers/details/613ee32d607b5a739a13a915bc268640/
