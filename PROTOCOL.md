# NeoGMA recording protocol (SOP for clinical staff)

A General Movements Assessment is only valid if the recording follows the
protocol. The software **refuses to score** a video that breaks the hard rules,
because a confident-looking number from an invalid recording is worse than no
number at all.

---

## Hard rules — the tool will REFUSE the video if these are broken

| Rule | Why |
|---|---|
| **Corrected age 9–20 weeks** | Fidgety movements only exist in this window. Their absence at 6 weeks or 30 weeks means nothing. This is the single most important field. |
| **At least 60 seconds** of usable recording | Fidgety movements are intermittent; a short clip can miss them entirely. |
| Infant **supine** (on the back) | Every published model, and the GMA itself, is defined on supine infants. |
| Filmed **top-down** (camera above, looking straight down) | An oblique angle distorts every kinematic measurement. |
| Wrists and ankles **visible and in frame** | Fidgety movements are *distal*. If the hands and feet are not tracked, the signal is not measurable. |

**Corrected age must be recorded.** Chronological age is not acceptable for
preterm infants.

---

## Recording

- **Duration:** 60–120 seconds of good-quality recording (aim for ~2 minutes;
  extra is fine, the software takes the compliant segment).
- **Attire:** nappy only, or minimal clothing. Sleeves and sleepsuits hide the
  wrists and ankles, which is exactly where the signal is.
- **Surface:** flat, firm, plain-coloured. Avoid patterned blankets.
- **Camera:** an ordinary phone or tablet is sufficient — no special equipment.
  Hold it **directly above** the infant, level, framing the whole body with a
  little margin.
- **Lighting:** even, no strong shadows, no backlighting.

## The infant must be

- **Awake and active** — this is the "active wakefulness" state
- **Not crying**, not fussing
- **Not drowsy** or falling asleep
- **Not hungry, not immediately post-feed**

If the infant cries or falls asleep, stop and try again later. A recording of a
crying baby is not a GMA.

## Absolutely not during recording

- ❌ **No pacifier/dummy**
- ❌ **No toys**, no objects in the hands
- ❌ **No talking to, touching, or interacting with the infant**
- ❌ No hands of a parent or clinician in the frame

Any of these change the infant's spontaneous movement, which is the entire thing
being measured.

---

## What to record alongside the video

These go into the database and are needed for the analysis:

**Required**
- Subject ID (stable, one per infant — never reuse)
- Date of birth, gestational age at birth → corrected age at recording
- Date of recording

**Strongly recommended**
- Reason for high risk (HIE/therapeutic hypothermia, prematurity, IVH, etc.)
- Neonatal imaging result (MRI/cranial ultrasound: normal / abnormal)
  — GMA plus neonatal imaging reaches 95.3% accuracy, versus ~82% for GMA alone
- Sex, birth weight

**The labels**
- **GMA score** by a certified assessor: fidgety movements **present / absent /
  abnormal**
- **CP status at 12–24 months corrected age**, when follow-up matures, with GMFCS
  level if known

---

## A note on what the tool can and cannot tell you

The model is trained to predict the **GMA score** (fidgety present vs absent),
which is a validated *surrogate* for CP — not CP itself. CP status at 2 years is
the real endpoint and requires follow-up.

Even expert human GMA, in real-world cohorts, has a **positive predictive value
of about 33%** — because only ~10% of high-risk infants develop CP. Two out of
three "abnormal" results will be children who do not develop CP. The test's
strength is its **negative** predictive value (~97%): a normal result is genuinely
reassuring.

This is a **research tool. It is not a medical device and must not be used to
make clinical decisions.**
