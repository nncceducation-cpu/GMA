"""Protocol gate + pose quality control.

A GMA is only valid if the recording meets the protocol. The tool must REFUSE to
score an invalid recording rather than return a confident-looking number from a
video of a swaddled, crying, 4-week-old infant filmed from the side.

Protocol (Prechtl standard; as operationalised by Segado 2026):
  * 9-20 weeks CORRECTED age          <- hard gate; FMs do not exist outside it
  * supine, filmed top-down
  * minimal attire (nappy only)
  * no pacifier, no toys, no interaction
  * 60-120 s of usable video
  * infant awake, active; not crying, not drowsy
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pipeline.normalise import COCO, NormalisedPose

# The fidgety-movement window. Outside it, absence of FMs means nothing.
AGE_MIN_WEEKS = 9.0
AGE_MAX_WEEKS = 20.0
MIN_DURATION_S = 60.0
MAX_DURATION_S = 300.0
MIN_MEAN_CONF = 0.4
MAX_LOW_CONF_FRACTION = 0.20
# Top-down check: Segado used the wingspan/body-length ratio as a proxy for a
# near-orthogonal overhead view, reporting 0.77 +/- 0.12 for valid recordings.
WINGSPAN_RATIO_RANGE = (0.45, 1.15)


def protocol_gate(corrected_age_weeks: Optional[float],
                  duration_s: float) -> Dict:
    """Hard gate. Returns {'pass': bool, 'blocking': [...], 'warnings': [...]}"""
    blocking: List[str] = []
    warnings: List[str] = []

    if corrected_age_weeks is None:
        blocking.append(
            "Corrected age is missing. GMA is only interpretable at 9-20 weeks "
            "corrected age; without it this video cannot be scored.")
    elif not (AGE_MIN_WEEKS <= corrected_age_weeks <= AGE_MAX_WEEKS):
        blocking.append(
            f"Corrected age {corrected_age_weeks:.1f} weeks is outside the "
            f"fidgety-movement window ({AGE_MIN_WEEKS:.0f}-{AGE_MAX_WEEKS:.0f} "
            "weeks). Absence of fidgety movements outside this window carries no "
            "predictive meaning. REFUSING to score.")

    if duration_s < MIN_DURATION_S:
        blocking.append(
            f"Video is {duration_s:.0f}s; the protocol requires at least "
            f"{MIN_DURATION_S:.0f}s of usable recording.")
    elif duration_s > MAX_DURATION_S:
        warnings.append(f"Video is {duration_s:.0f}s, longer than usual; "
                        "only the protocol-compliant segment should be scored.")

    return {"pass": len(blocking) == 0, "blocking": blocking, "warnings": warnings}


def pose_quality(np_: NormalisedPose) -> Dict:
    """QC on the extracted pose. Bad tracking must not be scored as bad movement."""
    issues: List[Dict] = []
    conf = np_.conf
    mean_conf = float(np.nanmean(conf))
    low_frac = float(np_.meta.get("low_conf_fraction", np.nan))

    if mean_conf < MIN_MEAN_CONF:
        issues.append({"severity": "ERROR", "kind": "low_pose_confidence",
                       "detail": f"mean keypoint confidence {mean_conf:.2f} < "
                                 f"{MIN_MEAN_CONF}. Tracking is unreliable; a low "
                                 "movement score here may be a tracking failure, "
                                 "not an absence of movement."})
    if np.isfinite(low_frac) and low_frac > MAX_LOW_CONF_FRACTION:
        issues.append({"severity": "WARN", "kind": "interpolated_keypoints",
                       "detail": f"{low_frac:.0%} of keypoints were below the "
                                 "confidence threshold and were interpolated."})

    # per-joint reliability, distal joints matter most for GMA
    for j in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        c = float(np.nanmean(conf[:, COCO[j]]))
        if c < MIN_MEAN_CONF:
            issues.append({"severity": "ERROR", "kind": "distal_joint_unreliable",
                           "detail": f"{j} mean confidence {c:.2f}. Fidgety "
                                     "movements are DISTAL; if the wrists/ankles "
                                     "are not tracked, the signal is not measurable."})

    # camera-angle proxy: wingspan (x-extent) / body length (y-extent)
    xy = np_.xy
    body = [COCO[k] for k in ("left_shoulder", "right_shoulder", "left_hip",
                              "right_hip", "left_wrist", "right_wrist",
                              "left_ankle", "right_ankle", "left_knee",
                              "right_knee", "left_elbow", "right_elbow")]
    xr = float(np.ptp(np.median(xy[:, body, 0], axis=0)))
    yr = float(np.ptp(np.median(xy[:, body, 1], axis=0)))
    ratio = xr / yr if yr > 1e-9 else np.nan
    if np.isfinite(ratio) and not (WINGSPAN_RATIO_RANGE[0] <= ratio <= WINGSPAN_RATIO_RANGE[1]):
        issues.append({"severity": "WARN", "kind": "camera_angle",
                       "detail": f"wingspan/body-length ratio {ratio:.2f} is "
                                 "outside the expected range for a top-down view "
                                 f"{WINGSPAN_RATIO_RANGE}. The camera may be "
                                 "oblique, which distorts every kinematic feature."})

    return {"mean_confidence": mean_conf,
            "low_conf_fraction": low_frac,
            "wingspan_ratio": ratio,
            "n_frames": int(len(xy)),
            "issues": issues,
            "usable": not any(i["severity"] == "ERROR" for i in issues)}
