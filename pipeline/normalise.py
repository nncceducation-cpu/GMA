"""Pose normalisation — remove the camera from the signal.

Three confounds are removed here. Each one has burned a published study.

1. FRAME RATE. Velocity/acceleration are per-frame quantities. A 60 fps clip
   yields half the per-frame displacement of a 30 fps clip for identical
   movement. Segado (GigaScience 2026): "All pose-data processing was normalized
   to each video's frame rate." We learned the same lesson the hard way in
   Nmotion, where the camera out-predicted the infant.

2. SCALE. Camera distance / zoom changes every pixel coordinate. Normalising the
   torso to unit length makes features invariant to how close the phone was held.

3. ROTATION. A video shot with the infant's head to the left is not the same
   feature vector as head-up, unless you rotate it. Left/right symmetry features
   are meaningless without this.

Nothing downstream should ever see raw pixel coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO-17 keypoint order, which is what ViTPose / MMPose emit.
COCO = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}
# The joints GMA actually cares about. Segado's permutation importance:
# ankle 41%, knee 39%, elbow 11%, wrist 9% — distal joints dominate.
DISTAL = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
PROXIMAL = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
GMA_JOINTS = DISTAL + PROXIMAL


@dataclass
class NormalisedPose:
    xy: np.ndarray          # [T, 17, 2] normalised coordinates
    conf: np.ndarray        # [T, 17] per-keypoint confidence
    fps: float              # analysis frame rate (all clips share this)
    source_fps: float
    fps_standardised: bool
    torso_px: float         # median torso length in pixels (scale reference)
    meta: Dict


def _mid(xy: np.ndarray, a: str, b: str) -> np.ndarray:
    return 0.5 * (xy[:, COCO[a]] + xy[:, COCO[b]])


def resample_to_fps(xy: np.ndarray, conf: np.ndarray, src_fps: float,
                    target_fps: float) -> tuple:
    """Resample the pose time-series onto a common time base.

    Unlike video decimation, pose is a continuous signal, so we can interpolate
    rather than drop frames. That means we can also handle src_fps < target_fps
    without fabricating duplicate (zero-motion) frames.
    """
    if abs(src_fps - target_fps) < 1e-6:
        return xy, conf, True
    T = len(xy)
    dur = T / src_fps
    n_out = max(2, int(round(dur * target_fps)))
    t_src = np.arange(T) / src_fps
    t_out = np.arange(n_out) / target_fps
    t_out = t_out[t_out <= t_src[-1]]

    out_xy = np.empty((len(t_out), xy.shape[1], 2), dtype=np.float32)
    out_cf = np.empty((len(t_out), conf.shape[1]), dtype=np.float32)
    for j in range(xy.shape[1]):
        out_xy[:, j, 0] = np.interp(t_out, t_src, xy[:, j, 0])
        out_xy[:, j, 1] = np.interp(t_out, t_src, xy[:, j, 1])
        out_cf[:, j] = np.interp(t_out, t_src, conf[:, j])
    return out_xy, out_cf, True


def lowpass(xy: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Remove tracker jitter, keep the movement.

    This exists because of a concrete failure. Scale/rotation/fps normalisation
    is NOT sufficient: every camera rig has a characteristic keypoint-noise
    level, and an unsupervised model will happily use that noise as a fingerprint
    for the recording site. Our own nuisance probe caught exactly this — the
    embedding could identify the site at 0.78 balanced accuracy (chance 0.50)
    purely from jitter.

    Cutoff is well above the fidgety band (0.5-6 Hz), so the clinical signal is
    preserved while broadband tracker noise is attenuated.
    """
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fps
    wn = min(cutoff_hz / nyq, 0.99)
    if wn <= 0 or wn >= 0.99 or len(xy) < 15:
        return xy
    b, a = butter(4, wn, btype="low")
    out = xy.copy()
    for j in range(xy.shape[1]):
        for d in range(2):
            out[:, j, d] = filtfilt(b, a, xy[:, j, d], method="gust")
    return out.astype(np.float32)


def normalise(xy: np.ndarray, conf: np.ndarray, src_fps: float,
              target_fps: float = 30.0,
              min_conf: float = 0.3,
              smooth_hz: float = 8.0) -> NormalisedPose:
    """Full normalisation: time base, jitter, rotation, scale.

    Args:
        xy:   [T, 17, 2] raw pixel keypoints (COCO-17 order)
        conf: [T, 17] keypoint confidences
        src_fps: frame rate of the source video
        smooth_hz: low-pass cutoff. Set 0 to disable (not recommended — see
            lowpass(): tracker jitter is a site fingerprint).
    """
    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1] != 17:
        raise ValueError(f"expected [T,17,2] COCO keypoints, got {xy.shape}")

    # --- 1. common time base ------------------------------------------------
    xy, conf, ok = resample_to_fps(xy, conf, src_fps, target_fps)
    fps = target_fps

    # --- 2. low-confidence keypoints are NOT data ---------------------------
    # Interpolating over them is better than feeding the model a hallucinated
    # ankle. Track how much we had to repair; it becomes a QC metric.
    bad = conf < min_conf
    for j in range(xy.shape[1]):
        b = bad[:, j]
        if b.all():
            continue
        if b.any():
            idx = np.arange(len(xy))
            for d in (0, 1):
                xy[b, j, d] = np.interp(idx[b], idx[~b], xy[~b, j, d])

    # --- 2b. low-pass: strip tracker jitter (a per-site camera fingerprint)
    if smooth_hz and smooth_hz > 0:
        xy = lowpass(xy, fps, smooth_hz)

    # --- 3. rotate to head-up ----------------------------------------------
    # Torso vector = mid-hip -> mid-shoulder. Rotate so it points "up" (-y).
    sh = _mid(xy, "left_shoulder", "right_shoulder")
    hp = _mid(xy, "left_hip", "right_hip")
    torso = sh - hp                                     # [T, 2]
    torso_len = np.linalg.norm(torso, axis=1)           # [T]
    med_len = float(np.median(torso_len[torso_len > 1e-6])) if (torso_len > 1e-6).any() else 0.0
    if med_len <= 1e-6:
        raise ValueError("degenerate torso: cannot normalise (pose likely failed)")

    # Use the MEDIAN torso direction over the clip, not per-frame: the infant's
    # trunk wobbles, and rotating each frame independently would cancel out real
    # trunk movement, which is itself part of the GMA signal.
    v = np.median(torso, axis=0)
    ang = np.arctan2(v[0], -v[1])                       # angle to the -y axis
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    centre = np.median(hp, axis=0)                      # origin at the pelvis
    xy = (xy - centre) @ R.T

    # --- 4. scale so torso length = 1 --------------------------------------
    xy = xy / med_len

    return NormalisedPose(
        xy=xy.astype(np.float32), conf=conf.astype(np.float32),
        fps=float(fps), source_fps=float(src_fps),
        fps_standardised=bool(abs(src_fps - target_fps) < 1e-6 or ok),
        torso_px=med_len,
        meta={"n_frames": int(len(xy)),
              "low_conf_fraction": float(bad.mean()),
              "rotation_deg": float(np.degrees(ang)),
              "smooth_hz": float(smooth_hz)},
    )
