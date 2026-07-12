"""GMA feature battery — computed on NORMALISED pose, in short windows.

Two families:

A) KINEMATIC (after Segado 2026, GigaScience). Position, velocity, acceleration,
   joint angles, left-right symmetry, movement complexity, for wrists, ankles,
   elbows and knees. Their permutation importance: ankle 41%, knee 39%,
   elbow 11%, wrist 9%.

B) FIDGETY-SPECIFIC (after Morais 2023, IEEE JBHI). Fidgety movements are
   "small amplitude, moderate speed, variable acceleration" movements of the
   distal joints. The signature is *movement-direction variability at small
   amplitude*. Morais measure exactly this and it is more interpretable and more
   accurate than whole-video summaries.

THE CARDINAL SIN this module exists to avoid — Segado's own stated limitation:

    "infrequent, small amplitude rolls of the wrists and ankles carry significant
     clinical meaning, but are infrequent and may be smoothed out when averaged
     over an entire video."

So: everything here is computed per WINDOW (default 5 s), never over the whole
clip. A video's score is an aggregate over windows, not a single average.

Every feature is scale-free and frame-rate-free by construction, because the
pose was normalised (torso = 1 unit, common fps) before it got here.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import welch

from pipeline.normalise import COCO, DISTAL, GMA_JOINTS, PROXIMAL

NAN = float("nan")

WINDOW_SECONDS = float(os.environ.get("NEOGMA_WINDOW_SECONDS", "5.0"))
OVERLAP = float(os.environ.get("NEOGMA_OVERLAP", "0.5"))

# Fidgety movements: small amplitude, moderate speed. Amplitude threshold is in
# TORSO UNITS (pose is normalised), so it is camera-independent.
FIDGETY_AMP_MAX = float(os.environ.get("NEOGMA_FIDGETY_AMP_MAX", "0.15"))
# Fidgety frequency band. Prechtl describes them as continuous, ~1-2 Hz-ish
# elegant movements; we search a generous band and let the model decide.
FID_FMIN = float(os.environ.get("NEOGMA_FID_FMIN", "0.5"))
FID_FMAX = float(os.environ.get("NEOGMA_FID_FMAX", "6.0"))

ANGLE_TRIPLETS = {   # joint -> (proximal, joint, distal)
    "left_elbow":  ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee":   ("left_hip", "left_knee", "left_ankle"),
    "right_knee":  ("right_hip", "right_knee", "right_ankle"),
}


def _iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.subtract(*np.percentile(x, [75, 25]))) if len(x) > 3 else NAN


def _entropy(x: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy of the value distribution (movement complexity)."""
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.ptp(x) < 1e-9:
        return NAN
    h, _ = np.histogram(x, bins=bins)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b, in degrees."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    cos = np.clip((v1 * v2).sum(axis=1) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _direction_variability(vel: np.ndarray, amp_max: float) -> Dict[str, float]:
    """Morais's fidgety signature: how much does the movement DIRECTION change,
    among the small-amplitude movements?

    A fidgety infant's distal joints wander continuously in varying directions.
    An infant without fidgety movements either barely moves, or moves in large,
    monotonous, stereotyped sweeps. Both are distinguishable here, and neither
    is captured by a mean velocity.
    """
    speed = np.linalg.norm(vel, axis=1)
    small = speed < amp_max                 # amplitude gate, in torso units/s
    n_small = int(small.sum())
    if n_small < 8:
        return {"dirvar_circstd": NAN, "dirvar_entropy": NAN,
                "dirvar_meanabs_dtheta": NAN, "small_amp_fraction": float(small.mean())}

    theta = np.arctan2(vel[small, 1], vel[small, 0])
    # circular standard deviation of movement direction
    R = np.sqrt(np.cos(theta).mean() ** 2 + np.sin(theta).mean() ** 2)
    circstd = float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))
    # entropy of direction over 12 sectors
    h, _ = np.histogram(theta, bins=12, range=(-np.pi, np.pi))
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    dent = float(-(p * np.log2(p)).sum())
    # mean absolute frame-to-frame turn
    dth = np.diff(theta)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    return {"dirvar_circstd": circstd,
            "dirvar_entropy": dent,
            "dirvar_meanabs_dtheta": float(np.abs(dth).mean()) if len(dth) else NAN,
            "small_amp_fraction": float(small.mean())}


def _spectral(x: np.ndarray, fps: float, prefix: str) -> Dict[str, float]:
    """Band-limited spectral peak. Deliberately band-limited: an unrestricted
    argmax over the PSD returns the frequency-resolution floor (= fps/nperseg),
    which reports the CAMERA, not a rhythm. That exact bug cost us a week in
    Nmotion; it is not repeated here."""
    x = x[np.isfinite(x)]
    nper = min(256, len(x))
    if nper < 16:
        return {f"{prefix}_peak_hz": NAN, f"{prefix}_peak_prom": NAN,
                f"{prefix}_band_power": NAN}
    f, psd = welch(x, fs=fps, nperseg=nper, detrend="linear")
    band = (f >= FID_FMIN) & (f <= FID_FMAX)
    if not band.any() or psd[band].max() <= 0:
        return {f"{prefix}_peak_hz": NAN, f"{prefix}_peak_prom": NAN,
                f"{prefix}_band_power": NAN}
    bf, bp = f[band], psd[band]
    k = int(np.argmax(bp))
    med = float(np.median(bp))
    return {f"{prefix}_peak_hz": float(bf[k]),
            f"{prefix}_peak_prom": float(bp[k] / (med + 1e-12)),
            f"{prefix}_band_power": float(bp.sum() / (psd.sum() + 1e-12))}


def window_features(xy: np.ndarray, fps: float) -> Dict[str, float]:
    """Features for ONE window of normalised pose. xy: [T, 17, 2], torso units."""
    out: Dict[str, float] = {}
    T = len(xy)
    dt = 1.0 / fps

    # derivatives, in torso-units per second (frame-rate free)
    vel = np.gradient(xy, dt, axis=0)          # [T,17,2]
    acc = np.gradient(vel, dt, axis=0)

    # ---------- A. kinematic, per GMA joint ----------
    for j in GMA_JOINTS:
        k = COCO[j]
        pos, v, a = xy[:, k], vel[:, k], acc[:, k]
        sp = np.linalg.norm(v, axis=1)
        ac = np.linalg.norm(a, axis=1)
        out[f"{j}_pos_median_x"] = float(np.median(pos[:, 0]))
        out[f"{j}_pos_median_y"] = float(np.median(pos[:, 1]))
        out[f"{j}_pos_iqr_x"] = _iqr(pos[:, 0])
        out[f"{j}_pos_iqr_y"] = _iqr(pos[:, 1])
        out[f"{j}_speed_median"] = float(np.median(sp))
        out[f"{j}_speed_iqr"] = _iqr(sp)
        out[f"{j}_acc_iqr"] = _iqr(ac)
        out[f"{j}_pos_entropy"] = _entropy(np.linalg.norm(pos, axis=1))
        out[f"{j}_speed_entropy"] = _entropy(sp)

    # ---------- B. fidgety signature, distal joints only ----------
    for j in DISTAL:
        k = COCO[j]
        dv = _direction_variability(vel[:, k], FIDGETY_AMP_MAX)
        for name, val in dv.items():
            out[f"{j}_{name}"] = val
        out.update(_spectral(np.linalg.norm(vel[:, k], axis=1), fps, f"{j}_speed"))

    # ---------- C. joint angles ----------
    for j, (p, m, d) in ANGLE_TRIPLETS.items():
        ang = _angle(xy[:, COCO[p]], xy[:, COCO[m]], xy[:, COCO[d]])
        av = np.gradient(ang, dt)
        out[f"{j}_angle_mean"] = float(np.mean(ang))
        out[f"{j}_angle_std"] = float(np.std(ang))
        out[f"{j}_angvel_median"] = float(np.median(np.abs(av)))
        out[f"{j}_angvel_iqr"] = _iqr(av)
        out[f"{j}_angle_entropy"] = _entropy(ang)

    # ---------- D. left-right symmetry + inter-limb coordination ----------
    for L, R, tag in [("left_wrist", "right_wrist", "wrist"),
                      ("left_ankle", "right_ankle", "ankle"),
                      ("left_knee", "right_knee", "knee"),
                      ("left_elbow", "right_elbow", "elbow")]:
        sl = np.linalg.norm(vel[:, COCO[L]], axis=1)
        sr = np.linalg.norm(vel[:, COCO[R]], axis=1)
        tot = sl.mean() + sr.mean()
        out[f"{tag}_symmetry"] = float(sl.mean() / tot) if tot > 1e-9 else NAN
        if T > 3 and sl.std() > 1e-9 and sr.std() > 1e-9:
            out[f"{tag}_xcorr"] = float(np.corrcoef(sl, sr)[0, 1])
        else:
            out[f"{tag}_xcorr"] = NAN

    # ---------- E. whole-body ----------
    dist_speed = np.stack([np.linalg.norm(vel[:, COCO[j]], axis=1) for j in DISTAL])
    out["distal_speed_mean"] = float(dist_speed.mean())
    out["distal_speed_std"] = float(dist_speed.std())
    out["n_frames_window"] = float(T)     # LEAK COLUMN - flagged, never modelled
    return out


def window_starts(n: int, win: int, overlap: float) -> List[int]:
    if n <= win:
        return [0]
    stride = max(1, round(win * (1.0 - overlap)))
    return list(range(0, n - win + 1, stride))


def extract_windows(xy: np.ndarray, fps: float,
                    window_seconds: float = WINDOW_SECONDS,
                    overlap: float = OVERLAP) -> List[Dict[str, float]]:
    """One feature row per window. This is the design matrix the model sees."""
    win = max(8, int(round(window_seconds * fps)))
    rows = []
    for i, s in enumerate(window_starts(len(xy), win, overlap)):
        e = min(s + win, len(xy))
        r = window_features(xy[s:e], fps)
        r["window_index"] = i
        r["start_frame"] = s
        r["end_frame"] = e
        rows.append(r)
    return rows
