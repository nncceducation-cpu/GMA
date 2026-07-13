"""Per-FRAME time series — what the clinician actually looks at.

The window features (features_gma.py) are what the MODEL sees. They are 134
numbers per 5-second window and they are unreadable to a human. This module
produces the other view: continuous traces over the whole clip, which is how a
GMA assessor thinks — "she had a quiet spell in the middle", "the left leg is
doing all the work", "there is a burst every few seconds".

Everything here is derived from L2 NORMALISED pose, so every trace is in torso
units per second. That means traces from two infants, two cameras and two frame
rates are directly comparable on the same axis. A trace in pixels/frame would
not be — it would mostly encode how close the phone was held.

WHAT EACH TRACE MEANS CLINICALLY

  distal_speed      Fidgety movements are DISTAL. This is the headline trace:
                    mean speed of both wrists and both ankles.
  limb speeds       The four limbs separately. Asymmetry is a CP sign; a
                    hemiplegic pattern shows here as one limb persistently flat.
  small_amp_frac    Fraction of joints moving at fidgety amplitude right now.
                    High and continuous = fidgety present. Near zero = absent.
  direction_change  Frame-to-frame turn in movement direction, distal joints.
                    This is the fidgety signature (Morais): fidgety movement
                    wanders; stereotyped movement does not.
  fidgety_power     Rolling share of spectral power in the 0.5-6 Hz band.
  lr_balance        Left share of total limb movement. 0.5 = symmetric.
                    Sustained departure from 0.5 is the asymmetry to look at.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import welch

from pipeline.features_gma import FID_FMAX, FID_FMIN, FIDGETY_AMP_MAX
from pipeline.normalise import COCO, DISTAL

LIMBS = {
    "left_arm": "left_wrist",
    "right_arm": "right_wrist",
    "left_leg": "left_ankle",
    "right_leg": "right_ankle",
}


def _roll(x: np.ndarray, win: int) -> np.ndarray:
    """Centred rolling mean, same length. Smooths for the eye without shifting
    events in time — a lagging filter would move a burst away from where it
    happened, which matters when a clinician cross-checks against the video."""
    if win <= 1 or len(x) < win:
        return x
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")[: len(x)]


def compute_series(xy: np.ndarray, fps: float, smooth_s: float = 0.5) -> Dict:
    """Per-frame traces from normalised pose [T,17,2]. Units: torso/second."""
    T = len(xy)
    dt = 1.0 / fps
    vel = np.gradient(xy, dt, axis=0)                       # [T,17,2]
    t = np.arange(T) / fps
    win = max(1, int(round(smooth_s * fps)))

    # distal speed: the headline trace
    dist = np.stack([np.linalg.norm(vel[:, COCO[j]], axis=1) for j in DISTAL])
    distal_speed = dist.mean(axis=0)

    out: Dict = {
        "fps": float(fps),
        "n_frames": int(T),
        "duration_s": float(T / fps),
        "t": t.round(3).tolist(),
        "distal_speed": _roll(distal_speed, win).round(4).tolist(),
    }

    # per-limb speed
    limb_mean = {}
    for name, joint in LIMBS.items():
        s = np.linalg.norm(vel[:, COCO[joint]], axis=1)
        limb_mean[name] = float(s.mean())
        out[name] = _roll(s, win).round(4).tolist()

    # left/right balance: 0.5 is symmetric
    left = limb_mean["left_arm"] + limb_mean["left_leg"]
    right = limb_mean["right_arm"] + limb_mean["right_leg"]
    out["lr_balance"] = float(left / (left + right)) if (left + right) > 1e-9 else None

    # fraction of distal joints at fidgety amplitude, per frame
    small = (dist < FIDGETY_AMP_MAX).mean(axis=0)
    out["small_amp_frac"] = _roll(small, win).round(4).tolist()

    # movement-direction change, distal joints (the fidgety signature)
    dth = np.zeros(T)
    for j in DISTAL:
        v = vel[:, COCO[j]]
        th = np.arctan2(v[:, 1], v[:, 0])
        d = np.diff(th, prepend=th[0])
        d = (d + np.pi) % (2 * np.pi) - np.pi
        dth += np.abs(d)
    out["direction_change"] = _roll(dth / len(DISTAL), win).round(4).tolist()

    # rolling fidgety-band power share (2 s windows, 1 s hop)
    w = max(16, int(round(2.0 * fps)))
    hop = max(1, int(round(1.0 * fps)))
    ft, fv = [], []
    for s in range(0, max(1, T - w + 1), hop):
        seg = distal_speed[s:s + w]
        if len(seg) < 16:
            continue
        f, psd = welch(seg, fs=fps, nperseg=min(len(seg), 128), detrend="linear")
        band = (f >= FID_FMIN) & (f <= FID_FMAX)
        tot = psd.sum()
        ft.append(float((s + w / 2) / fps))
        fv.append(float(psd[band].sum() / tot) if tot > 0 else np.nan)
    out["fidgety_t"] = [round(v, 2) for v in ft]
    out["fidgety_power"] = [None if not np.isfinite(v) else round(v, 4) for v in fv]

    # summary line the UI shows above the charts
    out["summary"] = {
        "distal_speed_mean": float(np.mean(distal_speed)),
        "distal_speed_iqr": float(np.subtract(*np.percentile(distal_speed, [75, 25]))),
        "small_amp_fraction": float(small.mean()),
        "direction_change_mean": float(np.mean(dth / len(DISTAL))),
        "fidgety_power_mean": float(np.nanmean(fv)) if fv else None,
        "lr_balance": out["lr_balance"],
        "limb_speed_mean": limb_mean,
    }
    return out


def series_frame_table(series: Dict) -> "object":
    """Long-format per-frame table for ML export (one row per frame)."""
    import pandas as pd
    cols = {"t_s": series["t"], "distal_speed": series["distal_speed"],
            "small_amp_frac": series["small_amp_frac"],
            "direction_change": series["direction_change"]}
    for name in LIMBS:
        cols[name + "_speed"] = series[name]
    return pd.DataFrame(cols)
