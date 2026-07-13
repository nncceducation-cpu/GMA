"""ViTPose-H with NO mmpose, NO mmcv, NO mmengine, NO mmdet. torch + torchvision.

WHY THIS EXISTS

The container needs mmpose. mmpose needs mmcv, which has no wheel for
torch 2.11 / cu128 and therefore compiles from source — which needs a C++
toolchain. Asking a clinical centre to install Visual Studio Build Tools before
it can score a baby is not a one-click install; it is how a deployment dies
quietly and the tool never gets used.

But mmpose only does three things at inference:

  1. crop the person box to 256x192 with the UDP affine warp
  2. run a ViT + deconv head           <- the only part worth preserving exactly
  3. decode the heatmaps (UDP + DARK)

Step 2 is exported once, from the working container, to a TorchScript file
(tools/export_vitpose.py). Trace-vs-eager difference: 0.00e+00 — it is the same
network, not a reimplementation. Steps 1 and 3 are ordinary array code, ported
here from mmpose's own source so the numbers match rather than merely resemble.

The result installs with `pip install torch torchvision` on any machine.

VERIFY, DO NOT TRUST: tools/verify_native.py compares this against mmpose on a
real clip, keypoint by keypoint. If it does not agree to sub-pixel, it is wrong
and must not ship.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("neogma.pose_native")

# From the exported model's own config — printed by tools/export_vitpose.py.
INPUT_W, INPUT_H = 192, 256          # network input  (W, H)
HEAT_W, HEAT_H = 48, 64              # heatmap size   (W, H)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)   # RGB
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
BLUR_KERNEL = 11                     # UDPHeatmap default
BBOX_PADDING = 1.25                  # mmpose GetBBoxCenterScale default

ASPECT = INPUT_W / INPUT_H           # 0.75


# ─────────────────────────────────────────────────────── 1. the UDP affine crop
def _bbox_to_cs(bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """xyxy -> (centre, scale), padded, then forced to the network aspect ratio.

    The aspect fix matters: a tall crop squeezed into a 3:4 network input would
    distort every limb length, and limb length is what the features measure.
    """
    x1, y1, x2, y2 = bbox.astype(np.float32)
    centre = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([(x2 - x1), (y2 - y1)], dtype=np.float32) * BBOX_PADDING
    w, h = scale
    if w > h * ASPECT:
        scale = np.array([w, w / ASPECT], dtype=np.float32)
    else:
        scale = np.array([h * ASPECT, h], dtype=np.float32)
    return centre, scale


def _udp_warp_matrix(centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """mmpose.structures.bbox.get_udp_warp_matrix, rot = 0."""
    input_size = centre * 2
    m = np.zeros((2, 3), dtype=np.float32)
    sx = (INPUT_W - 1) / scale[0]
    sy = (INPUT_H - 1) / scale[1]
    m[0, 0] = sx
    m[0, 1] = 0.0
    m[0, 2] = sx * (-0.5 * input_size[0] + 0.5 * scale[0])
    m[1, 0] = 0.0
    m[1, 1] = sy
    m[1, 2] = sy * (-0.5 * input_size[1] + 0.5 * scale[1])
    return m


# ─────────────────────────────────────────────────────────── 3. the UDP decode
def _gaussian_blur(heatmaps: np.ndarray, kernel: int = BLUR_KERNEL) -> np.ndarray:
    """mmpose.codecs.utils.gaussian_blur — peak-preserving blur."""
    import cv2
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    out = heatmaps.copy()
    for k in range(K):
        omax = out[k].max()
        if omax <= 0:
            continue
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = out[k]
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        out[k] = dr[border:-border, border:-border]
        m = out[k].max()
        if m > 0:
            out[k] *= omax / m
    return out


def _heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K, H, W = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    idx = np.argmax(flat, axis=1)
    y, x = np.unravel_index(idx, (H, W))
    locs = np.stack([x, y], axis=-1).astype(np.float32)
    vals = flat[np.arange(K), idx]
    locs[vals <= 0.0] = -1
    return locs, vals


def _refine_dark_udp(kpts: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """mmpose.codecs.utils.refine_keypoints_dark_udp — sub-pixel refinement.

    Without this, keypoints are quantised to the 48x64 heatmap grid: one cell is
    ~4 px in the crop, which is the same order as a fidgety movement. This is not
    a cosmetic refinement; it is the difference between measuring the movement
    and measuring the grid.
    """
    K, H, W = heatmaps.shape
    hm = _gaussian_blur(heatmaps)
    np.clip(hm, 1e-3, 50.0, hm)
    np.log(hm, hm)
    pad = np.pad(hm, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    k = kpts.copy()
    valid = (k[:, 0] >= 0) & (k[:, 1] >= 0)
    idx = (k[:, 0] + 1 + (k[:, 1] + 1) * (W + 2)
           + (W + 2) * (H + 2) * np.arange(K))
    idx = idx.astype(int).reshape(-1, 1)

    i_ = pad[idx]
    ix1 = pad[idx + 1]
    iy1 = pad[idx + W + 2]
    ix1y1 = pad[idx + W + 3]
    ix1_y1_ = pad[idx - W - 3]
    ix1_ = pad[idx - 1]
    iy1_ = pad[idx - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    deriv = np.concatenate([dx, dy], axis=1).reshape(K, 2, 1)

    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hess = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(K, 2, 2)
    hess = np.linalg.inv(hess + np.finfo(np.float32).eps * np.eye(2))

    k = k - np.einsum("imn,ink->imk", hess, deriv).squeeze(-1)
    k[~valid] = kpts[~valid]
    return k


DEFAULT_MODEL = os.getenv("NEOGMA_VITPOSE_TS", "models/vitpose_h.ts")


class NativeViTPose:
    """ViTPose-H on torch alone. Same weights, same maths, no mm* packages."""

    def __init__(self, ts_path: str = DEFAULT_MODEL, device: str = "cuda",
                 fp16: bool = True):
        import torch
        p = Path(ts_path)
        if not p.exists():
            raise FileNotFoundError(
                f"ViTPose weights not found at {p}. Run the installer, or fetch "
                "vitpose_h.ts from the NeoGMA release page. It is 1.3 GB and is "
                "downloaded once.")
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            logger.warning("CUDA not available — running ViTPose on CPU. This "
                           "works, but expect minutes per clip, not seconds.")
        self.fp16 = fp16 and self.device == "cuda"
        self.net = torch.jit.load(str(p), map_location=self.device).eval()
        logger.info("ViTPose-H (native TorchScript) on %s, fp16=%s",
                    self.device, self.fp16)

    def __call__(self, frames, boxes) -> Tuple[np.ndarray, np.ndarray]:
        """frames: list of BGR images. boxes: list of xyxy or None.

        Returns (xy [B,17,2] in IMAGE pixels, conf [B,17]).
        """
        import cv2
        import torch

        crops, keep, cs = [], [], []
        for i, (f, b) in enumerate(zip(frames, boxes)):
            if b is None:
                continue
            centre, scale = _bbox_to_cs(np.asarray(b, dtype=np.float32))
            M = _udp_warp_matrix(centre, scale)
            crop = cv2.warpAffine(f, M, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            crop = (crop - MEAN) / STD
            crops.append(crop.transpose(2, 0, 1))
            keep.append(i)
            cs.append((centre, scale))

        xy = np.full((len(frames), 17, 2), np.nan, dtype=np.float32)
        conf = np.zeros((len(frames), 17), dtype=np.float32)
        if not crops:
            return xy, conf

        x = torch.from_numpy(np.stack(crops)).to(self.device)
        if self.fp16:
            x = x.half()
        with torch.no_grad():
            hm = self.net(x).float().cpu().numpy()      # [B,17,64,48]

        for j, i in enumerate(keep):
            k, v = _heatmap_maximum(hm[j])
            k = _refine_dark_udp(k, hm[j])
            # heatmap grid -> network input space
            k = k / np.array([HEAT_W - 1, HEAT_H - 1], dtype=np.float32) \
                  * np.array([INPUT_W, INPUT_H], dtype=np.float32)
            # network input space -> original image pixels
            centre, scale = cs[j]
            k = k / np.array([INPUT_W, INPUT_H], dtype=np.float32) * scale \
                + centre - 0.5 * scale
            xy[i] = k
            conf[i] = v
        return xy, conf
