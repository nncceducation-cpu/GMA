"""Pose extraction — ViTPose-H via MMPose.

WHY ViTPose and not the obvious choices:

A systematic comparison of seven pose estimators on infants in supine position
(arXiv 2406.17382, 2024) concluded:

    "state-of-the-art human pose estimation methods work well to estimate infant
     poses without the need for additional training or finetuning. ViTPose has
     the best accuracy, followed by HRNet (top-down)... DeepLabCut... as well as
     MediaPipe with BlazePose does not provide competitive results at all."

Segado (GigaScience 2026) independently chose ViTPose-H over HRNet, PVTv2 and a
fine-tuned OpenPose.

So: **MediaPipe is the wrong choice for infants**, despite being the easiest to
install. Do not "simplify" this module to MediaPipe.

No fine-tuning is required — the pretrained COCO weights work on infants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("neogma.pose")

# ViTPose-H, pretrained. MMPose model-zoo alias; weights pulled on first use.
DEFAULT_MODEL = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192"


class PoseExtractor:
    """Lazy wrapper so importing this module does not require MMPose."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cuda",
                 det_cat_id: int = 0):
        self.model = model
        self.device = device
        self.det_cat_id = det_cat_id
        self._inferencer = None

    def _load(self):
        if self._inferencer is None:
            from mmpose.apis import MMPoseInferencer   # heavy import
            logger.info("loading %s on %s", self.model, self.device)
            self._inferencer = MMPoseInferencer(pose2d=self.model,
                                                device=self.device)
        return self._inferencer

    def extract(self, video_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (xy [T,17,2], conf [T,17], fps).

        Keeps only the highest-scoring detection per frame. The infant-pose
        benchmark found that "using the highest-scored detection resulted in the
        closest performance to the optimal detection for all methods".
        """
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        inferencer = self._load()
        xs, cs = [], []
        for result in inferencer(str(video_path), show=False):
            preds = result["predictions"][0]
            if not preds:
                # No detection: emit NaN and let normalise() interpolate. Never
                # silently drop frames - that would fabricate a time base.
                xs.append(np.full((17, 2), np.nan, dtype=np.float32))
                cs.append(np.zeros(17, dtype=np.float32))
                continue
            best = max(preds, key=lambda p: float(np.mean(p["keypoint_scores"])))
            xs.append(np.asarray(best["keypoints"], dtype=np.float32)[:17])
            cs.append(np.asarray(best["keypoint_scores"], dtype=np.float32)[:17])

        xy = np.stack(xs)
        conf = np.stack(cs)

        # forward/backward fill any all-NaN frames before handing on
        for j in range(17):
            col = xy[:, j, :]
            bad = ~np.isfinite(col).all(axis=1)
            if bad.all():
                raise ValueError(f"keypoint {j} never detected - pose failed")
            if bad.any():
                idx = np.arange(len(col))
                for d in (0, 1):
                    col[bad, d] = np.interp(idx[bad], idx[~bad], col[~bad, d])
        return xy, conf, float(fps)
