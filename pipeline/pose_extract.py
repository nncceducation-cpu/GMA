"""Pose extraction — COCO-17 keypoints, two interchangeable backends.

WHICH MODEL, AND WHY

A systematic comparison of seven pose estimators on infants in supine position
(arXiv 2406.17382, 2024) concluded:

    "state-of-the-art human pose estimation methods work well to estimate infant
     poses without the need for additional training or finetuning. ViTPose has
     the best accuracy, followed by HRNet (top-down)... DeepLabCut... as well as
     MediaPipe with BlazePose does not provide competitive results at all."

Segado (GigaScience 2026) independently chose ViTPose-H over HRNet, PVTv2 and a
fine-tuned OpenPose. So MediaPipe is the WRONG choice for infants, despite being
the easiest to install. Do not "simplify" this module to MediaPipe.

No fine-tuning is required — the pretrained COCO weights work on infants.

TWO BACKENDS

    vitpose       ViTPose-H via MMPose.         Accuracy target. Preferred.
    keypointrcnn  Keypoint R-CNN (torchvision). Always available, no extra deps.

The fallback exists for a practical reason: MMPose depends on mmcv, which has no
prebuilt wheel for the torch-2.11 / CUDA-12.8 combination this project needs for
Blackwell GPUs, so it must compile from source and may fail. When it does, the
app must still run — a pipeline you cannot execute teaches you nothing.

Both backends emit the SAME contract: COCO-17 xy [T,17,2], conf [T,17], fps.
Everything downstream (normalise, features, motifs) is backend-agnostic.

THE HONEST CAVEAT
Keypoint R-CNN is NOT as accurate as ViTPose on infants, and its weakest
keypoints are wrists and ankles — exactly the joints GMA depends on (Segado's
permutation importance: ankle 41%, knee 39%). Therefore:

  * The backend is recorded in every recording's manifest row and shown in the UI.
  * Do NOT mix backends within a cohort. Switching backend mid-study is a site
    effect with a new name, and probes.py will flag it as one.
  * Use keypointrcnn to develop, debug and demo. Use vitpose for anything you
    intend to publish.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger("neogma.pose")

# MMPose model-zoo alias; weights pulled on first use.
VITPOSE_MODEL = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192"

BACKENDS = ("vitpose", "keypointrcnn")

_FALLBACK_WARNING = (
    "MMPose unavailable — falling back to torchvision Keypoint R-CNN. Accuracy "
    "on distal joints (wrist/ankle) is materially worse than ViTPose-H, and "
    "those are precisely the joints GMA depends on. Fine for development; do "
    "not publish from it, and never mix backends within a cohort."
)

_VITPOSE_MISSING = (
    "backend 'vitpose' was requested but MMPose/mmcv is not installed in this "
    "image. Either install it, or set NEOGMA_POSE_BACKEND=keypointrcnn and "
    "accept the accuracy cost (which is recorded in the manifest)."
)


def mmpose_available() -> bool:
    """Import what we ACTUALLY use, not just the top-level package.

    `import mmpose` succeeds even when the inference stack is broken, because the
    breakages live deeper: xtcocotools compiled against the wrong NumPy ABI, or
    mmengine reaching for pkg_resources on a setuptools that no longer ships it.
    Both of those raise only when mmpose.apis is imported — i.e. on the first
    clip, in a worker thread, after the user has waited. Probing the real import
    path here means a broken MMPose degrades to the fallback backend at startup,
    visibly, instead of crashing mid-job.
    """
    try:
        from mmpose.apis import MMPoseInferencer  # noqa: F401
        return True
    except Exception as exc:
        logger.warning("MMPose present but not importable (%s: %s)",
                       type(exc).__name__, exc)
        return False


def resolve_backend(requested: str = "auto") -> str:
    """'auto' picks ViTPose when MMPose imports, else Keypoint R-CNN."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if mmpose_available():
            return "vitpose"
        logger.warning(_FALLBACK_WARNING)
        return "keypointrcnn"
    if requested not in BACKENDS:
        raise ValueError(f"unknown pose backend '{requested}'; use one of {BACKENDS}")
    if requested == "vitpose" and not mmpose_available():
        raise RuntimeError(_VITPOSE_MISSING)
    return requested


class PoseExtractor:
    """Lazy wrapper — importing this module must not require MMPose or torch."""

    def __init__(self, backend: str = "auto", device: str = "cuda",
                 model: str = VITPOSE_MODEL):
        if backend == "auto":
            backend = os.getenv("NEOGMA_POSE_BACKEND", "auto")
        self.backend = resolve_backend(backend)
        self.device = device
        self.model = model
        self._impl = None

    # ------------------------------------------------------------------ load
    def _load(self):
        if self._impl is not None:
            return self._impl
        if self.backend == "vitpose":
            from mmpose.apis import MMPoseInferencer
            logger.info("loading ViTPose-H (%s) on %s", self.model, self.device)
            self._impl = MMPoseInferencer(pose2d=self.model, device=self.device)
        else:
            import torch
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn)
            dev = self.device if torch.cuda.is_available() else "cpu"
            if dev != self.device:
                logger.warning("CUDA not available; pose will run on CPU (slow)")
            w = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            m = keypointrcnn_resnet50_fpn(weights=w, box_score_thresh=0.5)
            m.eval().to(dev)
            self._impl = (m, dev)
            logger.info("loaded Keypoint R-CNN on %s", dev)
        return self._impl

    # --------------------------------------------------------------- extract
    def extract(self, video_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (xy [T,17,2], conf [T,17], fps) in COCO-17 order.

        Keeps only the highest-scoring detection per frame. The infant-pose
        benchmark found that "using the highest-scored detection resulted in the
        closest performance to the optimal detection for all methods".
        """
        if self.backend == "vitpose":
            xy, conf, fps = self._extract_vitpose(video_path)
        else:
            xy, conf, fps = self._extract_krcnn(video_path)
        return self._fill_gaps(xy), conf, fps

    def _extract_vitpose(self, video_path: Path):
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        inferencer = self._load()
        xs, cs = [], []
        for result in inferencer(str(video_path), show=False):
            preds = result["predictions"][0]
            if not preds:
                xs.append(np.full((17, 2), np.nan, dtype=np.float32))
                cs.append(np.zeros(17, dtype=np.float32))
                continue
            best = max(preds, key=lambda p: float(np.mean(p["keypoint_scores"])))
            xs.append(np.asarray(best["keypoints"], dtype=np.float32)[:17])
            cs.append(np.asarray(best["keypoint_scores"], dtype=np.float32)[:17])
        return np.stack(xs), np.stack(cs), float(fps)

    def _extract_krcnn(self, video_path: Path, batch: int = 8):
        import cv2
        import torch

        model, dev = self._load()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        xs, cs = [], []

        def flush(frames):
            if not frames:
                return
            with torch.no_grad():
                outs = model([f.to(dev) for f in frames])
            for o in outs:
                if len(o["keypoints"]) == 0:
                    xs.append(np.full((17, 2), np.nan, dtype=np.float32))
                    cs.append(np.zeros(17, dtype=np.float32))
                    continue
                i = int(torch.argmax(o["scores"]).item())
                k = o["keypoints"][i].cpu().numpy()           # [17,3] x,y,vis
                s = o["keypoints_scores"][i].cpu().numpy()    # [17] logits
                xs.append(k[:17, :2].astype(np.float32))
                # Keypoint R-CNN scores are unbounded logits. Squash to (0,1) so
                # the min_conf gate in normalise() means the same thing for both
                # backends — otherwise the same threshold silently means two
                # different things depending on which model ran.
                cs.append((1.0 / (1.0 + np.exp(-s[:17]))).astype(np.float32))

        buf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            buf.append(t)
            if len(buf) >= batch:
                flush(buf)
                buf = []
        flush(buf)
        cap.release()

        if not xs:
            raise ValueError(f"no frames decoded from {video_path}")
        return np.stack(xs), np.stack(cs), float(fps)

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _fill_gaps(xy: np.ndarray) -> np.ndarray:
        """Frames with no detection are NaN. Interpolate them — never drop them.
        Dropping frames silently fabricates a time base, and every downstream
        feature is a per-frame derivative."""
        for j in range(17):
            col = xy[:, j, :]
            bad = ~np.isfinite(col).all(axis=1)
            if bad.all():
                raise ValueError(
                    f"keypoint {j} was never detected — pose estimation failed. "
                    "Check the framing: the infant must be fully in view, supine, "
                    "filmed from above (see PROTOCOL.md).")
            if bad.any():
                idx = np.arange(len(col))
                for d in (0, 1):
                    col[bad, d] = np.interp(idx[bad], idx[~bad], col[~bad, d])
        return xy
