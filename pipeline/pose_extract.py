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
import time
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger("neogma.pose")

# MMPose model-zoo alias; weights pulled on first use.
VITPOSE_MODEL = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192"

# Throughput knobs. See _extract_vitpose() for why these are safe.
POSE_BATCH = int(os.getenv("NEOGMA_POSE_BATCH", "24"))
DET_EVERY = int(os.getenv("NEOGMA_DET_EVERY", "20"))
PAD_FRAC = float(os.getenv("NEOGMA_BOX_PAD", "0.18"))

# Half precision for INFERENCE ONLY. Measured on this machine (RTX 5080):
#   ViTPose-H, batch 24, fp32 -> 3.9 fps
#   ViTPose-H, batch 24, fp16 -> 14.8 fps   (3.8x)
# A 637M-parameter vision transformer in fp32 was the entire reason a 45 s clip
# took 7 minutes. fp16 is standard for pose inference: keypoint coordinates are
# quantised to pixels downstream anyway, so the last bits of mantissa cannot
# change a wrist position. We verify this empirically rather than assuming it —
# see the confidence figures logged by webapp/smoke.py before and after.
# Set NEOGMA_FP16=0 to fall back to fp32 if you ever want to check.
FP16 = os.getenv("NEOGMA_FP16", "1") == "1"

BACKENDS = ("vitpose", "vitpose_mmpose", "keypointrcnn")

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


NATIVE_TS = os.getenv("NEOGMA_VITPOSE_TS", "/app/models/vitpose_h.ts")


def native_available() -> bool:
    """The mm-free ViTPose-H runner: torch + torchvision, no mmcv, no compiler.

    THIS IS THE PREFERRED PATH, and not only because it installs anywhere.

    A centre running the Docker image and a centre running the desktop installer
    must produce the SAME numbers. If Docker used mmpose and the installer used
    the native runner, the two would differ by ~5% on distal speed — small, but
    systematic, and systematically different BY INSTALL METHOD. That is a site
    effect wearing a disguise: probes.py would eventually flag it, and by then a
    year of contributions would be contaminated.

    So both use this. mmpose is now only the tool that EXPORTED the weights.
    """
    return Path(NATIVE_TS).exists()


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
        from mmpose.apis import inference_topdown  # noqa: F401
        from mmpose.apis.inferencers import Pose2DInferencer  # noqa: F401
        return True
    except Exception as exc:
        logger.warning("MMPose present but not importable (%s: %s)",
                       type(exc).__name__, exc)
        return False


def resolve_backend(requested: str = "auto") -> str:
    """'auto': native ViTPose-H if the weights are here, then mmpose, then the
    Keypoint R-CNN fallback."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if native_available():
            return "vitpose"                  # native TorchScript — the default
        if mmpose_available():
            logger.warning(
                "Native ViTPose weights not found at %s; falling back to mmpose. "
                "Numbers will differ slightly from centres running the installer "
                "— export the weights with tools/export_vitpose.py.", NATIVE_TS)
            return "vitpose_mmpose"
        logger.warning(_FALLBACK_WARNING)
        return "keypointrcnn"
    if requested == "vitpose" and not native_available():
        if mmpose_available():
            return "vitpose_mmpose"
    if requested not in BACKENDS:
        raise ValueError(f"unknown pose backend '{requested}'; use one of {BACKENDS}")
    if requested == "vitpose" and not native_available():
        raise RuntimeError(
            f"Native ViTPose weights not found at {NATIVE_TS}. Run the installer, "
            "or export them once with tools/export_vitpose.py.")
    if requested == "vitpose_mmpose" and not mmpose_available():
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
            # Native TorchScript ViTPose-H. No mmcv, no mmengine, no compiler.
            from pipeline.pose_native import NativeViTPose
            self._impl = (NativeViTPose(NATIVE_TS, device=self.device),
                          self._detector())
            return self._impl
        if self.backend == "vitpose_mmpose":
            # We build ViTPose through Pose2DInferencer (which resolves the model
            # alias to a config + checkpoint for us) but then use ONLY its pose
            # model, driving it with our own person boxes. See _detector().
            from mmpose.apis.inferencers import Pose2DInferencer
            logger.info("loading ViTPose-H (%s) on %s", self.model, self.device)
            p2d = Pose2DInferencer(model=self.model, device=self.device)
            self._impl = (p2d.model, self._detector())
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
    def extract(self, video_path: Path, progress=None
                ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (xy [T,17,2], conf [T,17], fps) in COCO-17 order.

        `progress(done, total, fps)` is called as frames are consumed. It exists
        because this step takes minutes on a real clip, and a progress bar frozen
        at 10% is indistinguishable from a crash — which is how the first version
        of this looked, and users rightly do not wait on something that appears
        dead.

        Keeps only the highest-scoring detection per frame. The infant-pose
        benchmark found that "using the highest-scored detection resulted in the
        closest performance to the optimal detection for all methods".
        """
        if self.backend == "vitpose":
            xy, conf, fps = self._extract_native(video_path, progress)
        elif self.backend == "vitpose_mmpose":
            xy, conf, fps = self._extract_vitpose(video_path, progress)
        else:
            xy, conf, fps = self._extract_krcnn(video_path, progress=progress)
        return self._fill_gaps(xy), conf, fps

    def _extract_native(self, video_path: Path, progress=None):
        """Same batching and same detector cadence as the mmpose path — only the
        pose network is driven directly instead of through mmpose."""
        import cv2

        net, det = self._load()
        dev = next(det.parameters()).device

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        xs, cs = [], []
        buf_f, buf_b, last, idx = [], [], None, 0
        t0 = time.time()

        def flush():
            if not buf_f:
                return
            a, b = net(buf_f, buf_b)
            xs.append(a); cs.append(b)
            buf_f.clear(); buf_b.clear()
            if progress:
                done = sum(len(v) for v in xs)
                progress(done, total, done / max(time.time() - t0, 1e-9))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if idx % DET_EVERY == 0 or last is None:
                bb = self._detect_one(frame, det, dev)
                if bb is not None:
                    last = self._pad_box(bb, w, h)
            buf_f.append(frame); buf_b.append(last); idx += 1
            if len(buf_f) >= POSE_BATCH:
                flush()
        flush()
        cap.release()

        if not xs:
            raise ValueError(f"no frames decoded from {video_path}")
        return np.concatenate(xs), np.concatenate(cs), float(fps)

    def _detector(self):
        """Person detector — torchvision Faster R-CNN, NOT mmdet.

        WHY NOT mmdet, which MMPose would use by default:

        mmcv has no prebuilt wheel for torch 2.11 / cu128, so it compiles from
        source in this image — and without the CUDA toolkit (nvcc) present it
        builds CPU-only operators. MMPose then runs the RTMDet detector on the
        GPU, its NMS looks for a CUDA kernel that was never compiled, and you get
        `RuntimeError: nms_impl: implementation for device cuda:0 not found` —
        several minutes into the first clip, after a 2.4 GB download.

        The alternatives were: ship a ~3 GB CUDA toolkit into the image and
        recompile mmcv's kernels, or stop asking mmcv to do detection. Detection
        is the ONLY thing here that needs NMS, and torchvision's NMS is a
        compiled CUDA op we already have and already verified. So we detect with
        torchvision and pose with ViTPose. mmcv is still used inside mmpose for
        image transforms, which are CPU ops and work fine.

        This is not a downgrade: the pose model — the part that actually decides
        the wrist and ankle positions GMA depends on — is still ViTPose-H.
        """
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn)
        dev = self.device if torch.cuda.is_available() else "cpu"
        det = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            box_score_thresh=0.5)
        det.eval().to(dev)
        logger.info("person detector: torchvision Faster R-CNN on %s", dev)
        return det

    def _detect_one(self, frame, det, dev):
        """Highest-scoring COCO 'person' (label 1) in one frame, as xyxy.

        The infant-pose benchmark found that "using the highest-scored detection
        resulted in the closest performance to the optimal detection for all
        methods". The protocol also guarantees one infant alone in shot, so any
        second detection is an adult's hand or a toy — neither of which we want
        to pose-estimate.
        """
        import cv2
        import torch
        t = (torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             .permute(2, 0, 1).float().div_(255).to(dev))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=FP16 and dev.type == "cuda"):
            out = det([t])[0]
        person = out["labels"] == 1
        if not bool(person.any()):
            return None
        scores = out["scores"][person]
        bb = out["boxes"][person][int(torch.argmax(scores))]
        return bb.cpu().numpy().astype(np.float32)

    @staticmethod
    def _pad_box(bb, w, h, frac=PAD_FRAC):
        """Grow the box so limbs that move between detections stay inside it."""
        x1, y1, x2, y2 = bb
        dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
        return np.array([max(0, x1 - dx), max(0, y1 - dy),
                         min(w, x2 + dx), min(h, y2 + dy)], dtype=np.float32)

    def _extract_vitpose(self, video_path: Path, progress=None):
        """Batched top-down pose. Two things make this ~10x faster than the
        obvious loop:

        1. The DETECTOR runs every DET_EVERY frames, not every frame. The infant
           does not travel across the frame — the camera is fixed and the baby is
           supine. Only the limbs move, and a padded box absorbs that. Re-running
           a 40M-parameter detector 30 times a second to re-discover a box that
           barely changes is pure waste.

        2. The POSE MODEL runs in batches. ViTPose-H is 632M parameters; called
           one frame at a time the GPU spends most of its life idle, waiting on
           Python and the preprocessing pipeline. `inference_topdown` already
           batches multiple boxes within ONE image — we do the same across
           frames.

        Correctness is unchanged: every frame still gets its own pose estimate.
        Only the redundant work is removed.
        """
        import cv2
        import torch
        from mmengine.dataset import Compose, pseudo_collate
        from mmengine.registry import init_default_scope

        pose_model, det = self._load()
        dev = next(det.parameters()).device

        # Building the detector left mmengine's default registry scope set to
        # "mmdet", so Compose() would look for MMPose's transforms in mmdet's
        # registry and fail with "LoadImage is not in the mmdet::transform
        # registry". inference_topdown does this internally; since we drive the
        # model ourselves, we must do it ourselves.
        init_default_scope(pose_model.cfg.get("default_scope", "mmpose"))
        pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        xs: list = []
        cs: list = []
        buf_frames: list = []
        buf_boxes: list = []
        last_box = None
        t0 = time.time()

        def nan_row():
            xs.append(np.full((17, 2), np.nan, dtype=np.float32))
            cs.append(np.zeros(17, dtype=np.float32))

        def flush():
            """Run ViTPose on the buffered frames in a single batch."""
            if not buf_frames:
                return
            data_list, keep = [], []
            for i, (frame, bb) in enumerate(zip(buf_frames, buf_boxes)):
                if bb is None:
                    continue
                info = dict(img=frame)
                info["bbox"] = bb[None, :]              # [1, 4] xyxy
                info["bbox_score"] = np.ones(1, dtype=np.float32)
                info.update(pose_model.dataset_meta)
                data_list.append(pipeline(info))
                keep.append(i)

            out = [None] * len(buf_frames)
            if data_list:
                batch = pseudo_collate(data_list)
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.float16,
                        enabled=FP16 and dev.type == "cuda"):
                    results = pose_model.test_step(batch)
                for i, r in zip(keep, results):
                    out[i] = r.pred_instances

            for pi in out:
                if pi is None:
                    # No infant detected. Emit NaN and let normalise() interpolate
                    # — never drop the frame, because dropping frames fabricates a
                    # time base and every downstream feature is a derivative.
                    nan_row()
                else:
                    xs.append(np.asarray(pi.keypoints[0], dtype=np.float32)[:17])
                    cs.append(np.asarray(pi.keypoint_scores[0],
                                         dtype=np.float32)[:17])
            buf_frames.clear()
            buf_boxes.clear()

            if progress:
                done = len(xs)
                rate = done / max(time.time() - t0, 1e-9)
                progress(done, total, rate)

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if idx % DET_EVERY == 0 or last_box is None:
                bb = self._detect_one(frame, det, dev)
                if bb is not None:
                    last_box = self._pad_box(bb, w, h)
            buf_frames.append(frame)
            buf_boxes.append(last_box)
            idx += 1
            if len(buf_frames) >= POSE_BATCH:
                flush()
        flush()
        cap.release()

        if not xs:
            raise ValueError(f"no frames decoded from {video_path}")
        return np.stack(xs), np.stack(cs), float(fps)

    def _extract_krcnn(self, video_path: Path, batch: int = 8, progress=None):
        import cv2
        import torch

        model, dev = self._load()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        t0 = time.time()

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
            if progress:
                progress(len(xs), total,
                         len(xs) / max(time.time() - t0, 1e-9))

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
