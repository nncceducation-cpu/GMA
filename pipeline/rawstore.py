"""Immutable raw-data store — the substrate for unsupervised learning.

RATIONALE

Supervised training on GMA labels is capped by the GMA's own accuracy. Segado:
"FMs, while highly indicative, are still not a perfect biomarker for CP." If the
predictive signal for CP lives in movement patterns the GMA taxonomy does not
name, only representation learning over ALL movement can find it — and that needs
the raw data, retained forever, in a form that can be re-analysed with methods
that do not exist yet.

So we keep everything, at four levels of fidelity:

  L0  video.mp4        the original recording          (identifiable PHI)
  L1  pose.npz         keypoints [T,17,2] + confidence (identifiable movement)
  L2  pose_norm.npz    normalised pose (fps/scale/rotation removed)   <- SSL input
  L3  features.parquet windowed feature rows                          <- supervised input

L2 is the important one. It is the *canonical* representation: the camera has
been divided out, so two recordings of the same infant on different phones are
the same signal. Any self-supervised model should be trained on L2, never L0.

WHY NOT TRAIN SSL ON RAW VIDEO
An unsupervised model on L0 will learn the camera, the room, the blanket, the
lighting and the site before it learns anything about the brain — and unlike
supervised leakage there is no label to expose it. Normalised pose is a
deliberately impoverished input: almost everything it *can* encode is movement.

GOVERNANCE
L0 and L1 are identifiable health data. They never leave the machine, never enter
git, and require ethics approval and consent for retention. L2 and L3 are
derived, de-identified and shareable — which is exactly what Segado released,
and why: "Sharing videos and even keypoint time series across clinical sites is
prohibitive due to privacy and ethical constraints. In contrast, kinematic
features and model weights can be shared publicly."
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("neogma.rawstore")

MANIFEST = "manifest.csv"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class RawStore:
    """Append-only, content-addressed store. One directory per RECORDING,
    indexed by SUBJECT — because the subject is the unit that must never be
    split across train and test."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "recordings").mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST

    # ---------------------------------------------------------------- manifest
    def manifest(self) -> pd.DataFrame:
        if self.manifest_path.exists():
            try:
                return pd.read_csv(self.manifest_path, dtype={"subject_id": str})
            except Exception:
                logger.exception("manifest unreadable")
        return pd.DataFrame()

    def _write_manifest(self, df: pd.DataFrame) -> None:
        df.to_csv(self.manifest_path, index=False)

    def find_by_hash(self, sha: str) -> Optional[Dict]:
        m = self.manifest()
        if m.empty or "content_sha256" not in m:
            return None
        hit = m[m.content_sha256.astype(str) == str(sha)]
        return None if hit.empty else hit.iloc[0].to_dict()

    # ---------------------------------------------------------------- ingest
    def ingest(self, *, video: Path, subject_id: str, recording_id: str,
               corrected_age_weeks: float, site: str = "",
               camera_model: str = "", risk_group: str = "",
               extra: Optional[Dict] = None) -> Dict:
        """Store L0.

        A re-upload of a byte-identical video is ANALYSED, not refused — you may
        legitimately want to re-run it after a pipeline change, or simply look at
        it again. But it is recorded as a duplicate, because the danger was never
        the analysis; it was the TRAINING SET. The same infant appearing twice —
        or worse, the same video filed under two subject IDs — silently breaks
        every subject-level split and inflates every metric. So the link is kept
        here, and `Learner.add` uses it to refuse a second copy into memory.

        Analyse freely; train once.
        """
        video = Path(video)
        sha = sha256_file(video)
        dup = self.find_by_hash(sha)

        d = self.root / "recordings" / recording_id
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"video{video.suffix.lower()}"
        if not dst.exists():
            shutil.copyfile(video, dst)

        rec = {
            "recording_id": recording_id,
            "subject_id": str(subject_id),
            "content_sha256": sha,
            "corrected_age_weeks": float(corrected_age_weeks),
            "site": site,
            "camera_model": camera_model,
            "risk_group": risk_group,          # e.g. HIE / preterm / IVH
            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_file": dst.name,
            # Provenance of a re-upload. Null for an original.
            "duplicate_of": dup.get("recording_id") if dup else None,
            "duplicate_of_subject": dup.get("subject_id") if dup else None,
            "subject_id_conflict": bool(
                dup and str(dup.get("subject_id")) != str(subject_id)),
            # labels, filled in later — nullable by design
            "gma_label": None,                 # fm_present / fm_absent / fm_abnormal
            "gma_scored_by": None,
            "cp_status": None,                 # the TRUE endpoint, at 12-24 months
            "cp_gmfcs": None,
            "cp_assessed_age_months": None,
        }
        if extra:
            rec.update(extra)

        m = self.manifest()
        if not m.empty and "recording_id" in m:
            m = m[m.recording_id != recording_id]
        m = pd.concat([m, pd.DataFrame([rec])], ignore_index=True, sort=False)
        self._write_manifest(m)
        (d / "meta.json").write_text(json.dumps(rec, indent=2, default=str))
        logger.info("ingested %s (subject %s)", recording_id, subject_id)
        return rec

    # ---------------------------------------------------------------- levels
    def save_pose(self, recording_id: str, xy: np.ndarray, conf: np.ndarray,
                  fps: float, level: str = "L1") -> Path:
        """L1 raw keypoints, or L2 normalised pose. Both retained."""
        name = {"L1": "pose_raw.npz", "L2": "pose_norm.npz"}[level]
        p = self.root / "recordings" / recording_id / name
        np.savez_compressed(p, xy=xy.astype(np.float32),
                            conf=conf.astype(np.float32), fps=float(fps))
        return p

    def load_pose(self, recording_id: str, level: str = "L2"):
        name = {"L1": "pose_raw.npz", "L2": "pose_norm.npz"}[level]
        p = self.root / "recordings" / recording_id / name
        if not p.exists():
            return None
        z = np.load(p)
        return z["xy"], z["conf"], float(z["fps"])

    def save_features(self, recording_id: str, df: pd.DataFrame) -> Path:
        p = self.root / "recordings" / recording_id / "features.parquet"
        df.to_parquet(p, index=False)
        return p

    # ---------------------------------------------------------------- labels
    def set_label(self, recording_id: str, **labels) -> None:
        """Attach GMA and/or CP outcome. Both are nullable and may arrive years
        apart — the CP endpoint at 12-24 months is joined in later."""
        m = self.manifest()
        if m.empty or recording_id not in set(m.recording_id):
            raise KeyError(recording_id)
        for k, v in labels.items():
            if k not in m.columns:
                m[k] = None
            m.loc[m.recording_id == recording_id, k] = v
        self._write_manifest(m)

    # ---------------------------------------------------------------- SSL view
    def unlabelled_pose_corpus(self, level: str = "L2"):
        """Every recording with pose, LABELLED OR NOT.

        This is the whole point of retaining raw data: self-supervised learning
        consumes recordings that have no GMA score and no CP outcome. In a
        typical cohort that is the large majority of them.
        """
        m = self.manifest()
        out = []
        for _, r in m.iterrows():
            got = self.load_pose(r.recording_id, level)
            if got is None:
                continue
            xy, conf, fps = got
            out.append({"recording_id": r.recording_id,
                        "subject_id": str(r.subject_id),
                        "xy": xy, "fps": fps,
                        "site": r.get("site", ""),
                        "camera_model": r.get("camera_model", ""),
                        "gma_label": r.get("gma_label"),
                        "cp_status": r.get("cp_status")})
        return out
