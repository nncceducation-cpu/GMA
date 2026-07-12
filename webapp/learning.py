"""Labelled store + retrain + honest metrics. Ported from Nmotion, adapted for GMA.

Two label families, and they arrive years apart:
  gma_label : fm_present / fm_absent / fm_abnormal   (available now)
  cp_status : cp / no_cp at 12-24 months             (the TRUE endpoint, later)

Everything is keyed on SUBJECT, never on video or window.
"""
from __future__ import annotations

import logging, os, threading, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("neogma.learning")

GMA_LABELS: Dict[str, str] = {
    "fm_present":  "Fidgety movements PRESENT (normal)",
    "fm_abnormal": "Fidgety movements ABNORMAL",
    "fm_absent":   "Fidgety movements ABSENT (high CP risk)",
}
CP_LABELS: Dict[str, str] = {"no_cp": "No CP at 12-24 months",
                             "cp": "Cerebral palsy confirmed"}

# The binary that matters. Stoen 2019: SPORADIC fidgety movements did NOT predict
# CP. The clinically meaningful contrast is absent vs present, not a graded scale.
def to_binary(gma: str) -> Optional[int]:
    if gma == "fm_absent":
        return 1
    if gma == "fm_present":
        return 0
    return None            # fm_abnormal is deliberately NOT collapsed

_LOCK = threading.RLock()
XGB = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
           eval_metric="logloss", random_state=42)


class Learner:
    def __init__(self, memory_dir: Path, model_path: Path):
        self.dir = Path(memory_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(model_path)
        self.features_csv = self.dir / "window_features.csv"
        self.manifest_csv = self.dir / "manifest.csv"

    def _read(self, p: Path) -> pd.DataFrame:
        if p.exists():
            try:
                return pd.read_csv(p, dtype={"subject_id": str})
            except Exception:
                logger.exception("unreadable: %s", p)
        return pd.DataFrame()

    def find_duplicate(self, sha: str) -> Optional[Dict]:
        m = self._read(self.manifest_csv)
        if m.empty or not sha or "content_sha256" not in m:
            return None
        hit = m[m.content_sha256.astype(str) == str(sha)]
        return None if hit.empty else hit.iloc[0].to_dict()

    def add(self, *, recording_id: str, subject_id: str, windows: pd.DataFrame,
            gma_label: Optional[str] = None, cp_status: Optional[str] = None,
            content_sha256: str = "", meta: Optional[Dict] = None) -> Dict:
        dup = self.find_duplicate(content_sha256)
        if dup and str(dup.get("subject_id")) != str(subject_id):
            raise ValueError(
                f"Byte-identical to a recording already stored for subject "
                f"'{dup.get('subject_id')}'. Storing it under '{subject_id}' "
                "would make one infant look like two, and no subject-level split "
                "can undo that.")
        meta = meta or {}
        with _LOCK:
            w = windows.copy()
            w["recording_id"] = recording_id
            w["subject_id"] = str(subject_id)
            w["gma_label"] = gma_label
            w["cp_status"] = cp_status
            for k, v in meta.items():
                w[k] = v
            store = self._read(self.features_csv)
            if not store.empty and "recording_id" in store:
                store = store[store.recording_id != recording_id]
            pd.concat([store, w], ignore_index=True, sort=False).to_csv(
                self.features_csv, index=False)

            man = self._read(self.manifest_csv)
            if not man.empty and "recording_id" in man:
                man = man[man.recording_id != recording_id]
            row = {"recording_id": recording_id, "subject_id": str(subject_id),
                   "content_sha256": content_sha256, "gma_label": gma_label,
                   "cp_status": cp_status, "n_windows": int(len(w)),
                   "labeled_at": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
            pd.concat([man, pd.DataFrame([row])], ignore_index=True,
                      sort=False).to_csv(self.manifest_csv, index=False)
        return {"recording_id": recording_id, "n_windows": int(len(w))}

    def set_outcome(self, subject_id: str, cp_status: str, **extra) -> int:
        """Join the CP outcome in when follow-up matures — possibly years later."""
        with _LOCK:
            n = 0
            for p in (self.features_csv, self.manifest_csv):
                d = self._read(p)
                if d.empty:
                    continue
                m = d.subject_id.astype(str) == str(subject_id)
                d.loc[m, "cp_status"] = cp_status
                for k, v in extra.items():
                    d.loc[m, k] = v
                d.to_csv(p, index=False)
                n = int(m.sum())
            return n

    # ---------------------------------------------------------------- training
    def feature_cols(self, df: pd.DataFrame) -> List[str]:
        drop = {"recording_id", "subject_id", "gma_label", "cp_status", "y",
                "window_index", "start_frame", "end_frame", "n_frames_window",
                "content_sha256", "site", "camera_model", "source_fps",
                "torso_px", "corrected_age_weeks", "labeled_at", "n_windows"}
        return [c for c in df.columns
                if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    def retrain(self, target: str = "gma") -> Dict:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        from pipeline.evaluate import grouped_cv, leakage_selftest

        with _LOCK:
            store = self._read(self.features_csv)
        if store.empty:
            return {"trained": False, "reason": "No labelled recordings yet."}

        if target == "gma":
            store["y"] = store["gma_label"].map(to_binary)
        else:
            store["y"] = store["cp_status"].map({"cp": 1, "no_cp": 0})
        store = store.dropna(subset=["y"])
        if store.empty:
            return {"trained": False,
                    "reason": f"No usable {target} labels yet "
                              "(fm_abnormal is intentionally not collapsed)."}
        store["y"] = store["y"].astype(int)

        n_sub = store.groupby("y")["subject_id"].nunique()
        if len(n_sub) < 2:
            return {"trained": False, "n_subjects": int(store.subject_id.nunique()),
                    "reason": f"Only one class present ({n_sub.index.tolist()}). "
                              "Nothing to learn."}
        if n_sub.min() < 2:
            return {"trained": False, "n_subjects": int(store.subject_id.nunique()),
                    "reason": (f"Only {int(n_sub.min())} infant(s) in the smaller "
                               "class. Between-infant generalisation is not "
                               "estimable and grouped CV cannot run. The data "
                               "contain no information about a NEW infant.")}

        fc = self.feature_cols(store)
        mk = lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0)
        cv = grouped_cv(store, fc, mk)
        leak = leakage_selftest(store, fc, mk)

        m = mk(); m.fit(store[fc].fillna(0).to_numpy(), store["y"].to_numpy())
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": m, "feature_cols": fc, "target": target}, self.model_path)

        return {"trained": True, "target": target,
                "n_windows": int(len(store)),
                "n_subjects": int(store.subject_id.nunique()),
                "cv": cv, "leakage_selftest": leak}

    def summary(self) -> Dict:
        man = self._read(self.manifest_csv)
        out = {"labels": GMA_LABELS, "cp_labels": CP_LABELS,
               "total_recordings": 0, "total_subjects": 0, "per_class": {},
               "cp_known": 0, "model_exists": self.model_path.exists()}
        for k, name in GMA_LABELS.items():
            out["per_class"][k] = {"name": name, "subjects": 0, "recordings": 0}
        if man.empty:
            return out
        out["total_recordings"] = int(len(man))
        out["total_subjects"] = int(man.subject_id.nunique())
        out["cp_known"] = int(man["cp_status"].notna().sum()) if "cp_status" in man else 0
        if "gma_label" in man:
            for g, sub in man.dropna(subset=["gma_label"]).groupby("gma_label"):
                if g in out["per_class"]:
                    out["per_class"][g]["recordings"] = int(len(sub))
                    out["per_class"][g]["subjects"] = int(sub.subject_id.nunique())
        return out
