"""Nuisance probes — the safety catch for unsupervised learning.

THE PROBLEM

Supervised leakage is detectable: enforce subject-level splits and the inflated
score collapses (Segado: ROC-AUC 0.86 -> 0.60). Unsupervised contamination is
NOT detectable that way, because there is no label to leak. An encoder trained on
infant recordings will happily learn:

    which hospital        which camera / phone model
    which room / blanket  how big the baby is
    which INFANT it is    what the frame rate was

None of that is clinical. A representation dominated by site identity can still
linearly predict CP *within* your cohort — because site correlates with case-mix
— and then fail completely somewhere else. This is the mechanism by which
"unsupervised deep learning found a novel biomarker" papers fail to replicate.

THE TEST, AND THE SUBTLETY THAT MAKES IT CORRECT

Naively: try to predict the nuisance variable from the embedding. If you can, the
representation is contaminated.

That naive test CRIES WOLF. In any real cohort site is correlated with outcome
through case-mix (a tertiary NICU sees sicker infants), so a representation that
correctly captures the movement signal will ALSO appear to "predict site" — via
the outcome. That is epidemiology, not contamination.

The question that actually decides is CONDITIONAL:

    Among infants who share the SAME outcome, can the embedding still tell you
    which site/camera they came from?

    yes -> the embedding encodes the recording setup.  CONTAMINATED.
    no  -> the apparent site signal was just case-mix. CLEAN.

Both are reported. A large gap between them is itself the informative quantity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("neogma.probes")


def _probe(Z: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray],
           task: str, seed: int = 0) -> Dict:
    """Linear probe with subject-level CV. Returns the score and its chance level."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                         StratifiedKFold, cross_val_predict)
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = ~pd.isna(y)
    Z, y = Z[keep], np.asarray(y)[keep]
    g = None if groups is None else np.asarray(groups)[keep]
    if len(np.unique(y)) < 2:
        return {"ok": False, "reason": "nuisance variable is constant"}

    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        if g is not None:
            per = pd.DataFrame({"y": y, "g": g}).groupby("y")["g"].nunique()
            if per.min() < 2:
                return {"ok": False,
                        "reason": f"only {int(per.min())} group(s) in the smallest class"}
            k = int(min(5, per.min()))
            cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            split = list(cv.split(Z, y, g))
        else:
            k = int(min(5, counts.min()))
            if k < 2:
                return {"ok": False, "reason": "too few samples per class"}
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            split = list(cv.split(Z, y))

        model = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
        preds = np.empty_like(y)
        for tr, te in split:
            m = model.fit(Z[tr], y[tr])
            preds[te] = m.predict(Z[te])
        score = float(balanced_accuracy_score(y, preds))
        chance = 1.0 / len(classes)
        return {"ok": True, "metric": "balanced_accuracy", "score": score,
                "chance": chance, "n_classes": int(len(classes)),
                "leakage": float(score - chance)}

    # regression (frame rate, torso size in px, ...)
    ng = len(np.unique(g)) if g is not None else 5
    k = int(min(5, ng))
    if k < 2:
        return {"ok": False, "reason": "too few groups"}
    cv = GroupKFold(n_splits=k) if g is not None else k
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred = cross_val_predict(model, Z, y, cv=cv, groups=g)
    r2 = float(r2_score(y, pred))
    return {"ok": True, "metric": "r2", "score": r2, "chance": 0.0,
            "leakage": max(r2, 0.0)}


DEFAULT_NUISANCES = {
    "site": "classification",
    "camera_model": "classification",
    "source_fps": "regression",
    "torso_px": "regression",          # camera distance / zoom
}


def nuisance_report(Z: np.ndarray, meta: pd.DataFrame,
                    subject_col: str = "subject_id",
                    nuisances: Optional[Dict[str, str]] = None,
                    condition_on: Optional[str] = None,
                    warn_at: float = 0.20) -> pd.DataFrame:
    """Probe the representation for each nuisance variable.

    Pass `condition_on="gma_label"` (or the CP outcome). The CONDITIONAL leak is
    what decides the verdict; see the module docstring.
    """
    nuis = dict(nuisances or DEFAULT_NUISANCES)
    nuis.setdefault(subject_col, "classification")   # can it memorise the infant?

    rows: List[Dict] = []
    for col, task in nuis.items():
        if col not in meta.columns:
            continue
        groups = None if col == subject_col else meta[subject_col].to_numpy()

        uncond = _probe(Z, meta[col].to_numpy(), groups, task)

        cond_s, cond_c = [], []
        if condition_on and condition_on in meta.columns and col != subject_col:
            for _lev, idx in meta.groupby(condition_on).groups.items():
                idx = np.asarray(list(idx))
                if len(idx) < 6:
                    continue
                g = None if groups is None else groups[idx]
                rc = _probe(Z[idx], meta[col].to_numpy()[idx], g, task)
                if rc.get("ok"):
                    cond_s.append(rc["score"]); cond_c.append(rc["chance"])

        if not uncond.get("ok") and not cond_s:
            rows.append({"nuisance": col, "status": "skipped",
                         "score": np.nan, "chance": np.nan,
                         "leak_uncond": np.nan, "leak_conditional": np.nan,
                         "detail": uncond.get("reason", "")})
            continue

        u = uncond["leakage"] if uncond.get("ok") else np.nan
        c = (float(np.mean(cond_s) - np.mean(cond_c))) if cond_s else np.nan
        decide = c if np.isfinite(c) else u

        status = ("CONTAMINATED" if decide > warn_at else
                  "suspicious" if decide > warn_at / 2 else "clean")
        detail = ""
        if status == "CONTAMINATED":
            detail = ("Holding outcome constant, the embedding STILL identifies "
                      "this. It encodes the recording setup.")
        elif np.isfinite(c) and np.isfinite(u) and (u - c) > warn_at / 2:
            detail = ("Apparent leak was case-mix, not the camera: it disappears "
                      "once outcome is held constant.")

        rows.append({"nuisance": col, "status": status,
                     "metric": uncond.get("metric", ""),
                     "score": round(uncond["score"], 3) if uncond.get("ok") else np.nan,
                     "chance": round(uncond["chance"], 3) if uncond.get("ok") else np.nan,
                     "leak_uncond": round(u, 3) if np.isfinite(u) else np.nan,
                     "leak_conditional": round(c, 3) if np.isfinite(c) else np.nan,
                     "detail": detail})
    return pd.DataFrame(rows)


def outcome_probe(Z: np.ndarray, meta: pd.DataFrame, outcome_col: str,
                  subject_col: str = "subject_id") -> Dict:
    """Linear probe for the OUTCOME (GMA label, or CP status), grouped by subject.

    Run this AFTER nuisance_report. If the CONDITIONAL nuisance probes are
    contaminated, a score here means nothing: the model may be reading the camera.
    """
    return _probe(Z, meta[outcome_col].to_numpy(),
                  meta[subject_col].to_numpy(), "classification")


def verdict(nuis: pd.DataFrame, out: Dict) -> str:
    bad = nuis[nuis.status == "CONTAMINATED"]["nuisance"].tolist()
    if bad:
        return ("DO NOT USE. Holding outcome constant, the representation still "
                "identifies " + ", ".join(bad) + ". It encodes the recording "
                "setup, so any outcome prediction from it may be reading the "
                "camera rather than the infant.")
    if not out.get("ok"):
        return ("Representation is CLEAN of the tested nuisances, but the outcome "
                "cannot be probed yet: " + str(out.get("reason")))
    return (f"Representation is CLEAN of the tested nuisances (conditional on "
            f"outcome). Outcome probe: balanced accuracy {out['score']:.2f} "
            f"(chance {out['chance']:.2f}).")
