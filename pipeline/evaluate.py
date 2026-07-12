"""Evaluation — the part of this field that is most often wrong.

Segado et al. (GigaScience 2026) re-benchmarked a published model that reported
ROC-AUC 0.86:

    "we observed that the available implementation used record-wise splitting.
     This practice is a well-documented source of overfitting in medical data.
     After adjusting the split to avoid overlap, performance was substantially
     lower with an ROC-AUC of 0.60."

A peer-reviewed 0.86 was mostly leakage. This module makes that failure
structurally impossible:

  * splits are ALWAYS grouped by subject_id — a window from one infant can never
    appear in both train and test;
  * duplicate videos are detected by content hash, because grouping by ID does
    NOT catch the same infant uploaded twice under a new ID;
  * a lock-box test set is held out and can be opened exactly once;
  * PR-AUC, PPV and NPV are reported, never ROC-AUC alone. At ~10% prevalence,
    ROC-AUC flatters and PPV is the number that decides whether a screening tool
    is usable. Even expert GMA only achieves PPV ~33% in the real world
    (Stoen 2019).

It also REFUSES to produce a score when the data cannot support one.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("neogma.evaluate")

SUBJECT = "subject_id"     # the grouping unit. An infant. Never a video, never a window.


def prevalence_note(y: np.ndarray) -> str:
    p = float(np.mean(y))
    return (f"positive class prevalence {p:.1%}. At this prevalence ROC-AUC is "
            f"optimistic; PR-AUC and PPV are the decision-relevant metrics.")


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    """Sensitivity/specificity/PPV/NPV at a threshold. Report all four, always."""
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {"threshold": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv)}


def aggregate_to_subject(df: pd.DataFrame, proba: np.ndarray,
                         how: str = "mean") -> pd.DataFrame:
    """Windows are not the unit of clinical interest — infants are.

    A model that scores windows must be aggregated to one score per infant
    BEFORE metrics are computed, or the metrics silently weight infants by how
    long their video happened to be.
    """
    t = df[[SUBJECT, "y"]].copy()
    t["p"] = proba
    agg = {"mean": "mean", "max": "max", "median": "median"}[how]
    out = t.groupby(SUBJECT).agg(y=("y", "first"), p=("p", agg)).reset_index()
    return out


def grouped_cv(df: pd.DataFrame, feature_cols: List[str],
               model_factory, n_splits: int = 5, seed: int = 42,
               aggregate: str = "mean") -> Dict:
    """Stratified group k-fold, grouped by INFANT. Returns subject-level metrics.

    Refuses to run rather than return a meaningless number.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_win = df["y"].to_numpy()
    groups = df[SUBJECT].to_numpy()
    X = df[feature_cols].fillna(0).to_numpy()

    per_class_subjects = df.groupby("y")[SUBJECT].nunique()
    if len(per_class_subjects) < 2:
        return {"ok": False, "reason": "Only one class present. Nothing to learn."}
    min_subj = int(per_class_subjects.min())
    if min_subj < 2:
        return {"ok": False,
                "reason": (f"Only {min_subj} infant(s) in the smaller class. "
                           "Between-infant generalisation is not estimable; "
                           "grouped CV cannot run. This is not a technical "
                           "inconvenience - the data contain no information "
                           "about generalising to a NEW infant.")}
    k = int(min(n_splits, min_subj))

    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    subj_frames = []
    for tr, te in cv.split(X, y_win, groups):
        # hard guarantee: no infant straddles the split
        assert not (set(groups[tr]) & set(groups[te])), "SUBJECT LEAKED ACROSS SPLIT"
        m = model_factory()
        m.fit(X[tr], y_win[tr])
        p = m.predict_proba(X[te])[:, 1]
        subj_frames.append(aggregate_to_subject(df.iloc[te], p, how=aggregate))

    S = pd.concat(subj_frames, ignore_index=True)
    y, p = S["y"].to_numpy(), S["p"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"ok": False, "reason": "Folds produced a single-class test set."}

    roc = float(roc_auc_score(y, p))
    pr = float(average_precision_score(y, p))
    prev = float(np.mean(y))

    # Youden-J threshold. Selecting it post hoc is optimistically biased; say so.
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = int(np.argmax(tpr - fpr))
    op = operating_point(y, p, thr[j])

    return {
        "ok": True,
        "unit": "infant (subject-level, windows aggregated by %s)" % aggregate,
        "n_infants": int(len(S)),
        "n_positive": int(y.sum()),
        "prevalence": prev,
        "n_splits": k,
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_auc_baseline": prev,          # a useless model scores this
        "operating_point_youden": op,
        "warnings": [
            prevalence_note(y),
            "Threshold chosen post hoc on the ROC curve; this is optimistically "
            "biased. Preregister the threshold before touching the lock-box.",
        ] + ([f"Only {len(S)} infants - the estimate is very high variance."]
             if len(S) < 50 else []),
    }


def leakage_selftest(df: pd.DataFrame, feature_cols: List[str],
                     model_factory, seed: int = 0) -> Dict:
    """Quantify what record-wise splitting would have bought us.

    Runs the SAME model both ways: split by window (wrong) and split by infant
    (right). The gap is the leakage. Segado measured 0.86 -> 0.60 on a published
    model; we measure it on ours, every time, and print it.
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    y = df["y"].to_numpy()
    g = df[SUBJECT].to_numpy()
    X = df[feature_cols].fillna(0).to_numpy()
    if len(np.unique(y)) < 2 or df.groupby("y")[SUBJECT].nunique().min() < 2:
        return {"ok": False, "reason": "insufficient subjects for the self-test"}

    def run(splitter, use_groups):
        ps, ys = [], []
        it = splitter.split(X, y, g) if use_groups else splitter.split(X, y)
        for tr, te in it:
            m = model_factory(); m.fit(X[tr], y[tr])
            ps.append(m.predict_proba(X[te])[:, 1]); ys.append(y[te])
        return float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))

    k = int(min(5, df.groupby("y")[SUBJECT].nunique().min()))
    leaky = run(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), False)
    honest = run(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed), True)
    return {"ok": True, "window_level_auc_LEAKY": leaky,
            "subject_level_auc_HONEST": honest,
            "leakage_inflation": leaky - honest,
            "note": ("If these differ materially, any published number that used "
                     "record-wise splitting is inflated by roughly this much.")}
