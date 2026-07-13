"""Export the corpus in a form you can actually model from — including the
warnings you would otherwise have to rediscover the hard way.

FOUR LEVELS, because they answer different questions:

  frames.parquet    one row per FRAME     -> deep learning on raw trajectories
  windows.parquet   one row per WINDOW    -> the supervised design matrix
  clips.parquet     one row per RECORDING -> summaries, quick looks
  pose/*.npz        normalised pose [T,17,2] -> self-supervised / motif learning

Every table carries subject_id. That is not decoration. It is the ONLY column
that makes a valid split possible, and it is why this exporter exists rather than
a one-line df.to_csv().

WHAT WILL RUIN YOUR MODEL, AND IS THEREFORE FLAGGED HERE

1. LEAKY COLUMNS. n_frames, fps, duration — these describe the RECORDING, not the
   infant. If clips of sicker babies happen to be shorter, a model will learn
   "short = abnormal" and score beautifully in cross-validation and fail in
   clinic. They are exported (you may want them for QC) but listed in
   LEAKY_COLS and named in the README as never-model columns.

2. RECORD-WISE SPLITTING. Segado re-ran a published model with record-wise
   instead of subject-wise splits: ROC-AUC fell from 0.86 to 0.60. Windows from
   one infant are not independent samples. The README says this in the first
   paragraph, and every table carries the group column needed to obey it.

3. PROTOCOL NON-COMPLIANCE. Clips under 60 s are scored but marked. They are
   included with protocol_compliant=False so you can exclude them — an "FMs
   absent" from a 20 s clip may just mean "not filmed long enough".
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Columns that describe the RECORDING, not the INFANT. Never model on these.
LEAKY_COLS = [
    "n_frames_window", "n_frames", "fps", "source_fps", "duration_s",
    "start_frame", "end_frame", "window_index", "n_windows", "torso_px",
]

LABEL_COLS = ["gma_label", "gma_scored_by", "cp_status", "cp_gmfcs",
              "cp_assessed_age_months"]

GROUP_COL = "subject_id"


def _dq(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality report: what is missing, what is constant, what is leaky.

    A constant feature is not harmless — it means the code that computes it is
    broken, and you would rather learn that here than after fitting a model."""
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        nun = int(s.nunique(dropna=True))
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "missing_frac": round(float(s.isna().mean()), 4),
            "n_unique": nun,
            "constant": nun <= 1,
            "leaky": c in LEAKY_COLS,
            "is_label": c in LABEL_COLS,
            "is_group": c == GROUP_COL,
            "mean": round(float(s.mean()), 6) if num and nun > 1 else None,
            "std": round(float(s.std()), 6) if num and nun > 1 else None,
        })
    return pd.DataFrame(rows)


README = """# NeoGMA export

## Read this before you fit anything

**Split by `subject_id`, never by row.** Windows and frames from one infant are
not independent samples. Segado (GigaScience 2026) re-ran a published model with
record-wise instead of subject-wise splitting and ROC-AUC fell from **0.86 to
0.60**. Use `StratifiedGroupKFold(groups=subject_id)` or equivalent.

**Do not model the columns listed as `leaky=True` in `data_quality.csv`.** They
describe the recording (length, frame rate, camera distance), not the infant. If
sicker babies happen to be filmed for less time, a model will learn "short =
abnormal", cross-validate beautifully, and fail in clinic.

**Check `protocol_compliant`.** Clips shorter than the 60 s Prechtl standard are
included but flagged. Fidgety movements are intermittent, so a short clip can
miss a bout entirely — "FMs absent" there may mean "not filmed long enough",
which reads as false CP risk. Exclude them for any cohort analysis.

**Report PR-AUC, PPV and NPV, not just ROC-AUC.** CP prevalence is low even in a
high-risk NICU cohort. Expert GMA itself has a PPV of about 33% at 10% prevalence
(Stoen 2019). An ROC-AUC alone will flatter any model here.

## Files

| file | one row per | use |
|---|---|---|
| `windows.parquet` | 5 s window | the supervised design matrix |
| `frames.parquet` | frame | deep learning on trajectories |
| `clips.parquet` | recording | summaries, quick looks |
| `pose/<recording_id>.npz` | recording | normalised pose `[T,17,2]`, torso units, common fps — the input for self-supervised / motif learning |
| `manifest.csv` | recording | subject, age, site, labels, QC |
| `data_dictionary.csv` | column | what each feature means |
| `data_quality.csv` | column | missingness, constants, **leaky flags** |

## Units

Pose is **normalised**: torso length = 1, rotated head-up, resampled to a common
frame rate. So speeds are in **torso lengths per second** and are comparable
across infants, cameras and phones. Nothing here is in pixels; pixels would
mostly encode how close the camera was held.

## Labels

`gma_label` is the expert's fidgety-movement score: `fm_present` (normal),
`fm_abnormal`, `fm_absent` (high CP risk). `cp_status` is the true endpoint at
12–24 months and arrives years later — it is null until follow-up matures. Both
are deliberately kept separate: GMA is a *surrogate* for CP, not CP itself.
"""

DICT_ROWS = [
    ("subject_id", "GROUP KEY. One per infant. Split on this."),
    ("recording_id", "One per video. An infant may have several."),
    ("gma_label", "Expert label: fm_present / fm_abnormal / fm_absent."),
    ("cp_status", "Outcome at 12-24 months: no_cp / cp. Null until follow-up."),
    ("protocol_compliant", "False if the clip is under the 60 s standard."),
    ("pose_backend", "vitpose or keypointrcnn. NEVER mix within a cohort."),
    ("*_speed_median", "Median joint speed in the window, torso/s."),
    ("*_speed_iqr", "Interquartile range of joint speed — variability."),
    ("*_dirvar_circstd", "Circular SD of movement direction at small amplitude. "
                         "The fidgety signature (Morais 2023)."),
    ("*_dirvar_entropy", "Entropy of movement direction over 12 sectors."),
    ("*_small_amp_fraction", "Fraction of frames at fidgety amplitude."),
    ("*_speed_peak_hz", "Spectral peak, band-limited to 0.5-6 Hz. Band-limiting "
                        "is essential: an unrestricted argmax returns the "
                        "frequency-resolution floor, i.e. the CAMERA."),
    ("*_band_power", "Share of spectral power in the fidgety band."),
    ("*_angle_mean/std", "Joint angle (elbow, knee) in degrees."),
    ("*_symmetry", "Left share of left+right limb speed. 0.5 = symmetric."),
    ("*_xcorr", "Correlation between left and right limb speed over the window."),
    ("distal_speed_mean", "Mean speed of wrists+ankles. The headline feature."),
    ("n_frames_window", "LEAKY. Window length. Never model on it."),
]


def build_bundle(store, learner, out_path: Path) -> Path:
    """Zip the whole corpus: features, frames, pose, labels, docs, QC."""
    from pipeline.series import compute_series, series_frame_table

    man = store.manifest()
    if man.empty:
        raise ValueError("nothing to export — no recordings stored yet")

    win_rows: List[pd.DataFrame] = []
    frm_rows: List[pd.DataFrame] = []
    clip_rows: List[Dict] = []
    poses: Dict[str, tuple] = {}

    for _, r in man.iterrows():
        rid = r["recording_id"]
        d = store.root / "recordings" / rid
        meta = {c: r.get(c) for c in man.columns}

        fp = d / "features.parquet"
        if fp.exists():
            w = pd.read_parquet(fp)
            for k, v in meta.items():
                if k not in w.columns:
                    w[k] = v
            win_rows.append(w)

        got = store.load_pose(rid, "L2")
        if got is not None:
            xy, conf, fps = got
            poses[rid] = (xy, conf, fps)
            s = compute_series(xy, fps)
            f = series_frame_table(s)
            f.insert(0, "recording_id", rid)
            f.insert(0, "subject_id", r["subject_id"])
            f["gma_label"] = r.get("gma_label")
            f["cp_status"] = r.get("cp_status")
            frm_rows.append(f)

            row = dict(meta)
            row.update({f"clip_{k}": v for k, v in s["summary"].items()
                        if not isinstance(v, dict)})
            for limb, val in s["summary"]["limb_speed_mean"].items():
                row[f"clip_{limb}_speed_mean"] = val
            clip_rows.append(row)

    windows = pd.concat(win_rows, ignore_index=True) if win_rows else pd.DataFrame()
    frames = pd.concat(frm_rows, ignore_index=True) if frm_rows else pd.DataFrame()
    clips = pd.DataFrame(clip_rows)

    out_path = Path(out_path)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        def put_df(name: str, df: pd.DataFrame, parquet: bool = True) -> None:
            if df.empty:
                return
            if parquet:
                b = io.BytesIO()
                df.to_parquet(b, index=False)
                z.writestr(name, b.getvalue())
            z.writestr(name.replace(".parquet", ".csv"), df.to_csv(index=False))

        put_df("windows.parquet", windows)
        put_df("frames.parquet", frames)
        put_df("clips.parquet", clips)

        for rid, (xy, conf, fps) in poses.items():
            b = io.BytesIO()
            np.savez_compressed(b, xy=xy, conf=conf, fps=fps)
            z.writestr(f"pose/{rid}.npz", b.getvalue())

        z.writestr("manifest.csv", man.to_csv(index=False))
        z.writestr("README.md", README)
        z.writestr("data_dictionary.csv",
                   pd.DataFrame(DICT_ROWS, columns=["column", "meaning"])
                   .to_csv(index=False))
        if not windows.empty:
            z.writestr("data_quality.csv", _dq(windows).to_csv(index=False))

        summary = {
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_subjects": int(man.subject_id.nunique()),
            "n_recordings": int(len(man)),
            "n_windows": int(len(windows)),
            "n_frames": int(len(frames)),
            "n_gma_labelled": int(man.gma_label.notna().sum())
            if "gma_label" in man else 0,
            "n_cp_known": int(man.cp_status.notna().sum())
            if "cp_status" in man else 0,
            "group_column": GROUP_COL,
            "leaky_columns": LEAKY_COLS,
            "split_rule": "StratifiedGroupKFold on subject_id. Never row-wise.",
        }
        z.writestr("summary.json", json.dumps(summary, indent=2, default=str))

    return out_path


def all_data_csv(store) -> str:
    """Everything joined into ONE flat CSV — window features + labels + subject."""
    man = store.manifest()
    parts = []
    for _, r in man.iterrows():
        fp = store.root / "recordings" / r["recording_id"] / "features.parquet"
        if not fp.exists():
            continue
        w = pd.read_parquet(fp)
        for c in man.columns:
            if c not in w.columns:
                w[c] = r.get(c)
        parts.append(w)
    if not parts:
        return "no recordings with features yet\n"
    df = pd.concat(parts, ignore_index=True)
    front = [c for c in [GROUP_COL, "recording_id", "gma_label", "cp_status",
                         "protocol_compliant", "site", "corrected_age_weeks"]
             if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].to_csv(index=False)
