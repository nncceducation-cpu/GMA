"""Publication-quality dashboard figure for one recording.

Uses matplotlib's object-oriented Figure API, NOT pyplot. pyplot keeps global
state and is not thread-safe; calling it from the worker thread is what froze
Nmotion mid-job. Do not "simplify" this to plt.subplots().

300 DPI, so the PNG can go straight into a manuscript or a case discussion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib.figure import Figure

BG = "#0f172a"
FG = "#e2e8f0"
MUT = "#94a3b8"
ACC = "#38bdf8"
OK = "#34d399"
BAD = "#f87171"
GRID = "#334155"

LIMB_COLOURS = {
    "left_arm": "#38bdf8", "right_arm": "#818cf8",
    "left_leg": "#34d399", "right_leg": "#fbbf24",
}


def _style(ax, title: str, ylab: str = "") -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=10, loc="left", pad=6)
    ax.set_ylabel(ylab, color=MUT, fontsize=8)
    ax.tick_params(colors=MUT, labelsize=7)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)


def dashboard(series: Dict, meta: Dict, out_path: Path,
              label: Optional[str] = None) -> Path:
    """Six panels: the clinical read of one recording."""
    t = np.asarray(series["t"], dtype=float)
    fig = Figure(figsize=(12, 10), dpi=300, facecolor=BG)

    sub = meta.get("subject_id", "?")
    age = meta.get("corrected_age_weeks", "?")
    head = f"NeoGMA — {sub} · {age} weeks corrected · {series['duration_s']:.0f}s"
    if label:
        head += f" · expert label: {label}"
    fig.suptitle(head, color=FG, fontsize=13, x=0.02, ha="left", y=0.985)

    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.22,
                          left=0.07, right=0.98, top=0.93, bottom=0.06)

    # 1. distal speed — the headline trace ---------------------------------
    ax = fig.add_subplot(gs[0, :])
    d = np.asarray(series["distal_speed"], dtype=float)
    ax.plot(t, d, color=ACC, lw=0.8)
    ax.axhline(np.median(d), color=MUT, lw=0.6, ls="--")
    ax.fill_between(t, 0, d, color=ACC, alpha=0.15)
    _style(ax, "Distal movement speed (wrists + ankles) — fidgety movements live here",
           "torso/s")
    ax.set_xlim(t[0], t[-1])

    # 2. per-limb ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for name, col in LIMB_COLOURS.items():
        ax.plot(t, series[name], color=col, lw=0.6, label=name.replace("_", " "))
    _style(ax, "Limb speed — persistent flatness in one limb is an asymmetry sign",
           "torso/s")
    ax.legend(fontsize=6, labelcolor=FG, facecolor=BG, edgecolor=GRID, ncol=2)
    ax.set_xlim(t[0], t[-1])

    # 3. left/right balance --------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    lm = series["summary"]["limb_speed_mean"]
    names = list(lm.keys())
    vals = [lm[k] for k in names]
    ax.barh(names, vals, color=[LIMB_COLOURS[k] for k in names], height=0.6)
    bal = series.get("lr_balance")
    _style(ax, f"Mean speed per limb — L/R balance {bal:.2f} (0.50 = symmetric)"
           if bal is not None else "Mean speed per limb", "torso/s")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)

    # 4. small-amplitude fraction -------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    sa = np.asarray(series["small_amp_frac"], dtype=float)
    ax.plot(t, sa, color=OK, lw=0.7)
    ax.fill_between(t, 0, sa, color=OK, alpha=0.15)
    _style(ax, "Fraction of distal joints at fidgety amplitude", "fraction")
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1])

    # 5. direction change — the fidgety signature ---------------------------
    ax = fig.add_subplot(gs[2, 1])
    dc = np.asarray(series["direction_change"], dtype=float)
    ax.plot(t, dc, color="#c084fc", lw=0.7)
    _style(ax, "Movement-direction change — fidgety wanders, stereotyped does not",
           "rad/frame")
    ax.set_xlim(t[0], t[-1])

    # 6. fidgety band power --------------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    ft = np.asarray(series["fidgety_t"], dtype=float)
    fv = np.asarray([np.nan if v is None else v for v in series["fidgety_power"]],
                    dtype=float)
    if len(ft):
        ax.plot(ft, fv, color="#f472b6", lw=1.0, marker="o", ms=2)
        ax.fill_between(ft, 0, fv, color="#f472b6", alpha=0.15)
    _style(ax, "Share of movement power in the fidgety band (0.5–6 Hz), rolling 2 s",
           "power share")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)", color=MUT, fontsize=8)
    if len(ft):
        ax.set_xlim(ft[0], ft[-1])

    out_path = Path(out_path)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    return out_path
