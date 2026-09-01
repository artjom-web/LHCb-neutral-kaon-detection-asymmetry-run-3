"""Cross-sample comparison figures (stage 6).

Generalised from compare_hlt1cuts.py, which could only compare HLT1 lines
within a track.  Here the thing being compared is a *column name*, so the same
code compares HLT1 selections, polarities, ycsets or tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config.selection import HLT1_LABELS
from ..core.util import capped_ylim
from ..physics.combine import linear_trend, weighted_mean

XLABEL_LT = r"KS lifetime $(t/\tau)$"
YLABEL_A = "A blinded"


def _pretty(column: str, value: str) -> str:
    if column == "hlt1":
        return HLT1_LABELS.get(value, value or "no HLT1 cut")
    return str(value)


def _errorbar(ax, grp: pd.DataFrame, bias: float, label=None, color=None, offset=0.0):
    xerr = np.vstack(
        [grp["xerr_lo"].to_numpy(dtype=float), grp["xerr_hi"].to_numpy(dtype=float)]
    )
    ax.errorbar(
        grp["x"].to_numpy(dtype=float) + offset,
        grp["A"].to_numpy(dtype=float) + bias,
        xerr=xerr,
        yerr=grp["A_err"].to_numpy(dtype=float),
        fmt="o",
        ms=4,
        capsize=3,
        lw=1,
        label=label,
        color=color,
    )


def _finalize(ax, title, ylabel=None, legend=True):
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(XLABEL_LT)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize="small")


def plot_overlay(
    df: pd.DataFrame,
    compare: str,
    outpath: Path | str,
    bias: float = 0.0,
    title: str = "",
    fit_trend: bool = False,
) -> Path:
    """Overlay every value of ``compare`` on one axis."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ys, yerrs = [], []
    for k, value in enumerate(sorted(df[compare].dropna().unique())):
        grp = df[df[compare] == value].sort_values("bin")
        if grp.empty:
            continue
        _errorbar(ax, grp, bias, label=_pretty(compare, value), color=f"C{k}")
        ys.append(grp["A"].to_numpy(dtype=float) + bias)
        yerrs.append(grp["A_err"].to_numpy(dtype=float))

        if fit_trend:
            fit = linear_trend(grp["x"], grp["A"], grp["A_err"])
            if fit:
                xf = np.linspace(float(grp["lt_lo"].min()), float(grp["lt_hi"].max()), 100)
                ax.plot(
                    xf,
                    fit["slope"] * xf + fit["intercept"] + bias,
                    "--",
                    color=f"C{k}",
                    lw=1.2,
                )

    if not ys:
        plt.close(fig)
        print(f"WARNING: nothing to draw for {outpath.name}")
        return outpath

    _finalize(ax, title or f"comparison by {compare}", YLABEL_A)
    ax.set_ylim(*capped_ylim(np.concatenate(ys), np.concatenate(yerrs)))
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_panels(
    df: pd.DataFrame,
    panel_by: str,
    compare: str,
    outpath: Path | str,
    bias: float = 0.0,
    suptitle: str = "",
    fit_trend: bool = False,
) -> Path:
    """One panel per value of ``panel_by``, overlaying ``compare`` inside each.

    Panels share a y-axis so the panels are actually comparable — the thing
    the eye is being asked to judge here is a difference between panels, and an
    independent y-range per panel would hide it.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    panels = sorted(df[panel_by].dropna().unique())
    if not panels:
        print(f"WARNING: no data to panel by {panel_by}")
        return outpath

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(5.5 * n, 5), sharey=True, constrained_layout=True, squeeze=False
    )
    all_y, all_yerr = [], []

    for idx, panel in enumerate(panels):
        ax = axes[0, idx]
        sub = df[df[panel_by] == panel]
        for k, value in enumerate(sorted(sub[compare].dropna().unique())):
            grp = sub[sub[compare] == value].sort_values("bin")
            if grp.empty:
                continue
            _errorbar(ax, grp, bias, label=_pretty(compare, value), color=f"C{k}")
            all_y.append(grp["A"].to_numpy(dtype=float) + bias)
            all_yerr.append(grp["A_err"].to_numpy(dtype=float))
            if fit_trend:
                fit = linear_trend(grp["x"], grp["A"], grp["A_err"])
                if fit:
                    xf = np.linspace(
                        float(grp["lt_lo"].min()), float(grp["lt_hi"].max()), 100
                    )
                    ax.plot(
                        xf,
                        fit["slope"] * xf + fit["intercept"] + bias,
                        "--",
                        color=f"C{k}",
                        lw=1.2,
                    )
        _finalize(ax, _pretty(panel_by, panel), YLABEL_A if idx == 0 else None)

    if all_y:
        lo, hi = capped_ylim(np.concatenate(all_y), np.concatenate(all_yerr))
        axes[0, 0].set_ylim(lo, hi)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, weight="bold")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_slope_summary(
    df: pd.DataFrame,
    group_by: Sequence[str],
    outpath: Path | str,
    title: str = "Linear-fit slope of A vs lifetime",
) -> Optional[Dict]:
    """Fit A vs lifetime for each group and plot the slopes side by side.

    The slopes, not the individual points, are the summary statistic worth
    comparing between configurations, and the weighted-mean chi2 printed here
    is the quantitative version of 'do these configurations agree'.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for keys, grp in df.groupby(list(group_by), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        fit = linear_trend(grp["x"], grp["A"], grp["A_err"])
        if fit is None:
            print(f"WARNING: too few points to fit {keys}")
            continue
        rows.append({**dict(zip(group_by, keys)), **fit})

    if not rows:
        print("WARNING: no slopes to summarise")
        return None

    slopes = np.array([r["slope"] for r in rows])
    errs = np.array([r["slope_err"] for r in rows])
    labels = [
        "\n".join(_pretty(k, r[k]) for k in group_by) for r in rows
    ]

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(labels)), 5))
    for i, (s, e) in enumerate(zip(slopes, errs)):
        ax.errorbar(i, s, yerr=e, fmt="o", ms=6, capsize=4, lw=1.2, color=f"C{i % 10}")

    summary = weighted_mean(slopes, errs)
    if summary and summary["dof"] > 0:
        ax.axhline(summary["mean"], color="k", ls="--", lw=1.2)
        ax.text(
            0.02,
            0.95,
            f"weighted mean: {summary['mean']:.2e} ± {summary['err']:.1e}\n"
            f"$\\chi^2$/dof = {summary['chi2']:.1f}/{summary['dof']}, "
            f"p = {summary['pvalue']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"slope of A vs $t/\tau$")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(alpha=0.3)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"fits": rows, "summary": summary, "path": outpath}