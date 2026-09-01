"""Figures for the statistical cost of reweighting (stage 4).

The drawing half of the old ``Analysis.plot_weighting_statistics``, now taking
the tidy DataFrame from :mod:`kspi_analysis.physics.stats` instead of
recomputing everything itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..physics.stats import final_iteration_by_bin, series_by_bin

XLABEL_LT = r"KS lifetime $(t/\tau)$"


def _vs_iteration(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    outpath: Path,
    logy: bool = False,
) -> Path:
    series = series_by_bin(df, column)
    iterations = sorted(df["iteration"].unique())

    fig, ax = plt.subplots(figsize=(6, 4))
    for ltbin in sorted(series):
        ax.plot(iterations, series[ltbin], label=f"ltbin {ltbin + 1}")
    if len(series) < 10:
        ax.legend(fontsize="small")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_statistics(
    df: pd.DataFrame,
    outdir: Path | str,
    bin_x: Optional[Sequence[float]] = None,
    bin_xerr=None,
    title: Optional[str] = None,
) -> Dict[str, Path]:
    """Write the four statistical-loss figures and return their paths.

    ``bin_x`` / ``bin_xerr`` are the weighted lifetime bin means from stage 2.
    If absent, geometric bin centres are used, which is only a cosmetic
    difference on the last plot.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    out["N_eff"] = _vs_iteration(
        df, "N_eff", r"$N_{eff}$", outdir / "Neff_vs_iteration.pdf"
    )
    out["S_w_inv"] = _vs_iteration(
        df, "S_w_inv", r"$S_w^{-1}$", outdir / "Swinv_vs_iteration.pdf", logy=True
    )
    out["R_w"] = _vs_iteration(
        df, "R_w", r"$R_w$", outdir / "Rw_vs_iteration.pdf"
    )

    # --- S_w^-1 across lifetime, at the final iteration ---
    last = final_iteration_by_bin(df)
    y = last["S_w_inv"].to_numpy(dtype=float)
    finite = y[np.isfinite(y)]
    norm = y / finite.max() if finite.size and finite.max() > 0 else y

    if bin_x is not None:
        x = np.asarray(bin_x, dtype=float)
        xerr = np.asarray(bin_xerr) if bin_xerr is not None else None
    else:
        edges = np.concatenate([last["lt_lo"].to_numpy(), [last["lt_hi"].iloc[-1]]])
        x = 0.5 * (edges[:-1] + edges[1:])
        xerr = 0.5 * (edges[1:] - edges[:-1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, norm, xerr=xerr, fmt="o", capsize=3)
    ax.set_xlabel(XLABEL_LT)
    ax.set_ylabel(r"$S_w^{-1}$ (normalized)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    path = outdir / "Swinv_vs_lifetime.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out["S_w_inv_vs_lt"] = path

    return out