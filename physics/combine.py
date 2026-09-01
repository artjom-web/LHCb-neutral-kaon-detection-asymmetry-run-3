"""Combining asymmetries across samples (stage 6).

Ported from ``load_asymmetry_results`` in multi_analysis.py, but reading the
stage-3 CSVs instead of re-opening every RooFit workspace.  That is the main
practical win of the new layout: combining a different set of polarities or
HLT1 lines is now a second-long CSV operation rather than a re-read of
thousands of ROOT files.

The combination is a per-bin inverse-variance weighted average:

    A     = sum(A_i / s_i^2) / sum(1 / s_i^2)
    s(A)  = 1 / sqrt(sum(1 / s_i^2))
    x     = sum(x_i / s_i^2) / sum(1 / s_i^2)

which is what the original did.  Bins where nothing is readable stay NaN so
they are visibly absent on a plot instead of looking like a measured zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Fields that identify a sample; anything in here can be grouped or combined over.
SAMPLE_KEYS: List[str] = ["track", "ycset", "polarity", "hlt1"]


def load_results(csv_paths: Iterable[Path | str]) -> pd.DataFrame:
    """Concatenate stage-3 CSVs, warning about (not silently dropping) missing ones."""
    frames = []
    for path in csv_paths:
        path = Path(path)
        if not path.exists():
            print(f"WARNING: missing results file {path}, skipping")
            continue
        df = pd.read_csv(path)
        df["source"] = str(path)
        frames.append(df)
    if not frames:
        raise SystemExit("No stage-3 asymmetry CSVs found to combine.")
    return pd.concat(frames, ignore_index=True)


def combine(
    df: pd.DataFrame,
    group_by: Sequence[str] = ("model", "procedure", "bin"),
    require_converged: bool = False,
) -> pd.DataFrame:
    """Inverse-variance combine every row sharing the same ``group_by`` values.

    Whatever is *not* in ``group_by`` is what gets combined over.  Keeping
    'track' in ``group_by`` combines polarities and ycsets within a track;
    dropping it combines the tracks too.

    ``require_converged=True`` drops fits that did not reach covQual 3.  It is
    off by default so the combined number matches the old behaviour, but it is
    worth running both ways: a large shift means a bad fit is carrying weight.
    """
    group_by = list(group_by)
    work = df.copy()

    if require_converged and "converged" in work.columns:
        before = len(work)
        work = work[work["converged"].astype(bool)]
        print(f"combine: kept {len(work)}/{before} converged fits")

    valid = (
        np.isfinite(work["A"])
        & np.isfinite(work["A_err"])
        & (work["A_err"] > 0)
    )
    dropped = int((~valid).sum())
    if dropped:
        print(f"combine: dropping {dropped} rows with unusable A / A_err")
    work = work[valid]

    if work.empty:
        raise SystemExit("combine: nothing left after filtering unusable fits.")

    work = work.assign(_w=1.0 / work["A_err"] ** 2)

    def _agg(grp: pd.DataFrame) -> pd.Series:
        w = grp["_w"].to_numpy()
        sw = w.sum()
        norm = np.sqrt(sw)
        out = {
            "A": float(np.average(grp["A"], weights=w)),
            "A_err": float(1.0 / norm) if norm > 0 else np.nan,
            "x": float(np.average(grp["x"], weights=w)),
            "lt_lo": float(grp["lt_lo"].iloc[0]),
            "lt_hi": float(grp["lt_hi"].iloc[0]),
            "n_samples": int(len(grp)),
            "chi2_consistency": float(
                np.sum(((grp["A"] - np.average(grp["A"], weights=w)) / grp["A_err"]) ** 2)
            ),
            "ndf_consistency": int(len(grp) - 1),
        }
        return pd.Series(out)

    combined = (
        work.groupby(group_by, dropna=False, sort=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    # error bars span each point's own bin, measured from its combined x
    combined["xerr_lo"] = combined["x"] - combined["lt_lo"]
    combined["xerr_hi"] = combined["lt_hi"] - combined["x"]
    return combined


def reindex_to_bins(
    combined: pd.DataFrame, n_bins: int, group_by: Sequence[str]
) -> pd.DataFrame:
    """Ensure every bin index appears for every group, filling gaps with NaN.

    Without this a lifetime bin where every sample failed would simply be
    missing from the table, and a downstream plot would join bin 2 straight to
    bin 4 as though nothing were wrong.
    """
    others = [g for g in group_by if g != "bin"]
    if not others:
        full = pd.DataFrame({"bin": range(n_bins)})
        return full.merge(combined, on="bin", how="left")

    keys = combined[others].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"bin": range(n_bins)}), how="cross")
    return grid.merge(combined, on=others + ["bin"], how="left")


def linear_trend(x, y, yerr):
    """Straight-line fit of A vs lifetime, with a chi2 p-value.

    A non-zero slope is the signature the analysis is looking for, so this
    (previously buried in compare_hlt1cuts.py) is part of the library.
    """
    from scipy.optimize import curve_fit
    from scipy.stats import chi2 as chi2_dist

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    x, y, yerr = x[ok], y[ok], yerr[ok]
    if len(x) < 3:
        return None

    def _lin(t, a, b):
        return a * t + b

    popt, pcov = curve_fit(_lin, x, y, sigma=yerr, absolute_sigma=True)
    resid = (y - _lin(x, *popt)) / yerr
    chi2_val = float(np.sum(resid**2))
    dof = len(x) - 2
    pval = float(chi2_dist.sf(chi2_val, dof)) if dof > 0 else np.nan
    return {
        "slope": float(popt[0]),
        "slope_err": float(np.sqrt(pcov[0, 0])),
        "intercept": float(popt[1]),
        "intercept_err": float(np.sqrt(pcov[1, 1])),
        "chi2": chi2_val,
        "dof": dof,
        "pvalue": pval,
    }


def weighted_mean(values, errors):
    """Weighted mean with its chi2 consistency test (used for slope summaries)."""
    from scipy.stats import chi2 as chi2_dist

    v = np.asarray(values, float)
    e = np.asarray(errors, float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[ok], e[ok]
    if len(v) == 0:
        return None
    w = 1.0 / e**2
    mean = float(np.sum(w * v) / np.sum(w))
    err = float(np.sqrt(1.0 / np.sum(w)))
    chi2_val = float(np.sum(((v - mean) / e) ** 2))
    dof = len(v) - 1
    return {
        "mean": mean,
        "err": err,
        "chi2": chi2_val,
        "dof": dof,
        "pvalue": float(chi2_dist.sf(chi2_val, dof)) if dof > 0 else np.nan,
    }