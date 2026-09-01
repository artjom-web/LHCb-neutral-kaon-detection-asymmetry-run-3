"""Statistical cost of the reweighting (stage 4).

Ported from the computation half of ``Analysis.plot_weighting_statistics``.
The old method computed and drew in one pass and returned nothing, so the
numbers behind the plots were unrecoverable.  Here the computation returns a
tidy DataFrame that gets written to CSV alongside the figures.

Quantities, per lifetime bin and per iteration:

    S_w      = sum(w^2) / (sum w)^2       — the statistical dilution factor
    S_w^-1                                — effective sample fraction retained
    R_w      = S_w(0) / S_w(i)            — dilution relative to the start
    N_eff    = sum(weight_0 * weight_acc) — surviving raw yield after cell
                                            rejection, i.e. the loss that comes
                                            from emptying sparse 4D cells rather
                                            than from the weights themselves
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def weighting_statistics(
    rdf, lt_bin_edges: Sequence[float], n_iterations: int
) -> pd.DataFrame:
    """Long-form table with one row per (lifetime bin, iteration)."""
    edges = list(map(float, lt_bin_edges))
    n_lt = len(edges) - 1
    iterations = list(range(n_iterations + 1))

    # --- book every sum across all bins and iterations before reading ---
    filtered = {}
    sum_w, sum_w2, sum_neff = {}, {}, {}
    for ltbin in range(n_lt):
        lo, hi = edges[ltbin], edges[ltbin + 1]
        filtered[ltbin] = rdf.Filter(f"KS_LT > {lo} && KS_LT < {hi}")
        for it in iterations:
            sum_w[(ltbin, it)] = filtered[ltbin].Sum(f"weight_{it}")
            sum_w2[(ltbin, it)] = filtered[ltbin].Sum(f"w2_{it}")
            sum_neff[(ltbin, it)] = filtered[ltbin].Sum(f"weight_eff_{it}")

    # --- the first GetValue() triggers the fused loop; the rest are free ---
    records: List[dict] = []
    for ltbin in range(n_lt):
        sw_series = []
        for it in iterations:
            w = sum_w[(ltbin, it)].GetValue()
            w2 = sum_w2[(ltbin, it)].GetValue()
            neff = sum_neff[(ltbin, it)].GetValue()
            sw = w2 / w**2 if w != 0 else np.nan
            sw_series.append(sw)
            records.append(
                {
                    "ltbin": ltbin,
                    "lt_lo": edges[ltbin],
                    "lt_hi": edges[ltbin + 1],
                    "iteration": it,
                    "sum_w": w,
                    "sum_w2": w2,
                    "N_eff": neff,
                    "S_w": sw,
                    "S_w_inv": 1.0 / sw if sw and np.isfinite(sw) else np.nan,
                }
            )
        sw0 = sw_series[0]
        for it in iterations:
            idx = ltbin * len(iterations) + it
            sw = records[idx]["S_w"]
            records[idx]["R_w"] = (
                sw0 / sw if sw and np.isfinite(sw) and np.isfinite(sw0) else np.nan
            )

    return pd.DataFrame.from_records(records)


def final_iteration_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lifetime bin, at the last iteration."""
    last = df["iteration"].max()
    return df[df["iteration"] == last].sort_values("ltbin").reset_index(drop=True)


def series_by_bin(df: pd.DataFrame, column: str) -> Dict[int, np.ndarray]:
    """``{ltbin: array over iterations}`` for one quantity."""
    out: Dict[int, np.ndarray] = {}
    for ltbin, grp in df.groupby("ltbin"):
        out[int(ltbin)] = grp.sort_values("iteration")[column].to_numpy(dtype=float)
    return out