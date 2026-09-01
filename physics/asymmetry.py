"""Asymmetry extraction: mass fits in lifetime bins (stage 3).

Ported from ``Analysis.plot_A_vs_kslt``, with the plotting split off.  The
function here *only* produces numbers, in a tidy long-form table; drawing them
is stage 3's script calling :mod:`kspi_analysis.plotting.asymmetry`, and
combining them across samples is stage 6.  That separation is what lets stage 6
re-plot without re-fitting, which the old code could not do.
"""

from __future__ import annotations

import array
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import ROOT

from ..config.massmodels import PROCEDURE_WEIGHTS
from ..config.selection import TrackConfig
from ..core import rootio
from ..core.runid import RunID
from ..core.util import weight_key
from .massfit import MassFitter


def _mass_axis(cfg: TrackConfig) -> array.array:
    return array.array("d", np.linspace(cfg.min_m, cfg.max_m, cfg.nbins_m + 1))


def _charge_axis() -> array.array:
    return array.array("d", np.linspace(-2, 2, 3))


def book_lifetime_mass_histograms(
    rdf, cfg: TrackConfig, procedures: Dict[str, str]
) -> Dict[str, object]:
    """One TH3(KS_LT, Dp_M, Dp_charge) per procedure, all booked before reading.

    Booking everything first matters: each ``GetValue()`` triggers an event
    loop, and RDataFrame only fuses results that were booked before the first
    trigger.  Reading them in a separate loop below turns N passes into one.
    """
    ma, ca = _mass_axis(cfg), _charge_axis()
    tedges = array.array("d", [float(e) for e in cfg.lt_bin_edges])

    booked: Dict[str, object] = {}
    for weight in dict.fromkeys(procedures.values()):
        wk = weight_key(weight)
        hkey = f"h3D_{wk}-KS_LT-Dp_M-Dp_charge"
        model = ROOT.RDF.TH3DModel(
            hkey, hkey, len(tedges) - 1, tedges, len(ma) - 1, ma, len(ca) - 1, ca
        )
        args = ("KS_LT", "Dp_M", "Dp_charge")
        booked[wk] = (
            rdf.Histo3D(model, *args, weight) if weight else rdf.Histo3D(model, *args)
        )
    return booked


def fit_lifetime_bins(
    rdf,
    cfg: TrackConfig,
    fitter: MassFitter,
    layout,
    runid: RunID,
    models: Sequence[str],
    procedures: Optional[Dict[str, str]] = None,
    bin_means: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """Fit every (procedure, model, lifetime bin) and return a long-form table.

    One row per fit, with the asymmetry, its error, the fit-quality flags and
    the x position of the bin.  NaN means 'no usable fit' and is preserved all
    the way to the CSV — never silently turned into zero.
    """
    procedures = procedures or dict(PROCEDURE_WEIGHTS)

    booked = book_lifetime_mass_histograms(rdf, cfg, procedures)
    filled: Dict[str, object] = {}
    for wk, ptr in booked.items():
        h = ptr.GetValue()
        if h.GetSumw2N() == 0:
            h.Sumw2()
        filled[wk] = h

    edges = np.asarray(cfg.lt_bin_edges, dtype=float)
    records: List[dict] = []

    for procedure, weight in procedures.items():
        wk = weight_key(weight)
        h = filled[wk]
        n_bins = h.GetXaxis().GetNbins()

        means = (bin_means or {}).get(wk, {})
        xs = means.get("x")
        xerr = np.asarray(means.get("xerr")) if means.get("xerr") is not None else None

        for i in range(1, n_bins + 1):
            h_plus = h.ProjectionY(f"h_plus_{i}_{wk}", i, i, 2, 2)
            h_minus = h.ProjectionY(f"h_minus_{i}_{wk}", i, i, 1, 1)

            for model in models:
                folder = str(layout.massfit_bin_dir(runid, wk, model, i - 1, create=True)) + "/"
                
                if wk == 'final_weight':
                    ws_file_path = str(layout.massfit_bin_dir(runid, 'unweighted', model, i - 1, create=True)) + "/model.root"
                    fitter.fit(h_plus, h_minus, folder, model_name=model, warm_start_file = ws_file_path)

                else: fitter.fit(h_plus, h_minus, folder, model_name=model)

                res = rootio.read_fit_result(folder + "model.root")

                x = float(xs[i - 1]) if xs is not None else float(
                    0.5 * (edges[i - 1] + edges[i])
                )
                if xerr is not None:
                    xerr_lo, xerr_hi = float(xerr[0, i - 1]), float(xerr[1, i - 1])
                else:
                    xerr_lo = x - edges[i - 1]
                    xerr_hi = edges[i] - x

                records.append(
                    {
                        **runid.to_dict(),
                        "model": model,
                        "procedure": procedure,
                        "weight": wk,
                        "bin": i - 1,
                        "lt_lo": float(edges[i - 1]),
                        "lt_hi": float(edges[i]),
                        "x": x,
                        "xerr_lo": xerr_lo,
                        "xerr_hi": xerr_hi,
                        "A": res["A"],
                        "A_err": res["A_err"],
                        "N_sig": res["N_sig"],
                        "status": res["status"],
                        "cov_qual": res["cov_qual"],
                        "edm": res["edm"],
                        "min_nll": res["min_nll"],
                        "converged": bool(res["converged"]),
                        "fit_dir": folder,
                    }
                )

    return pd.DataFrame.from_records(records)


def pivot_for_plotting(df: pd.DataFrame, models: Sequence[str],
                       procedures: Sequence[str]):
    """Reshape the long table into the nested dicts ``AsymmetryPlotter`` wants.

    Returns ``(A, A_err, x, xerr)`` with ``A[model][procedure]`` arrays, which
    is the shape both the per-sample plots and the combined plots use — so one
    plotting class serves both.
    """
    A: Dict[str, Dict[str, np.ndarray]] = {}
    A_err: Dict[str, Dict[str, np.ndarray]] = {}
    x: Dict[str, Dict[str, np.ndarray]] = {}
    xerr: Dict[str, Dict[str, np.ndarray]] = {}

    for model in models:
        A[model], A_err[model], x[model], xerr[model] = {}, {}, {}, {}
        for proc in procedures:
            sub = df[(df["model"] == model) & (df["procedure"] == proc)]
            sub = sub.sort_values("bin")
            A[model][proc] = sub["A"].to_numpy(dtype=float)
            A_err[model][proc] = sub["A_err"].to_numpy(dtype=float)
            x[model][proc] = sub["x"].to_numpy(dtype=float)
            xerr[model][proc] = np.vstack(
                [sub["xerr_lo"].to_numpy(dtype=float),
                 sub["xerr_hi"].to_numpy(dtype=float)]
            )
    return A, A_err, x, xerr


def failed_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose fit did not reach full convergence."""
    return df[~df["converged"].astype(bool)].copy()