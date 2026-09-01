"""Lifetime binning helpers shared by stages 2, 3 and 4."""

from __future__ import annotations

import array
from typing import Dict, List, Optional, Sequence

import numpy as np

import ROOT

from ..core.util import asymmetric_xerr, weight_key


def weighted_bin_means(
    rdf,
    param: str,
    bin_edges: Sequence[float],
    weights: Sequence[Optional[str]],
) -> Dict[str, Dict[str, object]]:
    """Mean of ``param`` inside each bin, once per weight column.

    Returns ``{weight_key: {"x": [...], "xerr": (2, N) array}}``.

    The plotted x for a lifetime bin is the weighted *mean* lifetime in that
    bin, not the bin centre, because the lifetime distribution is steeply
    falling and the two differ substantially in the widest bins.  All the
    profiles are booked before any is read so RDataFrame fuses them into a
    single pass over the data.
    """
    edges = array.array("d", [float(e) for e in bin_edges])
    booked = {}
    for w in weights:
        wk = weight_key(w)
        model = ROOT.RDF.TProfile1DModel(
            f"ltmean_{wk}", f"ltmean_{wk}", len(edges) - 1, edges
        )
        booked[wk] = (
            rdf.Profile1D(model, param, param)
            if not w
            else rdf.Profile1D(model, param, param, w)
        )

    out: Dict[str, Dict[str, object]] = {}
    for wk, prof_ptr in booked.items():
        prof = prof_ptr.GetValue()
        centers = [prof.GetBinContent(i) for i in range(1, prof.GetNbinsX() + 1)]
        out[wk] = {"x": centers, "xerr": asymmetric_xerr(edges, centers)}
    return out


def quantile_edges(rdf, param: str, nbins: int, lo: float, hi: float,
                   weight: Optional[str] = "weight_0",
                   fine_bins: int = 1000) -> List[float]:
    """Bin edges that put an equal weighted yield in each bin.

    This is how the fixed lifetime edges in ``config/selection.py`` were
    originally derived; keep it so they can be re-derived when the selection
    changes rather than being unexplained magic numbers.
    """
    args = ((f"tmp_{param}", "tmp", fine_bins, lo, hi), param)
    hist = (rdf.Histo1D(*args, weight) if weight else rdf.Histo1D(*args)).GetValue()

    probs = ROOT.std.vector("double")()
    for i in range(nbins + 1):
        probs.push_back(i / nbins)
    quant = ROOT.std.vector("double")(nbins + 1)
    hist.GetQuantiles(nbins + 1, quant.data(), probs.data())
    return [quant[i] for i in range(nbins + 1)]