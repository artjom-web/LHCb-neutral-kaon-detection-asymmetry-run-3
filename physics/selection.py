"""Data loading, derived columns and offline selection (stage 1).

Ported from ``Analysis.import_data`` / ``data_to_rdf`` / ``defs`` / ``cuts``.
The behaviour is unchanged; the difference is that each step is a free function
taking an RDataFrame and returning one, so they compose and can be reordered or
skipped without editing a class.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

import ROOT

from ..config import selection as sel
from ..config.selection import TrackConfig
from ..core import rootio
from ..core.runid import RunID


# ----------------------------------------------------------------------
# Input files
# ----------------------------------------------------------------------
def apd_files(runid: RunID, n_files: int) -> List[str]:
    """Resolve input file paths through the AnalysisProductions API."""
    from apd import AnalysisData

    datasets = AnalysisData("charm", "d_to_ksh")
    dataname = f"d_to_ksh_{runid.ycset}_{runid.polarity},_split_d2kspi_{runid.track}"
    year = runid.ycset.split("c")[0]
    result = datasets(
        config="lhcb",
        datatype=f"20{year}",
        filetype=f"d2kspi_{runid.track}.root",
        polarity=runid.polarity,
        eventtype="94000000",
        name=dataname,
    )
    if n_files > len(result):
        raise ValueError(
            f"{runid.label()}: asked for {n_files} files but the production "
            f"only has {len(result)}"
        )
    return [result[i] for i in range(n_files)]


def pfns_files(runid: RunID, n_files: int, base: str) -> List[str]:
    """Alternative input resolution: read PFNs from a text file (was import_PFNS)."""
    pol = "mu" if runid.polarity == "magup" else "md"
    path = f"{base}/{runid.track}/{pol}/{runid.ycset}_{pol}_{runid.track}.txt"
    with open(path) as f:
        rows = [line.strip() for line in f if line.strip()]
    if n_files > len(rows):
        raise ValueError(f"{path} has {len(rows)} lines, asked for {n_files}")
    return rows[:n_files]


def rdf_from_production(files: Sequence[str], track: str):
    tree = sel.TREE_NAMES.get(track)
    if tree is None:
        raise ValueError(f"No tree name configured for track {track!r}")
    return ROOT.RDataFrame(tree, list(files))


# ----------------------------------------------------------------------
# Derived columns
# ----------------------------------------------------------------------
def define_columns(rdf, columns=None):
    """Add the derived kinematic columns (was ``defs``)."""
    for name, expr in columns or sel.DERIVED_COLUMNS:
        rdf = rdf.Define(name, expr)
    return rdf


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
def apply_selection(
    rdf,
    cfg: TrackConfig,
    hlt1: str = "",
    *,
    mass_lt: bool = True,
    kinematic: bool = True,
    probnn: bool = True,
):
    """Apply the full offline selection in a fixed, documented order.

    Order matters for the cutflow report only, not for the final sample, but a
    fixed order makes two runs comparable line by line.  The old ``cuts()``
    had to be called twice (once for the trigger, once for everything else)
    because the trigger and offline blocks were mutually exclusive branches of
    the same ``if``; here they are independent flags.
    """
    if mass_lt:
        rdf = rootio.apply_filters(rdf, sel.mass_lt_filters(cfg))
    if hlt1:
        rdf = rootio.apply_filters(rdf, sel.hlt1_filters(hlt1, cfg.track))
    if kinematic:
        rdf = rootio.apply_filters(rdf, sel.kinematic_filters(cfg))
    if probnn:
        rdf = rootio.apply_filters(rdf, sel.PROBNN_FILTERS)
    return rdf


# ----------------------------------------------------------------------
# What to keep in the skim
# ----------------------------------------------------------------------
#: Columns every downstream stage needs, regardless of configuration.
ESSENTIAL_COLUMNS = ["Dp_M", "Dp_charge", "KS_LT"]


def skim_columns(
    cfg: TrackConfig, extra: Optional[Sequence[str]] = None
) -> List[str]:
    """Columns written by stage 1.

    Defaults to every variable that has histogram binning configured, plus the
    essentials.  That is deliberately generous: the skim is written once and
    read by every later stage, and adding a variable later means re-running the
    slowest stage in the chain.
    """
    hist_params = sel.build_hist_params(cfg)
    cols = set(hist_params) | set(ESSENTIAL_COLUMNS) | set(extra or [])
    return sorted(cols)