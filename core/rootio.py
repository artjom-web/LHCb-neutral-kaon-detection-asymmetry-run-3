"""Thin ROOT I/O helpers.

Isolating these means the rest of the code never opens a TFile by hand, never
forgets to close one, and never silently reports ``A = 0`` for a fit that
failed to produce a workspace.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import ROOT


def enable_mt(enable: bool = True) -> None:
    if enable:
        ROOT.EnableImplicitMT()
    else:
        ROOT.DisableImplicitMT()


@contextlib.contextmanager
def open_file(path: os.PathLike | str, mode: str = "READ"):
    """Open a TFile, guaranteeing it is closed and reporting zombies clearly."""
    f = ROOT.TFile.Open(str(path), mode)
    if not f or f.IsZombie():
        if f:
            f.Close()
        raise IOError(f"Could not open ROOT file {path}")
    try:
        yield f
    finally:
        f.Close()


def build_chain(files: Sequence[os.PathLike | str], tree: str = "DecayTree"):
    """TChain over one or more files; keeps the chain alive on the returned RDF."""
    chain = ROOT.TChain(tree)
    for path in files:
        chain.Add(str(path))
    return chain


def rdf_from_files(files: Sequence[os.PathLike | str], tree: str = "DecayTree"):
    chain = build_chain(files, tree)
    rdf = ROOT.RDataFrame(chain)
    rdf._chain_keepalive = chain  # chain must outlive the RDataFrame
    return rdf


def read_fit_result(model_file: os.PathLike | str) -> Dict[str, float]:
    """Read ``A_sig``, its error, and the fit-quality flags from a workspace.

    Returns NaNs (never zeros) for anything unreadable, so a missing fit shows
    up as a gap on a plot rather than as a real measurement of zero asymmetry.
    """
    out: Dict[str, float] = {
        "A": np.nan,
        "A_err": np.nan,
        "N_sig": np.nan,
        "f_sig": np.nan,
        "status": np.nan,
        "cov_qual": np.nan,
        "edm": np.nan,
        "min_nll": np.nan,
        "converged": False,
    }
    model_file = Path(model_file)
    if not model_file.exists():
        return out

    try:
        with open_file(model_file) as f:
            ws = f.Get("ws")
            if not ws:
                return out
            pv = ws.allVars()
            a = pv.find("A_sig")
            if a:
                out["A"] = a.getVal()
                out["A_err"] = a.getError()
            fs = pv.find("f_sig")
            if fs:
                out["f_sig"] = fs.getVal()
            nt = pv.find("N_tot")
            if nt and fs:
                out["N_sig"] = nt.getVal() * fs.getVal()

            res = f.Get("fit_results")
            if res:
                out["status"] = float(res.status())
                out["cov_qual"] = float(res.covQual())
                out["edm"] = float(res.edm())
                out["min_nll"] = float(res.minNll())
                out["converged"] = bool(
                    res.status() == 0
                    and res.covQual() == 3
                    and np.isfinite(res.minNll())
                    and np.isfinite(res.edm())
                )
    except IOError:
        return out
    return out


def snapshot(rdf, path: os.PathLike | str, columns: Sequence[str], tree: str = "DecayTree"):
    """Snapshot ``columns`` to ``path``, checking they all exist first.

    RDataFrame's own error for a missing column is a C++ exception thrown deep
    inside a JIT'd call; checking up front gives a message naming the column.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    available = set(str(c) for c in rdf.GetColumnNames())
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"Cannot snapshot to {path}: columns not defined: {missing}\n"
            f"Available: {sorted(available)}"
        )
    return rdf.Snapshot(tree, str(path), list(columns))


def column_names(rdf) -> List[str]:
    return sorted(str(c) for c in rdf.GetColumnNames())


def apply_filters(rdf, filters: Iterable[tuple]):
    """Apply a list of (expression, label) pairs, keeping the cutflow labels."""
    for expr, label in filters:
        rdf = rdf.Filter(expr, label)
    return rdf


def cutflow(rdf) -> List[Dict[str, float]]:
    """Trigger the event loop once and return the cutflow as plain dicts."""
    report = rdf.Report()
    rows = []
    for cut in report:
        rows.append(
            {
                "name": cut.GetName(),
                "all": float(cut.GetAll()),
                "pass": float(cut.GetPass()),
                "eff": float(cut.GetEff()),
            }
        )
    return rows