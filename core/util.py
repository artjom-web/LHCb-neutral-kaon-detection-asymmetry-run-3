"""Small pure helpers shared across stages.

These were static methods on ``Analysis`` (and duplicated again in
``multi_analysis.py`` and ``compare_hlt1cuts.py``).  There is now one copy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def weight_key(weight: Optional[str]) -> str:
    """Canonical, non-empty dict/directory key for a weight column name.

    '' and None both mean 'unweighted'.  Fit results are stored under this key,
    so it must stay stable: changing it invalidates every existing warm-start
    snapshot and output path.
    """
    return weight if weight else "unweighted"


def asymmetric_xerr(bin_edges: Sequence[float], bin_centers: Sequence[float]) -> np.ndarray:
    """(2, N) lower/upper distances from each plotted x to its own bin's edges.

    Used because the plotted x is a weighted bin *mean*, not the geometric
    centre, so the error bar is asymmetric within the bin.
    """
    bin_edges = np.asarray(bin_edges, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)
    lo, hi = bin_edges[:-1], bin_edges[1:]
    return np.vstack([bin_centers - lo, hi - bin_centers])


def capped_ylim(y, yerr, cap_lim: float = 1e-2, pad: float = 0.2) -> Tuple[float, float]:
    """y-range that ignores error bars larger than ``cap_lim``.

    The offending point is still drawn with its full error bar; it just is not
    allowed to blow out the axis so the other points become unreadable.
    """
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    valid = np.isfinite(y) & np.isfinite(yerr)
    if not valid.any():
        return 0.0, 1.0
    range_err = np.where(yerr > cap_lim, 0.0, yerr)
    lo = np.min((y - range_err)[valid])
    hi = np.max((y + range_err)[valid])
    rng = hi - lo if hi > lo else max(abs(lo), 1.0)
    return lo - pad * rng, hi + pad * rng


def particle_groups(params: Sequence[str]) -> Dict[str, List[str]]:
    particles: Dict[str, List[str]] = {"Pip": [], "KS": [], "Dp": []}
    for param in params:
        pid = param.split("_")[0]
        if pid in particles:
            particles[pid].append(param)
    return particles


def uniform_bin_centers_widths(p_args: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Centres/half-widths for plain linspace binning."""
    nbins, lo, hi = p_args[0], p_args[1], p_args[2]
    edges = np.linspace(lo, hi, nbins + 1)
    return 0.5 * (edges[:-1] + edges[1:]), 0.5 * (edges[1:] - edges[:-1])


def blinding_offset(seed: int = 0) -> float:
    """The blinding offset added to every plotted asymmetry.

    Deterministic in ``seed`` so that all stages and all samples share the same
    offset.  It is deliberately *not* applied to stored CSV values: the CSVs
    hold true fitted asymmetries and the offset is added at draw time only.
    """
    return float(np.random.RandomState(seed).uniform(-1, 1))