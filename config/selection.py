"""Per-track selection, binning and mass-window configuration.

Everything here used to live inside ``Analysis.__init__`` and ``Analysis.cuts``.
It is data, not behaviour, so it lives in config/ and is passed explicitly into
the physics code.  Nothing in this module imports ROOT, so it can be imported
(and unit-tested) anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence

import numpy as np

D_MASS = 1869.66  # MeV/c^2, PDG D+ mass, used to place the signal window


@dataclass(frozen=True)
class TrackConfig:
    """Everything that depends only on the KS track type ('ll' or 'dd')."""

    track: str
    ltmin: float
    ltmax: float
    lt_bin_edges: List[float]

    # mass window
    min_m: int = 1795
    max_m: int = 1943
    mass_bins_per_mev: int = 1

    # kinematic-variable binning used for the reweighting histograms
    kinvar_bins: int = 25
    kslt_plotbins: int = 100

    # reweighting acceptance threshold (min unweighted entries per 4D cell)
    threshold: float = 1.0

    @property
    def kslt_anabins(self) -> int:
        return len(self.lt_bin_edges) - 1

    @property
    def nbins_m(self) -> int:
        return (self.max_m - self.min_m) * self.mass_bins_per_mev

    # ---- sideband / signal window -------------------------------------
    @property
    def l_edge_sb(self) -> int:
        return self.min_m

    @property
    def r_edge_sb(self) -> int:
        return self.max_m

    @property
    def l_edge_sig(self) -> int:
        return int(np.ceil((self.min_m + D_MASS) / 2))

    @property
    def r_edge_sig(self) -> int:
        return int(np.floor((self.max_m + D_MASS) / 2))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            kslt_anabins=self.kslt_anabins,
            nbins_m=self.nbins_m,
            l_edge_sig=self.l_edge_sig,
            r_edge_sig=self.r_edge_sig,
        )
        return d


# The two production configurations.  Override per-run from the CLI if needed.
TRACK_CONFIGS: Dict[str, TrackConfig] = {
    "ll": TrackConfig(
        track="ll",
        ltmin=0.0,
        ltmax=0.4,
        lt_bin_edges=[0.0, 0.070, 0.093, 0.116, 0.142, 0.176, 0.4],
    ),
    "dd": TrackConfig(
        track="dd",
        ltmin=0.0,
        ltmax=3.0,
        lt_bin_edges=[0.0, 0.4686, 0.5727, 0.6518, 0.7385, 0.8259, 0.9733, 3.0],
    ),
}


def get_track_config(track: str, **overrides) -> TrackConfig:
    if track not in TRACK_CONFIGS:
        raise ValueError(f"Unexpected track {track!r}: use 'll' or 'dd'.")
    base = TRACK_CONFIGS[track]
    if not overrides:
        return base
    return TrackConfig(**{**asdict(base), **overrides})


# ----------------------------------------------------------------------
# Derived-column definitions (was Analysis.defs)
# ----------------------------------------------------------------------
DERIVED_COLUMNS: List[tuple] = [
    ("Pip_theta_x", "atan(Pip_PX/Pip_PZ)"),
    ("Pip_theta_y", "atan(Pip_PY/Pip_PZ)"),
    ("Pip_k", "1.0/sqrt(Pip_PX*Pip_PX*1e-6 + Pip_PZ*Pip_PZ*1e-6)"),
    ("Dp_theta_x", "atan(Dp_PX/Dp_PZ)"),
    ("Dp_theta_y", "atan(Dp_PY/Dp_PZ)"),
    ("Dp_k", "1.0/sqrt(Dp_PX*Dp_PX*1e-6 + Dp_PZ*Dp_PZ*1e-6)"),
    ("KS_theta_x", "atan(KS_PX/KS_PZ)"),
    ("KS_theta_y", "atan(KS_PY/KS_PZ)"),
    ("KS_k", "1.0/sqrt(KS_PX*KS_PX*1e-6 + KS_PZ*KS_PZ*1e-6)"),
    (
        "KS_FD",
        "sqrt((KS_END_VX-KS_OWNPVX)*(KS_END_VX-KS_OWNPVX)"
        " + (KS_END_VY-KS_OWNPVY)*(KS_END_VY-KS_OWNPVY)"
        " + (KS_END_VZ-KS_OWNPVZ)*(KS_END_VZ-KS_OWNPVZ))",
    ),
    ("Dp_charge", "Pip_PARTICLE_ID > 0 ? 1.0 : -1.0"),
    ("KS_LT", "KS_OWNPVLTIME / 0.08954"),
]

TREE_NAMES = {"ll": "D2KSpi_LL/DecayTree", "dd": "D2KSpi_DD/DecayTree"}


# ----------------------------------------------------------------------
# HLT1 trigger cut definitions (was hardcoded inside Analysis.cuts)
# ----------------------------------------------------------------------
DP_TIS_LINES_DD: List[str] = [
    "Dp_Hlt1LowPtMuonDecision_TIS",
    "Dp_Hlt1TrackMVADecision_TIS",
    "Dp_Hlt1TrackMuonMVADecision_TIS",
    "Dp_Hlt1DiPhotonHighMassDecision_TIS",
    "Dp_Hlt1D2KshhDecision_TIS",
    "Dp_Hlt1DiElectronDisplacedDecision_TIS",
    "Dp_Hlt1TwoTrackKsDecision_TIS",
    "Dp_Hlt1KsLLDetachedTrackDecision_TIS",
    "Dp_Hlt1OneMuonTrackLineDecision_TIS",
    "Dp_Hlt1TwoTrackMVADecision_TIS",
    "Dp_Hlt1DiMuonHighMassDecision_TIS",
    "Dp_Hlt1TrackElectronMVADecision_TIS",
    "Dp_Hlt1DiMuonDisplacedDecision_TIS",
    "Pip_Hlt1TrackMVADecision_TOS",
]

DP_TIS_LINES_LL: List[str] = [
    "Dp_Hlt1D2KshhDecision_TIS",
    "Dp_Hlt1DiElectronDisplacedDecision_TIS",
    "Dp_Hlt1DiMuonDisplacedDecision_TIS",
    "Dp_Hlt1DiMuonHighMassDecision_TIS",
    "Dp_Hlt1DiPhotonHighMassDecision_TIS",
    "Dp_Hlt1KsLLDetachedTrackDecision_TIS",
    "Dp_Hlt1LowPtMuonDecision_TIS",
    "Dp_Hlt1OneMuonTrackLineDecision_TIS",
    "Dp_Hlt1TrackElectronMVADecision_TIS",
    "Dp_Hlt1TrackMVADecision_TIS",
    "Dp_Hlt1TrackMuonMVADecision_TIS",
    "Dp_Hlt1TwoTrackKsDecision_TIS",
    "Dp_Hlt1TwoTrackMVADecision_TIS",
]


def dp_tis_expression(track: str) -> str:
    lines = DP_TIS_LINES_LL if track == "ll" else DP_TIS_LINES_DD
    return " || ".join(lines)


def hlt1_filters(name: str, track: str) -> List[tuple]:
    """Return a list of (expression, label) filters for an HLT1 selection name.

    ``name`` of '' or 'none' means no HLT1 requirement at all.  Add new
    trigger configurations here and they become available to every stage
    without touching any script.
    """
    if name in ("", None, "none"):
        return []
    tis = dp_tis_expression(track)
    table = {
        "KS_Hlt1TwoTrackKsDecision_TOS": [
            ("KS_Hlt1TwoTrackKsDecision_TOS", "HLT1: KS TwoTrackKsDecision_TOS")
        ],
        "Pip_Hlt1TrackMVADecision_TOS": [
            ("Pip_Hlt1TrackMVADecision_TOS", "HLT1: Pip_Hlt1TrackMVADecision_TOS")
        ],
        "DpTIS": [(tis, "HLT1: Dp TIS selection")],
        "DpTIS_PipMVATOS": [
            (tis, "HLT1: Dp TIS selection"),
            ("Pip_Hlt1TrackMVADecision_TOS", "HLT1: Pip TrackMVA TOS"),
        ],
        "DpTIS_KSTwoTracks": [
            (tis, "HLT1: Dp TIS selection"),
            ("KS_Hlt1TwoTrackKsDecision_TOS", "HLT1: KS TwoTrackKs TOS"),
        ],
    }
    if name not in table:
        raise ValueError(f"Unknown HLT1 selection {name!r}. Known: {sorted(table)}")
    return table[name]


HLT1_LABELS = {
    "KS_Hlt1TwoTrackKsDecision_TOS": "KS_TwoTracks",
    "Pip_Hlt1TrackMVADecision_TOS": "Pip_TrackMVA_TOS",
    "DpTIS": "Dp*TIS",
    "DpTIS_PipMVATOS": "Dp*TIS & Pip TOS",
    "DpTIS_KSTwoTracks": "Dp*TIS & KS TOS",
    "": "no HLT1 cut",
}


# ----------------------------------------------------------------------
# Offline selection (was the 'kin' and 'probnn' blocks of Analysis.cuts)
# ----------------------------------------------------------------------
def kinematic_filters(cfg: TrackConfig) -> List[tuple]:
    if cfg.track == "ll":
        track_specific = [
            ("Dp_OWNPVIP < 1", "Dp IP"),
            ("KS_FD > 20", "KS flight distance"),
            ("abs(KS_M - 497.611) < 10", "KS mass consistency"),
        ]
    else:
        track_specific = [
            ("Dp_OWNPVIP < 3.5", "Dp IP"),
            ("abs(KS_M - 497.611) < 20", "KS mass consistency"),
        ]

    common = [
        # D+
        ("Dp_HLT2_ETA > 2.2", "Dp eta min"),
        ("Dp_HLT2_ETA < 4.2", "Dp eta max"),
        ("Dp_HLT2_PT > 2800", "Dp PT min"),
        ("Dp_HLT2_PT < 12000", "Dp PT max"),
        ("Dp_k > 0.01", "Dp k min"),
        ("Dp_k < 0.12", "Dp k max"),
        # bachelor pion geometry
        ("Pip_k < (0.3 - abs(Pip_theta_x))", "Pip acceptance"),
        ("pow(Pip_theta_x/0.027,2) + pow(Pip_theta_y/0.017,2) > 1", "Beam ellipse"),
        ("abs(Pip_theta_y) > 0.001", "Pip_theta_y min"),
        (
            "!(abs(Pip_theta_y) < 0.005 && abs(Pip_theta_x) > 0.06 "
            "&& abs(Pip_theta_x) < 0.1)",
            "Dead region",
        ),
        # bachelor pion kinematics
        ("Pip_HLT2_PT > 1500", "Pip PT min"),
        ("Pip_HLT2_PT < 6000", "Pip PT max"),
        ("Pip_HLT2_ETA > 2.2", "Pip eta min"),
        ("Pip_HLT2_ETA < 4.2", "Pip eta max"),
        ("Pip_k > 0.005", "Pip k min"),
        ("Pip_k < 0.06", "Pip k max"),
        # KS
        ("KS_HLT2_ETA > 2.2", "KS eta min"),
        ("KS_HLT2_ETA < 4.2", "KS eta max"),
        ("KS_FD > 20", "KS flight distance"),
        ("abs(KS_M - 497.611) < 10", "KS mass consistency"),
    ]
    return track_specific + common


PROBNN_FILTERS: List[tuple] = [
    ("KSpip_PROBNN_PI > 0.5", "KSpip probnn > 0.5"),
    ("KSpim_PROBNN_PI > 0.5", "KSpim probnn > 0.5"),
    ("Pip_PROBNN_PI > 0.5", "Pip probnn > 0.5"),
    ("KSpip_PROBNN_GHOST < 0.5", "KSpip_PROBNN_GHOST < 0.5"),
    ("KSpim_PROBNN_GHOST < 0.5", "KSpim_PROBNN_GHOST < 0.5"),
    ("Pip_PROBNN_GHOST < 0.5", "Pip_PROBNN_GHOST < 0.5"),
]


def mass_lt_filters(cfg: TrackConfig) -> List[tuple]:
    return [
        (f"Dp_M > {cfg.min_m}", f"Dp_M > {cfg.min_m}"),
        (f"Dp_M < {cfg.max_m}", f"Dp_M < {cfg.max_m}"),
        (f"KS_LT > {cfg.ltmin}", "KS lifetime min"),
        (f"KS_LT < {cfg.ltmax}", "KS lifetime max"),
    ]


# ----------------------------------------------------------------------
# Histogram binning for every variable (was Analysis.init_hist_params)
# ----------------------------------------------------------------------
def build_hist_params(cfg: TrackConfig) -> Dict[str, List]:
    n = cfg.kinvar_bins
    return {
        "Dp_M": [cfg.nbins_m, cfg.min_m, cfg.max_m],
        "KS_LT": [cfg.kslt_plotbins, cfg.ltmin, cfg.ltmax],
        "Pip_k": [n, 0.01, 0.06],
        "Pip_theta_x": [n, -0.2, 0.2],
        "Pip_theta_y": [n, -0.2, 0.2],
        "Pip_HLT2_PHI": [n, -3.2, 3.2],
        "Pip_HLT2_ETA": [n, 2.2, 4.2],
        "Pip_HLT2_PT": [n, 1500, 6000],
        "Dp_HLT2_ETA": [n, 2.2, 4.2],
        "Dp_HLT2_PT": [n, 2800, 12000],
        "Dp_theta_x": [n, -0.2, 0.2],
        "Dp_theta_y": [n, -0.2, 0.2],
        "Dp_k": [n, 0.005, 0.04],
        "KS_k": [n, 0.0, 0.14],
        "KS_theta_x": [n, -0.15, 0.15],
        "KS_theta_y": [n, -0.15, 0.15],
        "KS_HLT2_PHI": [n, -3.2, 3.2],
        "KS_HLT2_ETA": [n, 2.2, 4.2],
        "KS_HLT2_PT": [n, 0, 6000],
    }