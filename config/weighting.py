"""Reweighting configuration: which variable triplets are used, in what order.

The cycle list is the single most important knob in the analysis, so it lives
in its own module.  Named schemes let you run an alternative sequence without
editing a script: ``--cycles nominal`` vs ``--cycles ks_only``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Cycle:
    """One reweighting iteration: a 3D kinematic template in lifetime bins."""

    name: str
    param1: str
    param2: str
    param3: str

    @property
    def params(self) -> tuple:
        return (self.param1, self.param2, self.param3)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "param1": self.param1,
            "param2": self.param2,
            "param3": self.param3,
        }


NOMINAL_CYCLES: List[Cycle] = [
    Cycle("cycle1", "Pip_k", "Pip_theta_x", "Pip_theta_y"),
    Cycle("cycle2", "Pip_k", "Dp_HLT2_PT", "Dp_HLT2_ETA"),
    Cycle("cycle3", "Pip_HLT2_PHI", "Pip_HLT2_ETA", "Dp_HLT2_PT"),
    Cycle("cycle4", "Pip_HLT2_PT", "Pip_HLT2_ETA", "Dp_HLT2_PT"),
    Cycle("cycle5", "Pip_HLT2_PT", "Dp_theta_x", "Dp_HLT2_PT"),
    Cycle("cycle6", "Pip_theta_x", "Pip_theta_y", "Dp_HLT2_PT"),
    Cycle("cycle7", "Pip_k", "Dp_HLT2_ETA", "Dp_HLT2_PT"),
]

CYCLE_SCHEMES: Dict[str, List[Cycle]] = {
    "nominal": NOMINAL_CYCLES,
    "short": NOMINAL_CYCLES[:3],
    "none": [],
}


def get_cycles(scheme: str = "nominal") -> List[Cycle]:
    if scheme not in CYCLE_SCHEMES:
        raise ValueError(
            f"Unknown cycle scheme {scheme!r}. Known: {sorted(CYCLE_SCHEMES)}"
        )
    return CYCLE_SCHEMES[scheme]


def cycle_variables(cycles: Sequence[Cycle]) -> List[str]:
    """Every kinematic column the given cycles touch, sorted and de-duplicated.
    Stage 1 uses this to decide which branches must survive into the skim."""
    seen: List[str] = []
    for c in cycles:
        for p in c.params:
            if p not in seen:
                seen.append(p)
    return sorted(seen)


#: Ratio clamp applied inside the C++ update_weight_3d helper.
WEIGHT_RATIO_MIN = 0.2
WEIGHT_RATIO_MAX = 5.0