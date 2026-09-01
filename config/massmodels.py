"""Mass-model configuration.

The model *shapes* stay where they already are, in your ``mass_models.py``
module: this file only says which of them to use, which one is the nominal
("standard") model, and where the warm-start parameter snapshots live.

Warm starting
-------------
``MassFitter`` writes a RooFit parameter snapshot to ``<folder>_out.txt`` after
every fully-converged fit and reads it back on the next run.  Setting
``param_seed_dir`` lets you point a fresh output directory at the snapshots
from a previous run, so a re-run starts from converged values instead of the
model defaults.  That is the "mass model parameters specified in another file"
hook: drop ``<model>_seed.txt`` files in that directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ALL_MODELS: List[str] = [
    "johnson_exp1",
    "johnson_exp",
    "johnson_tail_expo",
    "johnson_gauss_exp",
    "johnson_free_gauss_exp",
]

#: Models used for the systematic model-variation study in stage 3.
DEFAULT_MODELS: List[str] = [
    "johnson_exp1",
    "johnson_exp",
    "johnson_tail_expo",
    "johnson_gauss_exp",
]

DEFAULT_STANDARD_MODEL = "johnson_exp1"


@dataclass
class MassModelConfig:
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    standard_model: str = DEFAULT_STANDARD_MODEL
    max_attempts: int = 8
    seed: int = 1
    #: Directory holding ``<model>_seed.txt`` warm-start snapshots, or None.
    param_seed_dir: Optional[str] = None
    #: Draw a mass-fit figure whenever a fit fails to reach covQual == 3.
    plot_on_failure: bool = True

    def __post_init__(self) -> None:
        unknown = [m for m in self.models if m not in ALL_MODELS]
        if unknown:
            raise ValueError(
                f"Unknown mass model(s) {unknown}. Known models: {ALL_MODELS}"
            )
        if self.standard_model not in self.models:
            raise ValueError(
                f"standard_model {self.standard_model!r} is not in models {self.models}"
            )

    def seed_file(self, model: str) -> Optional[str]:
        """Path to the warm-start snapshot for ``model``, if one exists."""
        if not self.param_seed_dir:
            return None
        p = Path(self.param_seed_dir) / f"{model}_seed.txt"
        return str(p) if p.exists() else None

    def to_dict(self) -> dict:
        return {
            "models": list(self.models),
            "standard_model": self.standard_model,
            "max_attempts": self.max_attempts,
            "seed": self.seed,
            "param_seed_dir": self.param_seed_dir,
            "plot_on_failure": self.plot_on_failure,
        }


PROCEDURES: List[str] = ["before", "after"]

#: procedure name -> weight column used for it ('' means unweighted)
PROCEDURE_WEIGHTS: Dict[str, str] = {"before": "", "after": "final_weight"}

PROCEDURE_LABELS: Dict[str, str] = {
    "before": "before reweighting",
    "after": "after reweighting",
}