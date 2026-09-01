"""Filesystem layout.

One class owns every path the analysis writes.  If you want to move the output
tree, change ``Layout.root`` and nothing else.  Stages never build paths by
string concatenation, which is what made the old ``base_folder + 'result/'``
pattern impossible to reorganise.

Tree
----
::

    <root>/
      stage1_skim/<hlt1>/<key>.root                  + .json
      stage2_reweighted/<tag>/<hlt1>/<key>/
            reweighted.root                          + reweighted.json
            initial_massfit/...
      stage3_massfits/<tag>/<hlt1>/<key>/
            asymmetries.csv                          + asymmetries.json
            massfits/<procedure>/<model>/bin<i>/...
            failed_fits.txt
      stage4_statistics/<tag>/<hlt1>/<key>/
      stage5_weighting/<tag>/<hlt1>/<key>/
      stage6_combined/<tag>/<name>/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .runid import RunID

DEFAULT_ROOT = "/eos/user/a/ahulsber/scripts/analysis"


def default_tag() -> str:
    """Timestamped tag, used when the user does not supply ``--tag``."""
    now = datetime.now()
    return f"{now:%m-%d_%Hh%M}"


@dataclass(frozen=True)
class Layout:
    root: str = DEFAULT_ROOT
    tag: str = "latest"

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _mk(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    # ---- stage 1: skims ----------------------------------------------
    def skim_dir(self, runid: RunID, create: bool = False) -> Path:
        p = self.root_path / "stage1_skim" / runid.hlt1_dir
        return self._mk(p) if create else p

    def skim_file(self, runid: RunID, create: bool = False) -> Path:
        return self.skim_dir(runid, create) / f"{runid.key}.root"

    def skim_meta(self, runid: RunID, create: bool = False) -> Path:
        return self.skim_dir(runid, create) / f"{runid.key}.json"

    # ---- generic per-run stage directory ------------------------------
    def _stage_dir(self, stage: str, runid: RunID, create: bool = False) -> Path:
        p = self.root_path / stage / self.tag / runid.hlt1_dir / runid.key
        return self._mk(p) if create else p

    # ---- stage 2: reweighting ------------------------------------------
    def reweight_dir(self, runid: RunID, create: bool = False) -> Path:
        return self._stage_dir("stage2_reweighted", runid, create)

    def reweight_file(self, runid: RunID, create: bool = False) -> Path:
        return self.reweight_dir(runid, create) / "reweighted.root"

    def reweight_meta(self, runid: RunID, create: bool = False) -> Path:
        return self.reweight_dir(runid, create) / "reweighted.json"

    def initial_massfit_dir(self, runid: RunID, create: bool = False) -> Path:
        p = self.reweight_dir(runid, create) / "initial_massfit"
        return self._mk(p) if create else p

    # ---- stage 3: mass fits / asymmetries ------------------------------
    def massfit_dir(self, runid: RunID, create: bool = False) -> Path:
        return self._stage_dir("stage3_massfits", runid, create)

    def asymmetry_csv(self, runid: RunID, create: bool = False) -> Path:
        return self.massfit_dir(runid, create) / "asymmetries.csv"

    def asymmetry_meta(self, runid: RunID, create: bool = False) -> Path:
        return self.massfit_dir(runid, create) / "asymmetries.json"

    def massfit_bin_dir(
        self,
        runid: RunID,
        procedure_key: str,
        model: str,
        ibin: int,
        create: bool = False,
    ) -> Path:
        p = (
            self.massfit_dir(runid, create)
            / "massfits"
            / procedure_key
            / model
            / f"bin{ibin}"
        )
        return self._mk(p) if create else p

    def failed_fit_dir(self, runid: RunID, create: bool = False) -> Path:
        p = self.massfit_dir(runid, create) / "failed_fits"
        return self._mk(p) if create else p

    # ---- stage 4: statistical losses -----------------------------------
    def statistics_dir(self, runid: RunID, create: bool = False) -> Path:
        return self._stage_dir("stage4_statistics", runid, create)

    # ---- stage 5: weighting performance --------------------------------
    def weighting_dir(self, runid: RunID, create: bool = False) -> Path:
        return self._stage_dir("stage5_weighting", runid, create)

    # ---- stage 6: combination -------------------------------------------
    def combined_dir(self, name: str, create: bool = False) -> Path:
        p = self.root_path / "stage6_combined" / self.tag / name
        return self._mk(p) if create else p


def add_layout_args(parser) -> None:
    """Attach ``--out-root`` / ``--tag`` to an argparse parser."""
    parser.add_argument(
        "--out-root",
        default=os.environ.get("KSPI_OUT_ROOT", DEFAULT_ROOT),
        help="Base directory for all analysis output (env: KSPI_OUT_ROOT)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Run tag grouping stages 2-6; defaults to a timestamp",
    )


def layout_from_args(args, *, require_tag: bool = False) -> Layout:
    tag = args.tag
    if tag is None:
        if require_tag:
            raise SystemExit(
                "--tag is required here: it must match the tag used by the "
                "stage that produced this input."
            )
        tag = default_tag()
    return Layout(root=args.out_root, tag=tag)