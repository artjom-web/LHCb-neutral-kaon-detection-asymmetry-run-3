#!/usr/bin/env python
"""FILE 5 — weighting-performance diagnostics.

    python scripts/s5_weighting_performance.py --track dd --hlt1 DpTIS --tag v1
    python scripts/s5_weighting_performance.py --tag v1 --only w-kin

Input : stage-2 reweighted dataframe (+ sidecar)
Output: <out-root>/stage5_weighting/<tag>/<hlt1>/<key>/
          w_vs_kin/<particle>_<weight>.pdf   kinematic distributions per
                                             lifetime bin, before and after
          A_vs_kin/<particle>.pdf            asymmetry vs each kinematic var
          A_vs_LT/<particle>.pdf             those asymmetries integrated over
                                             each lifetime bin
          massfits/...                       every fit behind A_vs_kin

Note on cost: ``A_vs_kin`` runs one mass fit per bin of every kinematic
variable — with 17 variables at 25 bins that is well over 400 fits, far more
than stage 3 does. It is off by default for that reason; ask for it explicitly
with --only or --all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from kspi_analysis import cli
from kspi_analysis.config.massmodels import MassModelConfig
from kspi_analysis.config.selection import get_track_config
from kspi_analysis.core import metadata, rootio
from kspi_analysis.core.runid import RunID
from kspi_analysis.physics.massfit import MassFitter
from kspi_analysis.physics.performance import WeightingPerformance

CHOICES = ["w-kin", "A-kin", "A-LT"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_common_args(p, require_tag=True)
    p.add_argument("--only", nargs="*", choices=CHOICES, default=["w-kin"],
                   help="Which diagnostics to produce (default: w-kin)")
    p.add_argument("--all", action="store_true",
                   help="Produce all three diagnostics (slow: many mass fits)")
    p.add_argument("--standard-model", default=None,
                   help="Mass model used for the A-vs-kinematics fits")
    p.add_argument("--param-seed-dir", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    layout = cli.resolve_layout(args)
    rootio.enable_mt(not args.no_mt)

    wanted = set(CHOICES) if args.all else set(args.only)
    # A_LT is built from both the per-lifetime-bin distributions and the
    # per-variable asymmetries, so asking for it implies the other two.
    if "A-LT" in wanted:
        wanted |= {"w-kin", "A-kin"}

    runids = cli.selected_runids(args, from_inventory=False)

    def work(runid: RunID) -> None:
        in_file = layout.reweight_file(runid)
        if not in_file.exists():
            print(f"No stage-2 output at {in_file}; skipping")
            return
        meta_in = metadata.read(layout.reweight_meta(runid), expect_stage="reweight")

        cfg = get_track_config(runid.track, lt_bin_edges=list(meta_in["lt_bin_edges"]))
        model_cfg = MassModelConfig(param_seed_dir=args.param_seed_dir)
        if args.standard_model:
            model_cfg = MassModelConfig(
                standard_model=args.standard_model,
                param_seed_dir=args.param_seed_dir,
            )

        # Stage 2 already measured the weighted lifetime bin means; reuse them
        # so the x positions here line up exactly with the stage-3 plots.
        raw_means = meta_in.get("bin_means") or {}
        bin_x, bin_xerr = {}, {}
        for wk, d in raw_means.items():
            bin_x[f"KS_LT_{wk}"] = d.get("x")
            if d.get("xerr") is not None:
                bin_xerr[f"KS_LT_{wk}"] = np.asarray(d["xerr"])

        rdf = rootio.rdf_from_files([in_file])
        perf = WeightingPerformance(
            rdf,
            cfg,
            MassFitter(model_cfg),
            iteration=int(meta_in["n_iterations"]),
            hist_params=meta_in.get("config", {}).get("hist_params"),
            bin_x=bin_x,
            bin_xerr=bin_xerr,
        )

        outdir = layout.weighting_dir(runid, create=True)
        print(f"Producing: {sorted(wanted)}")
        perf.run(
            outdir,
            plot_w_kin="w-kin" in wanted,
            plot_A_kin="A-kin" in wanted,
            plot_A_LT="A-LT" in wanted,
        )

        metadata.write(
            outdir / "weighting_performance.json",
            stage="weighting_performance",
            runid=runid,
            inputs=[in_file],
            config={
                "track_config": cfg.to_dict(),
                "mass_model": model_cfg.to_dict(),
                "diagnostics": sorted(wanted),
            },
            n_iterations=int(meta_in["n_iterations"]),
            n_failed_fits=len(perf.fitter.failed),
            failed_fits=perf.fitter.failed,
            output=str(outdir),
        )
        if perf.fitter.failed:
            print(f"WARNING: {len(perf.fitter.failed)} fit(s) did not converge")
        print(f"Wrote figures to {outdir}")

    return cli.run_over_samples(args, runids, work)


if __name__ == "__main__":
    raise SystemExit(main())