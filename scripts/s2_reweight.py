#!/usr/bin/env python
"""FILE 2 — load the skim, run the reweighting, write weights per iteration.

    python scripts/s2_reweight.py --track dd --hlt1 DpTIS --tag v1
    python scripts/s2_reweight.py --track ll --cycles short --threshold 2 --tag test

Input : stage-1 skim (+ its sidecar)
Output: <out-root>/stage2_reweighted/<tag>/<hlt1>/<key>/reweighted.root
          — every input column plus weight_i, w2_i, weight_eff_i,
            weight_acc_i for every iteration i, and final_weight
        reweighted.json
          — cycles run, lifetime binning, weighted bin means, sideband ratio,
            per-iteration accepted fraction, and the full configuration
        initial_massfit/ — model.root, fit report and figures

Snapshotting *every* iteration's weights (not just the final one) is what lets
stage 4 study the statistical cost without re-running the reweighting, and lets
stage 3 and stage 5 run in any order or in parallel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kspi_analysis import cli
from kspi_analysis.config.massmodels import MassModelConfig
from kspi_analysis.config.selection import build_hist_params, get_track_config
from kspi_analysis.config.weighting import cycle_variables, get_cycles
from kspi_analysis.core import metadata, rootio
from kspi_analysis.core.runid import RunID
from kspi_analysis.core.util import weight_key
from kspi_analysis.physics.binning import weighted_bin_means
from kspi_analysis.physics.massfit import MassFitter
from kspi_analysis.physics.reweighting import Reweighter


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_common_args(p)
    p.add_argument("--cycles", default="nominal",
                   help="Named reweighting-cycle scheme (see config/weighting.py)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Minimum unweighted entries per 4D cell (default: config)")
    p.add_argument("--kinvar-bins", type=int, default=None,
                   help="Bins per kinematic variable in the reweighting templates")
    p.add_argument("--lt-bin-edges", nargs="*", type=float, default=None,
                   help="Override the lifetime bin edges")
    p.add_argument("--standard-model", default=None,
                   help="Mass model used for the initial sideband fit")
    p.add_argument("--param-seed-dir", default=None,
                   help="Directory of <model>_seed.txt warm-start snapshots")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    layout = cli.resolve_layout(args)
    rootio.enable_mt(not args.no_mt)

    cycles = get_cycles(args.cycles)
    runids = cli.selected_runids(args, from_inventory=False)

    def work(runid: RunID) -> None:
        skim = layout.skim_file(runid)
        skim_meta = layout.skim_meta(runid)
        if not skim.exists():
            print(f"No stage-1 skim at {skim}; skipping")
            return
        meta_in = metadata.read(skim_meta, expect_stage="skim")

        out_file = layout.reweight_file(runid, create=True)
        json_outfile = layout.reweight_meta(runid, create=False)
        if json_outfile.exists() and not args.overwrite:
            print(f"{json_outfile} exists; skipping (use --overwrite to remake)")
            return

        # --- configuration -------------------------------------------------
        overrides = {}
        if args.threshold is not None:
            overrides["threshold"] = args.threshold
        if args.kinvar_bins is not None:
            overrides["kinvar_bins"] = args.kinvar_bins
        if args.lt_bin_edges:
            overrides["lt_bin_edges"] = list(args.lt_bin_edges)
        cfg = get_track_config(runid.track, **overrides)

        model_cfg = MassModelConfig(param_seed_dir=args.param_seed_dir)
        if args.standard_model:
            model_cfg = MassModelConfig(
                models=[args.standard_model] if args.standard_model
                not in model_cfg.models else model_cfg.models,
                standard_model=args.standard_model,
                param_seed_dir=args.param_seed_dir,
            )

        missing = [v for v in cycle_variables(cycles)
                   if v not in meta_in.get("columns", [])]
        if missing:
            raise KeyError(
                f"The skim does not contain {missing}, which cycle scheme "
                f"'{args.cycles}' needs. Re-run stage 1 with --extra-columns."
            )

        # --- run -------------------------------------------------------------
        rdf = rootio.rdf_from_files([skim])
        fitter = MassFitter(model_cfg)
        rw = Reweighter(rdf, cfg, fitter, hist_params=build_hist_params(cfg))

        print(f"{rw.N_total_events} events in, running {len(cycles)} cycle(s)")
        rw.run(cycles, layout.initial_massfit_dir(runid, create=True))

        # Weighted lifetime bin means: stage 3 and 4 both need them, and
        # computing them here means they come from the same event loop as
        # the snapshot rather than being recomputed twice downstream.
        means = weighted_bin_means(
            rw.rdf, "KS_LT", cfg.lt_bin_edges, ["", "weight_0", "final_weight"]
        )

        columns = sorted(set(meta_in["columns"]) | set(rw.weight_columns()))
        rootio.snapshot(rw.rdf, out_file, columns)

        metadata.write(
            layout.reweight_meta(runid),
            stage="reweight",
            runid=runid,
            inputs=[skim],
            config={
                "track_config": cfg.to_dict(),
                "cycles_scheme": args.cycles,
                "cycles": [c.to_dict() for c in cycles],
                "mass_model": model_cfg.to_dict(),
                "hist_params": rw.hist_params,
            },
            n_iterations=rw.iteration,
            n_total_events=rw.N_total_events,
            lt_bin_edges=list(cfg.lt_bin_edges),
            weight_columns=rw.weight_columns(),
            final_weight="final_weight",
            sideband_ratio=rw.sideband_ratio,
            sideband_ratio_flat=rw.sideband_ratio_flat,
            accepted_fraction=rw.accepted_fraction,
            bin_means={k: {"x": v["x"], "xerr": v["xerr"]} for k, v in means.items()},
            columns=columns,
            output=str(out_file),
        )
        print(f"Wrote {out_file}  ({rw.iteration} iterations)")

    return cli.run_over_samples(args, runids, work)


if __name__ == "__main__":
    raise SystemExit(main())