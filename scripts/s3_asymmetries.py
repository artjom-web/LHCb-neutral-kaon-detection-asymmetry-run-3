#!/usr/bin/env python
"""FILE 3 — mass fits in lifetime bins; asymmetries to CSV.

    python scripts/s3_asymmetries.py --track dd --hlt1 DpTIS --tag v1
    python scripts/s3_asymmetries.py --track ll --tag v1 --models johnson_exp1

Input : stage-2 reweighted dataframe (+ sidecar)
Output: <out-root>/stage3_massfits/<tag>/<hlt1>/<key>/
          asymmetries.csv    one row per (model, procedure, lifetime bin) with
                             A, A_err, the bin mean and edges, and the fit
                             quality flags
          asymmetries.json   run metadata + a convergence summary
          massfits/<weight>/<model>/bin<i>/  model.root, fit report, figures
          failed_fits/       figures for every fit that did not converge

The CSV holds *true* fitted asymmetries. The blinding offset is added at draw
time only, so the file itself is safe to keep and to diff between runs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from kspi_analysis import cli
from kspi_analysis.config.massmodels import (
    DEFAULT_MODELS,
    PROCEDURE_LABELS,
    PROCEDURE_WEIGHTS,
    MassModelConfig,
)
from kspi_analysis.config.selection import get_track_config
from kspi_analysis.core import metadata, rootio
from kspi_analysis.core.runid import RunID
from kspi_analysis.core.util import blinding_offset
from kspi_analysis.physics import asymmetry
from kspi_analysis.physics.binning import weighted_bin_means
from kspi_analysis.physics.massfit import MassFitter
from kspi_analysis.plotting.asymmetry import AsymmetryPlotter


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_common_args(p, require_tag=True)
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS),
                   help="Mass models to fit (the first is used for the nominal plots)")
    p.add_argument("--standard-model", default=None,
                   help="Nominal model; defaults to the first of --models")
    p.add_argument("--procedures", nargs="*", default=list(PROCEDURE_WEIGHTS),
                   help="Which weightings to fit: before, after")
    p.add_argument("--param-seed-dir", default=None,
                   help="Directory of <model>_seed.txt warm-start snapshots")
    p.add_argument("--no-plots", action="store_true",
                   help="Write the CSV only; skip the summary figures")
    p.add_argument("--blind-seed", type=int, default=0,
                   help="Seed for the blinding offset used in the figures")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    layout = cli.resolve_layout(args)
    rootio.enable_mt(not args.no_mt)

    procedures = {p: PROCEDURE_WEIGHTS[p] for p in args.procedures}
    standard_model = args.standard_model or args.models[0]
    bias = blinding_offset(args.blind_seed)

    runids = cli.selected_runids(args, from_inventory=False)

    def work(runid: RunID) -> None:
        in_file = layout.reweight_file(runid)
        in_meta = layout.reweight_meta(runid)
        if not in_file.exists():
            print(f"No stage-2 output at {in_file}; skipping")
            return
        meta_in = metadata.read(in_meta, expect_stage="reweight")

        csv_path = layout.asymmetry_csv(runid, create=True)
        if csv_path.exists() and not args.overwrite:
            print(f"{csv_path} exists; skipping (use --overwrite to remake)")
            return

        cfg = get_track_config(
            runid.track, lt_bin_edges=list(meta_in["lt_bin_edges"])
        )
        model_cfg = MassModelConfig(
            models=list(args.models),
            standard_model=standard_model,
            param_seed_dir=args.param_seed_dir,
        )

        rdf = rootio.rdf_from_files([in_file])
        fitter = MassFitter(model_cfg)

        # Bin means come from stage 2 when available; recomputing them here
        # would give the same numbers but cost an extra pass over the data.
        bin_means = meta_in.get("bin_means") or {}
        if not bin_means:
            print("Sidecar has no bin means; recomputing from the dataframe")
            bin_means = weighted_bin_means(
                rdf, "KS_LT", cfg.lt_bin_edges, list(dict.fromkeys(procedures.values()))
            )

        df = asymmetry.fit_lifetime_bins(
            rdf, cfg, fitter, layout, runid,
            models=args.models, procedures=procedures, bin_means=bin_means,
        )
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}  ({len(df)} fits)")

        # --- collect the fits that did not converge --------------------
        bad = asymmetry.failed_fits(df)
        if not bad.empty:
            fail_dir = layout.failed_fit_dir(runid, create=True)
            for _, row in bad.iterrows():
                src = Path(row["fit_dir"]) / "figures"
                if not src.is_dir():
                    continue
                dest = fail_dir / f"{row['weight']}_{row['model']}_bin{row['bin']}"
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(src, dest)
            (fail_dir / "failed_fits.csv").write_text(bad.to_csv(index=False))
            print(f"{len(bad)} fit(s) did not converge; figures in {fail_dir}")

        # --- summary figures for this sample ---------------------------
        if not args.no_plots:
            A, A_err, x, xerr = asymmetry.pivot_for_plotting(
                df, args.models, list(procedures)
            )
            available = [
                m for m in args.models
                if any(np.isfinite(A[m][p]).any() for p in procedures)
            ]
            if available:
                AsymmetryPlotter(
                    models=args.models,
                    procedures=list(procedures),
                    xlabel=r"KS lifetime $(t/\tau)$",
                    procedure_labels=PROCEDURE_LABELS,
                    A_bias=bias,
                ).plot_standard_set(
                    A, A_err, x, xerr, str(layout.massfit_dir(runid)),
                    available_models=available,
                    standard_model=standard_model,
                    after_procedure="after",
                )
            else:
                print("WARNING: no model produced a usable fit; skipping figures")

        metadata.write(
            layout.asymmetry_meta(runid),
            stage="asymmetries",
            runid=runid,
            inputs=[in_file],
            config={
                "track_config": cfg.to_dict(),
                "mass_model": model_cfg.to_dict(),
                "procedures": procedures,
                "blind_seed": args.blind_seed,
            },
            n_fits=int(len(df)),
            n_converged=int(df["converged"].sum()),
            n_failed=int(len(bad)),
            lt_bin_edges=list(cfg.lt_bin_edges),
            n_iterations=meta_in.get("n_iterations"),
            output=str(csv_path),
        )

    return cli.run_over_samples(args, runids, work)


if __name__ == "__main__":
    raise SystemExit(main())