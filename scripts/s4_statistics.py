#!/usr/bin/env python
"""FILE 4 — statistical cost of the reweighting.

    python scripts/s4_statistics.py --track dd --hlt1 DpTIS --tag v1

Input : stage-2 reweighted dataframe (+ sidecar)
Output: <out-root>/stage4_statistics/<tag>/<hlt1>/<key>/
          weighting_statistics.csv   S_w, S_w^-1, R_w and N_eff for every
                                     (lifetime bin, iteration)
          Neff_vs_iteration.pdf
          Swinv_vs_iteration.pdf
          Rw_vs_iteration.pdf
          Swinv_vs_lifetime.pdf

This stage reads weights only, never refits, so it is cheap and safe to re-run.
The CSV is the point of it: the old code drew these curves and threw the
numbers away, which made it impossible to compare the statistical cost of two
cycle schemes without eyeballing two PDFs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from kspi_analysis import cli
from kspi_analysis.core import metadata, rootio
from kspi_analysis.core.runid import RunID
from kspi_analysis.physics.stats import final_iteration_by_bin, weighting_statistics
from kspi_analysis.plotting.stats import plot_statistics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_common_args(p, require_tag=True)
    p.add_argument("--no-plots", action="store_true", help="Write the CSV only")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    layout = cli.resolve_layout(args)
    rootio.enable_mt(not args.no_mt)

    runids = cli.selected_runids(args, from_inventory=False)

    def work(runid: RunID) -> None:
        in_file = layout.reweight_file(runid)
        if not in_file.exists():
            print(f"No stage-2 output at {in_file}; skipping")
            return
        meta_in = metadata.read(layout.reweight_meta(runid), expect_stage="reweight")

        outdir = layout.statistics_dir(runid, create=True)
        csv_path = outdir / "weighting_statistics.csv"
        if csv_path.exists() and not args.overwrite:
            print(f"{csv_path} exists; skipping (use --overwrite to remake)")
            return

        n_iterations = int(meta_in["n_iterations"])
        lt_bin_edges = list(meta_in["lt_bin_edges"])

        rdf = rootio.rdf_from_files([in_file])
        df = weighting_statistics(rdf, lt_bin_edges, n_iterations)
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

        # A one-line readout that is usually the thing you actually want:
        # how much effective statistics survived, per lifetime bin.
        last = final_iteration_by_bin(df)
        for _, row in last.iterrows():
            print(
                f"  ltbin {int(row['ltbin'])}: "
                f"S_w^-1 = {row['S_w_inv']:.4g}, "
                f"R_w = {row['R_w']:.4g}, "
                f"N_eff = {row['N_eff']:.4g}"
            )

        if not args.no_plots:
            means = (meta_in.get("bin_means") or {}).get("final_weight", {})
            paths = plot_statistics(
                df,
                outdir,
                bin_x=means.get("x"),
                bin_xerr=np.asarray(means["xerr"]) if means.get("xerr") else None,
                title=runid.label(),
            )
            for name, path in paths.items():
                print(f"  {name}: {path}")

        metadata.write(
            outdir / "statistics.json",
            stage="statistics",
            runid=runid,
            inputs=[in_file],
            config={
                "n_iterations": n_iterations,
                "lt_bin_edges": lt_bin_edges,
                "cycles": meta_in.get("config", {}).get("cycles"),
            },
            final_S_w_inv=last["S_w_inv"].tolist(),
            final_R_w=last["R_w"].tolist(),
            final_N_eff=last["N_eff"].tolist(),
            output=str(csv_path),
        )

    return cli.run_over_samples(args, runids, work)


if __name__ == "__main__":
    raise SystemExit(main())