#!/usr/bin/env python
"""FILE 1 — load data, apply cuts, write a skimmed ROOT dataframe.

    python scripts/s1_skim.py --track dd --hlt1 DpTIS
    python scripts/s1_skim.py --track ll --ycset 25c4 --polarity magup

Input : the AnalysisProductions samples listed in config/datasets.py
Output: <out-root>/stage1_skim/<hlt1>/<track>_<ycset>_<polarity>.root
        plus a .json sidecar recording the cutflow, the columns kept, the
        number of input files and the exact selection configuration.

The skim is written once and read by every later stage, so it deliberately
keeps every column that has histogram binning configured rather than only the
ones today's reweighting cycles happen to use.  Adding a variable later would
otherwise mean re-running the slowest stage in the chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kspi_analysis import cli
from kspi_analysis.config import datasets
from kspi_analysis.config.selection import build_hist_params, get_track_config
from kspi_analysis.core import metadata, rootio
from kspi_analysis.core.runid import RunID
from kspi_analysis.physics import selection


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_common_args(p)
    p.add_argument("--n-files", type=int, default=None,
                   help="Override the number of input files (default: all available)")
    p.add_argument("--source", choices=["apd", "pfns"], default="apd",
                   help="Where to resolve input file paths from")
    p.add_argument("--pfns-base", default="/eos/user/a/ahulsber/scripts/data/PFNS",
                   help="Base directory for --source pfns")
    p.add_argument("--kinvar-bins", type=int, default=None,
                   help="Override the kinematic-variable bin count")
    p.add_argument("--extra-columns", nargs="*", default=[],
                   help="Additional branches to keep in the skim")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-make skims that already exist")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    layout = cli.resolve_layout(args)
    rootio.enable_mt(not args.no_mt)

    runids = cli.selected_runids(args, from_inventory=True)

    def work(runid: RunID) -> None:
        out_file = layout.skim_file(runid, create=True)
        meta_file = layout.skim_meta(runid)

        if out_file.exists() and not args.overwrite:
            print(f"{out_file} exists; skipping (use --overwrite to remake)")
            return

        overrides = {}
        if args.kinvar_bins is not None:
            overrides["kinvar_bins"] = args.kinvar_bins
        cfg = get_track_config(runid.track, **overrides)

        n_avail = datasets.n_files(runid.track, runid.ycset, runid.polarity)
        n_files = args.n_files if args.n_files is not None else n_avail
        if n_files is None:
            raise ValueError(f"{runid.label()} is not in the dataset inventory")

        if args.source == "apd":
            files = selection.apd_files(runid, n_files)
        else:
            files = selection.pfns_files(runid, n_files, args.pfns_base)
        print(f"Resolved {len(files)} input file(s)")

        rdf = selection.rdf_from_production(files, runid.track)
        rdf = selection.define_columns(rdf)
        rdf = selection.apply_selection(rdf, cfg, hlt1=runid.hlt1)

        columns = selection.skim_columns(cfg, extra=args.extra_columns)

        # Snapshot triggers the event loop; take the cutflow from the same pass.
        report = rdf.Report()
        rootio.snapshot(rdf, out_file, columns)
        report.Print()
        flow = [
            {"name": c.GetName(), "all": float(c.GetAll()),
             "pass": float(c.GetPass()), "eff": float(c.GetEff())}
            for c in report
        ]
        n_selected = int(flow[-1]["pass"]) if flow else None

        metadata.write(
            meta_file,
            stage="skim",
            runid=runid,
            inputs=files,
            config={
                "track_config": cfg.to_dict(),
                "hlt1": runid.hlt1,
                "source": args.source,
                "hist_params": build_hist_params(cfg),
            },
            n_input_files=len(files),
            n_selected=n_selected,
            columns=columns,
            cutflow=flow,
            output=str(out_file),
        )
        print(f"Wrote {out_file}  ({n_selected} events)")

    return cli.run_over_samples(args, runids, work)


if __name__ == "__main__":
    raise SystemExit(main())