#!/usr/bin/env python
"""FILE 6 — combine stage-3 results across samples.

    # combine polarities and ycsets within each track and HLT1 line
    python scripts/s6_combine.py --tag v1 --name nominal

    # combine everything, then compare HLT1 lines (replaces compare_hlt1cuts.py)
    python scripts/s6_combine.py --tag v1 --name hlt1_study \\
        --hlt1-set KS_Hlt1TwoTrackKsDecision_TOS Pip_Hlt1TrackMVADecision_TOS DpTIS \\
        --keep track hlt1 --compare hlt1 --panel-by track

Input : the stage-3 asymmetries.csv of every selected sample
Output: <out-root>/stage6_combined/<tag>/<name>/
          combined_asymmetry.csv   the inverse-variance combined result
          input_asymmetry.csv      every input row, for traceability
          slope_fits.csv           linear A-vs-lifetime fit per group
          *.pdf                    overlay, panel and slope-summary figures

``--keep`` is the whole interface: it lists the fields that stay *separate*.
Everything else gets combined over. So ``--keep track hlt1`` averages the
polarities and ycsets together but keeps the tracks and trigger lines apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


from kspi_analysis.config import datasets
from kspi_analysis.config.massmodels import (
    DEFAULT_STANDARD_MODEL,
    PROCEDURE_LABELS,
)
from kspi_analysis.core import metadata
from kspi_analysis.core.paths import add_layout_args, layout_from_args
from kspi_analysis.core.runid import RunID, skipped
from kspi_analysis.core.util import blinding_offset
from kspi_analysis.physics import combine as cmb
from kspi_analysis.plotting.asymmetry import AsymmetryPlotter
from kspi_analysis.plotting.comparison import (
    plot_overlay,
    plot_panels,
    plot_slope_summary,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_layout_args(p)
    p.add_argument("--name", required=True,
                   help="Name of this combination, used as the output directory")
    p.add_argument("--track", choices=["ll", "dd"], default=None)
    p.add_argument("--ycset", nargs="*", default=None)
    p.add_argument("--polarity", nargs="*", default=None,
                   choices=["magup", "magdown"])
    p.add_argument("--hlt1-set", nargs="*", default=[""],
                   help="HLT1 selections to include ('' means the no-cut skim)")
    p.add_argument("--keep", nargs="*", default=["track", "hlt1"],
                   help="Sample fields that stay separate; the rest are combined")
    p.add_argument("--models", nargs="*", default=None,
                   help="Restrict to these mass models (default: all present)")
    p.add_argument("--procedures", nargs="*", default=["before", "after"])
    p.add_argument("--compare", default=None,
                   help="Field to overlay in the comparison figures, e.g. hlt1")
    p.add_argument("--panel-by", default=None,
                   help="Field to give one panel each, e.g. track")
    p.add_argument("--standard-model", default=DEFAULT_STANDARD_MODEL)
    p.add_argument("--require-converged", action="store_true",
                   help="Use only fits that reached covQual 3")
    p.add_argument("--blind-seed", type=int, default=0)
    return p.parse_args(argv)


def collect_inputs(args, layout):
    """Every stage-3 CSV matching the selectors, with the RunIDs that made them."""
    paths, runids = [], []
    for hlt1 in args.hlt1_set:
        for track, ycset, polarity, _ in datasets.iter_samples(args.track):
            if args.ycset and ycset not in args.ycset:
                continue
            if args.polarity and polarity not in args.polarity:
                continue
            rid = RunID(track=track, ycset=ycset, polarity=polarity, hlt1=hlt1 or "")
            if skipped(rid):
                continue
            csv = layout.asymmetry_csv(rid)
            if csv.exists():
                paths.append(csv)
                runids.append(rid)
    return paths, runids


def main(argv=None) -> int:
    args = parse_args(argv)
    layout = layout_from_args(args, require_tag=True)
    outdir = layout.combined_dir(args.name, create=True)
    bias = blinding_offset(args.blind_seed)

    paths, runids = collect_inputs(args, layout)
    if not paths:
        print("No stage-3 CSVs matched. Check --tag, --hlt1-set and the selectors.")
        return 1
    print(f"Combining {len(paths)} sample(s):")
    for rid in runids:
        print(f"  - {rid.label()}")

    df = cmb.load_results(paths)
    df["hlt1"] = df["hlt1"].fillna("")

    if args.models:
        df = df[df["model"].isin(args.models)]
    df = df[df["procedure"].isin(args.procedures)]
    if df.empty:
        print("Nothing left after filtering by --models / --procedures.")
        return 1

    df.to_csv(outdir / "input_asymmetry.csv", index=False)

    group_by = list(dict.fromkeys(list(args.keep) + ["model", "procedure", "bin"]))
    combined = cmb.combine(df, group_by=group_by,
                           require_converged=args.require_converged)

    n_bins = int(df["bin"].max()) + 1
    combined = cmb.reindex_to_bins(combined, n_bins, group_by)
    combined_path = outdir / "combined_asymmetry.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nWrote {combined_path}  ({len(combined)} rows)")

    models = sorted(combined["model"].dropna().unique())
    procedures = [p for p in args.procedures if p in set(combined["procedure"])]
    standard = args.standard_model if args.standard_model in models else (
        models[0] if models else None
    )

    # --- the standard four-figure set, one per kept group ---------------
    slice_keys = [k for k in args.keep if k in combined.columns]
    groups = (
        combined.groupby(slice_keys, dropna=False, sort=True)
        if slice_keys
        else [((), combined)]
    )
    for keys, grp in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        label = "_".join(str(k) if k != "" else "nohlt1" for k in keys) or "all"
        sub_dir = outdir / label
        sub_dir.mkdir(parents=True, exist_ok=True)

        A, A_err, x, xerr = {}, {}, {}, {}
        for model in models:
            A[model], A_err[model], x[model], xerr[model] = {}, {}, {}, {}
            for proc in procedures:
                s = grp[(grp["model"] == model) & (grp["procedure"] == proc)]
                s = s.sort_values("bin")
                A[model][proc] = s["A"].to_numpy(dtype=float)
                A_err[model][proc] = s["A_err"].to_numpy(dtype=float)
                x[model][proc] = s["x"].to_numpy(dtype=float)
                xerr[model][proc] = np.vstack(
                    [s["xerr_lo"].to_numpy(dtype=float),
                     s["xerr_hi"].to_numpy(dtype=float)]
                )

        available = [m for m in models
                     if any(np.isfinite(A[m][p]).any() for p in procedures)]
        if not available:
            print(f"WARNING: no usable combined result for {label}")
            continue

        AsymmetryPlotter(
            models=models, procedures=procedures,
            xlabel=r"KS lifetime $(t/\tau)$",
            procedure_labels=PROCEDURE_LABELS,
            A_bias=bias,
        ).plot_standard_set(
            A, A_err, x, xerr, str(sub_dir),
            available_models=available,
            standard_model=standard if standard in available else available[0],
            after_procedure="after" if "after" in procedures else procedures[0],
        )
        print(f"  figures for {label} -> {sub_dir}")

    # --- cross-configuration comparison figures --------------------------
    nominal = combined[
        (combined["model"] == standard)
        & (combined["procedure"] == ("after" if "after" in procedures else procedures[0]))
    ]

    if args.compare and args.compare in nominal.columns:
        plot_overlay(
            nominal, args.compare, outdir / f"compare_{args.compare}.pdf",
            bias=bias,
            title=f"{standard}, after reweighting",
            fit_trend=True,
        )
        if args.panel_by and args.panel_by in nominal.columns:
            plot_panels(
                nominal, args.panel_by, args.compare,
                outdir / f"compare_{args.compare}_by_{args.panel_by}.pdf",
                bias=bias,
                suptitle=f"{standard}, after reweighting",
                fit_trend=True,
            )

    # --- slope summary ---------------------------------------------------
    slope_group = [k for k in args.keep if k in nominal.columns]
    if slope_group and not nominal.empty:
        result = plot_slope_summary(
            nominal, slope_group, outdir / "slope_summary.pdf",
            title=f"Slope of A vs lifetime ({standard}, after reweighting)",
        )
        if result:
            pd.DataFrame(result["fits"]).to_csv(outdir / "slope_fits.csv", index=False)
            s = result["summary"]
            if s:
                print(
                    f"\nSlope consistency: chi2/dof = {s['chi2']:.2f}/{s['dof']}, "
                    f"p = {s['pvalue']:.3f}"
                )

    metadata.write(
        outdir / "combined.json",
        stage="combined",
        inputs=paths,
        config={
            "keep": list(args.keep),
            "group_by": group_by,
            "models": models,
            "procedures": procedures,
            "standard_model": standard,
            "require_converged": args.require_converged,
            "blind_seed": args.blind_seed,
        },
        n_input_samples=len(paths),
        n_input_rows=int(len(df)),
        n_combined_rows=int(len(combined)),
        samples=[r.to_dict() for r in runids],
        output=str(combined_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())