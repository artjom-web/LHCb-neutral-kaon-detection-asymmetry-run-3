"""Shared command-line plumbing.

Every stage takes the same sample selectors, so they are defined once.  Leaving
a selector out means "all of them", which is what turns

    python -m scripts.s2_reweight --track dd

into a loop over every ycset and polarity without the script containing a loop
over hardcoded lists.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Iterator, List, Optional

from .config import datasets
from .core.paths import Layout, add_layout_args, layout_from_args
from .core.runid import RunID, skipped


def add_sample_args(parser: argparse.ArgumentParser, *, hlt1_required: bool = False) -> None:
    parser.add_argument("--track", choices=["ll", "dd"], default=None,
                        help="Restrict to one track (default: both)")
    parser.add_argument("--ycset", default=None,
                        help="Restrict to one ycset (default: all)")
    parser.add_argument("--polarity", choices=["magup", "magdown"], default=None,
                        help="Restrict to one polarity (default: both)")
    parser.add_argument("--hlt1", default="" if not hlt1_required else None,
                        required=hlt1_required,
                        help="HLT1 selection name; '' means no HLT1 requirement")


def add_common_args(parser: argparse.ArgumentParser, *, hlt1_required: bool = False,
                    require_tag: bool = False) -> None:
    add_sample_args(parser, hlt1_required=hlt1_required)
    add_layout_args(parser)
    parser.add_argument("--no-mt", action="store_true",
                        help="Disable ROOT implicit multithreading")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the work that would be done, then exit")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Log and skip a failing sample instead of aborting")
    parser._kspi_require_tag = require_tag  # consumed by resolve_layout


def resolve_layout(args) -> Layout:
    return layout_from_args(args, require_tag=getattr(args, "_kspi_require_tag", False))


def selected_runids(args, *, from_inventory: bool = True) -> List[RunID]:
    """Expand the CLI selectors into concrete RunIDs.

    ``from_inventory=True`` (stage 1) consults the dataset table so that only
    samples that actually exist are produced; later stages instead discover
    what stage 1 wrote, so they pass False and filter by what is on disk.
    """
    hlt1 = args.hlt1 or ""
    out: List[RunID] = []
    if from_inventory:
        for track, ycset, polarity, _ in datasets.iter_samples(
            args.track, args.ycset, args.polarity
        ):
            rid = RunID(track=track, ycset=ycset, polarity=polarity, hlt1=hlt1)
            if skipped(rid):
                continue
            out.append(rid)
    else:
        for track, ycset, polarity, _ in datasets.iter_samples(
            args.track, args.ycset, args.polarity
        ):
            out.append(RunID(track=track, ycset=ycset, polarity=polarity, hlt1=hlt1))
    return out


def run_over_samples(args, runids, work) -> int:
    """Call ``work(runid)`` for each sample, honouring --dry-run / --continue-on-error.

    Returns a process exit code: non-zero if anything failed, so a batch
    submission notices.  The old scripts always exited 0 even when a sample
    raised, which made silent partial results easy to miss.
    """
    if not runids:
        print("No samples selected. Check --track/--ycset/--polarity.")
        return 1

    print(f"{len(runids)} sample(s) selected:")
    for rid in runids:
        print(f"  - {rid.label()}")
    if args.dry_run:
        print("(dry run: nothing executed)")
        return 0

    failures = []
    for rid in runids:
        print("\n" + "=" * 70)
        print(f"  {rid.label()}")
        print("=" * 70)
        try:
            work(rid)
        except Exception as exc:  # noqa: BLE001 - we deliberately continue
            failures.append((rid, exc))
            traceback.print_exc()
            if not args.continue_on_error:
                print(f"\nAborting after failure on {rid.label()}.")
                return 1
            print(f"\nWARNING: {rid.label()} failed, continuing: {exc}")

    if failures:
        print(f"\n{len(failures)} sample(s) failed:")
        for rid, exc in failures:
            print(f"  - {rid.label()}: {exc}")
        return 1
    print("\nAll samples completed.")
    return 0