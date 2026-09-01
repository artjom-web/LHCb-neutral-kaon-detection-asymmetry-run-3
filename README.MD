# D+ → KS π+ asymmetry analysis

Six-stage pipeline, restructured from `AnalysisFunctions.py`, `make_snapshot.py`,
`multi_analysis.py` and `compare_hlt1cuts.py`.

## Why it is shaped this way

The old code was one 2,278-line `Analysis` class holding the RDataFrame, the
track configuration, the mass window, the fitter, the weighting state and every
plot function at once. Because the weights existed only as lazy RDataFrame
columns on a live object, nothing downstream could run without redoing the
reweighting — which is why `multi_analysis.py` had to do everything in a single
process.

Three changes fix that:

1. **Stage 2 snapshots every iteration's weights**, not just `final_weight`.
   That single decision is what makes stages 3, 4 and 5 independent of each
   other and re-runnable without refitting.
2. **Stages talk through a file plus a JSON sidecar.** Each stage records what
   it produced — iteration count, lifetime edges, weight column names, sideband
   ratio, weighted bin means, the full configuration, the git revision — and
   the next stage reads that instead of re-deriving it.
3. **Scripts are thin.** All physics lives in the package; each script is
   argparse plus one call into it. Stage 3 and stage 6 therefore share the same
   plotting and combination code.

## Layout

```
kspi_analysis/
  config/       selection.py   per-track cuts, mass window, binning, HLT1 lines
                datasets.py    the (track, ycset, polarity) → n_files inventory
                weighting.py   reweighting cycle schemes
                massmodels.py  model list, standard model, warm-start seeds
  core/         runid.py       sample identity
                paths.py       every output path, in one place
                metadata.py    the sidecar contract between stages
                rootio.py      TFile/RDataFrame helpers
                util.py        shared pure helpers
  physics/      selection.py   loading, derived columns, cuts        (stage 1)
                reweighting.py the iterative reweighter              (stage 2)
                massfit.py     MassFitter (RooFit)                   (stages 2,3,5)
                binning.py     weighted lifetime bin means
                asymmetry.py   fits in lifetime bins → table         (stage 3)
                stats.py       S_w, R_w, N_eff                       (stage 4)
                performance.py weighting diagnostics                 (stage 5)
                combine.py     inverse-variance combination          (stage 6)
  plotting/     asymmetry.py   AsymmetryPlotter (shared by 3 and 6)
                weighting.py   w-vs-kin, A-vs-kin, A-vs-LT
                stats.py       statistical-loss figures
                comparison.py  cross-configuration figures
  cli.py        shared argparse plumbing
scripts/        s1_skim.py … s6_combine.py
```

Mass-fit *drawing* lives on `MassFitter.plot`, not in `plotting/`, because
RooFit's plotting needs the workspace the fitter just wrote.

## Output tree

```
<out-root>/
  stage1_skim/<hlt1>/<track>_<ycset>_<polarity>.root   + .json
  stage2_reweighted/<tag>/<hlt1>/<key>/reweighted.root + .json
                                       initial_massfit/
  stage3_massfits/<tag>/<hlt1>/<key>/asymmetries.csv   + .json
                                     massfits/<weight>/<model>/bin<i>/
                                     failed_fits/
  stage4_statistics/<tag>/<hlt1>/<key>/
  stage5_weighting/<tag>/<hlt1>/<key>/
  stage6_combined/<tag>/<name>/
```

Stage 1 is *not* under `<tag>`: the skim depends only on the selection, so one
skim feeds many tagged reweighting runs. Stages 2–6 share a `--tag` so a full
chain is reproducible and two configurations can coexist.

## Running it

```bash
export KSPI_OUT_ROOT=/eos/user/a/ahulsber/scripts/analysis

# 1. skim (slow, run once per HLT1 selection)
python scripts/s1_skim.py --track dd --hlt1 DpTIS
python scripts/s1_skim.py --track ll --hlt1 DpTIS

# 2. reweight
python scripts/s2_reweight.py --hlt1 DpTIS --tag v1 --cycles nominal

# 3-5. independent, run in any order or in parallel
python scripts/s3_asymmetries.py --hlt1 DpTIS --tag v1
python scripts/s4_statistics.py  --hlt1 DpTIS --tag v1
python scripts/s5_weighting_performance.py --hlt1 DpTIS --tag v1 --only w-kin

# 6. combine
python scripts/s6_combine.py --tag v1 --name nominal --keep track hlt1
```

Every selector left out means "all of them", so `--track dd` loops over all
ycsets and polarities. Add `--dry-run` to see the work list first, and
`--continue-on-error` for batch jobs.

### Replacing compare_hlt1cuts.py

```bash
python scripts/s6_combine.py --tag v1 --name hlt1_study \
  --hlt1-set KS_Hlt1TwoTrackKsDecision_TOS Pip_Hlt1TrackMVADecision_TOS DpTIS \
  --keep track hlt1 --compare hlt1 --panel-by track
```

`--keep` lists the fields that stay *separate*; everything else is combined
over. `--keep track hlt1` averages polarities and ycsets together while keeping
tracks and trigger lines apart.

## Configuration

Nothing that a physicist tunes should require editing a script.

| What | Where |
|---|---|
| cuts, mass window, lifetime bins, histogram binning | `config/selection.py` |
| which samples exist, how many files | `config/datasets.py` |
| reweighting cycles, weight-ratio clamp | `config/weighting.py` |
| mass models, warm-start seeds, retry budget | `config/massmodels.py` |

Mass-model *shapes* stay in your existing `mass_models.py`, which must be
importable. To warm-start a fresh run from a previous one's converged
parameters, put `<model>_seed.txt` files in a directory and pass
`--param-seed-dir`.

## Blinding

CSVs hold **true** fitted asymmetries; the blinding offset is added at draw
time only (`core/util.blinding_offset`, deterministic in `--blind-seed` so
every stage and sample shares one offset). This means result files are safe to
diff between runs, which was not true before — the offset used to be generated
independently in `multi_analysis.py` and `compare_hlt1cuts.py`.

## Behaviour changes from the old code

Everything numerical is preserved. `mass_fit`, `plot_massfit`,
`weighting_cycle`, `normalize_th4d`, `apply_weight_acc`, `AsymmetryPlotter` and
the three `_draw_*` functions were ported **mechanically** — a script extracted
the exact line ranges and rewrote only the `self.` couplings — so the fit
strategy, jitter schedule, workspace contents and report format are unchanged.
Deliberate differences:

- `plot_massfit` looped `range(self.nbins_m)` while filling a histogram with
  `nbins` bins. These were equal for every fit actually performed, so results
  are identical; using `nbins` removed the fitter's last dependency on the
  global mass binning.
- Failed fits are collected in `MassFitter.failed` and copied to
  `stage3_massfits/.../failed_fits/`, instead of the warning scrolling past.
- Missing or unreadable fits stay `NaN` end to end. They previously risked
  becoming a plotted `A = 0` that looks like a real measurement.
- Stage 4 writes the numbers behind its plots to CSV. Previously they were
  computed, drawn and discarded.
- Non-zero exit codes on failure, so batch submissions notice.

## Bugs found in the old code

- `multi_analysis.py:104` uses `runfolder`, which is never defined — the script
  cannot run as written.
- `open_file` / `open_folder` derive sample identity by splitting filenames on
  `_`, which breaks on HLT1 names like `Pip_Hlt1TrackMVADecision_TOS`. Identity
  is now a `RunID`; HLT1 lives in a directory and file counts in metadata.
- `make_snapshot.py` shadows the `time` module with a `time` variable.
- `Analysis.cuts` had to be called twice (once for the trigger, once for
  everything else) because the two were mutually exclusive branches of one
  `if`. They are independent flags now.
- The `(magdown, 25c3)` / `(magup, 25c2)` skip guard was needed because the old
  loop iterated the full cartesian product. `datasets.iter_samples` yields only
  samples that exist, so the guard (kept in `runid.skipped`) is now redundant.

## Status

Built and checked without ROOT available, so:

- **Verified:** all 27 modules import; every internal import resolves; all six
  scripts parse and `--dry-run` correctly; combination, trend fits, pivoting
  and all figure types were exercised on synthetic data.
- **Not verified:** anything touching real ROOT I/O. Expect the first live run
  to surface small issues, most likely in the stage-2 snapshot column list.
- `mass_models.py` was not supplied; `MassFitter` imports it as-is and it must
  remain on `PYTHONPATH`.