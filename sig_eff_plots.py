import ROOT
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


def init_data():
    params = ["KS_LT"]
    tracks = ['ll', 'dd']
    ycsets = ['25c1', '25c2', '25c3', '25c4']
    polarities = ['magup', 'magdown']
    data = {
        "ll": {
            "magup": "25c4_80-25c3_156-25c1_95-24c4a_33-24c3a_83-24c2a_240",
            "magdown": "25c4_59-25c2_246-25c1_22-24c4a_52-24c3a_69-24c2a_72",
        },
        "dd": {
            "magup": "25c4_188-25c3_320-25c1_426-24c4a_132-24c3a_305-24c2a_57",
            "magdown": "25c4_126-25c2_1007-25c1_100-24c4a_240-24c3a_250-24c2a_14",
        },
    }

    result = {}

    for track, polarities in data.items():
        for polarity, values in polarities.items():
            for item in values.split("-"):
                ycset, value = item.rsplit("_", 1)
                result[f"{track}_{ycset}_{polarity}"] = int(value)
    return tracks, ycsets, polarities, result, params

# ---------------------------------------------------------------------------
# Reading fit results
# ---------------------------------------------------------------------------

def read_fit_yields(folder, cut):
    """
    Reads N_tot / f_sig from <folder>/<cut>/model.root and returns
    (Nsig, Nsig_err, Nbg, Nbg_err), or None if the file is missing/broken.
    N_tot is fixed (setConstant(True) in mass_fit), so it contributes no
    error; f_sig's fit error is the sole source of uncertainty on both yields.
    """
    file = os.path.join(folder, cut, "model.root")
    if not os.path.exists(file):
        print(f"WARNING: missing {file}")
        return None
    f = ROOT.TFile.Open(file)
    if not f or f.IsZombie():
        print(f"WARNING: could not open {file}")
        return None
    ws = f.Get("ws")
    fit_params = ws.allVars()
    N_tot = fit_params.find("N_tot").getVal()
    f_sig = fit_params.find("f_sig").getVal()
    f_sig_err = fit_params.find("f_sig").getError()
    f.Close()

    Nsig = N_tot * f_sig
    Nsig_err = N_tot * f_sig_err
    Nbg = N_tot * (1 - f_sig)
    Nbg_err = N_tot * f_sig_err
    return (Nsig, Nsig_err, Nbg, Nbg_err)


def gather_yields(base_folder, files_dict, tracks, ycsets, polarities, cuts, skip_combo=None, N_files_fit=10):
    """
    Reads every (track, ycset, polarity, cut) combination's yields into a
    nested dict: yields[track][ycset][polarity][cut] -> tuple or None.
    skip_combo(track, ycset, polarity) -> bool lets you replicate the
    dataset-availability gaps from the run script (e.g. missing polarity
    for a given ycset).
    """
    yields = {t: {y: {p: {} for p in polarities} for y in ycsets} for t in tracks}
    for track in tracks:
        for ycset in ycsets:
            for polarity in polarities:
                if skip_combo and skip_combo(track, ycset, polarity):
                    continue
                folder = os.path.join(base_folder, f"{track}_{ycset}_{polarity}")
                for cut in cuts:
                    N_files_tot = files_dict[f"{track}_{ycset}_{polarity}"]
                    (Nsig, Nsig_err, Nbg, Nbg_err) = read_fit_yields(folder, cut)
                    Nsigtot = (N_files_tot / N_files_fit) * Nsig
                    yields[track][ycset][polarity][cut] = (Nsig, Nsig_err, Nbg, Nbg_err, Nsigtot)
    return yields


# ---------------------------------------------------------------------------
# Combining across independent datasets
# ---------------------------------------------------------------------------

def combine_yields(yield_list):
    """Sums Nsig/Nbg across a list of (independent) yield tuples, adding
    errors in quadrature. Entries that are None (missing data) are skipped."""
    valid = [y for y in yield_list if y is not None]
    if not valid:
        return None
    Nsig = sum(v[0] for v in valid)
    Nsig_err = np.sqrt(sum(v[1] ** 2 for v in valid))
    Nbg = sum(v[2] for v in valid)
    Nbg_err = np.sqrt(sum(v[3] ** 2 for v in valid))
    Nsigtot = sum(v[4] for v in valid)
    return (Nsig, Nsig_err, Nbg, Nbg_err, Nsigtot)


def combine_polarities(yields, track, ycset, cuts, polarities):
    return {cut: combine_yields([yields[track][ycset][p].get(cut) for p in polarities])
            for cut in cuts}


def combine_ycsets(yields, track, polarity, cuts, ycsets):
    return {cut: combine_yields([yields[track][y][polarity].get(cut) for y in ycsets])
            for cut in cuts}


def combine_all(yields, track, cuts, ycsets, polarities):
    return {cut: combine_yields([yields[track][y][p].get(cut) for y in ycsets for p in polarities])
            for cut in cuts}


# python sig_eff_plots.py --base_folder /eos/user/a/ahulsber/scripts/data/sig_eff/08-16/Pip_Hlt1TrackMVADecision_TOS_16-28 --hlt1_cut Pip_Hlt1TrackMVADecision_TOS

# ---------------------------------------------------------------------------
# Efficiency / rejection with error propagation
# ---------------------------------------------------------------------------

def eff_and_rej(yields_by_cut, cuts):
    """
    Returns (sig_eff, sig_eff_err, bg_rej, bg_rej_err, Nsig_final) arrays
    over `cuts`, relative to cuts[0] as baseline (N_init).
    Ratios are propagated treating numerator/denominator as independent
    (see caveat above) except at the baseline point itself, which is
    fixed to eff=1.0/rej=0.0 with zero error since it's the same
    measurement as itself, not two independent ones.
    """
    base = yields_by_cut.get(cuts[0])
    if base is None:
        return None
    Nsig0, Nsig0_err, Nbg0, Nbg0_err, Nsigtot0 = base

    n = len(cuts)
    sig_eff = np.full(n, np.nan)
    sig_eff_err = np.full(n, np.nan)
    bg_rej = np.full(n, np.nan)
    bg_rej_err = np.full(n, np.nan)

    for idx, cut in enumerate(cuts):
        if cut == cuts[0]:
            sig_eff[idx], sig_eff_err[idx] = 1.0, 0.0
            bg_rej[idx], bg_rej_err[idx] = 0.0, 0.0
            continue
        y = yields_by_cut.get(cut)
        if y is None:
            continue
        Nsig, Nsig_err, Nbg, Nbg_err, Nsigtot = y

        if Nsig0 > 0 and Nsig > 0:
            eff = Nsig / Nsig0
            eff_err = eff * np.sqrt((Nsig_err / Nsig) ** 2 + (Nsig0_err / Nsig0) ** 2)
            sig_eff[idx], sig_eff_err[idx] = eff, eff_err

        if Nbg0 > 0 and Nbg > 0:
            r = Nbg / Nbg0
            r_err = r * np.sqrt((Nbg_err / Nbg) ** 2 + (Nbg0_err / Nbg0) ** 2)
            bg_rej[idx], bg_rej_err[idx] = 1 - r, r_err

    final = yields_by_cut.get(cuts[-1])
    Nsig_final = final[4] if final is not None else np.nan

    return sig_eff, sig_eff_err, bg_rej, bg_rej_err, Nsig_final


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eff_rej_hlt1_baseline(series, title, outpath, annotate_fontsize=13):
    """
    Like plot_eff_rej, but each series entry supplies its OWN `cuts` list
    (series[i]['cuts'][0] is that series' baseline, e.g. its HLT1 cut name)
    instead of sharing one global cuts list — needed because different HLT1
    cut variants are different baselines, not different points on a shared
    x-axis of cut names.
    series: list of {'label': str, 'yields_by_cut': {cut: yield_tuple}, 'cuts': [...]}
    """
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    markers = ["o", "s", "D", "^", "v", "<", ">"]

    for i, s in enumerate(series):
        cuts_local = s['cuts']
        result = eff_and_rej(s['yields_by_cut'], cuts_local)
        if result is None:
            print(f"WARNING: no baseline data for '{s['label']}', skipping")
            continue
        sig_eff, sig_eff_err, bg_rej, bg_rej_err, Nsig_final = result
        label = _millions_label(s['label'], Nsig_final)
        valid = np.isfinite(sig_eff) & np.isfinite(bg_rej)
        if not valid.any():
            continue
        m = markers[i % len(markers)]
        ax.errorbar(bg_rej[valid], sig_eff[valid],
                    xerr=bg_rej_err[valid], yerr=sig_eff_err[valid],
                    fmt=m + "-", ms=6, capsize=3, lw=1, color=f"C{i}", label=label)
        for j, cut in enumerate(cuts_local):
            if valid[j]:
                if j in (0,3,7):
                    continue
                ax.annotate(cut, (bg_rej[j], sig_eff[j]),
                            fontsize=annotate_fontsize, ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points")

    ax.set_xlabel("Background rejection (relative to HLT1 cut)")
    ax.set_ylabel("Signal efficiency (relative to HLT1 cut)")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.04, 1.04)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(title, fontsize=14, weight="bold")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _millions_label(base_label, Nsig_final):
    """Appends the final-cut signal yield in millions to a legend label,
    e.g. '25c1 (0.42M events)'."""
    if Nsig_final is None or not np.isfinite(Nsig_final) or Nsig_final <= 0:
        return base_label
    millions = Nsig_final / 1e6
    return base_label + fr" ($\approx${millions:.0f}M events)"



def _oom_label(base_label, Nsig_final):
    """Appends the order-of-magnitude of the final-cut signal yield to a
    legend label, e.g. '25c1 (~1e5)'."""
    if Nsig_final is None or not np.isfinite(Nsig_final) or Nsig_final <= 0:
        return base_label
    oom = int(np.floor(np.log10(Nsig_final)))
    return base_label + r" ($\sim 10^{%d}$ events)" % oom


def plot_efficiency_pair(series, cuts, title, outpath):
    """
    series: list of {'label': str, 'yields_by_cut': {cut: yield_tuple}}
    Draws a 1x2 figure: signal efficiency panel + background rejection panel,
    one line per series entry, x-axis = cuts (in order, cuts[0]=baseline).
    """
    fig, (ax_eff, ax_rej) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(cuts))

    for s in series:
        result = eff_and_rej(s['yields_by_cut'], cuts)
        if result is None:
            print(f"WARNING: no baseline data for '{s['label']}', skipping")
            continue
        sig_eff, sig_eff_err, bg_rej, bg_rej_err, Nsig_final = result
        label = _oom_label(s['label'], Nsig_final)
        ax_eff.errorbar(x, sig_eff, yerr=sig_eff_err, fmt="o-", ms=4, capsize=3, lw=1, label=label)
        ax_rej.errorbar(x, bg_rej, yerr=bg_rej_err, fmt="o-", ms=4, capsize=3, lw=1, label=label)

    for ax, ylabel in ((ax_eff, "Signal efficiency"), (ax_rej, "Background rejection")):
        ax.set_xticks(x)
        ax.set_xticklabels(cuts, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize='small')
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlim(-0.04, 1.04)

    fig.suptitle(title, fontsize=14, weight='bold')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_eff_rej(series, cuts, title, outpath):
    """Scatter of bg_rej vs sig_eff, one point per cut per series, connected
    in cut order to show the progression from the baseline (mass_lt) cut."""
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    markers = ["o", "s", "D", "^", "v", "<", ">"]

    for i, s in enumerate(series):
        result = eff_and_rej(s['yields_by_cut'], cuts)
        if result is None:
            print(f"WARNING: no baseline data for '{s['label']}', skipping")
            continue
        sig_eff, sig_eff_err, bg_rej, bg_rej_err, Nsig_final = result
        label = _oom_label(s['label'], Nsig_final)
        valid = np.isfinite(sig_eff) & np.isfinite(bg_rej)
        if not valid.any():
            continue
        m = markers[i % len(markers)]
        ax.errorbar(bg_rej[valid], sig_eff[valid],
                    xerr=bg_rej_err[valid], yerr=sig_eff_err[valid],
                    fmt=m + "-", ms=6, capsize=3, lw=1, color=f"C{i}", label=label)
        for j, cut in enumerate(cuts):
            if valid[j]:
                ax.annotate(cut, (bg_rej[j], sig_eff[j]),
                            fontsize=7, ha="center", va="bottom",
                            xytext=(0, 3), textcoords="offset points")

    ax.set_xlabel("Background rejection")
    ax.set_ylabel("Signal efficiency")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.04, 1.04)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.suptitle(title, fontsize=14, weight="bold")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_skip_combo(track, ycset, polarity):
    """Mirrors the run script's dataset-availability gaps."""
    return (polarity == 'magdown' and ycset == '25c3') or (polarity == 'magup' and ycset == '25c2')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, nargs='+', required=True,
                         help="One base_folder per --hlt1_cut, same order.")
    parser.add_argument("--hlt1_cut", type=str, nargs='+', required=True,
                         help="One or more HLT1 cut names, e.g. two or three runs to compare.")
    args = parser.parse_args()

    if len(args.base_folder) != len(args.hlt1_cut):
        raise ValueError("--base_folder and --hlt1_cut must have the same number of entries "
                          f"(got {len(args.base_folder)} and {len(args.hlt1_cut)})")

    runs = list(zip(args.hlt1_cut, args.base_folder))  # [(hlt1_cut, base_folder), ...]

    tracks = ['ll', 'dd']
    ycsets = ['25c1', '25c2', '25c3', '25c4']
    polarities = ['magup', 'magdown']
    _, _, _, files_dict, _ = init_data()

    # --- existing 3 plots + yield gathering, once per HLT1-cut run ---
    yields_by_run = {}   # hlt1_cut -> yields dict, kept for the new cross-run plot below

    for hlt1_cut, base_folder in runs:
        if not base_folder.endswith("/"):
            base_folder += "/"
        cuts = ['mass_lt', hlt1_cut, 'kin', 'probnn']

        yields = gather_yields(base_folder, files_dict, tracks, ycsets, polarities, cuts,
                                skip_combo=default_skip_combo, N_files_fit=10)
        yields_by_run[hlt1_cut] = yields

        outdir = base_folder + "sig_eff_plots/"
        os.makedirs(outdir, exist_ok=True)

        for track in tracks:
            series = [
                {'label': ycset, 'yields_by_cut': combine_polarities(yields, track, ycset, cuts, polarities)}
                for ycset in ycsets
            ]
            plot_eff_rej(series, cuts, f'{track}: ycset dependence (combined polarities)',
                         outdir + f'{track}_ycset_dependence_eff_rej.pdf')

        for track in tracks:
            series = [
                {'label': polarity, 'yields_by_cut': combine_ycsets(yields, track, polarity, cuts, ycsets)}
                for polarity in polarities
            ]
            plot_eff_rej(series, cuts, f'{track}: polarity dependence (combined ycsets)',
                         outdir + f'{track}_polarity_dependence_eff_rej.pdf')

        series = [
            {'label': track, 'yields_by_cut': combine_all(yields, track, cuts, ycsets, polarities)}
            for track in tracks
        ]
        plot_eff_rej(series, cuts, 'Track comparison (combined ycset & polarity)',
                     outdir + 'track_comparison_eff_rej.pdf')

        print(f"Plots saved to: {outdir}")

    # --- new plot: HLT1-cut comparison, one figure per track, HLT1 cut's own point = 100% ---
    combined_outdir = os.path.commonpath([bf for _, bf in runs]) + "/sig_eff_plots_hlt1_comparison/"
    os.makedirs(combined_outdir, exist_ok=True)

    for track in tracks:
        series = []
        for hlt1_cut, _ in runs:
            cuts_rel = [hlt1_cut, 'kin', 'probnn']   # this run's HLT1 point is now the 100% baseline
            yields_by_cut = combine_all(yields_by_run[hlt1_cut], track, cuts_rel, ycsets, polarities)
            series.append({'label': hlt1_cut, 'yields_by_cut': yields_by_cut, 'cuts': cuts_rel})

        plot_eff_rej_hlt1_baseline(
            series,
            f'{track}: effect of kinematic cuts, per HLT1 cut (HLT1 point = 100%)',
            combined_outdir + f'{track}_hlt1_comparison_eff_rej.pdf',
            annotate_fontsize=13,
        )

    print(f"HLT1-comparison plots saved to: {combined_outdir}")



# python sig_eff_plots.py \
#   --hlt1_cut Pip_Hlt1TrackMVADecision_TOS DpTIS KS_Hlt1TwoTrackKsDecision_TOS DpTIS_PipMVATOS DpTIS_KSTwoTracks \
#   --base_folder /eos/user/a/ahulsber/scripts/data/sig_eff/08-16/Pip_Hlt1TrackMVADecision_TOS_16-28 \
# /eos/user/a/ahulsber/scripts/data/sig_eff/08-16/DpTIS_16-28 \
# /eos/user/a/ahulsber/scripts/data/sig_eff/08-16/KS_Hlt1TwoTrackKsDecision_TOS_16-28 \
# /eos/user/a/ahulsber/scripts/data/sig_eff/08-16/DpTIS_PipMVATOS_16-28 \
# /eos/user/a/ahulsber/scripts/data/sig_eff/08-16/DpTIS_KSTwoTracks_16-28
