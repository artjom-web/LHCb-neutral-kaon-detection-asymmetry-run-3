import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACKS = ["ll", "dd"]
CUTS = ["KS_Hlt1TwoTrackKsDecision_TOS", "Pip_Hlt1TrackMVADecision_TOS", "DpTIS"]
MODELS = ["johnson_exp", "johnson_exp1", "johnson_tail_expo", "johnson_gauss_exp"]
PROCEDURES = ["before", "after"]
SM = ["johnson_exp1"]

parser = argparse.ArgumentParser()
parser.add_argument("--day", type=str, required=True)
parser.add_argument("--time", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="/eos/user/a/ahulsber/scripts/data")
args = parser.parse_args()

outfolder = f"{args.data_dir}/{args.day}/compare_hlt1cuts_{args.time}/"
os.makedirs(outfolder, exist_ok=True)
A_bias = np.random.RandomState(0).uniform(-1, 1)
frames = []
for track in TRACKS:
    for cut in CUTS:
        csv_path = f"{args.data_dir}/{args.day}/{track}_{cut}_{args.time}/asymmetry_results.csv"
        if not os.path.exists(csv_path):
            print(f"WARNING: no results for {track}_{cut}: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["track"], df["cut"] = track, cut
        frames.append(df)

if not frames:
    raise SystemExit("No asymmetry CSVs found to compare")

combined = pd.concat(frames, ignore_index=True)
combined.to_csv(f"{outfolder}combined_asymmetry.csv", index=False)
print(f"Combined results saved to {outfolder}combined_asymmetry.csv")


def _errorbar_series(ax, x, xerr, y, yerr, label = None, color = None, offset=0.0):
    if label != None:
        ax.errorbar(x + offset, y, xerr=xerr, yerr=yerr,
                    fmt="o", ms=4, capsize=3, lw=1, label=label, color=color)
    else:
        ax.errorbar(x + offset, y, xerr=xerr, yerr=yerr,
                    fmt="o", ms=4, capsize=3, color=color)


def _finalize_axis(ax, title, xlabel, ylabel=None):
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()


def _set_shared_ylim(y, yerr, cap_lim=1e-2, pad=0.2):
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    valid = np.isfinite(y) & np.isfinite(yerr)
    if not valid.any():
        return 0.0, 1.0
    outlier = yerr > cap_lim
    range_err = np.where(outlier, 0.0, yerr)
    lo = np.min((y - range_err)[valid])
    hi = np.max((y + range_err)[valid])
    rng = hi - lo if hi > lo else max(abs(lo), 1.0)
    return lo - pad * rng, hi + pad * rng

def _fmt_sci(val, err):
    exp = int(np.floor(np.log10(abs(val))))
    scale = 10 ** exp
    return f"({val/scale:.1f} ± {err/scale:.1f}) × 10$^{{{exp}}}$"


# PLOT 1: hlt1 comparison for ll: JSUexp1 after weighting

# ── PDF 1 & 2: SM asymmetry per track ──
for track in TRACKS:
    sm_after = combined[
        (combined["model"].isin(SM))
        & (combined["procedure"] == "after")
        & (combined["track"] == track)
    ]
    if sm_after.empty:
        print(f"WARNING: no SM after rewegihting data for {track}, skipping")
        continue

    fig, ax = plt.subplots(figsize=(7, 5))
    cuts_present = sm_after["cut"].unique()
    for k, cut in enumerate(cuts_present):
        grp = sm_after[sm_after["cut"] == cut].sort_values("bin")
        if grp.empty:
            continue
        xerr = np.vstack([grp["xerr_lo"].to_numpy(), grp["xerr_hi"].to_numpy()])
        A_biased = grp["A"].to_numpy() + A_bias
        _errorbar_series(ax, grp["x"].to_numpy(), xerr,
                         A_biased, grp["A_err"].to_numpy(),
                         cut, f"C{k}")
    _finalize_axis(ax, f"SM after reweighting ({track})", r"KS lifetime $(t/\tau)$", "A blinded")
    y_all = sm_after["A"].to_numpy() + A_bias
    yerr_all = sm_after["A_err"].to_numpy()
    ymin, ymax = _set_shared_ylim(y_all, yerr_all)
    ax.set_ylim(ymin, ymax)

    plt.savefig(f"{outfolder}A_SM_{track}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ── PDF 3: SM ll & dd side by side, shared y ──
sm_data = {}
for track in TRACKS:
    sm_data[track] = combined[
        (combined["model"].isin(SM))
        & (combined["procedure"] == "after")
        & (combined["track"] == track)
    ]

if sm_data["ll"].empty and sm_data["dd"].empty:
    print("WARNING: no SM after data for either track, skipping shared plot")
else:
    fig, (ax_ll, ax_dd) = plt.subplots(1, 2, sharey=True, figsize=(12, 5),
                                        constrained_layout=True)
    for ax, track in [(ax_ll, "ll"), (ax_dd, "dd")]:
        sub = sm_data[track]
        if sub.empty:
            ax.set_title(f"SM after reweighting({track})\n(no data)")
            continue
        cuts_present = sub["cut"].unique()
        for k, cut in enumerate(cuts_present):
            grp = sub[sub["cut"] == cut].sort_values("bin")
            if grp.empty:
                continue
            xerr = np.vstack([grp["xerr_lo"].to_numpy(), grp["xerr_hi"].to_numpy()])

            A_biased = grp["A"].to_numpy() + A_bias
            _errorbar_series(ax, grp["x"].to_numpy(), xerr,
                                    A_biased, grp["A_err"].to_numpy(),
                                    cut, f"C{k}")
        _finalize_axis(ax, f"SM after reweighting ({track})", r"KS lifetime $(t/\tau)$", "A blinded")

    y_all = np.concatenate([(sm_data[t]["A"].to_numpy() + A_bias) for t in TRACKS if not sm_data[t].empty])
    yerr_all = np.concatenate([sm_data[t]["A_err"].to_numpy() for t in TRACKS if not sm_data[t].empty])
    ymin, ymax = _set_shared_ylim(y_all, yerr_all)
    ax_ll.set_ylim(ymin, ymax)

    fig.suptitle("SM after reweighting", fontsize=14, weight="bold")
    plt.savefig(f"{outfolder}A_SM_shared.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
# ── Compute fits once ──
from scipy.optimize import curve_fit

def _linear(x, a, b):
    return a * x + b

sm_after = combined[
    (combined["model"].isin(SM))
    & (combined["procedure"] == "after")
]

fits = []
for track in TRACKS:
    for cut in CUTS:
        grp = sm_after[(sm_after["track"] == track) & (sm_after["cut"] == cut)].sort_values("bin")
        if grp.empty:
            print(f"WARNING: no data for {track}_{cut}, skipping fit")
            continue
        popt, pcov = curve_fit(
            _linear, grp["x"].to_numpy(), grp["A"].to_numpy(),
            sigma=grp["A_err"].to_numpy(), absolute_sigma=True,
        )
        fits.append({
            "track": track, "cut": cut, "grp": grp,
            "popt": popt, "slope_err": np.sqrt(pcov[0, 0]),
        })

# ── PDF 4: per-cut ll vs dd overlay with fits ──
cuts_with_data = sm_after["cut"].unique()
if len(cuts_with_data) == 0:
    print("WARNING: no SM after reweighting data for any cut, skipping track comparison")
else:
    n = len(cuts_with_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True,
                             constrained_layout=True, squeeze=False)
    for idx, cut in enumerate(cuts_with_data):
        ax = axes[0, idx]
        y_vals, y_errs = [], []
        for k, fit in enumerate(f for f in fits if f["cut"] == cut):
            grp = fit["grp"]
            xerr = np.vstack([grp["xerr_lo"].to_numpy(), grp["xerr_hi"].to_numpy()])
            A_biased = grp["A"].to_numpy() + A_bias
            _errorbar_series(ax, grp["x"].to_numpy(), xerr,
                             A_biased, grp["A_err"].to_numpy(),
                                color= f"C{k}")
            y_vals.append(A_biased)
            y_errs.append(grp["A_err"].to_numpy())
            x_fit = np.linspace(0.0, 3.0, 100)
            ax.plot(x_fit, _linear(x_fit, *fit["popt"]) + A_bias, "--", color=f"C{k}", lw=1.2,
                    label=f"{fit['track']} fit (a={_fmt_sci(fit['popt'][0], fit['slope_err'])})")

        _finalize_axis(ax, cut, r"KS lifetime $(t/\tau)$", "A blinded" if idx == 0 else None)
        ax.legend(fontsize='large')
        if y_vals:
            y_all = np.concatenate(y_vals)
            yerr_all = np.concatenate(y_errs)
            ymin, ymax = _set_shared_ylim(y_all, yerr_all)
            ax.set_ylim(ymin, ymax)

    fig.suptitle("SM after reweighting: ll vs dd", fontsize=14, weight="bold")
    plt.savefig(f"{outfolder}A_compare_tracks.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ── Slope summary plot ──
slopes = [f["popt"][0] for f in fits]
slope_errs = [f["slope_err"] for f in fits]
labels = [f"{f['track']}\n{f['cut']}" for f in fits]
colors = ["C0" if f["track"] == "ll" else "C1" for f in fits]

fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(labels)), 5))
x_pos = np.arange(len(labels))
ax.errorbar(x_pos, slopes, yerr=slope_errs, fmt="o", ms=6, capsize=4, lw=1.2, color="k")
for i, (s, s_err, c) in enumerate(zip(slopes, slope_errs, colors)):
    ax.errorbar(i, s, yerr=s_err, fmt="o", ms=6, capsize=4, lw=1.2, color=c,
                label= fit["track"] if i == 0 or colors[i] != colors[i-1] else "")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Slope (a) of A vs t/τ")
ax.set_title("Linear fit slope: SM after weighting", fontsize=13, weight="bold")
ax.grid(alpha=0.3)
# Manual legend for tracks
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker="o", color="C0", label="ll", lw=0),
                   Line2D([0], [0], marker="o", color="C1", label="dd", lw=0)]
ax.legend(handles=legend_elements, fontsize="small")
plt.savefig(f"{outfolder}A_slope_fit.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)


# # ── PDF: fits overlaid on data ──
# from scipy.optimize import curve_fit

# def _linear(x, a, b):
#     return a * x + b

# sm_after = combined[
#     (combined["model"].isin(SM))
#     & (combined["procedure"] == "after")
# ]

# fits = []
# for track in TRACKS:
#     for cut in CUTS:
#         grp = sm_after[(sm_after["track"] == track) & (sm_after["cut"] == cut)].sort_values("bin")
#         if grp.empty:
#             print(f"WARNING: no data for {track}_{cut}, skipping fit plot")
#             continue
#         popt, pcov = curve_fit(
#             _linear, grp["x"].to_numpy(), grp["A"].to_numpy(),
#             sigma=grp["A_err"].to_numpy(), absolute_sigma=True,
#         )
#         fits.append((track, cut, grp, popt, np.sqrt(pcov[0, 0])))

# n = len(fits)
# cols = min(n, 3)
# rows = (n + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
#                          sharex=True, sharey=True, constrained_layout=True,
#                          squeeze=False)
# for idx, (track, cut, grp, popt, slope_err) in enumerate(fits):
#     ax = axes[idx // cols, idx % cols]
#     xerr = np.vstack([grp["xerr_lo"].to_numpy(), grp["xerr_hi"].to_numpy()])
#     ax.errorbar(grp["x"].to_numpy(), grp["A"].to_numpy(), xerr=xerr,
#                 yerr=grp["A_err"].to_numpy(), fmt="o", ms=4, capsize=3, lw=1)
#     x_fit = np.linspace(grp["x"].min(), grp["x"].max(), 100)
#     ax.plot(x_fit, _linear(x_fit, *popt), "r-", lw=1.5,
#             label=f"a={popt[0]:.4f} ± {slope_err:.4f}")
#     ax.set_title(f"{track} / {cut}", fontsize=9)
#     ax.legend(fontsize="small")
#     ax.grid(alpha=0.3)

# for idx in range(n, rows * cols):
#     axes[idx // cols, idx % cols].set_visible(False)

# fig.suptitle("Linear fits: SM after weighting", fontsize=14, weight="bold")
# plt.savefig(f"{outfolder}A_fit_overlays.pdf", dpi=300, bbox_inches="tight")
# plt.close(fig)