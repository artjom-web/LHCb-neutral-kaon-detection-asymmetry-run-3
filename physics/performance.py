"""Weighting-performance diagnostics (stage 5).

Computed by :class:`WeightingPerformance` and drawn by the free functions
``draw_w_kin`` / ``draw_A_kin`` / ``draw_A_LT``.  Three independent
diagnostics, each switchable:

  ``w_kin``  the weighted kinematic distributions in each lifetime bin,
             overlaid — if the reweighting worked, the lifetime bins should
             lie on top of each other after weighting and not before.
  ``A_kin``  the raw asymmetry as a function of each kinematic variable,
             from a mass fit in every bin of that variable.
  ``A_LT``   those per-variable asymmetries integrated over each lifetime
             bin's kinematic distribution — i.e. how much apparent lifetime
             dependence is induced purely by kinematics.
"""

from __future__ import annotations

import array
import math
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt

from collections import defaultdict

import numpy as np

import ROOT

from ..config.selection import TrackConfig, build_hist_params
from ..core.util import (
    capped_ylim,
    particle_groups,
    uniform_bin_centers_widths,
    weight_key,
)
from .massfit import MassFitter


class WeightingPerformance:
    """Holds the histograms and fit results behind the stage-5 figures."""

    def __init__(self, rdf, cfg: TrackConfig, fitter: MassFitter, iteration: int,
                 hist_params: Optional[Dict[str, list]] = None,
                 bin_x: Optional[Dict[str, list]] = None,
                 bin_xerr: Optional[Dict[str, object]] = None):
        self.rdf = rdf
        self.cfg = cfg
        self.fitter = fitter
        self.iteration = iteration
        self.hist_params = hist_params or build_hist_params(cfg)
        self.lt_bin_edges = list(cfg.lt_bin_edges)

        self.hists: Dict[str, object] = {}
        self.hists_filled: Dict[str, object] = {}
        self.A_fit_cache: Dict[str, object] = {}
        self.A_integrated = defaultdict(dict)
        self.A_integrated_err = defaultdict(dict)
        self.bin_x = bin_x or {}
        self.bin_xerr = bin_xerr or {}

    def run(self, folder, plot_w_kin=False, plot_A_kin=False, plot_A_LT=False):
        folder = str(folder).rstrip('/') + '/'
        os.makedirs(folder, exist_ok=True)

        ##########
        # Objective 1: plot w vs kin (for w in signal weights)
        ########

        if plot_w_kin or plot_A_LT:
            dist_params = {k: v for k, v in self.hist_params.items() if k!= "KS_LT"}
            signal_weights = ['weight_0', f'weight_{self.iteration}']
            tedges = array.array('d', [float(e) for e in self.lt_bin_edges])

            # --- book: one Histo2D per (weight, param), fused into a single pass ---
            for w in signal_weights:
                for param, p_args in dist_params.items():
                    hkey = f'h2D_{weight_key(w)}-{param}-KSLT'
                    if hkey not in self.hists:
                        self.hists[hkey] = self.rdf.Histo2D(
                            (hkey, hkey, p_args[0], p_args[1], p_args[2], self.cfg.kslt_anabins, tedges),
                            param, "KS_LT", w
                        )

            # --- trigger + slice into per-lt-bin TH1Ds via ProjectionX (no extra passes) ---
            for w in signal_weights:
                for param in dist_params:
                    hkey = f'h2D_{weight_key(w)}-{param}-KSLT'
                    h2 = self.hists[hkey].GetValue()
                    for i in range(self.cfg.kslt_anabins):
                        pkey = f'h1D_{weight_key(w)}-{param}_ltbin{i}'
                        self.hists_filled[pkey] = h2.ProjectionX(f'px_{pkey}', i + 1, i + 1)


        ##########
        # Objective 2: plot A vs kin (unweighted)
        ########
        if plot_A_kin or plot_A_LT:
            A_params = {k: v for k, v in self.hist_params.items() if k != "Dp_M" and k != "KS_LT"}
            ma = array.array("d", np.linspace(self.cfg.min_m, self.cfg.max_m, self.cfg.nbins_m + 1))
            ca = array.array("d", np.linspace(-2, 2, 3))

            # init hists
            for param, p_args in A_params.items():
                hkey = f'h3D_N-{param}-Dp_M-Dp_charge'
                pa = array.array("d", np.linspace(p_args[1], p_args[2], p_args[0] + 1))
                args = ((hkey, hkey, len(pa) - 1, pa, len(ma) - 1, ma, len(ca) - 1, ca), param, "Dp_M", "Dp_charge")
                self.hists[hkey] = self.rdf.Histo3D(*args)

            # fill hists
            for param in A_params:
                hkey = f'h3D_N-{param}-Dp_M-Dp_charge'
                self.hists_filled[hkey] = self.hists[hkey].GetValue()

            # execute mass fits
            for param, p_args in A_params.items():
                hkey = f'h3D_N-{param}-Dp_M-Dp_charge'
                pid = param.split("_")[0]
                h = self.hists_filled[hkey]
                for i in range(1, h.GetXaxis().GetNbins() + 1):
                    h_plus = h.ProjectionY(f"h_plus_{i}_{param}", i, i, 2, 2)
                    h_minus = h.ProjectionY(f"h_minus_{i}_{param}", i, i, 1, 1)
                    self.fitter.fit(h_plus, h_minus,
                                    folder + f"massfits/{pid}/{param}/bin{i-1}/")

            # --- save the asymmetries (dedented: runs once, after all fits) ---
            self.A_fit_cache = {}
            for param, p_args in A_params.items():
                pid = param.split("_")[0]
                self.A_fit_cache[param] = ([], [])
                for bin_idx in range(p_args[0]):
                    fit_file = folder + f"massfits/{pid}/{param}/bin{bin_idx}/model.root"
                    A_val, A_err_val = np.nan, np.nan
                    if os.path.exists(fit_file):
                        f = ROOT.TFile.Open(fit_file)
                        if f and not f.IsZombie():
                            ws = f.Get("ws")
                            pv = ws.allVars()
                            A_val = pv.find("A_sig").getVal()
                            A_err_val = pv.find("A_sig").getError()
                        if f:
                            f.Close()
                    else:
                        print(f"WARNING: missing {fit_file}")
                    self.A_fit_cache[param][0].append(A_val)
                    self.A_fit_cache[param][1].append(A_err_val)


        # --- find integrated asymmetries ---
        if plot_A_LT:
            for w in signal_weights:
                self.A_integrated[weight_key(w)] = {}
                self.A_integrated_err[weight_key(w)] = {}
                for param, p_args in A_params.items():
                    self.A_integrated[weight_key(w)][param] = []
                    self.A_integrated_err[weight_key(w)][param] = []
                    Asig_arr = self.A_fit_cache[param][0]
                    Asig_err_arr = self.A_fit_cache[param][1]

                    for i in range(self.cfg.kslt_anabins):
                        A_int = 0.0
                        A_int_err_sq = 0.0
                        hkey = f'h1D_{weight_key(w)}-{param}_ltbin{i}'
                        hist = self.hists_filled[hkey]
                        total = hist.Integral()
                        if total <= 0:
                            self.A_integrated[weight_key(w)][param].append(np.nan)
                            self.A_integrated_err[weight_key(w)][param].append(np.nan)
                            continue

                        for parambin in range(1, p_args[0] + 1):
                            count = hist.GetBinContent(parambin)
                            norm_count = count / total
                            idx = parambin - 1
                            if idx >= len(Asig_arr):
                                continue
                            A_val, A_err_val = Asig_arr[idx], Asig_err_arr[idx]
                            if not np.isfinite(A_val) or not np.isfinite(A_err_val):
                                continue
                            A_int += A_val * norm_count
                            A_int_err_sq += (A_err_val * norm_count) ** 2

                        self.A_integrated[weight_key(w)][param].append(A_int)
                        self.A_integrated_err[weight_key(w)][param].append(np.sqrt(A_int_err_sq))


        ##########
        # plotting
        ###########

        if plot_w_kin:
            for w in signal_weights:
                draw_w_kin(self, folder, w)
        if plot_A_kin:
            draw_A_kin(self, folder)
        if plot_A_LT:
            draw_A_LT(self, folder)
        return folder


#: Line colours used for the per-lifetime-bin overlays.
ROOTCOLORS = [
    ROOT.kBlue,
    ROOT.kRed,
    ROOT.kGreen + 2,
    ROOT.kMagenta,
    ROOT.kOrange + 1,
    ROOT.kCyan + 2,
    ROOT.kBlack,
    ROOT.kBrown,
]


def draw_w_kin(perf, folder, weight):
    w = weight
    outdir =  folder + 'w_vs_kin/'
    os.makedirs(outdir, exist_ok=True)
    dist_params = {k: v for k, v in perf.hist_params.items() if k!= "KS_LT"}
    colors = ROOTCOLORS

    particles = {'Pip': [], 'KS': [], 'Dp': []}
    for param in dist_params:
        particles[param.split("_")[0]].append(param)

    for pid in particles:
        if not particles[pid]:
            continue
        n_plots = len(particles[pid])
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols

        c = ROOT.TCanvas(f"c_wperf_{pid}_{perf.iteration}", pid, 800 * ncols, 700 * nrows)
        c.Divide(ncols, nrows)

        legends, stacks = [], []

        for pad_idx, param in enumerate(particles[pid], start=1):
            c.cd(pad_idx)

            leg = ROOT.TLegend(0.68, 0.55, 0.88, 0.90)
            leg.SetFillColor(ROOT.kWhite)
            leg.SetBorderSize(0)
            leg.SetTextSize(0.035)
            leg.SetTextFont(132)

            hs = ROOT.THStack(f"hs_wperf_{param}", param)


            for i in range(perf.cfg.kslt_anabins):
                pkey = f'h1D_{weight_key(w)}-{param}_ltbin{i}'
                hist = perf.hists_filled[pkey]
                integral = hist.Integral()
                if integral > 1e-13:
                    hist.Scale(1 / integral)
                else:
                    print(f'WARNING: Integral too small for {pkey} = {integral}')

                color = colors[i]
                hist.SetLineColor(color)
                hist.SetLineStyle(1)
                hist.SetLineWidth(1)

                hs.Add(hist)
                label = f"lt_bin {i}"
                leg.AddEntry(hist, label, "l")

            hs.Draw("nostack hist")
            hs.GetXaxis().SetTitle(param)
            leg.Draw()
            legends.append(leg)
            stacks.append(hs)

        c.Modified()
        c.Update()
        c.SaveAs(outdir + f"{pid}_{weight_key(w)}.pdf")


def draw_A_kin(perf, folder):
    outdir = folder + 'A_vs_kin/'
    os.makedirs(outdir, exist_ok=True)

    A_params = {k: v for k, v in perf.hist_params.items() if k not in ("Dp_M", "KS_LT")}
    particles = particle_groups(A_params)

    for pid, params in particles.items():
        if not params:
            continue
        ncols = 3
        nrows = math.ceil(len(params) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3.8 * nrows),
            constrained_layout=True, squeeze=False,
        )
        axes = axes.flatten()

        for ax, param in zip(axes, params):
            if param not in perf.A_fit_cache:
                ax.set_visible(False)
                continue
            A_arr = np.asarray(perf.A_fit_cache[param][0])
            A_err_arr = np.asarray(perf.A_fit_cache[param][1])
            x, xerr = uniform_bin_centers_widths(A_params[param])
            n = min(len(A_arr), len(x))

            ax.errorbar(x[:n], A_arr[:n], xerr=xerr[:n], yerr=A_err_arr[:n],
                        fmt="o", ms=4, capsize=3, lw=1, color="C0")
            ax.grid(alpha=0.3)
            ax.set_title(param, fontsize=11)
            ax.set_xlabel(param)
            ax.set_ylabel("A")
            ax.set_ylim(*capped_ylim(A_arr[:n], A_err_arr[:n]))

        for ax in axes[len(params):]:
            ax.set_visible(False)

        fig.suptitle(f'{pid}: asymmetry vs kinematics', fontsize=16, weight="bold")
        plt.savefig(outdir + f"{pid}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def draw_A_LT(perf, folder):
    outdir = folder + 'A_vs_LT/'
    os.makedirs(outdir, exist_ok=True)

    signal_weights = list(perf.A_integrated.keys())
    A_params = {k: v for k, v in perf.hist_params.items() if k not in ("Dp_M", "KS_LT")}
    particles = particle_groups(A_params)

    for pid, params in particles.items():
        if not params:
            continue
        ncols = 3
        nrows = math.ceil(len(params) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3.8 * nrows),
            constrained_layout=True, squeeze=False,
        )
        axes = axes.flatten()

        for ax, param in zip(axes, params):
            y_all, yerr_all = [], []
            for i, w in enumerate(signal_weights):
                prof_key = f"KS_LT_{weight_key(w)}"
                if perf.bin_x.get(prof_key) is not None:
                    x = np.asarray(perf.bin_x[prof_key])
                    xerr = perf.bin_xerr[prof_key]
                else:
                    lt_edges = np.asarray(perf.lt_bin_edges)
                    x = 0.5 * (lt_edges[:-1] + lt_edges[1:])
                    xerr = 0.5 * (lt_edges[1:] - lt_edges[:-1])

                if param not in perf.A_integrated.get(weight_key(w), {}):
                    continue
                y = np.asarray(perf.A_integrated[w][param])
                yerr = np.asarray(perf.A_integrated_err[w][param])
                label = 'before' if w == signal_weights[0] else 'after'
                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=4, capsize=3, lw=1,
                            label=label, color=f"C{i}")
                y_all.append(y)
                yerr_all.append(yerr)

            ax.grid(alpha=0.3)
            ax.set_title(param, fontsize=11)
            ax.set_xlabel(r'KS lifetime $(t/\tau)$')
            ax.set_ylabel("Integrated A")
            ax.legend(fontsize='small')
            if y_all:
                ax.set_ylim(*capped_ylim(np.concatenate(y_all), np.concatenate(yerr_all)))

        for ax in axes[len(params):]:
            ax.set_visible(False)

        fig.suptitle(f'{pid}: integrated asymmetry vs KS lifetime', fontsize=16, weight="bold")
        plt.savefig(outdir + f"{pid}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)