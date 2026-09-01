import ROOT
from ROOT import (RooRealVar, RooGaussian, RooExponential, RooAddPdf, 
    RooDataHist, RooFit, RooPlot, TCanvas, RooArgList, TLegend, RooArgSet, RooFormulaVar,
    RooBifurGauss, RooJohnson, RooProdPdf, RooGenericPdf,
    RooSimultaneous, RooCategory, RooFit, RooStats)
import io
import contextlib
import math

from apd import AnalysisData
import numpy as np
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
import time
import array
import cppyy
import sys
import mass_models as mm
from collections import defaultdict


class AsymmetryPlotter:
    """
    Shared plotting logic for 'asymmetry vs KS lifetime' figures. Used by
    Analysis.plot_A_vs_kslt (single dataset) and can be reused identically
    for the cross-dataset weighted-average plots in multi_analysis.py —
    it only draws, it never knows where A/A_err/x/xerr came from.

    Expected data shapes for every method:
      A[model][procedure]      -> array over lifetime bins
      A_err[model][procedure]  -> array over lifetime bins
      x[procedure]             -> array of bin x-positions (e.g. bin means)
      xerr[procedure]          -> (2, N) or scalar array for errorbar(xerr=...)
    """

    def __init__(self, models, procedures, xlabel=r'KS_LT $(t/\tau)$',
                 procedure_labels=None, A_bias = 0, cap_lim=1e-2, pad=0.2):
        self.models = list(models)
        self.procedures = list(procedures)
        self.xlabel = xlabel
        self.procedure_labels = procedure_labels or {}
        self.cap_lim = cap_lim
        self.pad = pad
        self.A_bias = A_bias

    @staticmethod
    def _grid_shape(n, max_cols=3):
        if n <= 0:
            return 1, 1
        if n == 4:
            return 2, 2
        ncols = min(n, max_cols)
        nrows = math.ceil(n / ncols)
        return nrows, ncols

    @staticmethod
    def _checkx(x, xerr, model, proc):
        if model in x:
            x_plt = x[model][proc]
            xerr_plt = xerr[model][proc]
        else:
            x_plt = x[proc]
            xerr_plt = xerr[proc]

        return x_plt, xerr_plt



    def _capped_ylim(self, y, yerr):
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        valid = np.isfinite(y) & np.isfinite(yerr)
        if not valid.any():
            return 0.0, 1.0
        outlier = yerr > self.cap_lim
        range_err = np.where(outlier, 0.0, yerr)
        lo = np.min((y - range_err)[valid])
        hi = np.max((y + range_err)[valid])
        rng = hi - lo if hi > lo else max(abs(lo), 1.0)
        return lo - self.pad * rng, hi + self.pad * rng

    def _label(self, proc):
        return self.procedure_labels.get(proc, proc)

    def _errorbar_series(self, ax, x, xerr, y, yerr, label, color):
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=4, capsize=3,
                    lw=1, label=self._label(label), color=color)


    def _finalize_axis(self, ax, title, ylabel=None):
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(self.xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize='small')

    def plot_grid(self, A, A_err, x, xerr, outpath, suptitle='Asymmetry vs KS lifetime'):
        fig, axes = plt.subplots(
            len(self.models), len(self.procedures),
            figsize=(5 * len(self.procedures), 3.8 * len(self.models)),
            sharex=True, constrained_layout=True, squeeze=False,
        )
        for j, model in enumerate(self.models):
            y_all = np.concatenate([(A[model][p] + self.A_bias) for p in self.procedures])
            yerr_all = np.concatenate([A_err[model][p] for p in self.procedures])
            ymin, ymax = self._capped_ylim(y_all, yerr_all)
            for i, proc in enumerate(self.procedures):
                ax = axes[j, i]
                x_plt, xerr_plt = self._checkx(x, xerr, model, proc)
                A_biased = np.asarray(A[model][proc]) + self.A_bias
                self._errorbar_series(ax, x_plt, xerr_plt, A_biased, A_err[model][proc], proc, f"C{i}")
                self._finalize_axis(ax, f"{model} ({self._label(proc)})", "A blinded" if i == 0 else None)
                ax.set_ylim(ymin, ymax)
        fig.suptitle(suptitle, fontsize=16, weight="bold")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_overlay_procedures(self, A, A_err, x, xerr, outpath,
                                 suptitle='Before vs after, per model', models=None):
        models = models or self.models
        n = len(models)
        nrows, ncols = self._grid_shape(n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                                  constrained_layout=True, squeeze=False, sharey=True)
        axes = axes.flatten()
        for j, model in enumerate(models):
            ax = axes[j]
            y_all, yerr_all = [], []
            for i, proc in enumerate(self.procedures):
                x_plt, xerr_plt = self._checkx(x, xerr, model, proc)
                A_biased = np.asarray(A[model][proc]) + self.A_bias
                self._errorbar_series(ax, x_plt, xerr_plt, A_biased, A_err[model][proc], proc, f"C{i}")
                y_all.append(A_biased)
                yerr_all.append(A_err[model][proc])
            self._finalize_axis(ax, model, "A blinded" if j % ncols == 0 else None)
            ax.set_ylim(*self._capped_ylim(np.concatenate(y_all), np.concatenate(yerr_all)))
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(suptitle, fontsize=16, weight="bold")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_overlay_models(self, A, A_err, x, xerr, outpath,
                             suptitle='Model comparison, per procedure', procedures=None):
        procedures = procedures or self.procedures
        fig, axes = plt.subplots(1, len(procedures), figsize=(5 * len(procedures), 4.2),
                                  constrained_layout=True, squeeze=False, sharey=True)
        axes = axes[0]
        for i, proc in enumerate(procedures):
            ax = axes[i]
            y_all, yerr_all = [], []
            for j, model in enumerate(self.models):
                x_plt, xerr_plt = self._checkx(x, xerr, model, proc)
                A_biased = np.asarray(A[model][proc]) + self.A_bias
                self._errorbar_series(ax, x_plt, xerr_plt, A_biased, A_err[model][proc], model, f"C{j}")
                y_all.append(A_biased)
                yerr_all.append(A_err[model][proc])
            self._finalize_axis(ax, self._label(proc), "A blinded" if i == 0 else None)
            ax.set_ylim(*self._capped_ylim(np.concatenate(y_all), np.concatenate(yerr_all)))
        fig.suptitle(suptitle, fontsize=16, weight="bold")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_single_model_procedures(self, A, A_err, x, xerr, model, outpath, suptitle=None):
        fig, ax = plt.subplots(figsize=(5, 4.2), constrained_layout=True)
        y_all, yerr_all = [], []
        for i, proc in enumerate(self.procedures):
            x_plt, xerr_plt = self._checkx(x, xerr, model, proc)
            A_biased = np.asarray(A[model][proc]) + self.A_bias
            self._errorbar_series(ax, x_plt, xerr_plt, A_biased, A_err[model][proc], proc, f"C{i}")
            y_all.append(A_biased)
            yerr_all.append(A_err[model][proc])
        self._finalize_axis(ax, model, "A blinded")
        ax.set_ylim(*self._capped_ylim(np.concatenate(y_all), np.concatenate(yerr_all)))
        fig.suptitle(suptitle or f'{model}: reweighting comparison', fontsize=14, weight="bold")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_single_procedure_models(self, A, A_err, x, xerr, procedure, outpath, suptitle=None):
        fig, ax = plt.subplots(figsize=(5, 4.2), constrained_layout=True)
        y_all, yerr_all = [], []
        for j, model in enumerate(self.models):
            x_plt, xerr_plt = self._checkx(x, xerr, model, procedure)
            A_biased = np.asarray(A[model][procedure]) + self.A_bias
            self._errorbar_series(ax, x_plt, xerr_plt, A_biased, A_err[model][procedure], model, f"C{j}")
            y_all.append(A_biased)
            yerr_all.append(A_err[model][procedure])
        self._finalize_axis(ax, self._label(procedure), "A blinded")
        ax.set_ylim(*self._capped_ylim(np.concatenate(y_all), np.concatenate(yerr_all)))
        fig.suptitle(suptitle or f'Model comparison ({self._label(procedure)})', fontsize=14, weight="bold")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_standard_set(self, A, A_err, x, xerr, outdir, available_models=None,
                           standard_model=None, after_procedure='after'):
        """Reproduces exactly the four PDFs from the original Analysis.plot_A_vs_kslt:
        check_reweighting.pdf, check_massmodels.pdf, result_reweighting.pdf,
        result_massmodels.pdf — same filenames, same titles."""
        os.makedirs(outdir, exist_ok=True)
        avail = available_models or self.models

        self.plot_overlay_procedures(
            A, A_err, x, xerr, os.path.join(outdir, "check_reweighting.pdf"),
            suptitle='A vs KS_LT: before vs after reweighting, per mass model', models=avail,
        )
        self.plot_overlay_models(
            A, A_err, x, xerr, os.path.join(outdir, "check_massmodels.pdf"),
            suptitle='A vs KS_LT: model comparison, before and after reweighting',
        )
        if standard_model is not None:
            if standard_model not in avail:
                print(f"WARNING: standard_model '{standard_model}' has no data, skipping result_reweighting.pdf")
            else:
                self.plot_single_model_procedures(
                    A, A_err, x, xerr, standard_model,
                    os.path.join(outdir, "result_reweighting.pdf"),
                    suptitle='A vs KS_LT: reweighting comparison',
                )
        if after_procedure in self.procedures:
            self.plot_single_procedure_models(
                A, A_err, x, xerr, after_procedure,
                os.path.join(outdir, "result_massmodels.pdf"),
                suptitle='A vs KS_LT: mass model comparison',
            )
        else:
            print(f"WARNING: procedure '{after_procedure}' not defined, skipping result_massmodels.pdf")


class Analysis:
    _instance_counter = 0   # class-level, shared across all instances ever created
    def __init__(self, track):
        Analysis._instance_counter += 1
        self._instance_id = Analysis._instance_counter

        if track == 'll':
            self.kslt_anabins = 6
            self.ltmin = 0.0
            self.ltmax = 0.4
            self.lt_bin_edges = [0.0, 0.070, 0.093, 0.116, 0.142, 0.176, 0.4]
        elif track == 'dd':
            self.kslt_anabins = 7
            self.ltmin = 0.0
            self.ltmax = 3.0
            self.lt_bin_edges = [0.0, 0.4686, 0.5727, 0.6518, 0.7385, 0.8259, 0.9733, 3.0]
        else:
            raise ValueError("Unexpected track: Use 'll' or 'dd'.")
        self.track = track
        self.base_folder = '/eos/user/a/ahulsber/scripts/multi_analysis/'
        self.standard_model = 'johnson_exp1'
        self.models = ['johnson_exp1', 'johnson_exp', 'johnson_tail_expo', 'johnson_gauss_exp', 'johnson_free_gauss_exp']
        self.min_m = 1795
        self.max_m = 1943 
        self.nbins_m = (self.max_m - self.min_m) * 2
        self.l_edge_sb = self.min_m
        self.r_edge_sb = self.max_m
        self.l_edge_sig = int(np.ceil((self.min_m + 1869.66) / 2))
        self.r_edge_sig = int(np.floor((self.max_m + 1869.66) / 2))

        self.threshold = 1
        self.kinvar_bins = 25

        self.mfit_iter = 0
        self.empty_bins = {}
        self.files = []
        self.rdfs = {}
        self.N_sigw0 = {}
        self._pending_acc_sum = None
        self._pending_acc_iter = None
        self.objects_keepalive = {}
        self.bin_x = {}
        self.bin_xerr = {}
        self.A_bias = np.random.RandomState(0).uniform(-1, 1)
        self.A_fit_cache = defaultdict(dict)  # A_fit_cache[weight][param][model] = (A, A_err)
        self.hists = {}
        self.hists_filled = {}
        self.profiles = {}
        self.profiles_filled = {}
        self.A_integrated = defaultdict(lambda: defaultdict(dict))
        self.A_integrated_err = defaultdict(lambda: defaultdict(dict))
        self.rootcolors = [
            ROOT.kBlue,
            ROOT.kRed,
            ROOT.kGreen + 2,
            ROOT.kMagenta,
            ROOT.kOrange + 1,
            ROOT.kCyan + 2,
            ROOT.kBlack,
            ROOT.kBrown,
        ]

        if not hasattr(ROOT, "_update_weight_3d_declared"):
            ROOT.gInterpreter.Declare(r"""
            double update_weight_3d(double x, double y, double z, double t, double w, TH3D* h3, THnD* h4)
            {
                double num = h3->GetBinContent(h3->FindBin(x, y, z));
                Int_t bins[4];
                bins[0] = h4->GetAxis(0)->FindBin(x);
                bins[1] = h4->GetAxis(1)->FindBin(y);
                bins[2] = h4->GetAxis(2)->FindBin(z);
                bins[3] = h4->GetAxis(3)->FindBin(t);
                double den = h4->GetBinContent(bins);

                if (den <= 0.0) return 0.0;
                num = std::max(num, 0.0);
                double ratio = num / den;

                const double lo = 0.2 , hi = 5 ;   // tune to taste
                if (ratio < lo) ratio = lo;
                if (ratio > hi) ratio = hi;

                return w * ratio;
            }
            """)
            ROOT._update_weight_3d_declared = True
            ROOT.TH1.SetDefaultSumw2(True)

        


    #################################
    # SETUP
    ################################3
    def import_PFNS(self, polarity, ycset, datasetnumbers, track):
        self.track = track
        if polarity == 'magup': pol = 'mu'
        else: pol = 'md'
        filepath = f"/eos/user/a/ahulsber/scripts/data/PFNS/{track}/{pol}/{ycset}_{pol}_{track}.txt"

        with open(filepath) as f:
            rows = [line.strip() for line in f]

        for i in datasetnumbers:
            if 0 <= i < len(rows):
                self.files.append(rows[i])
            else:
                raise ValueError(f"Warning: line {i} does not exist in {filepath}")



    def import_data(self, polarity, ycset, datasetnumbers, track):
        datasets = AnalysisData("charm", "d_to_ksh")
        dataname = f"d_to_ksh_{ycset}_{polarity},_split_d2kspi_{track}"
        # print(dataname)
        y = ycset.split("c")[0]
        result = datasets(
            config="lhcb",
            datatype=f"20{y}",
            filetype=f"d2kspi_{track}.root",
            polarity=polarity,
            eventtype="94000000",
            name = dataname
        )
        self.files.extend(result[i] for i in datasetnumbers)
        # for i in datasetnumbers:
        #     print(result[i])


    def data_to_rdf(self):
        if self.track == 'dd':
            self.rdf = ROOT.RDataFrame("D2KSpi_DD/DecayTree", self.files)
        elif self.track == 'll':
            self.rdf = ROOT.RDataFrame("D2KSpi_LL/DecayTree", self.files)
        else:
            raise ValueError("Wrong track inserted in data_to_rdf")
        self.files = []
        

    def defs(self):
        rdf1 = (
            self.rdf
            .Define("Pip_theta_x", "atan(Pip_PX/Pip_PZ)")
            .Define("Pip_theta_y", "atan(Pip_PY/Pip_PZ)")
            .Define("Pip_k", "1.0/sqrt(Pip_PX*Pip_PX*1e-6 + Pip_PZ*Pip_PZ*1e-6)")
            .Define("Dp_theta_x", "atan(Dp_PX/Dp_PZ)")
            .Define("Dp_theta_y", "atan(Dp_PY/Dp_PZ)")
            .Define("Dp_k", "1.0/sqrt(Dp_PX*Dp_PX*1e-6 + Dp_PZ*Dp_PZ*1e-6)")
            .Define("KS_theta_x", "atan(KS_PX/KS_PZ)")
            .Define("KS_theta_y", "atan(KS_PY/KS_PZ)")
            .Define("KS_k", "1.0/sqrt(KS_PX*KS_PX*1e-6 + KS_PZ*KS_PZ*1e-6)")
            .Define(
                "KS_FD",
                "sqrt((KS_END_VX-KS_OWNPVX)*(KS_END_VX-KS_OWNPVX)"
                " + (KS_END_VY-KS_OWNPVY)*(KS_END_VY-KS_OWNPVY)"
                " + (KS_END_VZ-KS_OWNPVZ)*(KS_END_VZ-KS_OWNPVZ))"
            )
            .Define("Dp_charge", "Pip_PARTICLE_ID > 0 ? 1.0 : -1.0")
            .Define("KS_LT", "KS_OWNPVLTIME / 0.08954" )
        )
        self.rdf = rdf1
        report = rdf1.Report()
        report.Print()
        self.N_total_events = self.rdf.Count().GetValue()
        return rdf1

 
    def cuts(self, cut=''):
        rdf1 = self.rdf

        # DpTIS filters
        trigger_cols_dd = [
            "Dp_Hlt1LowPtMuonDecision_TIS",
            "Dp_Hlt1TrackMVADecision_TIS",
            "Dp_Hlt1TrackMuonMVADecision_TIS",
            "Dp_Hlt1DiPhotonHighMassDecision_TIS",
            "Dp_Hlt1D2KshhDecision_TIS",
            "Dp_Hlt1DiElectronDisplacedDecision_TIS",
            "Dp_Hlt1TwoTrackKsDecision_TIS",
            "Dp_Hlt1KsLLDetachedTrackDecision_TIS",
            "Dp_Hlt1OneMuonTrackLineDecision_TIS",
            "Dp_Hlt1TwoTrackMVADecision_TIS",
            "Dp_Hlt1DiMuonHighMassDecision_TIS",
            "Dp_Hlt1TrackElectronMVADecision_TIS",
            "Dp_Hlt1DiMuonDisplacedDecision_TIS",
            "Pip_Hlt1TrackMVADecision_TOS",
        ]

        trigger_cols_ll = [
            "Dp_Hlt1D2KshhDecision_TIS",
            "Dp_Hlt1DiElectronDisplacedDecision_TIS",
            "Dp_Hlt1DiMuonDisplacedDecision_TIS",
            "Dp_Hlt1DiMuonHighMassDecision_TIS",
            "Dp_Hlt1DiPhotonHighMassDecision_TIS",
            "Dp_Hlt1KsLLDetachedTrackDecision_TIS",
            "Dp_Hlt1LowPtMuonDecision_TIS",
            "Dp_Hlt1OneMuonTrackLineDecision_TIS",
            "Dp_Hlt1TrackElectronMVADecision_TIS",
            "Dp_Hlt1TrackMVADecision_TIS",
            "Dp_Hlt1TrackMuonMVADecision_TIS",
            "Dp_Hlt1TwoTrackKsDecision_TIS",
            "Dp_Hlt1TwoTrackMVADecision_TIS",
        ]

        if self.track == 'll':
            trigger_filter = " || ".join(trigger_cols_ll)
        else:
            trigger_filter = " || ".join(trigger_cols_dd)


        if cut == '' or cut == 'mass_lt':
            rdf1 = (rdf1
                    .Filter(f"Dp_M > {self.min_m}", f"Dp_M > {self.min_m}")
                    .Filter(f"Dp_M < {self.max_m}", f"Dp_M < {self.max_m}")
                    .Filter(f"KS_LT > {self.ltmin}", "KS lifetime min")
                    .Filter(f"KS_LT < {self.ltmax}", "KS lifetime max")        
            )
        
        if cut == 'KS_Hlt1TwoTrackKsDecision_TOS':
            rdf1 = rdf1.Filter("KS_Hlt1TwoTrackKsDecision_TOS", "HLT1: KS TwoTrackKsDecision_TOS")
        if cut == 'Pip_Hlt1TrackMVADecision_TOS':
            rdf1 = rdf1.Filter("Pip_Hlt1TrackMVADecision_TOS", "HLT1: Pip_Hlt1TrackMVADecision_TOS")
        if cut == 'DpTIS':
            rdf1 = rdf1.Filter(trigger_filter, "HLT1 trigger selection")
        if cut == 'DpTIS_PipMVATOS':
            rdf1 = rdf1.Filter(f'{trigger_filter}', "DpTIS HLT1 trigger selection")
            rdf1 = rdf1.Filter(f'Pip_Hlt1TrackMVADecision_TOS', "PipTOS HLT1 trigger selection")

        if cut == 'DpTIS_KSTwoTracks':
            rdf1 = rdf1.Filter(f'{trigger_filter}', "DpTIS HLT1 trigger selection")
            rdf1 = rdf1.Filter(f'KS_Hlt1TwoTrackKsDecision_TOS', "KsTOS HLT1 trigger selection")


        if cut == '' or cut == 'kin':
            if self.track == 'll':
                rdf1 = (rdf1
                    .Filter("Dp_OWNPVIP < 1", "Dp IP")
                    .Filter("KS_FD > 20", "KS flight distance")
                    .Filter("abs(KS_M - 497.611) < 10", "KS mass consistency")
                )

            if self.track == 'dd':
                rdf1 = (rdf1
                    .Filter("Dp_OWNPVIP < 3.5", "Dp IP")
                    .Filter("abs(KS_M - 497.611) < 20", "KS mass consistency")
                )
 

            rdf1 = (
                rdf1
                # D+
                .Filter("Dp_HLT2_ETA > 2.2", "Dp eta min")
                .Filter("Dp_HLT2_ETA < 4.2", "Dp eta max")
                .Filter("Dp_HLT2_PT > 2800", "Dp PT min") 
                .Filter("Dp_HLT2_PT < 12000", "Dp PT max")
                .Filter("Dp_k > 0.01", "Dp k min")
                .Filter("Dp_k < 0.12", "Dp k max")

                # bachelor pion geometry
                .Filter("Pip_k < (0.3 - abs(Pip_theta_x))", "Pip acceptance")
                .Filter("pow(Pip_theta_x/0.027,2) + pow(Pip_theta_y/0.017,2) > 1", "Beam ellipse")
                .Filter("abs(Pip_theta_y) > 0.001", "Pip_theta_y min")
                .Filter(
                    "!(abs(Pip_theta_y) < 0.005 && abs(Pip_theta_x) > 0.06 && abs(Pip_theta_x) < 0.1)",
                    "Dead region"
                )

                # bachelor pion kinematics
                .Filter("Pip_HLT2_PT > 1500", "Pip PT min")
                .Filter("Pip_HLT2_PT < 6000", "Pip PT max")
                .Filter("Pip_HLT2_ETA > 2.2", "Pip eta min")
                .Filter("Pip_HLT2_ETA < 4.2", "Pip eta max")
                .Filter("Pip_k > 0.005", "Pip k min")
                .Filter("Pip_k < 0.06", "Pip k max")

                # KS
                .Filter("KS_HLT2_ETA > 2.2", "KS eta min")
                .Filter("KS_HLT2_ETA < 4.2", "KS eta max")
                .Filter("KS_FD > 20", "KS flight distance")
                .Filter("abs(KS_M - 497.611) < 10", "KS mass consistency")


            )

        if cut == '' or cut == 'probnn':
            rdf1 = (rdf1
                .Filter("KSpip_PROBNN_PI > 0.5", "KSpip probnn > 0.5")
                .Filter("KSpim_PROBNN_PI > 0.5", "KSpim probnn > 0.5")
                .Filter("Pip_PROBNN_PI > 0.5", "Pip probnn > 0.5")

                .Filter("KSpip_PROBNN_GHOST < 0.5", "KSpip_PROBNN_GHOST < 0.5")
                .Filter("KSpim_PROBNN_GHOST < 0.5", "KSpim_PROBNN_GHOST < 0.5")
                .Filter("Pip_PROBNN_GHOST < 0.5", "Pip_PROBNN_GHOST < 0.5")
            )
                

        # in cuts()/cuts_DD(): fuse Report() and Count() into one pass instead of two
        self.rdf = rdf1
        report = rdf1.Report()
        count = rdf1.Count()
        self.N_total_events = count.GetValue()   # triggers once, computing Report too
        report.Print()                            # free now

        return rdf1

    def init_hist_params(self, kslt_plotbins = 100):
        self.hist_params = {
            "Dp_M":             [self.nbins_m, self.min_m, self.max_m],
            "KS_LT":            [kslt_plotbins, self.ltmin, self.ltmax],

            "Pip_k":            [self.kinvar_bins, 0.01, 0.06],
            "Pip_theta_x":      [self.kinvar_bins, -0.2, 0.2], 
            "Pip_theta_y":      [self.kinvar_bins, -0.2, 0.2], 
            "Pip_HLT2_PHI":     [self.kinvar_bins, -3.2,  3.2], 
            "Pip_HLT2_ETA":     [self.kinvar_bins, 2.2,   4.2],  
            "Pip_HLT2_PT":      [self.kinvar_bins, 1500, 6000 ],  

            "Dp_HLT2_ETA":      [self.kinvar_bins, 2.2,   4.2], 
            "Dp_HLT2_PT":       [self.kinvar_bins, 2800,  12000], 
            "Dp_theta_x":       [self.kinvar_bins, -0.2, 0.2], 
            "Dp_theta_y":       [self.kinvar_bins, -0.2, 0.2],
            "Dp_k":             [self.kinvar_bins, 0.005, 0.04],

            "KS_k":            [self.kinvar_bins, 0.0, 0.14],
            "KS_theta_x":      [self.kinvar_bins, -0.15, 0.15], 
            "KS_theta_y":      [self.kinvar_bins, -0.15, 0.15], 
            "KS_HLT2_PHI":     [self.kinvar_bins, -3.2,  3.2], 
            "KS_HLT2_ETA":     [self.kinvar_bins, 2.2,   4.2],  
            "KS_HLT2_PT":      [self.kinvar_bins, 0, 6000 ],  


        }

    #############################################
    # Initial weights
    #############################################

    def init_massfit(self, folder):
        keys = ["Dp_M_p", 'Dp_M_m', "Dp_M"]
        for key in keys:
            if key.endswith("_p"):
                column = key[:-2]
                rdf = self.rdf.Filter("Dp_charge > 0")
            elif key.endswith("_m"):
                column = key[:-2]
                rdf = self.rdf.Filter("Dp_charge < 0")
            else:
                column = key
                rdf = self.rdf

            self.hists[key] = rdf.Histo1D(
                (f"h_{key}", key, *self.hist_params[column]),
                column,
            )

        # fill the hists
        for key in keys:
            self.hists_filled[key] = self.hists[key].GetValue()

        h_p = self.hists_filled['Dp_M_p']
        h_m = self.hists_filled['Dp_M_m']

        # massfit
        self.mass_fit(h_p, h_m, folder, self.standard_model)


    def init_weights(self):
        self.iteration = 0
        
        massfit_folder = self.base_folder + 'initial_massfit/'
        self.init_massfit(massfit_folder)

        # finding the range
        f = ROOT.TFile.Open(massfit_folder + "model.root")
        ws = f.Get("ws")
        params = ws.allVars()
        m = ws.var("Dp_M")
        bkg_tot = ws.pdf("bkg_tot")
        m.setRange("leftSB",  self.min_m, self.l_edge_sig)
        m.setRange("rightSB", self.r_edge_sig, self.max_m)
        m.setRange("sig_region", self.l_edge_sig, self.r_edge_sig)

        # calculating the Nbkg_sb and Nbkg_sig
        N_bkg_sb = bkg_tot.createIntegral(ROOT.RooArgSet(m), ROOT.RooFit.Range("leftSB")).getVal() + bkg_tot.createIntegral(ROOT.RooArgSet(m), ROOT.RooFit.Range("rightSB")).getVal()
        N_bkg_sig = bkg_tot.createIntegral(ROOT.RooArgSet(m), ROOT.RooFit.Range("sig_region")).getVal()

        print(f'N_bkg_sig / N_bkg_sb from fit = {N_bkg_sig / N_bkg_sb}')
        print(f'N_bkg_sig / N_bkg_sb assuming constant bkg = {(self.r_edge_sig - self.l_edge_sig) / ((self.l_edge_sig - self.min_m) + (self.max_m - self.r_edge_sig))}')
      

        self.rdf = (self.rdf.Filter(f"Dp_M > {self.min_m}", f"Dp_M > {self.min_m}")
            .Filter(f"Dp_M < {self.max_m}", f"Dp_M < {self.max_m}"))

        self.rdf = (self.rdf
                .Define(
                    f"weight_hist_{self.iteration}",
                    f"""
                    ((Dp_M >= {self.l_edge_sig} && Dp_M <= {self.r_edge_sig}) ? 1.0 :
                    ((Dp_M >= {self.l_edge_sb} && Dp_M < {self.l_edge_sig}) ||
                    (Dp_M > {self.r_edge_sig} && Dp_M <= {self.r_edge_sb}))
                        ? {-N_bkg_sig / N_bkg_sb}
                        : 0.0)
                    """)
                .Define(f"weight_acc_{self.iteration}", "1")
                .Define(
                    f"weight_{self.iteration}",
                    f"weight_hist_{self.iteration} * weight_acc_{self.iteration}"
                )
                .Define(f"w2_{self.iteration}", f"weight_{self.iteration} * weight_{self.iteration}")
                .Define(f"weight_eff_{self.iteration}", f"weight_0 * weight_acc_{self.iteration}")
        )
        f.Close()
        self.plot_massfit(massfit_folder)
        return 



    def equal_bins(self, hist=None, bins=None):
        if hist is None:
            hist = self.rdf.Histo1D(
                ("tmp", "tmp", 1000,
                self.hist_params["KS_LT"][1],
                self.hist_params["KS_LT"][2]),
                "KS_LT",
                "weight_0"
            ).GetValue()

        if bins is None:
            bins = self.kslt_anabins
        probs = ROOT.std.vector("double")()
        for i in range(bins + 1):
            probs.push_back(i / bins)

        quant = ROOT.std.vector("double")(bins + 1)

        hist.GetQuantiles(
            bins + 1,
            quant.data(),
            probs.data()
        )

        bin_edges = [quant[i] for i in range(bins + 1)]
        return bin_edges

    ######################################
    # mass fit
    ###################################
    def mass_fit(
        self,
        h_plus,                          # TH1D: positive candidates
        h_minus,                         # TH1D: negative candidates
        folder,                          # result folder
        model_name = 'johnson_exp',
        new_run = False,           # bool: if True, ignore existing param file

    ):

        self.mfit_iter += 1
        params_file_in  = folder + "_in.txt"
        params_file_out = folder + "_out.txt"
        model_file_out = folder + "model.root"
        os.makedirs(folder, exist_ok=True)
        fit_converges = True

        xaxis = h_plus.GetXaxis()

        xmin = xaxis.GetXmin()
        xmax = xaxis.GetXmax()
        nbins = xaxis.GetNbins()

        m = RooRealVar("Dp_M", "Dp_M", xmin, xmax, "MeV/c^{2}")
        m.setBins(nbins)

        data_p   = RooDataHist("data_p",   "Dp_M tag +", RooArgList(m), ROOT.RooFit.Import(h_plus))
        data_m   = RooDataHist("data_m",   "Dp_M tag -", RooArgList(m), ROOT.RooFit.Import(h_minus))
        h_tot    = h_plus.Clone("h_tot")
        h_tot.Add(h_minus)
        data_tot = RooDataHist("data_tot", "Dp_M", RooArgList(m), ROOT.RooFit.Import(h_tot))
        N_tot = h_tot.Integral()



        model = getattr(mm, model_name)

        mm_result = model(m)

        signal_plus = mm_result['sp']
        signal_minus = mm_result['sm']
        background_plus = mm_result['bp']
        background_minus = mm_result['bm']
        self.objects_keepalive[self.mfit_iter] = mm_result['objects']

        # Yields and asymmetries
        N_tot = RooRealVar("N_tot", "Total events",     N_tot, 0., 1.5*(N_tot))
        N_tot.setConstant(True)
        f_sig = RooRealVar("f_sig", "Signal Fraction", 0.5, 0., 1.)
        N_sig = RooFormulaVar("N_sig", "@0*@1", RooArgList(N_tot, f_sig))
        N_bkg = RooFormulaVar("N_bkg", "@0*(1-@1)", RooArgList(N_tot, f_sig))
        A_sig = RooRealVar("A_sig", "Signal Asymmetry", 0., -1., 1.)
        A_bkg = RooRealVar("A_bkg", "Background asymmetry",   0., -1., 1.)

        Y_sig_plus  = RooFormulaVar("Y_sig_plus",  "@0*0.5*(1+@1)", RooArgList(N_sig, A_sig))
        Y_sig_minus = RooFormulaVar("Y_sig_minus", "@0*0.5*(1-@1)", RooArgList(N_sig, A_sig))
        Y_bkg_plus  = RooFormulaVar("Y_bkg_plus",  "@0*0.5*(1+@1)", RooArgList(N_bkg, A_bkg))
        Y_bkg_minus = RooFormulaVar("Y_bkg_minus", "@0*0.5*(1-@1)", RooArgList(N_bkg, A_bkg))

        # model 1
        pdf_positive = RooAddPdf(
            "pdf_positive", "pdf_positive",
            RooArgList(signal_plus, background_plus),
            RooArgList(Y_sig_plus, Y_bkg_plus)
        )
        pdf_negative = RooAddPdf(
            "pdf_negative", "pdf_negative",
            RooArgList(signal_minus, background_minus),
            RooArgList(Y_sig_minus, Y_bkg_minus)
        )


        tag = RooCategory("Dp_charge", "Tag")
        tag.defineType("positive",  1)
        tag.defineType("negative", -1)

        obs = RooArgSet(m, tag)

        combData = RooDataHist("combData", "Combined data", RooArgList(m),
                            ROOT.RooFit.Index(tag),
                            ROOT.RooFit.Import("positive", data_p),
                            ROOT.RooFit.Import("negative", data_m))


        model_tot = RooSimultaneous("model_tot", "model_tot", tag)
        model_tot.addPdf(pdf_positive, "positive")
        model_tot.addPdf(pdf_negative, "negative")

        # ------------------------------------------------------------------
        # Fit
        # ------------------------------------------------------------------
        fit_args = [
            ROOT.RooFit.Extended(True),
            ROOT.RooFit.Save(True),
            ROOT.RooFit.EvalBackend("legacy"),
            ROOT.RooFit.SumW2Error(True),
            ROOT.RooFit.Offset(True),   # NEW — improves numerical conditioning of the NLL for large N
        ]

        params = model_tot.getParameters(obs)

        def construct_plot_objects():
            f_sig_plus = RooFormulaVar("f_sig_plus", "0.5*(1+@0)", RooArgList(A_sig))
            f_bkg_plus = RooFormulaVar("f_bkg_plus", "0.5*(1+@0)", RooArgList(A_bkg))

            signal_tot = RooAddPdf(
                "signal_tot", "Total signal (for plotting)",
                RooArgList(signal_plus, signal_minus),
                RooArgList(f_sig_plus)
            )
            bkg_tot = RooAddPdf(
                "bkg_tot", "Total background (for plotting)",
                RooArgList(background_plus, background_minus),
                RooArgList(f_bkg_plus)
            )

            # NEW: plain extended total-model pdf, only for the unsliced "total" plot
            model_tot_plot = RooAddPdf(
                "model_tot_plot", "Total model (for plotting)",
                RooArgList(signal_tot, bkg_tot),
                RooArgList(N_sig, N_bkg)
            )

            signal_tot._f_sig_plus_keepalive = f_sig_plus
            bkg_tot._f_bkg_plus_keepalive = f_bkg_plus
            model_tot_plot._signal_tot_keepalive = signal_tot
            model_tot_plot._bkg_tot_keepalive = bkg_tot

            return signal_tot, bkg_tot, model_tot_plot



        def save_ws(results):
            signal_tot, bkg_tot, model_tot_plot = construct_plot_objects()
            ws = ROOT.RooWorkspace("ws")

            for pdf in (model_tot, bkg_tot, signal_tot, model_tot_plot):
                getattr(ws, "import")(pdf, ROOT.RooFit.RecycleConflictNodes())

            for data in (combData, data_p, data_m, data_tot):
                getattr(ws, "import")(data)

            ws.writeToFile(model_file_out)

            outfile = ROOT.TFile(model_file_out, "UPDATE")
            outfile.cd()
            h_plus.Write("h_plus")
            h_minus.Write("h_minus")

            # ---- fit report ----
            report_path = folder + "_fit_report.txt"

            params = results.floatParsFinal()
            names = [params.at(i).GetName() for i in range(params.getSize())]

            col_width = 25

            with open(report_path, "w") as f:

                # ============================================================
                # Fit summary
                # ============================================================
                f.write("============================================================\n")
                f.write("FIT RESULT\n")
                f.write("============================================================\n\n")

                f.write(f"Minimized FCN value: {results.minNll():.6f}\n")
                f.write(f"EDM:                 {results.edm():.6e}\n")
                f.write(f"Fit status:           {results.status()}\n")
                f.write(f"Covariance quality:   {results.covQual()}\n")

                # ============================================================
                # Floating parameters
                # ============================================================
                f.write("\n--- Floating Parameters ---\n\n")

                f.write(
                    f"{'Parameter':30s}"
                    f"{'Value':>18s}"
                    f"{'Error':>18s}\n"
                )

                f.write("-" * 70 + "\n")

                for i in range(params.getSize()):
                    par = params.at(i)

                    f.write(
                        f"{par.GetName():30s}"
                        f"{par.getVal():18.8e}"
                        f"{par.getError():18.8e}\n"
                    )

                # ============================================================
                # Parameter ordering
                # ============================================================
                f.write("\n--- Parameter ordering ---\n\n")

                for i, name in enumerate(names):
                    f.write(f"{i:3d}: {name}\n")

                # ============================================================
                # Covariance Matrix
                # ============================================================
                f.write("\n--- Covariance Matrix ---\n\n")

                f.write(f"{'':25s}")

                for name in names:
                    f.write(f"{name:{col_width}s}")

                f.write("\n")

                cov = results.covarianceMatrix()

                for i, name in enumerate(names):

                    f.write(f"{name:25s}")

                    for j in range(len(names)):
                        f.write(f"{cov[i][j]:{col_width}.6e}")

                    f.write("\n")

                # ============================================================
                # Correlation Matrix
                # ============================================================
                f.write("\n--- Correlation Matrix ---\n\n")

                f.write(f"{'':25s}")

                for name in names:
                    f.write(f"{name:{col_width}s}")

                f.write("\n")

                corr = results.correlationMatrix()

                for i, name in enumerate(names):

                    f.write(f"{name:25s}")

                    for j in range(len(names)):
                        f.write(f"{corr[i][j]:{col_width}.6f}")

                    f.write("\n")


            # ================================================================
            # Save the RooFitResult into the ROOT output file
            # ================================================================
            results.Write("fit_results")

            outfile.Close()


        def do_fit(strategy):
            args = fit_args + [ROOT.RooFit.Strategy(strategy)]
            return model_tot.fitTo(combData, *args)


        def is_good(res):
            """Full convergence: RooFit's strictest, most trustworthy tier."""
            return (res.status() == 0 and res.covQual() == 3
                    and np.isfinite(res.minNll()) and np.isfinite(res.edm()))


        def is_acceptable(res):
            """Usable but not ideal — accepted only as a fallback if nothing
            reaches is_good() in the attempt budget."""
            return (res.status() == 0 and res.covQual() >= 2
                    and np.isfinite(res.minNll()) and np.isfinite(res.edm()))


        def better_result(new, old):
            """Tiered comparison: status, then covQual, then NLL, then EDM.
            Used both to track the best-so-far attempt and to decide what to
            jitter from next."""
            if old is None:
                return True
            new_ok, old_ok = new.status() == 0, old.status() == 0
            if new_ok != old_ok:
                return new_ok
            if new.covQual() != old.covQual():
                return new.covQual() > old.covQual()
            new_nll, old_nll = new.minNll(), old.minNll()
            if np.isfinite(new_nll) and np.isfinite(old_nll) and not np.isclose(new_nll, old_nll):
                return new_nll < old_nll
            return new.edm() < old.edm()

        def jitter_floating_params(rng, sigma_mult, max_frac_of_range=0.15):
            for var in params:
                if not isinstance(var, RooRealVar) or var.isConstant():
                    continue
                lo, hi = var.getMin(), var.getMax()
                span = hi - lo
                sigma = var.getError()
                # Don't trust a missing OR already-inflated error as a jitter scale —
                # a degenerate fit's blown-up error would otherwise feed a runaway
                # jitter on the next attempt, walking the parameter to its boundary.
                if not np.isfinite(sigma) or sigma <= 0 or sigma > max_frac_of_range * span:
                    sigma = 0.05 * span
                step_sigma = min(sigma_mult * sigma, max_frac_of_range * span)
                new_val = var.getVal() + rng.normal(0, step_sigma)
                var.setVal(float(np.clip(new_val, lo, hi)))


        MAX_ATTEMPTS = 8
        seed = 1
        rng = np.random.default_rng(seed)

        if os.path.exists(params_file_out) and not new_run:
            params.readFromFile(params_file_out)

        initial_params = params.snapshot()   # attempt 0's controlled starting point (warm-started, if available)

        best_results = None
        best_params = None

        for attempt in range(MAX_ATTEMPTS):
            if attempt == 0:
                params.assign(initial_params)
            elif best_params is not None:
                params.assign(best_params)
                sigma_mult = 1.0 + 0.75 * (attempt - 1)   # widen the jump the longer it takes
                jitter_floating_params(rng, sigma_mult)
            else:
                # nothing usable found yet even once — restart from the controlled
                # point with a mild jitter rather than compounding on a bad state
                params.assign(initial_params)
                jitter_floating_params(rng, 0.5)

            strategy = 1 if attempt < 2 else 2   # cheap first two tries, escalate only if needed
            results = do_fit(strategy)
            current_params = params.snapshot()

            print(f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: status={results.status()}, "
                f"covQual={results.covQual()}, EDM={results.edm():.3g}, "
                f"NLL={results.minNll():.6g}, strategy={strategy}")

            if better_result(results, best_results):
                best_results = results
                best_params = current_params

            if is_good(results):
                print(f"Converged (covQual=3) after {attempt + 1} attempt(s).")
                break
        else:
            print(f"WARNING: did not reach full convergence in {MAX_ATTEMPTS} attempts; "
                f"using best result found (status={best_results.status()}, "
                f"covQual={best_results.covQual()}, edm={best_results.edm():.3g}).")

        results = best_results
        fit_converges = is_good(results)

        # restore the model's LIVE parameters to the best snapshot found — fitTo
        # leaves them at whatever the *last* attempt produced, which isn't
        # necessarily best_results if a later jitter made things worse. Everything
        # below (save_ws, boundary checks, N_sig.getVal(), A_sig.getError()) reads
        # live parameter state, so this must happen before any of that runs.
        params.assign(best_params)

        if fit_converges:
            params.writeToFile(params_file_out)   # only ever persist a fully-converged snapshot
        elif is_acceptable(results):
            print("WARNING: best result is only partially acceptable (covQual < 3); "
                "not persisting to warm-start file.")
        else:
            print("WARNING: no acceptable fit found in any attempt; not persisting to warm-start file.")

        print("h_tot entries:", h_tot.GetEntries())
        print("data_tot.sumEntries():", data_tot.sumEntries())
        print("combData.sumEntries():", combData.sumEntries())
        print("N_tot (fixed):", N_tot.getVal())
        print("N_sig fitted:", N_sig.getVal())
        print("N_bkg fitted:", N_bkg.getVal())
        print("N_sig + N_bkg:", N_sig.getVal() + N_bkg.getVal())
        save_ws(results)

        if not fit_converges:
            self.plot_massfit(folder)

        # Warn if parameters near boundaries
        for var in params:
            if isinstance(var, RooRealVar) and not var.isConstant():
                val, err = var.getVal(), var.getError()
                if err > 0:
                    if abs(val - var.getMin()) / err < 3 or abs(val - var.getMax()) / err < 3:
                        print(f"WARNING: {var.GetName()} = {val:.4f} +/- {err:.4f} "
                            f"in [{var.getMin():.4f}, {var.getMax():.4f}] — close to boundary!")

        Nsig = N_sig.getVal()
        if Nsig > 1e-7:
            print(f"1/sqrt(N) = {(1.0/Nsig)**0.5:.6f}")
        else:
            print(f"Warning: nsig too small: {Nsig}")
        print(f"A_sig_err = {A_sig.getError():.6f}\n")

        return fit_converges
        
    ########################################
    # weighting cycle
    ########################################
    def weighting_cycle(
        self,
        param1,
        param2,
        param3
    ):
        self.iteration += 1
        

        print(
            f"Starting iteration {self.iteration}: "
            f"{param1}, {param2}, {param3}"
        )

        def make_edges(params):
            nbins, lo, hi = params
            return array.array(
                "d",
                np.linspace(lo,hi,nbins+1)
            )

        name = ( f"h4d_{param1}_{param2}_{param3}_{self.iteration}")
        name2 = name + '_unw'


        xedges = make_edges(self.hist_params[param1])
        yedges = make_edges(self.hist_params[param2])
        zedges = make_edges(self.hist_params[param3])
        tedges = array.array("d", self.lt_bin_edges)

        model = ROOT.RDF.THnDModel(
            name,
            name,
            4,
            [
                len(xedges)-1,
                len(yedges)-1,
                len(zedges)-1,
                len(tedges)-1
            ],
            [
                xedges,
                yedges,
                zedges,
                tedges
            ]
        )
        model2 = ROOT.RDF.THnDModel(
            name2,
            name2,
            4,
            [
                len(xedges)-1,
                len(yedges)-1,
                len(zedges)-1,
                len(tedges)-1
            ],
            [
                xedges,
                yedges,
                zedges,
                tedges
            ]
        )

        result = self.rdf.HistoND(
            model,
            [
                param1,
                param2,
                param3,
                "KS_LT"
            ],
            f"weight_{self.iteration - 1}"
        )

        self.rdf = self.rdf.Define(f"test_weight_{self.iteration}", f"weight_acc_{self.iteration-1} * weight_0")        
        result2 = self.rdf.HistoND(
            model2,
            [
                param1,
                param2,
                param3,
                "KS_LT"
            ],
            f"test_weight_{self.iteration}" 

        )

        self.hists[name] = result
        self.hists[name2] = result2
        h4d = self.hists[name].GetValue()
        h4d_unw = self.hists[name2].GetValue()
        if self._pending_acc_sum is not None:
            N = self._pending_acc_sum.GetValue()
            print(f'[iter {self._pending_acc_iter}] accepted events = {N}, fraction = {N/self.N_total_events}')
            self._pending_acc_sum = None

        h4d = self.normalize_th4d(h4d, h4d_unw, param1, param2, param3)

        del h4d_unw                 # local, would fall out of scope anyway — explicit for clarity
        self._purge([name2])        # the *_unw RResultPtr — safe, nothing else points at it

        h3d = h4d.Projection(0, 1, 2)

        h3d.Scale(1/self.kslt_anabins)

        self.hists_filled[f"{name}"] = h4d
        self.hists_filled[f"{name}_h3d"] = h3d

        address3d = cppyy.addressof(h3d)
        address4d = cppyy.addressof(h4d)
        self.rdf = self.rdf.Define(
            f"weight_hist_{self.iteration}",

            f"""
            update_weight_3d(
                {param1},
                {param2},
                {param3},
                KS_LT,
                weight_{self.iteration-1},
                (TH3D*){address3d},
                (THnD*){address4d}
            )
            """
        )
        self.rdf = (self.rdf.Define(
                f"weight_{self.iteration}",
                f"weight_hist_{self.iteration} * weight_acc_{self.iteration}"
            )
            .Define(f"w2_{self.iteration}", f"weight_{self.iteration} * weight_{self.iteration}")
            .Define(f"weight_eff_{self.iteration}", f"weight_0 * weight_acc_{self.iteration}"))
        print(f"Column weight_{self.iteration} is created." )
        

    def normalize_th4d(self, h4d, h4d_unw, param1, param2, param3):
        h = h4d
        h2 = h4d_unw
        new_empty_bins = []
        key = f"{param1}_{param2}_{param3}"
        self.empty_bins.setdefault(key, set())

        axis0 = h.GetAxis(0)
        axis1 = h.GetAxis(1)
        axis2 = h.GetAxis(2)
        axis3 = h.GetAxis(3)

        n0 = axis0.GetNbins()
        n1 = axis1.GetNbins()
        n2 = axis2.GetNbins()
        n3 = axis3.GetNbins()
        n_bins = n0 * n1 * n2
        coord = array.array('i', [0, 0, 0, 0])
        axis3_tots = np.zeros(n3)
        axis3_totsw0 = np.zeros(n3)
        axis3_thresholds = np.zeros(n3)

        GetBinContent = h.GetBinContent
        SetBinContent = h.SetBinContent
        GetBinError = h.GetBinError
        SetBinError = h.SetBinError

        GetBinContent2 = h2.GetBinContent

        # # determine thresholds
        # for i3 in range(1, n3 + 1):
        #     axis3.SetRange(i3, i3)
        #     h3 = h.Projection(0, 1, 2)
        #     total = h3.Integral()
        #     axis3_thresholds[i3 - 1] = (total / n_bins) * self.threshold
        #     h3.Delete()

        # empty bins and find integral
        for i0 in range(1, n0 + 1):
            coord[0] = i0
            for i1 in range(1, n1 + 1):
                coord[1] = i1
                for i2 in range(1, n2 + 1):
                    coord[2] = i2
                    is_empty = False
                    for i3 in range(1, n3 + 1):
                        coord[3] = i3
                        content = GetBinContent2(coord)
                        if content <= self.threshold:
                            is_empty = True
                            break
                    if is_empty:
                        coord3d = (i0, i1, i2)
                        if coord3d not in self.empty_bins[f'{param1}_{param2}_{param3}']:
                            new_empty_bins.append(coord3d)
                            for i3 in range(1, n3 + 1):
                                coord[3] = i3
                                SetBinContent(coord, 0.)
                                SetBinError(coord, 0.)
                        continue
                    for i3 in range(1, n3 + 1):
                        coord[3] = i3
                        axis3_tots[i3 - 1] += GetBinContent(coord)
                        axis3_totsw0[i3-1] += GetBinContent2(coord)

        for i3 in range(1, n3 + 1):
            self.N_sigw0[f'{self.iteration}_{i3}'] = axis3_totsw0[i3-1]
    
        # normalize. note that the histogram is not perfectly normalized: 
        # To do this a new threshold should be calculated, 
        # but then also a new bin rejection should be applied and you end up in an infinite loop
        for i3 in range(1, n3 + 1):
            if axis3_tots[i3 - 1] > 1e-13:
                inv_total = 1.0 / axis3_tots[i3 - 1]
            else:
                print(f'WARNING: Integral too small')
                inv_total = 0
            print(f'Integral h4d ({param1}_{param2}_{param3}) ltbin {i3} = {axis3_tots[i3 - 1]}')
            for i0 in range(1, n0 + 1):
                coord[0] = i0
                for i1 in range(1, n1 + 1):
                    coord[1] = i1
                    for i2 in range(1, n2 + 1):
                        coord[2] = i2
                        coord[3] = i3
                        content = GetBinContent(coord)
                        error = GetBinError(coord)
                        SetBinContent(coord, content * inv_total)
                        SetBinError(coord, error * inv_total)

        axis3.SetRange()  # reset range back to full, otherwise it stays restricted
    
        self.empty_bins[f'{param1}_{param2}_{param3}'].update(new_empty_bins)
        self.apply_weight_acc(axis0, axis1, axis2, new_empty_bins, param1, param2, param3)

        print(f'emptied events in {len(new_empty_bins)} / {n0*n1*n2} bins')
        self._pending_acc_sum = self.rdf.Sum(f"weight_acc_{self.iteration}")   # booked, not triggered
        self._pending_acc_iter = self.iteration

        return h


    def apply_weight_acc(self, axis0, axis1, axis2, empty_bins, param1, param2, param3):

        n1 = axis1.GetNbins()
        n2 = axis2.GetNbins()

        ns = f"weightacc_{self._instance_id}_{self.iteration}"

        for name, ax in (("axis0", axis0), ("axis1", axis1), ("axis2", axis2)):
            try:
                _ = ax.GetNbins()
            except Exception as e:
                raise RuntimeError(
                    f"instance {self._instance_id}, iteration {self.iteration}: "
                    f"{name} appears invalid before assignment into {ns} — {e}"
                )
        ROOT.gInterpreter.Declare(f"""
        #include <unordered_set>
        namespace {ns} {{
            std::unordered_set<long long> empty_flat;
            TAxis* axis0 = nullptr;
            TAxis* axis1 = nullptr;
            TAxis* axis2 = nullptr;
            int n1 = 0, n2 = 0;

            inline double factor(double x, double y, double z) {{
                int i0 = axis0->FindBin(x);
                int i1 = axis1->FindBin(y);
                int i2 = axis2->FindBin(z);
                long long flat = ((long long)i0 * (n1 + 2) + i1) * (n2 + 2) + i2;
                return empty_flat.count(flat) ? 0.0 : 1.0;
            }}
        }}
        """)

        ns_obj = getattr(ROOT, ns)
        ns_obj.axis0 = axis0
        ns_obj.axis1 = axis1
        ns_obj.axis2 = axis2
        ns_obj.n1 = n1
        ns_obj.n2 = n2

        for i0, i1, i2 in empty_bins:
            flat = (i0 * (n1 + 2) + i1) * (n2 + 2) + i2
            ns_obj.empty_flat.insert(flat)

        # first call: initialize the column
        self.rdf = self.rdf.Define(
            f"weight_acc_{self.iteration}",
            f"weight_acc_{self.iteration - 1} * {ns}::factor({param1}, {param2}, {param3})"
        )



    def ltbin_means(self, weights=None):
        param = 'KS_LT'
        tedges = array.array('d', [float(e) for e in self.lt_bin_edges])

        if weights == None:
            weights = ['', 'weight_0', f'weight_{self.iteration}', 'final_weight']

        for w in weights:
            profile_model = ROOT.RDF.TProfile1DModel(f'ltmean_{self._weight_key(w)}', f'ltmean_{self._weight_key(w)}', len(tedges) - 1, tedges)
            if w == '':
                self.profiles[self._weight_key(w)] = self.rdf.Profile1D(profile_model, param, param)
            else:
                self.profiles[self._weight_key(w)] = self.rdf.Profile1D(
                    profile_model,
                    param,
                    param,
                    w
                )

        for w in weights:
            self.profiles_filled[self._weight_key(w)] = self.profiles[self._weight_key(w)].GetValue()

        for w in weights:
            prof = self.profiles_filled[self._weight_key(w)]
            centers = [prof.GetBinContent(i) for i in range(1, prof.GetNbinsX() + 1)]
            self.bin_x[f"{param}_{self._weight_key(w)}"] = centers
            self.bin_xerr[f"{param}_{self._weight_key(w)}"] = self._asymmetric_xerr(tedges, centers)

        

            
    def _purge(self, keys):
        for key in keys:
            self.hists.pop(key, None)
            self.hists_filled.pop(key, None)





    ############################
    # PLOTTING
    ############################

    def plot_massfit(
        self,
        folder,
    ):

        model_file_out = folder + "model.root"
        output_path = folder + "figures/"
        os.makedirs(output_path, exist_ok=True)

        margin = 0.05

        # ------------------------------------------------------------------
        # Load file + workspace
        # ------------------------------------------------------------------
        f = ROOT.TFile.Open(model_file_out)

        ws = f.Get("ws")

        model_tot = ws.pdf("model_tot")
        model_tot_plot = ws.pdf("model_tot_plot")
        combData  = ws.data("combData")
        data_p    = ws.data("data_p")
        data_m    = ws.data("data_m")
        data_tot  = ws.data("data_tot")
        
        if not model_tot_plot:
            raise RuntimeError("model_tot_plot not found in workspace — was save_ws() updated to import it?")

        h_plus = f.Get("h_plus")
        h_minus = f.Get("h_minus")

        if not h_plus or not isinstance(h_plus, ROOT.TH1D):
            print(f"WARNING: h_plus not a TH1D in {model_file_out} (type={type(h_plus)}), skipping plot")
            f.Close()
            return
        if not h_minus or not isinstance(h_minus, ROOT.TH1D):
            print(f"WARNING: h_minus not a TH1D in {model_file_out} (type={type(h_minus)}), skipping plot")
            f.Close()
            return

        m   = ws.var("Dp_M")
        tag = ws.cat("Dp_charge")

        xaxis = h_plus.GetXaxis()
        xmin = xaxis.GetXmin()
        xmax = xaxis.GetXmax()
        nbins = xaxis.GetNbins()

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def make_pads(canvas):
            upper = ROOT.TPad("upper", "", 0., 0.25, 1., 1.)
            lower = ROOT.TPad("lower", "", 0., 0.,   1., 0.23)

            for pad in (upper, lower):
                pad.SetRightMargin(margin)
                pad.SetLeftMargin(3 * margin)

            upper.SetTopMargin(margin)
            lower.SetBottomMargin(2 * margin)

            canvas.cd()
            lower.Draw()
            upper.Draw()
            return upper, lower

        def make_legend():
            leg = ROOT.TLegend(0.20, 0.65, 0.32, 0.90)
            leg.SetFillColor(ROOT.kWhite)
            leg.SetTextSize(0.055)
            leg.SetBorderSize(0)
            leg.SetTextFont(132)
            return leg

        # ------------------------------------------------------------------
        # Core plotting function
        # ------------------------------------------------------------------

        def draw_fit(canvas, data, slice_label, comp_bkg, save_name):
            plot_pdf = model_tot if slice_label else model_tot_plot
            upper, lower = make_pads(canvas)
            frame = m.frame(ROOT.RooFit.Bins(nbins))



            # ---------------- full model ----------------
            model_args = [
                ROOT.RooFit.Precision(1e-6),
                ROOT.RooFit.LineColor(ROOT.kRed),
                ROOT.RooFit.Name("model"),
            ]

            # Only the total plot needs fit-range normalization
            if slice_label:
                model_args += [
                    ROOT.RooFit.Slice(tag, slice_label),
                    ROOT.RooFit.ProjWData(tag, combData),
                ]



            # ---------------- background ----------------
            bkg_args = [
                ROOT.RooFit.Components(comp_bkg),
                ROOT.RooFit.FillColor(ROOT.kBlue),
                ROOT.RooFit.LineColor(ROOT.kBlue),
                ROOT.RooFit.DrawOption("F"),
                ROOT.RooFit.Precision(1e-6),
                ROOT.RooFit.Name("bkg"),
            ]

            if slice_label:
                bkg_args += [
                    ROOT.RooFit.Slice(tag, slice_label),
                    ROOT.RooFit.ProjWData(tag, combData),
                ]
            plot_pdf.plotOn(frame, *bkg_args)

    
            # ---------------- data ----------------
            if slice_label == "positive":
                data_p.plotOn(frame, ROOT.RooFit.Name("data"))
            elif slice_label == "negative":
                data_m.plotOn(frame, ROOT.RooFit.Name("data"))
            else:
                data.plotOn(frame, ROOT.RooFit.Name("data"))

            plot_pdf.plotOn(frame, *model_args)



            # ---------------- legend ----------------
            leg = make_legend()
            leg.AddEntry(frame.findObject("data"), "Data", "pe")
            leg.AddEntry(frame.findObject("model"), "Fit", "l")
            leg.AddEntry(frame.findObject("bkg"), "Bkg", "f")

            # ---------------- pull ----------------
            pull = frame.pullHist("data", "model")
            pull.SetLineColor(ROOT.kBlack)

            pull_frame = m.frame(
                ROOT.RooFit.Range(xmin,xmax),
                ROOT.RooFit.Bins(nbins)
            )
            pull_frame.addPlotable(pull, "BX")

            # ---------------- draw ----------------
            upper.cd()
            frame.SetTitle("")
            frame.Draw()

            leg.Draw()

            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextSize(0.04)
            latex.DrawLatex(0.65, 0.82, f"Entries = {data.sumEntries():.2e}")

            lower.cd()
            pull_frame.SetTitle("")
            pull_frame.GetYaxis().SetRangeUser(-5, 5)
            pull_frame.Draw("B")

            canvas.SaveAs(save_name + ".pdf")

            return frame

        # ------------------------------------------------------------------
        # Make plots
        # ------------------------------------------------------------------
    

        c_plus  = ROOT.TCanvas("c_plus", "", 900, 700)
        c_minus = ROOT.TCanvas("c_minus", "", 900, 700)
        c_tot   = ROOT.TCanvas("c_tot", "", 900, 700)

        draw_fit(c_plus,  data_p,   "positive", "expo_plus",              output_path + "plus")
        draw_fit(c_minus, data_m,   "negative", "expo_minus",              output_path + "minus")
        draw_fit(c_tot, data_tot, "", "bkg_tot", output_path + "total")

        # ------------------------------------------------------------------
        # Asymmetry plot
        # ------------------------------------------------------------------
        c_asym = ROOT.TCanvas("c_asym", "", 900, 700)

        upper = ROOT.TPad("u", "", 0, 0.25, 1, 1)
        lower = ROOT.TPad("l", "", 0, 0, 1, 0.23)

        upper.Draw()
        lower.Draw()

        frame_p = m.frame(ROOT.RooFit.Bins(nbins))
        frame_m = m.frame(ROOT.RooFit.Bins(nbins))

        model_tot.plotOn(
            frame_p,
            ROOT.RooFit.Slice(tag, "positive"),
            ROOT.RooFit.ProjWData(tag, combData),
            ROOT.RooFit.LineColor(ROOT.kRed),
            ROOT.RooFit.Name("p"),
        )

        model_tot.plotOn(
            frame_m,
            ROOT.RooFit.Slice(tag, "negative"),
            ROOT.RooFit.ProjWData(tag, combData),
            ROOT.RooFit.LineColor(ROOT.kRed),
            ROOT.RooFit.Name("m"),
        )


        combData.plotOn(frame_p, ROOT.RooFit.Cut(f"Dp_charge==Dp_charge::positive"))
        combData.plotOn(frame_m, ROOT.RooFit.Cut(f"Dp_charge==Dp_charge::negative"))



        cp = frame_p.findObject("p")
        cm = frame_m.findObject("m")

        h_asym = h_plus.GetAsymmetry(h_minus)
        h_asym.SetXTitle("Dp_M")
        h_asym.SetYTitle("Asymmetry")

        pdf_asym = ROOT.TH1D("pdf_asym", "", nbins, xmin, xmax)

        for i in range(self.nbins_m):
            x = xmin + (i + 0.5) * (xmax - xmin) / nbins

            fp = cp.Eval(x)
            fm = cm.Eval(x)

            denom = fp + fm
            val = (fp - fm) / denom if denom != 0 else 0

            pdf_asym.SetBinContent(i + 1, val)

        upper.cd()
        h_asym.Draw("EP")
        pdf_asym.SetLineColor(ROOT.kRed)
        pdf_asym.Draw("L SAME")

        lower.cd()
        ROOT.gPad.SetGridy()

        c_asym.SaveAs(output_path + "asym.pdf")

        f.Close()


    def _draw_w_kin(self, folder, weight):
        w = weight
        outdir =  folder + 'w_vs_kin/'
        os.makedirs(outdir, exist_ok=True)
        dist_params = {k: v for k, v in self.hist_params.items() if k!= "KS_LT"}
        colors = self.rootcolors

        particles = {'Pip': [], 'KS': [], 'Dp': []}
        for param in dist_params:
            particles[param.split("_")[0]].append(param)

        for pid in particles:
            if not particles[pid]:
                continue
            n_plots = len(particles[pid])
            ncols = 3
            nrows = (n_plots + ncols - 1) // ncols

            c = ROOT.TCanvas(f"c_wperf_{pid}_{self.iteration}", pid, 800 * ncols, 700 * nrows)
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


                for i in range(self.kslt_anabins):
                    pkey = f'h1D_{self._weight_key(w)}-{param}_ltbin{i}'
                    hist = self.hists_filled[pkey]
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
            c.SaveAs(outdir + f"{pid}_{self._weight_key(w)}.pdf")

    def _draw_A_kin(self, folder):
        outdir = folder + 'A_vs_kin/'
        os.makedirs(outdir, exist_ok=True)

        A_params = {k: v for k, v in self.hist_params.items() if k not in ("Dp_M", "KS_LT")}
        particles = self._particle_groups(A_params)

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
                if param not in self.A_fit_cache:
                    ax.set_visible(False)
                    continue
                A_arr = np.asarray(self.A_fit_cache[param][0])
                A_err_arr = np.asarray(self.A_fit_cache[param][1])
                x, xerr = self._uniform_bin_centers_widths(A_params[param])
                n = min(len(A_arr), len(x))

                ax.errorbar(x[:n], A_arr[:n], xerr=xerr[:n], yerr=A_err_arr[:n],
                            fmt="o", ms=4, capsize=3, lw=1, color="C0")
                ax.grid(alpha=0.3)
                ax.set_title(param, fontsize=11)
                ax.set_xlabel(param)
                ax.set_ylabel("A")
                ax.set_ylim(*self._capped_ylim(A_arr[:n], A_err_arr[:n]))

            for ax in axes[len(params):]:
                ax.set_visible(False)

            fig.suptitle(f'{pid}: asymmetry vs kinematics', fontsize=16, weight="bold")
            plt.savefig(outdir + f"{pid}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _draw_A_LT(self, folder):
        outdir = folder + 'A_vs_LT/'
        os.makedirs(outdir, exist_ok=True)

        signal_weights = list(self.A_integrated.keys())
        A_params = {k: v for k, v in self.hist_params.items() if k not in ("Dp_M", "KS_LT")}
        particles = self._particle_groups(A_params)

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
                    prof_key = f"KS_LT_{self._weight_key(w)}"
                    if self.bin_x.get(prof_key) is not None:
                        x = np.asarray(self.bin_x[prof_key])
                        xerr = self.bin_xerr[prof_key]
                    else:
                        lt_edges = np.asarray(self.lt_bin_edges)
                        x = 0.5 * (lt_edges[:-1] + lt_edges[1:])
                        xerr = 0.5 * (lt_edges[1:] - lt_edges[:-1])

                    if param not in self.A_integrated.get(self._weight_key(w), {}):
                        continue
                    y = np.asarray(self.A_integrated[w][param])
                    yerr = np.asarray(self.A_integrated_err[w][param])
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
                    ax.set_ylim(*self._capped_ylim(np.concatenate(y_all), np.concatenate(yerr_all)))

            for ax in axes[len(params):]:
                ax.set_visible(False)

            fig.suptitle(f'{pid}: integrated asymmetry vs KS lifetime', fontsize=16, weight="bold")
            plt.savefig(outdir + f"{pid}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def plot_weighting_performance(self, plot_w_kin = False, plot_A_kin = False, plot_A_LT = False):
        folder = self.base_folder + 'weighting/'
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
                    hkey = f'h2D_{self._weight_key(w)}-{param}-KSLT'
                    if hkey not in self.hists:
                        self.hists[hkey] = self.rdf.Histo2D(
                            (hkey, hkey, p_args[0], p_args[1], p_args[2], self.kslt_anabins, tedges),
                            param, "KS_LT", w
                        )

            # --- trigger + slice into per-lt-bin TH1Ds via ProjectionX (no extra passes) ---
            for w in signal_weights:
                for param in dist_params:
                    hkey = f'h2D_{self._weight_key(w)}-{param}-KSLT'
                    h2 = self.hists[hkey].GetValue()
                    for i in range(self.kslt_anabins):
                        pkey = f'h1D_{self._weight_key(w)}-{param}_ltbin{i}'
                        self.hists_filled[pkey] = h2.ProjectionX(f'px_{pkey}', i + 1, i + 1)


        ##########
        # Objective 2: plot A vs kin (unweighted)
        ########
        if plot_A_kin or plot_A_LT:
            A_params = {k: v for k, v in self.hist_params.items() if k != "Dp_M" and k != "KS_LT"}
            ma = array.array("d", np.linspace(self.min_m, self.max_m, self.nbins_m + 1))
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
                    self.mass_fit(h_plus, h_minus,
                                folder + f"massfits/{pid}/{param}/bin{i-1}/",
                                model_name=self.standard_model)   # see note below on the kwarg name

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
                self.A_integrated[self._weight_key(w)] = {}
                self.A_integrated_err[self._weight_key(w)] = {}
                for param, p_args in A_params.items():
                    self.A_integrated[self._weight_key(w)][param] = []
                    self.A_integrated_err[self._weight_key(w)][param] = []
                    Asig_arr = self.A_fit_cache[param][0]
                    Asig_err_arr = self.A_fit_cache[param][1]

                    for i in range(self.kslt_anabins):
                        A_int = 0.0
                        A_int_err_sq = 0.0
                        hkey = f'h1D_{self._weight_key(w)}-{param}_ltbin{i}'
                        hist = self.hists_filled[hkey]
                        total = hist.Integral()
                        if total <= 0:
                            self.A_integrated[self._weight_key(w)][param].append(np.nan)
                            self.A_integrated_err[self._weight_key(w)][param].append(np.nan)
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

                        self.A_integrated[self._weight_key(w)][param].append(A_int)
                        self.A_integrated_err[self._weight_key(w)][param].append(np.sqrt(A_int_err_sq))


        ##########
        # plotting
        ###########

        if plot_w_kin:
            for w in signal_weights:
                self._draw_w_kin(folder, w)
        
        if plot_A_kin: self._draw_A_kin(folder)

        if plot_A_LT: self._draw_A_LT(folder)

    def plot_weighting_statistics(self):
        folder = self.base_folder + 'weighting_statistics/'
        os.makedirs(folder, exist_ok=True)
        iterations = np.arange(0, self.iteration + 1)
        

        # --- book everything first, across ALL bins and iterations ---
        sum_w, sum_w2, sum_Neff = {}, {}, {}
        for ltbin in range(self.kslt_anabins):
            key3 = f'ltbin_{ltbin}_{self.iteration}'
            if key3 not in self.rdfs:
                lt_lo, lt_hi = self.lt_bin_edges[ltbin], self.lt_bin_edges[ltbin + 1]
                self.rdfs[key3] = self.rdf.Filter(f"KS_LT > {lt_lo} && KS_LT < {lt_hi}")
            for it in iterations:
                sum_w[(ltbin, it)] = self.rdfs[key3].Sum(f"weight_{it}")
                sum_w2[(ltbin, it)] = self.rdfs[key3].Sum(f"w2_{it}")
                sum_Neff[(ltbin, it)] = self.rdfs[key3].Sum(f"weight_eff_{it}")

        # --- trigger once (first .GetValue() fuses everything above), then read back ---
        Rw, Swinv, Neff = {}, {}, {}
        Swinv_vs_lt = []
        for ltbin in range(self.kslt_anabins):
            Sw_vs_it, Neff_vs_it = [], []
            for it in iterations:
                w_val = sum_w[(ltbin, it)].GetValue()
                w2_val = sum_w2[(ltbin, it)].GetValue()
                neff_val = sum_Neff[(ltbin, it)].GetValue()
                sw = w2_val / w_val**2 if w_val != 0 else np.nan
                Sw_vs_it.append(sw)
                Neff_vs_it.append(neff_val)
            Sw_vs_it = np.asarray(Sw_vs_it)
            Rw[ltbin] = (Sw_vs_it[0] / Sw_vs_it )
            Swinv[ltbin] = 1.0 / Sw_vs_it
            Neff[ltbin] = np.asarray(Neff_vs_it)
            Swinv_vs_lt.append(Swinv[ltbin][-1])

        # --- x-axis for lifetime plot ---
        key5 = "KS_LT_final_weight"
        if self.bin_x.get(key5) is not None:
            x = np.asarray(self.bin_x[key5])
            xerr = self.bin_xerr[key5]

        else:
            lt_edges = np.asarray(self.lt_bin_edges)
            x = 0.5 * (lt_edges[:-1] + lt_edges[1:])
            xerr = 0.5 * (lt_edges[1:] - lt_edges[:-1])

        # --- PDF 1: Neff vs iteration ---
        fig, ax = plt.subplots(figsize=(6, 4))
        for ltbin in range(self.kslt_anabins):
            ax.plot(iterations, Neff[ltbin], label=f'ltbin {ltbin + 1}')
        if self.kslt_anabins < 10:
            ax.legend(fontsize='small')
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$N_{eff}$")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(folder, "Neff_vs_iteration.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- PDF 2: Swinv vs iteration ---
        fig, ax = plt.subplots(figsize=(6, 4))
        for ltbin in range(self.kslt_anabins):
            ax.plot(iterations, Swinv[ltbin], label=f'ltbin {ltbin + 1}')
        if self.kslt_anabins < 10:
            ax.legend(fontsize='small')
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$S_w^{-1}$")
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(folder, "Swinv_vs_iteration.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- PDF 3: Rw vs iteration ---
        fig, ax = plt.subplots(figsize=(6, 4))
        for ltbin in range(self.kslt_anabins):
            ax.plot(iterations, Rw[ltbin], label=f'ltbin {ltbin + 1}')
        if self.kslt_anabins < 10:
            ax.legend(fontsize='small')
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$R_w$")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(folder, "Rw_vs_iteration.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- PDF 4: Swinv vs lifetime (normalized) ---
        Swinv_vs_lt = np.asarray(Swinv_vs_lt)
        norm_Swinv_vs_lt = Swinv_vs_lt / np.max(Swinv_vs_lt)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(x, norm_Swinv_vs_lt, xerr=xerr, fmt="o", capsize=3)
        ax.set_xlabel(r"KS_LT $(t/\tau)$")
        ax.set_ylabel(r"$S_w^{-1}$ (normalized)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(folder, "Swinv_vs_lifetime.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        return
    def plot_A_vs_kslt(self):
        folder = self.base_folder + 'result/'
        os.makedirs(folder, exist_ok=True)
        procedures = {'before': '', 'after': 'final_weight'}
        proc_names = list(procedures.keys())
        weight_keys = {proc: self._weight_key(w) for proc, w in procedures.items()}
        weights = list({procedures[p] for p in proc_names})
        param = "KS_LT"

        ma = array.array("d", np.linspace(self.min_m, self.max_m, self.nbins_m + 1))
        ca = array.array("d", np.linspace(-2, 2, 3))
        tedges = array.array('d', [float(e) for e in self.lt_bin_edges])

        # init hists
        for weight in weights:
            wk = self._weight_key(weight)
            hkey = f'h3D_{wk}-{param}-Dp_M-Dp_charge'
            model = ROOT.RDF.TH3DModel(hkey, hkey, len(tedges) - 1, tedges, len(ma) - 1, ma, len(ca) - 1, ca)
            args = (param, "Dp_M", "Dp_charge", weight) if weight != '' else (param, "Dp_M", "Dp_charge")
            self.hists[hkey] = self.rdf.Histo3D(model, *args)

        # fill hists
        for proc in proc_names:
            wk = weight_keys[proc]
            hkey = f'h3D_{wk}-KS_LT-Dp_M-Dp_charge'
            self.hists_filled[hkey] = self.hists[hkey].GetValue()

            # Store proper weighted uncertainties
            if self.hists_filled[hkey].GetSumw2N() == 0:
                self.hists_filled[hkey].Sumw2()

        # mass fits
        for proc in proc_names:
            wk = weight_keys[proc]
            hkey = f'h3D_{wk}-KS_LT-Dp_M-Dp_charge'
            h = self.hists_filled[hkey]
            for i in range(1, h.GetXaxis().GetNbins() + 1):
                h_plus = h.ProjectionY(f"h_plus_{i}_{param}", i, i, 2, 2)
                h_minus = h.ProjectionY(f"h_minus_{i}_{param}", i, i, 1, 1)
                for model in self.models:
                    self.mass_fit(h_plus, h_minus,
                                folder + f"massfits/{wk}/{model}/bin{i-1}/",
                                model_name=model)

        # --- find asymmetries ---
        for proc in proc_names:
            wk = weight_keys[proc]
            self.A_fit_cache[wk] = {"KS_LT": {}}
            hkey = f'h3D_{wk}-KS_LT-Dp_M-Dp_charge'
            n_bins = self.hists_filled[hkey].GetXaxis().GetNbins()
            for model in self.models:
                self.A_fit_cache[wk]["KS_LT"][model] = ([], [])
                for i in range(1, n_bins + 1):
                    fit_file = folder + f"massfits/{wk}/{model}/bin{i-1}/model.root"
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
                    self.A_fit_cache[wk]["KS_LT"][model][0].append(A_val)
                    self.A_fit_cache[wk]["KS_LT"][model][1].append(A_err_val)

        # --- gather data ---
        A, A_err, x, xerr = {}, {}, {}, {}
        for proc in proc_names:
            wk = weight_keys[proc]
            x[proc] = np.asarray(self.bin_x[f"{param}_{wk}"])
            xerr[proc] = self.bin_xerr[f"{param}_{wk}"]

        for model in self.models:
            A[model], A_err[model] = {}, {}
            for proc in proc_names:
                wk = weight_keys[proc]
                cached = self.A_fit_cache.get(wk, {}).get(param, {}).get(model, ([], []))
                A[model][proc] = np.asarray(cached[0])
                A_err[model][proc] = np.asarray(cached[1])

        available_models = [m for m in self.models
                            if any(A[m].get(p, np.array([])).size > 0 for p in proc_names)]
        if not available_models:
            print("WARNING: no models with data for KS_LT")
            return

        # --- plotting: delegated to AsymmetryPlotter ---
        plotter = AsymmetryPlotter(
            models=self.models, procedures=proc_names,
            xlabel=r'KS_LT $(t/\tau)$',
            procedure_labels={'before': 'before reweighting', 'after': 'after reweighting'},
            A_bias = self.A_bias
        )
        plotter.plot_standard_set(
            A, A_err, x, xerr, folder,
            available_models=available_models,
            standard_model=self.standard_model,
            after_procedure='after',
        )

    def plot_dist(self, folder, params, bins):
        os.makedirs(folder, exist_ok=True)
        hists = {}

        for param in params:
            values = self.rdf.AsNumpy([param])[param]

            # Boolean variable
            if values.dtype == np.bool_:
                num = 3
                min_val = - num
                max_val = num
                hist_bins = num*4 + 1

            # Numeric variable
            else:
                min_val = np.percentile(values, 0.001)
                max_val = np.percentile(values, 99.999)
                hist_bins = bins

            print(param, f"min = {min_val}, max = {max_val}")

            h = self.rdf.Histo1D(
                (f"{param}_1D", f"{param}_1D", hist_bins, min_val, max_val),
                param
            )

            hists[param] = h

        for param in params:
            h = hists[param].GetValue()

            c = ROOT.TCanvas(f"c_{param}", "My Canvas", 1200, 800)
            h.Draw()
            c.Update()
            c.SaveAs(os.path.join(folder, f"{param}_1D.pdf"))
            c.Close()

    ##################################################
    # helper methods
    ##################################

    @staticmethod
    def _asymmetric_xerr(bin_edges, bin_centers):
        """
        Build a (2, N) array suitable for plt.errorbar(..., xerr=...), where
        each point's error bar spans exactly its own bin's edges — even when
        the plotted x-value (e.g. a weighted mean) isn't centered in the bin.

        bin_edges:   length N+1, the N bin boundaries (lo_0, hi_0=lo_1, ..., hi_{N-1})
        bin_centers: length N, the x-value actually being plotted for each bin
                    (e.g. a bin mean rather than the geometric bin center)
        """
        bin_edges = np.asarray(bin_edges, dtype=float)
        bin_centers = np.asarray(bin_centers, dtype=float)
        lo = bin_edges[:-1]
        hi = bin_edges[1:]
        lower = bin_centers - lo
        upper = hi - bin_centers
        return np.vstack([lower, upper])   # shape (2, N)

    @staticmethod
    def _capped_ylim(y, yerr, cap_lim=1e-2, pad=0.2):
        """
        Same y-range logic as before, except any error bar more than
        `cap_factor` times its own point's |A_sig| is excluded from the
        range calculation. The point is still plotted with its full error
        bar via errorbar() — it just won't be allowed to blow out the axis.
        """
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        valid = np.isfinite(y) & np.isfinite(yerr)
        if not valid.any():
            return 0.0, 1.0

        outlier = yerr > cap_lim
        range_err = np.where(outlier, 0.0, yerr)   # outliers contribute 0 to the range calc

        lo = np.min((y - range_err)[valid])
        hi = np.max((y + range_err)[valid])
        rng = hi - lo if hi > lo else max(abs(lo), 1.0)
        return lo - pad * rng, hi + pad * rng

    @staticmethod
    def _particle_groups(params):
        particles = {'Pip': [], 'KS': [], 'Dp': []}
        for param in params:
            pid = param.split("_")[0]
            if pid in particles:
                particles[pid].append(param)
        return particles

    @staticmethod
    def _uniform_bin_centers_widths(p_args):
        """A_params uses plain linspace binning (not quantile bins), so
        centers/half-widths are trivially symmetric."""
        nbins, lo, hi = p_args[0], p_args[1], p_args[2]
        edges = np.linspace(lo, hi, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        halfwidths = 0.5 * (edges[1:] - edges[:-1])
        return centers, halfwidths



    @staticmethod
    def _weight_key(weight):
        """Canonical, non-empty dict-key string for a weight column name.
        '' (and None) mean 'unweighted'."""
        return weight if weight else "unweighted"

