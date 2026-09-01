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
from ..user_uploads import mass_models as mm
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

