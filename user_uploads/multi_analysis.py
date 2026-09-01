from AnalysisFunctions import Analysis, AsymmetryPlotter
import ROOT
from ROOT import (RooRealVar, RooGaussian,
 RooExponential, RooAddPdf, RooDataHist, RooFit, 
 RooPlot, TCanvas, RooArgList, TLegend, RooArgSet)

from apd import AnalysisData
import numpy as np
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
import time
from itertools import combinations
from datetime import datetime
import argparse
from pathlib import Path
from collections import defaultdict

data = defaultdict(lambda: defaultdict(dict))

def open_file(runfile):
    data = runfile.split('_')
    if len(data) == 4:
        track, ycset, polarity, n_files = data
    elif len(data) == 3:
        ycset, track, n_files = data
        polarity = None
    elif len(data) == 2:
        track, n_files = data
        polarity, ycset = None, None
    else:
        raise ValueError("Wrong runfile format")
    n_files = int(n_files)

    return track, ycset, polarity, n_files

def open_folder(folder):
    runfiles = [f.stem for f in folder.glob("*.root")]
    analysis_plan = defaultdict(lambda: defaultdict(dict))
    ycsets = []
    length_check = len(runfiles[0].split('_'))
    track_check, _, _, _ = open_file(runfiles[0])

    for runfile in runfiles:
        length = len(runfile.split('_'))
        track, ycset, polarity, n_files = open_file(runfile)

        if length != length_check: 
            for runfile in runfiles:
                print(runfile)
            raise ValueError("Runfiles in folder don't have the same format")
        if track != track_check:
            for runfile in runfiles:
                print(runfile)
            raise ValueError("Runfiles in folder don't have the same track")

        if ycset is not None: 
            if polarity is not None:
                analysis_plan[f'{track}_{ycset}_{polarity}'] = runfile, track, n_files
            else: raise ValueError("files must contain polarity")
        else: raise ValueError("files must contain ycset")
        

    #     if ycset is not None: 
    #         if polarity is not None:
    #             analysis_plan[f'{track}_{ycset}'][polarity] = runfile, track, n_files
    #         else:
    #             analysis_plan[f'{track}_{ycset}']['both'] = runfile, track, n_files
    #     else:
    #         analysis_plan[f'{track}']['both'] =  runfile, track, n_files

    return analysis_plan, track_check







parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default=None)
parser.add_argument("--polarity", type=str, default=None)
parser.add_argument("--ycset", type=str, default=None)
parser.add_argument("--hlt1cut", type=str, default=None)
parser.add_argument("--name", type=str, default=None)

args = parser.parse_args()
day = f'{datetime.now().day}-{datetime.now().month}'
name = args.name if args.name else f'{datetime.now().hour}hr{datetime.now().minute}'


params = ["KS_LT"]

params = [
        "Pip_HLT2_PT", "Pip_HLT2_ETA", "Pip_k", "Pip_theta_x", "Pip_theta_y",
        "Dp_HLT2_PT", "Dp_HLT2_ETA", "Pip_HLT2_PHI", "Dp_theta_x", "Dp_theta_y", 
        "KS_HLT2_PHI", "KS_HLT2_PT", "KS_HLT2_ETA", "KS_k", "KS_theta_x", "KS_theta_y", "KS_LT"
]


base_folder = f'/eos/user/a/ahulsber/scripts/data/{day}/{name}/'
data_folder = Path('/eos/user/a/ahulsber/scripts/multi_analysis/' + runfolder)
ana_plan, track = open_folder(data_folder)

ltbin_x = {}
ltbin_xerr = {}
paths = []
path_to_key = {}   # NEW
os.makedirs(base_folder, exist_ok=True)
A_bias = np.random.RandomState(0).uniform(-1, 1)

model_names =  ['johnson_exp1', 'johnson_exp', 'johnson_tail_expo', 'johnson_gauss_exp']
standard_model = 'johnson_exp1'
if track == 'dd':
    lt_bin_edges = [0.0, 0.4686, 0.5727, 0.6518, 0.7385, 0.8259, 0.9733, 3.0]

elif track == 'll':
    lt_bin_edges = [0.0, 0.070, 0.093, 0.116, 0.142, 0.176, 0.4]

else: 
    raise ValueError("Unexpected amount of lt bins")


triplets = [ 
    (
        "cycle1",
        "Pip_k",
        "Pip_theta_x",
        "Pip_theta_y"
    ),

    (
        "cycle2",
        "Pip_k",
        "Dp_HLT2_PT",
        "Dp_HLT2_ETA"
    ),

    (
        "cycle3",
        "Pip_HLT2_PHI",
        "Pip_HLT2_ETA",
        "Dp_HLT2_PT"
    ),

    (
        "cycle4",
        "Pip_HLT2_PT",
        "Pip_HLT2_ETA",
        "Dp_HLT2_PT"
    ),

    (
        "cycle5",
        "Pip_HLT2_PT",
        "Dp_theta_x",
        "Dp_HLT2_PT"
    ),

    (
        "cycle6",
        "Pip_theta_x",
        "Pip_theta_y",
        "Dp_HLT2_PT"
    ),

    (
        "cycle7",
        "Pip_k",
        "Dp_HLT2_ETA",
        "Dp_HLT2_PT"
    )
]

ROOT.EnableImplicitMT()


for key in ana_plan:
    datafiles = []
    tot_files = 0

    datafile, _, n_files = ana_plan[key]
    datafiles.append(datafile)
    tot_files += n_files
        

    ana = Analysis(track)  
    ana.kinvar_bins = 20
    ana.base_folder = base_folder + key + f'_{tot_files}/'
    os.makedirs(ana.base_folder, exist_ok=True)
    ana.lt_bin_edges = lt_bin_edges
    ana.kslt_anabins = len(lt_bin_edges)-1
    ana.A_bias = A_bias
    ana.models = model_names
    ana.standard_model = standard_model
    chain = ROOT.TChain("DecayTree")
    for datafile in datafiles:
        chain.Add(str(data_folder / f"{datafile}.root"))

    ana.rdf = ROOT.RDataFrame(chain)
    ana.N_total_events = ana.rdf.Count().GetValue()

    with open(ana.base_folder + "config.txt", "w") as f:
        f.write(f"threshold = {ana.threshold}\n")
        f.write(f"ltmin = {ana.ltmin}\n")
        f.write(f"ltmax = {ana.ltmax}\n")
        f.write(f"nbins_params = {ana.kinvar_bins}\n")
        f.write(f"\n kslt_bin_edges\n")
        for i, edge in enumerate(lt_bin_edges):
            f.write(f"edge_{i} = {edge}\n")

    # this is the actual analysis
    ana.init_hist_params()
    ana.init_weights()

    for cycle, param1, param2, param3 in triplets: ana.weighting_cycle(param1=param1,param2=param2,param3=param3)
    ana.rdf = ana.rdf.Define("final_weight", f"weight_{ana.iteration} / weight_0")
    if ana._pending_acc_sum is not None:
        N = ana._pending_acc_sum.GetValue()
        print(f'[iter {ana._pending_acc_iter}] accepted events = {N}, fraction = {N/ana.N_total_events}')
    ana.ltbin_means()


    # ana.plot_weighting_performance(plot_w_kin = True, plot_A_kin = True, plot_A_LT = True)

    ana.plot_weighting_statistics()

    ana.plot_A_vs_kslt()


    ltbin_x[f'{key}_before'] = ana.bin_x["KS_LT_unweighted"]
    ltbin_xerr[f'{key}_before'] = ana.bin_xerr["KS_LT_unweighted"]
    ltbin_x[f'{key}_after'] = ana.bin_x["KS_LT_final_weight"]
    ltbin_xerr[f'{key}_after'] = ana.bin_xerr["KS_LT_final_weight"]

    path = ana.base_folder + 'result/'
    paths.append(path)
    path_to_key[path] = key   # NEW
    

def _asymmetric_xerr(bin_edges, bin_centers):
    """(2, N) lower/upper distances from each center to its own bin's edges."""
    bin_edges = np.asarray(bin_edges, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)
    lo, hi = bin_edges[:-1], bin_edges[1:]
    return np.vstack([bin_centers - lo, hi - bin_centers])


def load_asymmetry_results(paths, path_to_key, models, procedures, lt_bin_edges,
                            ltbin_x, ltbin_xerr):
    """
    Reads A_sig / A_sig_err from every fit workspace under
    <path>/<procedure>/KS/KS_LT_<suffix>/<model>/bin<i>/model.root
    (suffix is '' for 'before', 'mass_weight' for 'after'), combines them
    across `paths` bin-by-bin via inverse-variance weighting, and returns:
      - A_final[model][procedure]      -> np.array over lifetime bins
      - A_final_errs[model][procedure] -> np.array over lifetime bins
      - x, xerr                        -> bin centers/half-widths
      - a long-form DataFrame of the same data, for the CSV
    A missing/unreadable file for a given path is skipped with a warning;
    a bin with zero readable files is left as NaN rather than silently 0,
    so it's visibly absent on the plots instead of looking like a real A=0.
    """
    procedure_weight_suffix = {'before': 'unweighted', 'after': 'final_weight'}

    lt_bin_edges = np.asarray(lt_bin_edges)
    n_bins = len(lt_bin_edges) - 1
    x = 0.5 * (lt_bin_edges[:-1] + lt_bin_edges[1:])
    xerr = 0.5 * (lt_bin_edges[1:] - lt_bin_edges[:-1])

    A_final = {m: {} for m in models}
    A_final_errs = {m: {} for m in models}
    x_final = {m: {} for m in models}
    xerr_final = {m: {} for m in models}
    records = []

    for model in models:
        for procedure in procedures:
            suffix = procedure_weight_suffix.get(procedure)
            if suffix is None:
                print(f"WARNING: unknown procedure '{procedure}', skipping")
                continue

            A_model = np.full(n_bins, np.nan)
            A_model_errs = np.full(n_bins, np.nan)
            x_model = np.full(n_bins, np.nan)

            for ibin in range(n_bins):
                Abin, Abin_errs, xbin = [], [], []
                for path in paths:
                    key = path_to_key[path]
                    file = path + f"massfits/{suffix}/{model}/bin{ibin}/model.root"
                    if not os.path.exists(file):
                        print(f"WARNING: missing {file}, skipping")
                        continue
                    f = ROOT.TFile.Open(file)
                    if not f or f.IsZombie():
                        print(f"WARNING: could not open {file}, skipping")
                        continue
                    ws = f.Get("ws")
                    fit_params = ws.allVars()
                    Abin.append(fit_params.find("A_sig").getVal())
                    Abin_errs.append(fit_params.find("A_sig").getError())
                    xbin.append(ltbin_x[f'{key}_{procedure}'][ibin])
                    f.Close()



                Abin = np.asarray(Abin)
                Abin_errs = np.asarray(Abin_errs)
                xbin = np.asarray(xbin)

                valid = np.isfinite(Abin) & np.isfinite(Abin_errs) & (Abin_errs > 0)

                Abin = Abin[valid]
                Abin_errs = Abin_errs[valid]
                xbin = xbin[valid]

                if len(Abin) == 0:
                    continue

                weights = 1.0 / Abin_errs**2
                norm = np.sqrt(np.sum(weights))
                if norm > 0:
                    A_model[ibin] = np.average(Abin, weights=weights) 
                    A_model_errs[ibin] = 1.0 / norm
                    x_model[ibin] = np.average(xbin, weights=weights)

            A_final[model][procedure] = A_model
            A_final_errs[model][procedure] = A_model_errs
            x_final[model][procedure] = x_model
            xerr_final[model][procedure] = _asymmetric_xerr(lt_bin_edges, x_model)


            records.extend(
                {
                    "model": model, "procedure": procedure, "bin": ibin,
                    "x": x_model[ibin],
                    "xerr_lo": xerr_final[model][procedure][0, ibin],
                    "xerr_hi": xerr_final[model][procedure][1, ibin],
                    "A": A_model[ibin], "A_err": A_model_errs[ibin],
                }
                for ibin in range(n_bins)
            )

    return A_final, A_final_errs, x_final, xerr_final, pd.DataFrame(records)


procedures = ['before', 'after']
A_final, A_final_errs, x_final, xerr_final, df = load_asymmetry_results(
    paths, path_to_key, model_names, procedures, lt_bin_edges, ltbin_x, ltbin_xerr
)

plotter = AsymmetryPlotter(
    models=model_names, procedures=procedures,
    xlabel=r'KS lifetime $(t/\tau)$',
    procedure_labels={'before': 'before reweighting', 'after': 'after reweighting'},
    A_bias = A_bias
)

plotter.plot_standard_set(
    A_final, A_final_errs, x_final, xerr_final, base_folder,
    available_models = model_names,
    standard_model=standard_model,   # whatever your global standard model variable is called
    after_procedure='after',
)

output_file = os.path.join(base_folder, "asymmetry_results.csv")
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")




