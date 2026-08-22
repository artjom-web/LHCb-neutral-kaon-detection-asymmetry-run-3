from AnalysisFunctions import Analysis 
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


parser = argparse.ArgumentParser()
parser.add_argument("--ltmax", type=float, required=True)
parser.add_argument("--pbins", type=int, required=True)
parser.add_argument("--ltbins", type=int, required=True)
parser.add_argument("--file", type=str, required=False)



args = parser.parse_args()


day = f'{datetime.now().month}-{datetime.now().day}'
time = f'{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}'

params = ["KS_LT"]
ROOT.EnableImplicitMT()

ana = Analysis()  
ana.simple_threshold = 1
ana.ltmin = 0.0
ana.ltmax = args.ltmax
ana.nbins_params = args.pbins
ana.kslt_anabins = args.ltbins
file = args.file

snapshot_folder = '/eos/user/a/ahulsber/scripts/snapshots/'
ana.rdf = ROOT.RDataFrame("DecayTree", snapshot_folder + file + '.root')
ana.N_total_events = ana.rdf.Count().GetValue()
if len(file.split('_')) == 3:
    _, track, tot_files = file.split('/')[1].split('_')
elif len(file.split('_')) == 2:
    track, tot_files = file.split('/')[1].split('_')
else:
    print('WRONG FILENAME!')
tot_files = int(tot_files)
name = file.split('/')[1]
ana.init_hist_params()
base_folder = f'/eos/user/a/ahulsber/scripts/data/{day}/{name}_{time}/'

os.makedirs(base_folder, exist_ok=True)



ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.mass_fit(ana.hists_filled['Dp_M_p'], ana.hists_filled['Dp_M_m'], base_folder + 'fit0/')
ana.plot_massfit(base_folder + 'fit0/')
ana.init_weights(base_folder + 'fit0/')
if ana.kslt_anabins ==6: 
    ana.lt_bin_edges = lt_bin_edges = [0.0, 0.070, 0.095, 0.116, 0.141, 0.175, 0.4]


else: ana.lt_bin_edges = ana.equal_bins()
# ana.plot_kin_dist(base_folder + 'weighting/a_start')
# ana.plot_kin_dist(base_folder + 'weighting/cycle_0', 'weight_0')

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

for cycle, param1, param2, param3 in triplets:

    ana.weighting_cycle(
        param1=param1,
        param2=param2,
        param3=param3
    )

if ana._pending_acc_sum is not None:
    N = ana._pending_acc_sum.GetValue()
    print(f'[iter {ana._pending_acc_iter}] accepted events = {N}, fraction = {N/ana.N_total_events}')


# ana.plot_kin_dist(base_folder + f"weighting/cycle_7",  f"weight_7")
# ana.plot_kin_dist(base_folder + f"weighting/cycle_1",  f"weight_1")

edges, counts = ana.plot_w0(base_folder + f"weighting/")

def equal_count_bins(bin_edges, counts, n):
    counts = np.asarray(counts)
    bin_edges = np.asarray(bin_edges)

    cumulative = np.cumsum(counts)
    total = cumulative[-1]

    targets = np.linspace(0, total, n + 1)

    new_edges = np.interp(
        targets,
        np.r_[0, cumulative],
        bin_edges
    )

    return new_edges
new_edges_6 = equal_count_bins(edges, counts, 6)
new_edges_7 = equal_count_bins(edges, counts, 7)

with open(base_folder + "config.txt", "w") as f:
    f.write(f"threshold = {ana.simple_threshold}\n")
    f.write(f"ltmin = {ana.ltmin}\n")
    f.write(f"ltmax = {ana.ltmax}\n")
    f.write(f"nbins_params = {ana.nbins_params}\n\n")
    f.write(f"\n new_edges_6\n")
    for i, edge in enumerate(new_edges_6):
        f.write(f"edge_{i} = {edge}\n")

    f.write(f"\n new_edges_7\n")
    for i, edge in enumerate(new_edges_7):
        f.write(f"edge_{i} = {edge}\n")
    



# ana.find_Adep(params, base_folder + 'before/')
# ana.plot_Adep(params, base_folder + 'before/')
# ana.rdf = ana.rdf.Define("mass_weight", f"weight_{ana.iteration} / weight_0")
# ana.find_Adep(params, base_folder + 'after/', 'mass_weight')
# ana.plot_Adep(params, base_folder + 'after/', 'mass_weight')

