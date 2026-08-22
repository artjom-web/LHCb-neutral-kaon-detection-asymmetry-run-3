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
parser.add_argument("--threshold", type=int, required=True)
parser.add_argument("--ltmin", type=float, required=True)
parser.add_argument("--ltmax", type=float, required=True)
parser.add_argument("--nbins", type=int, required=True)
parser.add_argument("--data_up", type=str, required=False)
parser.add_argument("--data_down", type=str, required=False)
parser.add_argument("--track", type=str, required=False)
parser.add_argument("--file", type=str, required=False)



args = parser.parse_args()
day = f'{datetime.now().month}-{datetime.now().day}'
time = f'{datetime.now().hour}hr{datetime.now().minute}'
params = [
        "Pip_HLT2_PT", "Pip_HLT2_ETA", "Pip_k", "Pip_theta_x", "Pip_theta_y",
        "Dp_HLT2_PT", "Dp_HLT2_ETA", "Pip_HLT2_PHI", "Dp_theta_x", "Dp_theta_y", 
        "KS_HLT2_PHI", "KS_HLT2_PT", "KS_HLT2_ETA", "KS_k", "KS_theta_x", "KS_theta_y"
]
params = ["KS_LT"]
ROOT.EnableImplicitMT()

ana = Analysis()  
ana.simple_threshold = args.threshold
ana.ltmin = args.ltmin
ana.ltmax = args.ltmax
ana.nbins_params = args.nbins

if args.file:
    snapshot_folder = '/eos/user/a/ahulsber/scripts/snapshots/'
    ana.rdf = ROOT.RDataFrame("DecayTree", snapshot_folder + args.file + '.root')
    ana.N_total_events = ana.rdf.Count().GetValue()
    if len(args.file.split('_')) == 3:
        _, track, tot_files = args.file.split('/')[1].split('_')
    elif len(args.file.split('_')) == 2:
        track, tot_files = args.file.split('/')[1].split('_')
    else:
        print('WRONG FILENAME!')
    tot_files = int(tot_files)
    name = args.file.split('/')[1]


else:
    track = args.track
    tot_files = 0
    for polarity, data in [("magup", args.data_up), ("magdown", args.data_down)]:
        if data != '':
            for datastr in data.split("-"):
                try:
                    ycset, files = datastr.split("_")
                    files = int(files)
                except ValueError:
                    raise ValueError(
                        f"Invalid format '{datastr}'. Expected cset_files, e.g. 25c4_60."
                    )
                tot_files += files
                ana.import_PFNS(
                    polarity=polarity,
                    ycset=ycset,
                    datasetnumbers=np.arange(files),
                    track=track
                )
    name = f'{track}_{tot_files}'

    print('startup ok')
    
    ana.data_to_rdf()
    ana.defs()
    if ana.track == 'll':
        ana.cuts()
    else:
        ana.cuts_DD()
        
ana.init_hist_params()
base_folder = f'/eos/user/a/ahulsber/scripts/data/{day}/{name}_{time}/'

os.makedirs(base_folder, exist_ok=True)
with open(base_folder + "config.txt", "w") as f:
    f.write(f"threshold = {ana.threshold}\n")
    f.write(f"ltmin = {ana.ltmin}\n")
    f.write(f"ltmax = {ana.ltmax}\n")
    f.write(f"nbins_params = {ana.nbins_params}\n")


ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.mass_fit(ana.hists_filled['Dp_M_p'], ana.hists_filled['Dp_M_m'], base_folder + 'fit0/')
ana.plot_massfit(base_folder + 'fit0/')
ana.init_weights(base_folder + 'fit0/')
ana.lt_bin_edges = ana.equal_bins()
ana.plot_kin_dist(base_folder + 'weighting/a_start')
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


ana.plot_kin_dist(base_folder + f"weighting/cycle_7",  f"weight_7")
# ana.plot_kin_dist(base_folder + f"weighting/cycle_1",  f"weight_1")

ana.plot_w0(base_folder + f"weighting/")
ana.find_Adep(params, base_folder + 'before/')
ana.plot_Adep(params, base_folder + 'before/')
ana.rdf = ana.rdf.Define("mass_weight", f"weight_{ana.iteration} / weight_0")
ana.find_Adep(params, base_folder + 'after/', 'mass_weight')
ana.plot_Adep(params, base_folder + 'after/', 'mass_weight')

