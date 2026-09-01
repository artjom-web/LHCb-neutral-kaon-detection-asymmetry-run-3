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

parser = argparse.ArgumentParser()
parser.add_argument("--trigger", type=str, required=False, default='')
parser.add_argument("--track", type=str, default=None)
parser.add_argument("--ycset", type=str, default=None)
parser.add_argument("--polarity", type=str, default=None)
parser.add_argument("--day", type=str, default=None)
parser.add_argument("--time", type=str, default=None)
args = parser.parse_args()
trigger = args.trigger
day = args.day if args.day else f'{datetime.now().month}-{datetime.now().day}'
time = args.time if args.time else f'{datetime.now().hour}hr{datetime.now().minute}'
base_folder = f'/eos/user/a/ahulsber/scripts/snapshots/{day}/{trigger}/'
os.makedirs(base_folder, exist_ok=True)
tracks, ycsets, polarities, result, params = init_data()
ROOT.EnableImplicitMT()

combos = [(t, y, p) for t in tracks for y in ycsets for p in polarities]
if args.track is not None:
    combos = [c for c in combos if c[0] == args.track]
if args.ycset is not None:
    combos = [c for c in combos if c[1] == args.ycset]
if args.polarity is not None:
    combos = [c for c in combos if c[2] == args.polarity]

for track, ycset, polarity in combos:
    key = f"{track}_{ycset}_{polarity}"
    if key not in result:
        continue
    if (polarity == 'magdown' and ycset == '25c3') or (polarity == 'magup' and ycset == '25c2'):
        continue
    ana = Analysis(track)
    n_files = result[key]
    snapshot_path = base_folder + f'{track}_{ycset}_{polarity}_{n_files}.root'
    ana.import_data(
        polarity=polarity,
        ycset=ycset,
        datasetnumbers=np.arange(n_files),
        track=track
    )
    ana.data_to_rdf()
    ana.defs()
    ana.cuts(cut=trigger)
    ana.cuts()
    ana.init_hist_params()
    cols = sorted(set(ana.hist_params.keys()) | {"Dp_M", "Dp_charge", "KS_LT"})
    ana.rdf.Snapshot("DecayTree", snapshot_path, cols)

# args = parser.parse_args()
# trigger = args.trigger
# day = f'{datetime.now().month}-{datetime.now().day}'
# time = f'{datetime.now().hour}hr{datetime.now().minute}'
# base_folder =  f'/eos/user/a/ahulsber/scripts/snapshots/{day}/{trigger}/'
# os.makedirs(base_folder, exist_ok=True)
# tracks, ycsets, polarities, result, params = init_data()
# ROOT.EnableImplicitMT()

# for track in tracks:
#     for ycset in ycsets:
#         for polarity in polarities:
#             key = f"{track}_{ycset}_{polarity}"
#             if key not in result:
#                 continue
#             if (polarity == 'magdown' and ycset == '25c3') or (polarity == 'magup' and ycset == '25c2'):
#                 continue
#             ana = Analysis(track)  
#             n_files = result[key]
#             snapshot_path = base_folder + f'{track}_{ycset}_{polarity}_{n_files}.root'
#             ana.import_data(
#                 polarity=polarity,
#                 ycset=ycset,
#                 datasetnumbers=np.arange(n_files),
#                 track=track
#             )
#             ana.data_to_rdf()
#             ana.defs()
#             ana.cuts(cut=trigger)
#             ana.cuts()
#             ana.init_hist_params()
#             cols = sorted(set(ana.hist_params.keys()) | {"Dp_M", "Dp_charge", "KS_LT"})
#             ana.rdf.Snapshot("DecayTree", snapshot_path, cols)


