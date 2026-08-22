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


folder = "/eos/user/a/ahulsber/scripts/data/08-20/"
time = "15hr54"

for entry in sorted(os.listdir(folder)):
    entry_path = os.path.join(folder, entry)
    if not os.path.isdir(entry_path) or not entry.endswith(f"_{time}"):
        continue

    prefix = entry.split("_")[0]
    total_n_sig = 0.0

    for sub_entry in sorted(os.listdir(entry_path)):
        sub_path = os.path.join(entry_path, sub_entry)
        if not os.path.isdir(sub_path) or not sub_entry.startswith(f"{prefix}_"):
            continue

        root_path = os.path.join(sub_path, "initial_massfit", "model.root")
        if not os.path.exists(root_path):
            print(f"WARNING: {root_path} not found, skipping")
            continue

        f = ROOT.TFile.Open(root_path)
        ws = f.Get("ws")
        n_sig = ws.function("N_sig")
        if n_sig is None:
            print(f"WARNING: N_sig not found in {root_path}, skipping")
            f.Close()
            continue

        val = n_sig.getVal()
        total_n_sig += val
        f.Close()

    print(f"{entry}: N_sig sum = {total_n_sig:.2e}")