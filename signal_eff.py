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
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--hlt1_cut", type=str, required=True)
parser.add_argument("--track", type=str, required=True, choices=['ll', 'dd'])
parser.add_argument("--ycset", type=str, required=True, choices=['25c1', '25c2', '25c3', '25c4'])
parser.add_argument("--polarity", type=str, required=True, choices=['magup', 'magdown'])
parser.add_argument("--day", type=str, required=True)
parser.add_argument("--time", type=str, required=True)
args = parser.parse_args()
hlt1_cut = args.hlt1_cut

base_folder = f'/eos/user/a/ahulsber/scripts/data/sig_eff/{args.day}/{hlt1_cut}_{args.time}/'
cuts = ["mass_lt", hlt1_cut, "kin", "probnn"]

ana = Analysis(args.track)
folder = base_folder + f'{args.track}_{args.ycset}_{args.polarity}/'
os.makedirs(folder, exist_ok=True)

ana.import_data(polarity=args.polarity, ycset=args.ycset,
                datasetnumbers=np.arange(10), track=args.track)
ana.data_to_rdf()
ana.defs()
ana.init_hist_params()

for cut in cuts:
    ana.cuts(cut)
    ana.init_massfit(folder + cut + '/')
    ana.plot_massfit(folder + cut + '/')