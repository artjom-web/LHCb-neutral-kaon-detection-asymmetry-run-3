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


# def init_hists(ana):
#     for key, value in self.hist_params.items():
#         ana.hists[key] = self.rdf.Histo1D((f'h_{key}', key, *value), key)

# hist_params = {
#     "Dp_M":             [100, 1800, 1950],
#     "KS_OWNPVLTIME":    [7, 0, 0.4],
#     "Pip_k":            [100, 0.0, 0.08],
#     "Pip_theta_x":      [100, -0.15, 0.15], 
#     "Pip_theta_y":      [100, -0.05, 0.05], 
#     "Dp_HLT2_ETA":      [100, 2.2,   4.2], 
#     "Dp_HLT2_PT":       [100, 2500,  12000], 
#     "Pip_HLT2_PHI":     [100, -3.2,  3.2], 
#     "Pip_HLT2_ETA":     [100, 2.2,   4.2],  
#     "Pip_HLT2_PT":      [100, 1500, 6000 ],  
#     "Dp_theta_x":       [100, -1.6, 1.6], 
#     "Dp_theta_y":       [100, -1.6, 1.6], 
# }


# filenames = [
# 'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000142_1.d2kspi_ll.root',
# 'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000268_1.d2kspi_ll.root',
# 'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000230_1.d2kspi_ll.root',
# 'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000397_1.d2kspi_ll.root',
# 'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000360_1.d2kspi_ll.root'
# ]


ROOT.EnableImplicitMT()
ana = Analysis()        
track = 'dd'
ycset = '25c4'
ana.track = track
ana.min_m = 0
ana.max_m = 2500

ana.import_data(
polarity="magdown",
ycset=ycset,
datasetnumbers=np.arange(2),
track=track
)
print('import ok')
folder = f'/eos/user/a/ahulsber/scripts/data/paramer/{track}/{ycset}_nocut'
os.makedirs(folder, exist_ok=True)

ana.init_hist_params()
ana.data_to_rdf()
ana.defs()
# if track == 'll':
#         ana.cuts()
# elif track == 'dd':
#         ana.cuts_DD()
# else: print("WROONGNGN")

ana.plot_dist(folder, params = ["KS_Hlt1TwoTrackKsDecision_TOS","Pip_Hlt1TrackMVADecision_TOS",  ], bins = 100)

# KS0_Hlt1DownstreamKsToPiPiDecision_TOS

# Pip_Hlt1TrackMVADecision_TOS
# KS_Hlt1TwoTrackKsDecision_TOS
# KS_Hlt1DownstreamKsToPiPiDecision_TOS