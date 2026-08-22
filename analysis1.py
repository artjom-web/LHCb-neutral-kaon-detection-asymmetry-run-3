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
params = [
        "Pip_HLT2_PT", "Pip_HLT2_ETA", "Pip_k", "Pip_theta_x", "Pip_theta_y",
        "Dp_HLT2_PT", "Dp_HLT2_ETA", "Pip_HLT2_PHI", "Dp_theta_x", "Dp_theta_y", 
        "KS_HLT2_PHI", "KS_HLT2_PT", "KS_HLT2_ETA", "KS_k", "KS_theta_x", "KS_theta_y"
]
folder = './signal_eff/'
ROOT.EnableImplicitMT()
ana = Analysis()        
ana.import_data(polarity = "magup", csets = [4], datasetnumbers=np.arange(0,20)) 
# ana.import_data(polarity = "magup", csets = [3], datasetnumbers=np.arange(0,20)) # 130
ana.import_data(polarity = "magdown", csets = [4], datasetnumbers=np.arange(0,20))
# ana.import_data(polarity = "magdown", csets = [2], datasetnumbers=np.arange(0,2)) # 150

ana.data_to_rdf()
ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.mass_fit(ana.hists_filled['Dp_M_p'], ana.hists_filled['Dp_M_m'], folder, new_run= True)
ana.plot_massfit(folder)
# ana.cuts()
# ana.find_Adep(params, folder)
# ana.plot_Adep(params, folder)
# ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
# ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
# ana.plot_filter(folder)   