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




day = datetime.now().day
# params = [
#         "Pip_HLT2_PT", "Pip_HLT2_ETA", "Pip_k", "Pip_theta_x", "Pip_theta_y",
#         "Dp_HLT2_PT", "Dp_HLT2_ETA", "Pip_HLT2_PHI", "Dp_theta_x", "Dp_theta_y", 
#         "KS_HLT2_PHI", "KS_HLT2_PT", "KS_HLT2_ETA", "KS_k", "KS_theta_x", "KS_theta_y"
# ]
params = ["KS_OWNPVLTIME"]
ROOT.EnableImplicitMT()
ana = Analysis()  
ana.import_data(polarity = "magup", csets = [4], datasetnumbers=np.arange(0,50))
ana.import_data(polarity = "magdown", csets = [4], datasetnumbers=np.arange(0,50))
base_folder = f'/eos/user/a/ahulsber/scripts/data/31/ltdep/'
print('startup ok')
ana.data_to_rdf()
ana.defs()
ana.cuts()
ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.mass_fit(ana.hists_filled['Dp_M_p'], ana.hists_filled['Dp_M_m'], base_folder + 'fit0/')
ana.plot_massfit(base_folder + 'fit0/')
ana.init_weights(base_folder + 'fit0/')
ana.lt_bin_edges = ana.equal_bins()

ana.find_Adep(params, base_folder + 'before/')
ana.plot_Adep(params, base_folder + 'before/')


