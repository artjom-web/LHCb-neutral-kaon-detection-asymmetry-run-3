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


ROOT.EnableImplicitMT()
base_folder = f'/eos/user/a/ahulsber/scripts/data/31/signal_eff/'
ana = Analysis()        
ana.import_data(polarity = "magup", csets = [4], datasetnumbers=np.arange(0,25))
ana.import_data(polarity = "magdown", csets = [4], datasetnumbers=np.arange(0,25))
ana.data_to_rdf()
ana.defs()
# ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
# ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
# ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
# ana.mass_fit(ana.hists_filled['Dp_M_p'], ana.hists_filled['Dp_M_m'], base_folder + 'nocut/' + 'fit0/')

# ana.plot_massfit(base_folder + 'nocut/' + 'fit0/')
ana.cuts()
ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.mass_fit(ana.hists_filled['Dp_M_p'], ana.hists_filled['Dp_M_m'], base_folder + 'cut/' + 'fit0/')
ana.plot_massfit(base_folder + 'cut/' + 'fit0/')

