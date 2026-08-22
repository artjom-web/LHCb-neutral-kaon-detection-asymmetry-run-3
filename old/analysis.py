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
 
# module load lxbatch/eossubmit
# condor_rm ahulsber
# condor_submit /eos/user/a/ahulsber/scripts/HTCondor/subfiles/analysis1.sub
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

ROOT.EnableImplicitMT()
folder = './test2/'
ana = Analysis()        
ana.import_data(polarity = "magup", csets = [4], datasetnumbers=np.arange(0,35))
ana.import_data(polarity = "magdown", csets = [2], datasetnumbers=np.arange(0,35))
ana.data_to_rdf()
ana.cuts()
ana.init_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.fill_hists(["Dp_M_p", 'Dp_M_m', "Dp_M"])
ana.mass_fit(ana.hists_filled['Dp_M_p'], ana.hists_filled['Dp_M_m'], folder)
print("N_plus:", ana.hists_filled['Dp_M_p'].GetEntries())
print("N_minus:", ana.hists_filled['Dp_M_m'].GetEntries())
print("N_Dp_M (unfiltered, unused):", ana.hists_filled['Dp_M'].GetEntries())
print("N_plus + N_minus:", ana.hists_filled['Dp_M_p'].GetEntries() + ana.hists_filled['Dp_M_m'].GetEntries())
time.sleep(3)
ana.plot_massfit(folder)
print('filter done')
ana.init_weights(folder)
print('weighting done')
ana.lt_bin_edges = ana.equal_bins()
print('equal bins made')
ana.plot_kin_dist(folder + 'cycle0', 0)

# triplets2 =[     (
#         "KS_HLT2_PHI",
#         "KS_HLT2_PT",
#         "Dp_HLT2_ETA"
#     ),
    
#     (
#         "KS_HLT2_ETA",
#         "KS_theta_x",
#         "KS_theta_y"
#     ),

#     ]

triplets1 = [     (
        "Pip_HLT2_PHI",
        "Pip_HLT2_ETA",
        "Pip_HLT2_PT"
    ),
    (
        "Dp_k",
        "Dp_HLT2_PT",
        "Dp_HLT2_ETA"
    )
    
    ]

# "KS_HLT2_PHI", "KS_HLT2_PT", "KS_HLT2_ETA", "KS_k", "KS_theta_x", "KS_theta_y"
# triplets1 = [     (
#         "KS_HLT2_PHI", 
#         "Dp_HLT2_PT", 
#         "Pip_HLT2_ETA"
#     ),
#     (   "Pip_HLT2_PHI",
#         "KS_HLT2_PT",
#         "Dp_HLT2_ETA",
#     ),
#     (   "Dp_HLT2_PHI",
#         "Pip_HLT2_PT",
#         "KS_HLT2_ETA",
#     ),
#     ("KS_k", "KS_theta_x", "KS_theta_y"),

    
    
#     ]

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

        ana.init_h3d(
            param1=param1,
            param2=param2,
            param3=param3
        )


for cycle, param1, param2, param3 in triplets:

    ana.make_h3d(
        param1=param1,
        param2=param2,
        param3=param3
    )


 

for cycle, param1, param2, param3 in triplets:

    ana.weighting_cycle(
        param1=param1,
        param2=param2,
        param3=param3
    )
    ana.plot_kin_dist(folder + f"{cycle}" )







    

# # ana.make_h3d(
# #     param1="Pip_k",
# #     param2="Dp_HLT2_PT",
# #     param3="Pip_HLT2_PT"
# # )

# # ana.weighting_cycle_3d(
# #     param1="Pip_k",
# #     param2="Dp_HLT2_PT",
# #     param3="Pip_HLT2_PT"
# # )

# # ana.weighting_cycle_3d(
# #     param1="Pip_k",
# #     param2="Dp_HLT2_PT",
# #     param3="Pip_HLT2_PT"
# # )

# ana.plot_kin_dist(folder + "post_plots" )

triplets = [ 
    (
        "Pip_k",
        "Pip_theta_x",
        "Pip_theta_y"
    ),

    (
        "Pip_k",
        "Dp_HLT2_PT",
        "Dp_HLT2_ETA"
    ),

    (
        "Pip_HLT2_PHI",
        "Pip_HLT2_ETA",
        "Dp_HLT2_PT"
    ),

    (
        "Pip_HLT2_PT",
        "Pip_HLT2_ETA",
        "Dp_HLT2_PT"
    ),

    (
        "Pip_HLT2_PT",
        "Dp_theta_x",
        "Dp_HLT2_PT"
    ),

    (
        "Pip_theta_x",
        "Pip_theta_y",
        "Dp_HLT2_PT"
    ),

    (
        "Pip_k",
        "Dp_HLT2_ETA",
        "Dp_HLT2_PT"
    )
]