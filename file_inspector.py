import ROOT
from ROOT import (RooRealVar, RooGaussian, RooExponential, RooAddPdf, 
    RooDataHist, RooFit, RooPlot, TCanvas, RooArgList, TLegend, RooArgSet, RooFormulaVar,
    RooBifurGauss, RooJohnson, RooProdPdf, RooGenericPdf,
    RooSimultaneous, RooCategory, RooFit, RooStats)
import io
import contextlib
import math

from apd import AnalysisData
import numpy as np
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
import time
import array
import cppyy
import sys



def walk(directory, indent=0):
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        print("  " * indent + f"{obj.ClassName():20} {obj.GetName()}")
        if obj.InheritsFrom("TDirectory"):
            walk(obj, indent + 1)

def import_data(polarity, ycset, datasetnumbers, track):
        datasets = AnalysisData("charm", "d_to_ksh")
        dataname = f"d_to_ksh_{ycset}_{polarity},_split_d2kspi_{track}"
        print(dataname)
        y = ycset.split("c")[0]
        result = datasets(
            # config="lhcb",
            # datatype=f"20{y}",
            # filetype=f"d2kspi_{track}.root",
            # polarity=polarity,
            # eventtype="94000000",
            name = dataname,
            version = 'v1r5360'
        )
        for i in datasetnumbers:
            fname = result[i]
            print(fname)
            f = ROOT.TFile.Open(fname)
            walk(f)


# track = 'dd'
# import_data(
# polarity="magup",
# ycset='24c2a',
# datasetnumbers=np.arange(2),
# track=track
# )
# print('import ok')
files = [
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000268_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000230_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000397_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000360_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000086_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000256_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000306_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000404_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000083_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000039_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000303_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000229_1.d2kspi_ll.root',
'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000362_1.d2kspi_ll.root',

]

datasets = AnalysisData("charm", "d_to_ksh")
result = datasets(
    config="lhcb",
    datatype=f"2025",
    filetype=f"d2kspi_dd.root",
    polarity='magup',
    eventtype="94000000",
    name = "d_to_ksh_25c3_magup,_split_d2kspi_dd",
)

files_set = set(files)

import ROOT


# f = ROOT.TFile.Open(result[5])

# def print_root_structure(directory, indent=""):
#     for key in directory.GetListOfKeys():
#         obj = key.ReadObj()

#         if obj.InheritsFrom("TDirectory"):
#             print(f"{indent}📁 {obj.GetName()}/")
#             print_root_structure(obj, indent + "    ")

#         elif obj.InheritsFrom("TTree"):
#             print(f"{indent}🌳 {obj.GetName()}")

#         else:
#             print(f"{indent}{obj.ClassName()}: {obj.GetName()}")

# print(f"\nFile: {result[5]}")
# print_root_structure(f)

f = ROOT.TFile.Open(result[5])

print(f"\nFile: {result[5]}")
tree = f.Get("D2KSpi_DD/DecayTree")

print("Entries:", tree.GetEntries())
print("\nBranches:")

for branch in tree.GetListOfBranches():
    print("  ", branch.GetName())

f.Close()

# fname = 'root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision25/D2KSPI_LL.ROOT/00373382/0000/00373382_00000268_1.d2kspi_ll.root'

# f = ROOT.TFile.Open(fname)
# walk(f)


