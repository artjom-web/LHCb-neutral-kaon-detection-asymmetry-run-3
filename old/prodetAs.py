import ROOT
from ROOT import RooRealVar, RooGaussian, RooExponential, RooAddPdf, RooDataHist, RooFit, RooPlot, TCanvas, RooArgList, TLegend, RooArgSet
from apd import AnalysisData
import numpy as np
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
from massfit3 import mass_fit_simpl
import time
from test1 import cuts

datasetnumber = 0
datasetnumbers = np.arange(0,15)

cset = 2
ltime_bins = 10

datasets = AnalysisData("charm", "d_to_ksh")
dataname = f"d_to_ksh_25c{cset}_magdown,_split_d2kspi_ll"

result = datasets(
    config="lhcb",
    datatype="2025",
    filetype="d2kspi_ll.root",
    polarity="magdown",
    eventtype="94000000",
    name=dataname
)
files = [result[i] for i in datasetnumbers]
rdf = ROOT.RDataFrame("D2KSpi_DD/DecayTree", files)


rdf = cuts(rdf)

ROOT.gStyle.SetOptStat(0)
os.makedirs("plots", exist_ok=True)

variables = [
    ("Pip_k",        100, 0.0,   0.08),
    ("theta_x",      100, -0.15, 0.15),
    ("theta_y",      100, -0.05, 0.05),
    ("Dp_HLT2_ETA",  100, 2.2,   4.2),
    ("Dp_HLT2_PT",   100, 2500,  12000),
    ("Pip_HLT2_PHI", 100, -3.2,  3.2),
]

rdf_lt_lo  = rdf.Filter("charge==1")
rdf_lt_hi = rdf.Filter("charge==0")

rdf_plus = rdf_lt_lo
rdf_minus = rdf_lt_hi

hplus = {}
hminus = {}

# Book all histograms
for var, nbins, xmin, xmax in variables:
    hplus[var] = rdf_plus.Histo1D(
        (f"hplus_{var}", "", nbins, xmin, xmax),
        var
    )

    hminus[var] = rdf_minus.Histo1D(
        (f"hminus_{var}", "", nbins, xmin, xmax),
        var
    )

# Trigger ONE event loop
for h in hplus.values():
    h.GetValue()

# Produce plots
for var, nbins, xmin, xmax in variables:

    hp = hplus[var].GetValue()
    hm = hminus[var].GetValue()

    c = ROOT.TCanvas(f"c_{var}","",800,600)

    hp.SetLineColor(ROOT.kRed)
    hm.SetLineColor(ROOT.kBlue)

    hp.SetLineWidth(2)
    hm.SetLineWidth(2)

    hp.SetMaximum(1.15*max(hp.GetMaximum(), hm.GetMaximum()))

    hp.Draw("hist")
    hm.Draw("hist same")

    leg = ROOT.TLegend(0.70,0.75,0.88,0.88)
    leg.AddEntry(hp,"D^{+}","l")
    leg.AddEntry(hm,"D^{-}","l")
    leg.Draw()

    c.SaveAs(f"plots/{var}_overlay.png")

    hasym = hp.Clone(f"asym_{var}")
    hasym.Reset()

    for b in range(1, nbins+1):

        p = hp.GetBinContent(b)
        m = hm.GetBinContent(b)

        if p+m:

            hasym.SetBinContent(b, (p-m)/(p+m))
            hasym.SetBinError(
                b,
                (2*p*m/(p+m)**3)**0.5
            )

    c2 = ROOT.TCanvas(f"ca_{var}","",800,600)

    hasym.SetMinimum(-0.1)
    hasym.SetMaximum(0.1)
    hasym.SetMarkerStyle(20)
    hasym.Draw("E1")

    ROOT.TLine(xmin,0,xmax,0).Draw()

    c2.SaveAs(f"plots/{var}_asymmetry.png")
