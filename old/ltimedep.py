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
from test1.py import cuts

ROOT.EnableImplicitMT(10)

def run_analysis(cset):
    start_time = time.time()

    xmin, xmax = 1800, 1950
    nbins = 100
    ltime_bins = 10
    datasetnumbers = np.arange(0,11)

    folderpath = f'/afs/cern.ch/user/a/ahulsber/private/tests/results/cset{cset}/'
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
    end_time = time.time()
    print('loaded datasets, duration: ', f'{(end_time - start_time):.3f}')
    start_time = time.time()

    rdf = cuts(rdf)

    end_time = time.time()
    print('executed cuts, duration: ', f'{(end_time - start_time):.3f}')
    start_time = time.time()

    # ----------- Filling 3D histogram
    h3 = rdf.Histo3D(
        (
            "h3",
            "",
            200, 0.0, 0.4,      # KS lifetime
            nbins, xmin, xmax,  # Dp mass
            2, 0, 2             # charge (0,1)
        ),
        "KS_OWNPVLTIME",
        "Dp_M",
        "charge"
    )

    hist3 = h3.GetValue()

    end_time = time.time()
    h3_time = end_time - start_time
    h3_events = h3.GetEntries()


    print('h3 filled, duration: ', f'{h3_time:.3f}')
    print(f'h3 events = {h3_events}')
    print(f'time per million events = {(h3_time / h3_events)*10**6:.3f}')
    start_time = time.time()


    # ----------- Finding edges

    h_lt = hist3.ProjectionX()


    probs = ROOT.std.vector("double")()
    for i in range(ltime_bins + 1):
        probs.push_back(i / ltime_bins)

    quant = ROOT.std.vector("double")(ltime_bins + 1)

    h_lt.GetQuantiles(
        ltime_bins + 1,
        quant.data(),
        probs.data()
    )

    bin_edges = [quant[i] for i in range(ltime_bins + 1)]


        
    end_time = time.time()
    print('found edges, duration: ', f'{(end_time - start_time):.3f}')
    start_time = time.time()


    for i in range(ltime_bins):

        lt_lo = hist3.GetXaxis().FindBin(bin_edges[i])
        lt_hi = hist3.GetXaxis().FindBin(bin_edges[i + 1])

        # charge + (bin z=2)
        h_plus = hist3.ProjectionY(
            f"h_plus_{i}",
            lt_lo,
            lt_hi,
            2, 2
        )

        # charge - (bin z=1)
        h_minus = hist3.ProjectionY(
            f"h_minus_{i}",
            lt_lo,
            lt_hi,
            1, 1
        )

        end_time = time.time()
        print(f'made hists for bin {i}, duration: ', f'{(end_time - start_time):.3f}')
        start_time = time.time()

        mass_fit_simpl(
            h_plus,
            h_minus,
            "Dp_M",
            folderpath + f"bin{i}/",
            folderpath + f"bin{i}/figures/",
            first_time_runs=False,
            range_min=-1,
            range_max=-1,
            mass_name="Dp_M",
            tag_name="Pip_PARTICLE_ID",
            plot_figures=(i == 0)
        )



        end_time = time.time()
        print(f'Finished fit for bin {i}, duration: ', f'{(end_time - start_time):.3f}')
        start_time = time.time()

    return bin_edges, h3_time, h3_events

def plot_results(cset, bin_edges):
    lbins = 10
    A_sigs, A_sig_errs, xs, x_errs = [], [], [], []
    folderpath = f'/afs/cern.ch/user/a/ahulsber/private/tests/results/cset{cset}/'
    for i in range(lbins):
        file = folderpath + f'bin{i}/model.root'
        f = ROOT.TFile.Open(file)
        ws = f.Get("ws")
        params = ws.allVars()
        A_sig, A_sig_err = params.find("A_sig").getVal(), params.find("A_sig").getError()
        lt_lo, lt_hi = bin_edges[i], bin_edges[i+1]
        A_sigs.append(A_sig)
        A_sig_errs.append(A_sig_err)
        xs.append((lt_lo+lt_hi)/2)
        x_errs.append((lt_hi-lt_lo)/2)
    plt.errorbar(xs, A_sigs, xerr = x_errs, yerr = A_sig_errs, capsize=1, fmt = '.')
    plt.ylabel('A')
    plt.xlabel('Ks lifetime')
    plt.xlim(0, 0.1)
    plt.savefig(folderpath + 'A_ksltime.png')



bin_edges, h3_time, h3_events = run_analysis(cset=2)
print(bin_edges)
print('h3 filled, duration: ', f'{h3_time:.3f}')
print(f'h3 events = {h3_events}')
print(f'time per million events = {(h3_time / h3_events)*10**6:.3f}')
plot_results(cset=2, bin_edges=bin_edges)

.Snapshot