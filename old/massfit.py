import ROOT
from ROOT import RooRealVar, RooGaussian, RooExponential, RooAddPdf, RooDataHist, RooFit, RooPlot, TCanvas, RooArgList, TLegend
from apd import AnalysisData
import numpy as np
import os
from datetime import datetime
import csv

def massfit(cset = 2, datasetnumber = 0, charge = 0):
    xmin, xmax = 1800, 2040
    ltime_bins = 10
    nbins = 100
    timestring = datetime.now().strftime("%d_%H:%M")
    datapath = f"./results/cset{cset}/number{datasetnumber}/charge{charge}/"
    os.makedirs(datapath, exist_ok=True)
    outfile = datapath + '/fitresults.csv'

    # Write header once when file is created
    if not os.path.exists(outfile):
        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "fit_index",
                "mu", "mu_err",
                "sigma", "sigma_err",
                # "sig_frac", "sig_frac_err",
                "a", "a_err",
                "n_sig", "n_sig_err",
                "n_bkg", "n_bkg_err",
                "fit_status", "edm", 
                "maxltime", "n_evts"
            ])


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

    f = ROOT.TFile.Open(result[datasetnumber])
    d = f.Get("D2KSpi_DD")
    tree = d.Get("DecayTree")
    rdf = ROOT.RDataFrame(tree)
    rdf = rdf.Filter(f"Dp_M > {xmin} && Dp_M < {xmax}")
    rdf = rdf.Filter("Pip_PARTICLE_ID > 0") if charge > 0 else rdf.Filter("Pip_PARTICLE_ID < 0")   

    # --- Pull both columns ---
    Dp_M_values       = rdf.AsNumpy(columns=["Dp_M"])["Dp_M"]
    KS_OWNPVLTIME     = rdf.AsNumpy(columns=["KS_OWNPVLTIME"])["KS_OWNPVLTIME"]

    # Remove NaNs before computing bin edges
    valid_mask    = np.isfinite(KS_OWNPVLTIME)
    KS_ltime_clean = KS_OWNPVLTIME[valid_mask]
    Dp_M_clean     = Dp_M_values[valid_mask]   # keep arrays aligned!

    # --- Equal-statistics ltime bin edges ---
    quantiles  = np.linspace(0, 100, ltime_bins + 1)
    bin_edges  = np.percentile(KS_ltime_clean, quantiles)

    print(f"Total events: {len(KS_OWNPVLTIME)},  after NaN removal: {len(KS_ltime_clean)}")
    print("Ltime bin edges:", bin_edges)

    # --- Loop over ltime bins ---

    for i in range(ltime_bins):
        lt_lo, lt_hi = bin_edges[i], bin_edges[i + 1]

        # Select events in this ltime bin
        mask    = (KS_OWNPVLTIME >= lt_lo) & (KS_OWNPVLTIME < lt_hi)
        m_vals  = Dp_M_values[mask]
        n_evts  = len(m_vals)
        print(f"\n--- Bin {i+1}/{ltime_bins}  ltime=[{lt_lo:.4f}, {lt_hi:.4f})  N={n_evts} ---")

        if n_evts < 10:
            print("  Too few events, skipping.")
            continue

        # Build histogram
        hist = ROOT.TH1F(f"hist_{i}", f"hist_{i}", nbins, xmin, xmax)
        for v in m_vals:
            hist.Fill(v)

        # RooFit objects — fully independent per bin
        x = RooRealVar("x", "x", xmin, xmax)
        data = RooDataHist(f"data_{i}", f"data_{i}", RooArgList(x), hist)

        mu       = RooRealVar(f"mu_{i}",       "mu",       1880,  xmin, xmax)
        sigma    = RooRealVar(f"sigma_{i}",    "sigma",    10,    0.1,    100)
        # sig_frac = RooRealVar(f"sig_frac_{i}", "sig_frac", 0.008, 0,    1)
        a        = RooRealVar(f"a_{i}",        "a",        0.01,  0.000001, 100)

        gauss = RooGaussian(   f"gauss_{i}", "Gaussian",   x, mu, sigma)
        exp   = RooExponential(f"exp_{i}",   "Exponential", x, a)

        n_sig = RooRealVar("n_sig", "n_sig", n_evts*0.05, 0.1, n_evts)
        n_bkg = RooRealVar("n_bkg", "n_bkg", n_evts*0.95, 0.1, n_evts)

        model = RooAddPdf("model", "Gauss+Exp",
                        RooArgList(gauss, exp),
                        RooArgList(n_sig, n_bkg))  # two coefficients → extended

        fit_result = model.fitTo(data, 
            ROOT.RooFit.Extended(True),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Minimizer("Minuit2", "migrad"),
            ROOT.RooFit.Strategy(1),
            ROOT.RooFit.Save()
        )

        print(f"  Fit status: {fit_result.status()}  EDM: {fit_result.edm():.2e}")
        fit_result.floatParsFinal().Print("s")
        
        with open(outfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i,
                mu.getVal(),       mu.getError(),
                sigma.getVal(),    sigma.getError(),
                # sig_frac.getVal(), sig_frac.getError(),
                a.getVal(),        a.getError(),
                n_sig.getVal(),     n_sig.getError(),
                n_bkg.getVal(),     n_bkg.getError(),
                fit_result.status(),
                fit_result.edm(),
                lt_hi,
                n_evts
            ])



        # Plot
        # xframe = x.frame()
        # xframe.SetTitle(f"ltime bin {i+1}: [{lt_lo:.4f}, {lt_hi:.4f})  N={n_evts}")

        # data.plotOn(xframe,  ROOT.RooFit.Name("data"))
        # model.plotOn(xframe, ROOT.RooFit.Name("model"))
        # model.plotOn(xframe,
        #             ROOT.RooFit.Components(f"gauss_{i}"),
        #             ROOT.RooFit.LineColor(ROOT.kRed),
        #             ROOT.RooFit.LineStyle(ROOT.kDashed),
        #             ROOT.RooFit.Name("sig"))
        # model.plotOn(xframe,
        #             ROOT.RooFit.Components(f"exp_{i}"),
        #             ROOT.RooFit.LineColor(ROOT.kGreen + 2),
        #             ROOT.RooFit.LineStyle(ROOT.kDashed),
        #             ROOT.RooFit.Name("bkg"))

        # legend = TLegend(0.65, 0.65, 0.88, 0.88)
        # legend.SetTextSize(0.03)
        # legend.SetBorderSize(0)
        # legend.SetFillStyle(0)
        # legend.AddEntry(xframe.findObject("data"),  "Data",              "P")
        # legend.AddEntry(xframe.findObject("model"), "Total fit",         "L")
        # legend.AddEntry(xframe.findObject("sig"),   "Signal (Gaussian)", "L")
        # legend.AddEntry(xframe.findObject("bkg"),   "Background (Exp)",  "L")

        # c = TCanvas(f"c_{i}", f"c_{i}", 800, 600)
        # xframe.Draw()
        # legend.Draw("SAME")
        # c.SaveAs(datapath + f"/{i+1}.png")
        # c.Close()

        # Cleanup to avoid RooFit name-clash warnings on next iteration
        del x, data, mu, sigma, a, gauss, exp, model, fit_result, hist, n_bkg, n_sig
massfit()