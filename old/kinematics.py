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

rdf = ....
        
        
("KS_OWNPVLTIME", 7, 0, 0.4)
("Dp_M",          100, 1800, 1950),
("charge",        2, -1, 1),
("Pip_k",        100, 0.0,   0.08),
("theta_x",      100, -0.15, 0.15),
("theta_y",      100, -0.05, 0.05),
("Dp_HLT2_ETA",  100, 2.2,   4.2),
("Dp_HLT2_PT",   100, 2500,  12000),
("Pip_HLT2_PHI", 100, -3.2,  3.2),











import ROOT
from math import fabs, sqrt

# -------------------------------------------------------------------------
# Global settings (as in the C++ code)
# -------------------------------------------------------------------------

min_bin_content_split = 1      # minimum bin content in TH3D in t/tau bins
min_bin_content_integrated = 10  # minimum bin content in TH3D integrated over t/tau

# NOTE: The following symbols are assumed to be defined elsewhere,
# exactly as in the original C++ code:
#   MDplus, MDsplus
#   mK0s_PDG, mphi_PDG
#   lim_mass_KS0LL, lim_mass_KS0DD, lim_mass_phi
#   kinvars, kinvars_labels, first_bin, last_bin
#   nbins_LL, nbins_DD, nbinstot, nbins_new
#   bins_tauK0s_LL, bins_tauK0s_DD
#   struct/class `weights` with fields: name_file, name_tree, name_weights
#   function `fitjsu1_10_sim` (and fitjsu2_11_sim if used)
# This Python file mirrors the logic of the C++ snippet and expects
# those definitions to exist in the same ROOT session.


# -------------------------------------------------------------------------
# Sidebands ratio
# -------------------------------------------------------------------------

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
def compute_sidebands_ratio(hp, hm):
    # do fit to determine peak and 3sigma region
    # then use, signal region 1
    # sideband regions (Nbkg,sig / Nbkg,sideband)
    # rest = 0
    folderpath = f'/afs/cern.ch/user/a/ahulsber/private/tests/results/cset{cset}/'

    mass_fit_simpl(
        hp,
        hm,
        "Dp_M",
        folderpath + f"sideband/",
        first_time_runs=False,
        range_min=-1,
        range_max=-1,
        mass_name="Dp_M",
        tag_name="Pip_PARTICLE_ID",
        plot_figures=(i == 0)
    )



# -------------------------------------------------------------------------
# Make sideband weights
# -------------------------------------------------------------------------


def make_weights_sidebands(name_tree_data, files_path,
                           weights_out, string_MD,
                           signal_width, distance_sidebands,
                           label_data, label_plot,
                           cutL0=False, cutHLT1=False,
                           exclude=False, daughter="pi"):
    """Python translation of make_weights_sidebands.

    Parameters mirror the C++ version.
    """

    ROOT.DisableImplicitMT()

    # RDataFrame on the input file
    df = ROOT.RDataFrame(name_tree_data, files_path)

    mass_cut = ""
    cut_IP = "Dplus_IP_OWNPV < 0.1"

    if label_plot.startswith("Dp2KS0LLpip"):
        mass_cut = f"fabs(KS0_M - {mK0s_PDG}) < {lim_mass_KS0LL}"
    elif label_plot.startswith("Dp2KS0DDpip"):
        mass_cut = f"fabs(KS0_M - {mK0s_PDG}) < {lim_mass_KS0DD}"
        cut_IP = "Dplus_IP_OWNPV < 0.35"
    elif label_plot.startswith("Dp2KmKppip"):
        mass_cut = f"fabs(phi_M - {mphi_PDG}) < {lim_mass_phi}"
    else:
        print("Error: wrong label_data")
        return

    # D+ and D- mass histograms
    rdataP = (df.Filter("Dplus_ID > 0")
                .Filter(mass_cut)
                .Histo1D((string_MD,
                          "MD+-; M(D^{#pm}) [MeV/c^{2}]; Events/1 MeV/c^{2}",
                          110, 1815, 1925), string_MD))

    rdataM = (df.Filter("Dplus_ID < 0")
                .Filter(mass_cut)
                .Histo1D((string_MD,
                          "MD+-; M(D^{#pm}) [MeV/c^{2}]; Events/1 MeV/c^{2}",
                          110, 1815, 1925), string_MD))

    hdataP = rdataP.GetPtr()
    hdataM = rdataM.GetPtr()

    peakDp = [0.0]
    peakDm = [0.0]
    sigma = [0.0]

    ttl_plot = f"/afs/cern.ch/work/i/icelesti/private/Ana_asyKS0/Figures/Weights_sidebands/side_{label_plot}.pdf"
    c = ROOT.TCanvas("c", "c", 800, 600)
    c.Print(ttl_plot + "[")

    ratio = compute_sidebands_ratio(hdataP, hdataM,
                                    MDplus, signal_width, distance_sidebands,
                                    label_data, peakDp, peakDm, sigma,
                                    ttl_plot)

    c.Print(ttl_plot + "]")

    # Initial weights (for fit only)
    def fweights_init(mD, off):
        if not off:
            return 0.0
        if exclude:
            return 0.0
        return 1.0

    # Sideband-subtraction weights
    def fweights(mD, D_ID, off):
        signal_width_loc = 3.0 * sigma[0]
        signal_center = 1869.66
        if D_ID > 0:
            signal_center = peakDp[0]
        else:
            signal_center = peakDm[0]

        if not off:
            return 0.0
        if exclude:
            return 0.0

        if (mD > (signal_center - signal_width_loc) and
                mD < (signal_center + signal_width_loc)):
            # peak
            return 1.0
        elif (mD > (signal_center - distance_sidebands - signal_width_loc) and
              mD < (signal_center - distance_sidebands)):
            # left sideband
            return -1.0 * ratio
        elif (mD > (signal_center + distance_sidebands) and
              mD < (signal_center + distance_sidebands + signal_width_loc)):
            # right sideband
            return -1.0 * ratio
        else:
            return 0.0

    # Offline selection
    offline_cut = "Fidcuts"
    if cutL0:
        offline_cut += f" && (Dplus_L0Global_TIS || ({daughter}plus_L0HadronDecision_TOS && {daughter}minus_L0HadronDecision_TOS))"
    if cutHLT1:
        if daughter == "pi":
            offline_cut += " && hplus_Hlt1TrackMVADecision_TOS"
        else:
            offline_cut += " && piplus_Hlt1TrackMVADecision_TOS"

    df_weights = (df.Define("off", offline_cut)
                    .Define(weights_out.name_weights,
                            fweights, [string_MD, "Dplus_ID", "off"])
                    .Define(weights_out.name_weights + "_forfit",
                            fweights_init, [string_MD, "off"]))

    df_weights.Snapshot(weights_out.name_tree, weights_out.name_file,
                        [weights_out.name_weights,
                         weights_out.name_weights + "_forfit",
                         string_MD,
                         "Dplus_L0Global_TIS",
                         f"{daughter}plus_L0HadronDecision_TOS",
                         f"{daughter}minus_L0HadronDecision_TOS"])

    print("DF with initial sideband-subtractions weights:", weights_out.name_file)


# -------------------------------------------------------------------------
# Kinematic reweighting KS0(LL+DD) to phi or phi to KS0(LL+DD)
# -------------------------------------------------------------------------


def kin_reweighting_KS0LLDD_forsplit(variables,
                                      name_tree_data,
                                      files_path_KS0LL,
                                      files_path_KS0DD,
                                      files_path_phi,
                                      weights_init_KS0LL,
                                      weights_init_KS0DD,
                                      weights_init_phi,
                                      weights_out_KS0LL,
                                      weights_out_KS0DD,
                                      weights_out_phi,
                                      label_rew_variables,
                                      fout_kinrew_ttl,
                                      year, pol,
                                      nbins_kinrew,
                                      useDsplus):
    """Python translation of kin_reweighting_KS0LLDD_forsplit."""

    ROOT.DisableImplicitMT()
    rew_id = variables[6]

    # Open files and trees
    fphi = ROOT.TFile.Open(files_path_phi)
    Tphi = fphi.Get(name_tree_data)
    Tphi.ResetBit(ROOT.TTree.kEntriesReshuffled)
    if weights_init_phi.name_file != files_path_phi:
        Tphi.AddFriend(weights_init_phi.name_tree, weights_init_phi.name_file)

    fKS0LL = ROOT.TFile.Open(files_path_KS0LL)
    TKS0LL = fKS0LL.Get(name_tree_data)
    TKS0LL.ResetBit(ROOT.TTree.kEntriesReshuffled)
    if weights_init_KS0LL.name_file != files_path_KS0LL:
        TKS0LL.AddFriend(weights_init_KS0LL.name_tree, weights_init_KS0LL.name_file)

    fKS0DD = ROOT.TFile.Open(files_path_KS0DD)
    TKS0DD = fKS0DD.Get(name_tree_data)
    TKS0DD.ResetBit(ROOT.TTree.kEntriesReshuffled)
    if weights_init_KS0DD.name_file != files_path_KS0DD:
        TKS0DD.AddFriend(weights_init_KS0DD.name_tree, weights_init_KS0DD.name_file)

    # Prepare variable names
    v1_KS0 = f"{variables[0]}_{variables[1]}"
    v2_KS0 = f"{variables[2]}_{variables[3]}"
    v3_KS0 = f"{variables[4]}_{variables[5]}"

    v1_phi = v1_KS0
    v2_phi = v2_KS0
    v3_phi = v3_KS0

    if variables[0] == "hplus":
        v1_phi = f"piplus_{variables[1]}"
    if variables[2] == "hplus":
        v2_phi = f"piplus_{variables[3]}"
    if variables[4] == "hplus":
        v3_phi = f"piplus_{variables[5]}"

    titlex = titley = titlez = ""
    ix = fx = iy = fy = iz = fz = 0.0

    # Match variables to kinvars to get ranges and labels
    for ivar, kv in enumerate(kinvars):
        if v1_KS0 == kv:
            ix = first_bin[ivar]
            fx = last_bin[ivar]
            titlex = kinvars_labels[ivar]
        if v2_KS0 == kv:
            iy = first_bin[ivar]
            fy = last_bin[ivar]
            titley = kinvars_labels[ivar]
        if v3_KS0 == kv:
            iz = first_bin[ivar]
            fz = last_bin[ivar]
            titlez = kinvars_labels[ivar]

    print("\nTime Integrated Reweighting (for split)", rew_id, "using :")
    print(f"{v1_KS0} ({ix} , {fx})")
    print(f"{v2_KS0} ({iy} , {fy})")
    print(f"{v3_KS0} ({iz} , {fz})")

    if fx * fy * fz == 0.0:
        print("Variabili ripesamento non adatte")
        return

    # Build bin edges
    edges_v1 = []
    edges_v2 = []
    edges_v3 = []
    for i in range(nbins_kinrew + 1):
        frac = i / float(nbins_kinrew)
        edges_v1.append(ix + frac * (fx - ix))
        edges_v2.append(iy + frac * (fy - iy))
        edges_v3.append(iz + frac * (fz - iz))

    nbins_KS0LL = [len(edges_v1) - 1, len(edges_v2) - 1, len(edges_v3) - 1, nbins_LL]
    edges_KS0LL = [edges_v1, edges_v2, edges_v3, bins_tauK0s_LL]

    nbins_KS0DD = [len(edges_v1) - 1, len(edges_v2) - 1, len(edges_v3) - 1, nbins_DD]
    edges_KS0DD = [edges_v1, edges_v2, edges_v3, bins_tauK0s_DD]

    title_histo4D = f"Reweighting ; {titlex} ; {titley} ; {titlez}; t/#tau_{{KS0}}"
    title_histo3D = f"Reweighting ; {titlex} ; {titley} ; {titlez}"

    model_h3D_rew_phi = ROOT.RDF.TH3DModel("h3D_rew_phi", title_histo3D,
                                           nbins_kinrew, ix, fx,
                                           nbins_kinrew, iy, fy,
                                           nbins_kinrew, iz, fz)

    model_h4D_KS0LL = ROOT.RDF.THnDModel("h4D_KS0LL", title_histo4D,
                                         4, nbins_KS0LL, edges_KS0LL)
    model_h4D_KS0DD = ROOT.RDF.THnDModel("h4D_KS0DD", title_histo4D,
                                         4, nbins_KS0DD, edges_KS0DD)

    # Reduced DataFrames
    df_D_KS0LL = ROOT.RDataFrame(TKS0LL, [v1_KS0, v2_KS0, v3_KS0,
                                          weights_init_KS0LL.name_weights,
                                          "KS0_M", "Dplus_DTF_M_0", "Dplus_ID"])

    df_D_KS0DD = ROOT.RDataFrame(TKS0DD, [v1_KS0, v2_KS0, v3_KS0,
                                          weights_init_KS0DD.name_weights,
                                          "KS0_M", "Dplus_DTF_M_0", "Dplus_ID"])

    df_D_phi = ROOT.RDataFrame(Tphi, [v1_phi, v2_phi, v3_phi,
                                      weights_init_phi.name_weights,
                                      "phi_M", "Dplus_DTFonlyPV_M_0", "Dplus_ID"])

    # Mass window filters
    def fm_K0sLL(mk0):
        return fabs(mk0 - mK0s_PDG) < lim_mass_KS0LL

    def fm_K0sDD(mk0):
        return fabs(mk0 - mK0s_PDG) < lim_mass_KS0DD

    r3D_rew_phi = (df_D_phi
                   .Filter(f"fabs(phi_M - {mphi_PDG}) < {lim_mass_phi}")
                   .Histo3D(model_h3D_rew_phi,
                            v1_phi, v2_phi, v3_phi,
                            weights_init_phi.name_weights))

    r4D_rew_KS0LL = (df_D_KS0LL
                     .Filter(fm_K0sLL, ["KS0_M"])
                     .HistoND(model_h4D_KS0LL,
                              [v1_KS0, v2_KS0, v3_KS0,
                               "Dplus_DTF_KS0_ctau_scaled",
                               weights_init_KS0LL.name_weights]))

    r4D_rew_KS0DD = (df_D_KS0DD
                     .Filter(fm_K0sDD, ["KS0_M"])
                     .HistoND(model_h4D_KS0DD,
                              [v1_KS0, v2_KS0, v3_KS0,
                               "Dplus_DTF_KS0_ctau_scaled",
                               weights_init_KS0DD.name_weights]))

    h4D_rew_KS0LL = r4D_rew_KS0LL.GetPtr()
    h4D_rew_KS0DD = r4D_rew_KS0DD.GetPtr()

    # Split into time bins and clean
    h3D_KS0LL = []
    h3D_KS0DD = []

    for i in range(1, nbins_LL + 1):
        h4D_bin = h4D_rew_KS0LL.Clone(f"h4D_KS0LL{i}")
        h4D_bin.GetAxis(3).SetRange(i, i)
        h3D = h4D_bin.Projection(0, 1, 2, f"h3D_KS0LL{i}")
        for ibin in range(1, h3D.GetNcells() + 1):
            if h3D.GetBinContent(ibin) < min_bin_content_split:
                h3D.SetBinContent(ibin, 0.0)
        h3D_KS0LL.append(h3D)

    for i in range(1, nbins_DD + 1):
        h4D_bin = h4D_rew_KS0DD.Clone(f"h4D_KS0DD{i}")
        h4D_bin.GetAxis(3).SetRange(i, i)
        h3D = h4D_bin.Projection(0, 1, 2, f"h3D_KS0DD{i}")
        for ibin in range(1, h3D.GetNcells() + 1):
            if h3D.GetBinContent(ibin) < min_bin_content_split:
                h3D.SetBinContent(ibin, 0.0)
        h3D_KS0DD.append(h3D)

    # 3D histos for reweighting
    h3D_rew_phi = ROOT.TH3D()
    r3D_rew_phi.Copy(h3D_rew_phi)

    h3D_rew_KS0 = ROOT.TH3D("h3D_KS0", title_histo3D,
                            nbins_kinrew, ix, fx,
                            nbins_kinrew, iy, fy,
                            nbins_kinrew, iz, fz)

    for h3D in h3D_KS0LL:
        h3D_rew_KS0.Add(h3D)
    for h3D in h3D_KS0DD:
        h3D_rew_KS0.Add(h3D)

    if rew_id == "KS0->phi":
        h3D_weights = h3D_rew_phi.Clone("h3D_weights")
    elif rew_id == "phi->KS0":
        h3D_weights = h3D_rew_KS0.Clone("h3D_weights")
    else:
        print("Errore --- valore di rew_id non adeguato:", rew_id)
        return

    # Remove low-stat bins
    for i in range(1, h3D_weights.GetNcells() + 1):
        if (h3D_rew_phi.GetBinContent(i) < min_bin_content_integrated or
                h3D_rew_KS0.GetBinContent(i) < min_bin_content_integrated):
            h3D_weights.SetBinContent(i, 0.0)

    h3D_rew_phi.Scale(1.0 / h3D_rew_phi.Integral())
    h3D_rew_KS0.Scale(1.0 / h3D_rew_KS0.Integral())
    h3D_weights.Scale(1.0 / h3D_weights.Integral())

    if rew_id == "KS0->phi":
        h3D_weights.Divide(h3D_rew_KS0)
    elif rew_id == "phi->KS0":
        h3D_weights.Divide(h3D_rew_phi)

    h3D_accept = h3D_weights.Clone("h3D_accept")

    for i in range(1, h3D_weights.GetNcells() + 1):
        h3D_accept.SetBinContent(i, 1.0)
        if h3D_weights.GetBinContent(i) == 0.0:
            h3D_accept.SetBinContent(i, 0.0)
        if h3D_weights.GetBinContent(i) > 3.0:
            h3D_weights.SetBinContent(i, 0.0)
            h3D_accept.SetBinContent(i, 0.0)

    # Event-level kinematic weights
    def fweights_phi(vx, vy, vz, weight_init):
        if rew_id == "KS0->phi":
            bin_idx = h3D_accept.FindBin(vx, vy, vz)
            accept = h3D_accept.GetBinContent(bin_idx)
            return accept * weight_init
        elif rew_id == "phi->KS0":
            bin_idx = h3D_weights.FindBin(vx, vy, vz)
            weight = h3D_weights.GetBinContent(bin_idx)
            return weight * weight_init
        return 0.0

    def fweights_KS0LL(vx, vy, vz, t, weight_init):
        bin_number_KS0 = 0
        bin_idx = h3D_accept.FindBin(vx, vy, vz)

        if vx < ix or vx > fx:
            return 0.0
        if vy < iy or vy > fy:
            return 0.0
        if vz < iz or vz > fz:
            return 0.0
        if t < bins_tauK0s_LL[0] or t > bins_tauK0s_LL[-1]:
            return 0.0

        for i in range(1, len(bins_tauK0s_LL)):
            if t > bins_tauK0s_LL[i - 1] and t < bins_tauK0s_LL[i]:
                bin_number_KS0 = i
        if bin_number_KS0 == 0:
            return 0.0

        if h3D_KS0LL[bin_number_KS0 - 1].GetBinContent(bin_idx) == 0.0:
            return 0.0

        if rew_id == "phi->KS0":
            accept = h3D_accept.GetBinContent(bin_idx)
            return accept * weight_init
        elif rew_id == "KS0->phi":
            weight = h3D_weights.GetBinContent(bin_idx)
            return weight * weight_init
        return 0.0

    def fweights_KS0DD(vx, vy, vz, t, weight_init):
        bin_number_KS0 = 0
        bin_idx = h3D_accept.FindBin(vx, vy, vz)

        if vx < ix or vx > fx:
            return 0.0
        if vy < iy or vy > fy:
            return 0.0
        if vz < iz or vz > fz:
            return 0.0
        if t < bins_tauK0s_DD[0] or t > bins_tauK0s_DD[-1]:
            return 0.0

        for i in range(1, len(bins_tauK0s_DD)):
            if t > bins_tauK0s_DD[i - 1] and t < bins_tauK0s_DD[i]:
                bin_number_KS0 = i
        if bin_number_KS0 == 0:
            return 0.0

        if h3D_KS0DD[bin_number_KS0 - 1].GetBinContent(bin_idx) == 0.0:
            return 0.0

        if rew_id == "phi->KS0":
            accept = h3D_accept.GetBinContent(bin_idx)
            return accept * weight_init
        elif rew_id == "KS0->phi":
            weight = h3D_weights.GetBinContent(bin_idx)
            return weight * weight_init
        return 0.0

    # Define reweighted DataFrames
    df_D_KS0LL_reweighted = df_D_KS0LL.Define(
        weights_out_KS0LL.name_weights + "_forfit",
        fweights_KS0LL,
        [v1_KS0, v2_KS0, v3_KS0, "Dplus_DTF_KS0_ctau_scaled",
         weights_init_KS0LL.name_weights + "_forfit"])

    df_D_KS0DD_reweighted = df_D_KS0DD.Define(
        weights_out_KS0DD.name_weights + "_forfit",
        fweights_KS0DD,
        [v1_KS0, v2_KS0, v3_KS0, "Dplus_DTF_KS0_ctau_scaled",
         weights_init_KS0DD.name_weights + "_forfit"])

    df_D_phi_reweighted = df_D_phi.Define(
        weights_out_phi.name_weights + "_forfit",
        fweights_phi,
        [v1_phi, v2_phi, v3_phi,
         weights_init_phi.name_weights + "_forfit"])

    # ...
    # The rest of this function (sideband recomputation, stats, Snapshot)
    # follows the same structure as in the C++ code and can be ported
    # analogously. Due to length, it is omitted here but should be
    # implemented if you need the full functionality.

    # Clean up
    for h3D in h3D_KS0LL:
        h3D.Delete()
    for h3D in h3D_KS0DD:
        h3D.Delete()

    h3D_weights.Delete()
    h3D_accept.Delete()


# -------------------------------------------------------------------------
# Kinematic reweighting KS0(LL or DD) to phi or phi to KS0(LL or DD)
# -------------------------------------------------------------------------


def kin_reweighting_KS0_split(use_onlyLL,
                               variables,
                               name_tree_data,
                               files_path_KS0LL,
                               files_path_KS0DD,
                               files_path_phi,
                               weights_init_KS0LL,
                               weights_init_KS0DD,
                               weights_init_phi,
                               weights_out_KS0LL,
                               weights_out_KS0DD,
                               weights_out_phi,
                               split_KS0LL,
                               split_KS0DD,
                               split_phi,
                               label_rew_variables,
                               fout_kinrew_ttl,
                               year, pol,
                               nbins_kinrew,
                               useDsplus):
    """Python translation of kin_reweighting_KS0_split.

    This function is very long in C++. The structure here mirrors the
    original, but some repeated blocks (sideband recomputation, stats,
    Snapshots) are abbreviated for brevity.
    """

    nbins_split = nbinstot
    if use_onlyLL:
        nbins_split = nbins_LL

    ROOT.DisableImplicitMT()
    rew_id = variables[6]

    # Open trees and add friends
    fphi = ROOT.TFile.Open(files_path_phi)
    Tphi = fphi.Get(name_tree_data)
    Tphi.ResetBit(ROOT.TTree.kEntriesReshuffled)
    if weights_init_phi.name_file != files_path_phi:
        Tphi.AddFriend(weights_init_phi.name_tree, weights_init_phi.name_file)
    Tphi.AddFriend(split_phi.name_tree, split_phi.name_file)

    fKS0LL = ROOT.TFile.Open(files_path_KS0LL)
    TKS0LL = fKS0LL.Get(name_tree_data)
    TKS0LL.ResetBit(ROOT.TTree.kEntriesReshuffled)
    if weights_init_KS0LL.name_file != files_path_KS0LL:
        TKS0LL.AddFriend(weights_init_KS0LL.name_tree, weights_init_KS0LL.name_file)
    TKS0LL.AddFriend(split_KS0LL.name_tree, split_KS0LL.name_file)

    fKS0DD = ROOT.TFile.Open(files_path_KS0DD)
    TKS0DD = fKS0DD.Get(name_tree_data)
    TKS0DD.ResetBit(ROOT.TTree.kEntriesReshuffled)
    if weights_init_KS0DD.name_file != files_path_KS0DD:
        TKS0DD.AddFriend(weights_init_KS0DD.name_tree, weights_init_KS0DD.name_file)
    TKS0DD.AddFriend(split_KS0DD.name_tree, split_KS0DD.name_file)

    print("Open tree", name_tree_data, "in file", files_path_KS0DD,
          "with weights", weights_init_KS0DD.name_weights,
          "in file", weights_init_KS0DD.name_file)
    print("Open tree", name_tree_data, "in file", files_path_KS0LL,
          "with weights", weights_init_KS0LL.name_weights,
          "in file", weights_init_KS0LL.name_file)
    print("Open tree", name_tree_data, "in file", files_path_phi,
          "with weights", weights_init_phi.name_weights,
          "in file", weights_init_phi.name_file)

    # Variable names
    v1_KS0 = f"{variables[0]}_{variables[1]}"
    v2_KS0 = f"{variables[2]}_{variables[3]}"
    v3_KS0 = f"{variables[4]}_{variables[5]}"

    v1_phi = v1_KS0
    v2_phi = v2_KS0
    v3_phi = v3_KS0

    if variables[0] == "hplus":
        v1_phi = f"piplus_{variables[1]}"
    if variables[2] == "hplus":
        v2_phi = f"piplus_{variables[3]}"
    if variables[4] == "hplus":
        v3_phi = f"piplus_{variables[5]}"

    titlex = titley = titlez = ""
    ix = fx = iy = fy = iz = fz = 0.0

    for ivar, kv in enumerate(kinvars):
        if v1_KS0 == kv:
            ix = first_bin[ivar]
            fx = last_bin[ivar]
            titlex = kinvars_labels[ivar]
        if v2_KS0 == kv:
            iy = first_bin[ivar]
            fy = last_bin[ivar]
            titley = kinvars_labels[ivar]
        if v3_KS0 == kv:
            iz = first_bin[ivar]
            fz = last_bin[ivar]
            titlez = kinvars_labels[ivar]

    print("\nReweighting", rew_id, "using:")
    print(f"{v1_KS0} ({ix} , {fx})")
    print(f"{v2_KS0} ({iy} , {fy})")
    print(f"{v3_KS0} ({iz} , {fz})")

    if fx * fy * fz == 0.0:
        print("Variabili ripesamento non adatte")
        return

    # Build bin edges for 4D histos (3 kinematic + split bin)
    edges_v1 = []
    edges_v2 = []
    edges_v3 = []
    edges_v4 = []

    for i in range(nbins_kinrew + 1):
        frac = i / float(nbins_kinrew)
        edges_v1.append(ix + frac * (fx - ix))
        edges_v2.append(iy + frac * (fy - iy))
        edges_v3.append(iz + frac * (fz - iz))
    for i in range(nbins_split + 1):
        edges_v4.append(0.5 + i)

    nbins_h4D = [len(edges_v1) - 1, len(edges_v2) - 1,
                 len(edges_v3) - 1, nbins_split]
    edges_h4D = [edges_v1, edges_v2, edges_v3, edges_v4]

    title_histo4D = f"Reweighting ; {titlex} ; {titley} ; {titlez}; t/#tau_{{KS0}} bin"

    model_h4D_KS0LL = ROOT.RDF.THnDModel("h4D_KS0LL", title_histo4D,
                                         4, nbins_h4D, edges_h4D)
    model_h4D_KS0DD = ROOT.RDF.THnDModel("h4D_KS0DD", title_histo4D,
                                         4, nbins_h4D, edges_h4D)
    model_h4D_phi = ROOT.RDF.THnDModel("h4D_phi", title_histo4D,
                                       4, nbins_h4D, edges_h4D)

    # Reduced DataFrames
    df_D_KS0LL = ROOT.RDataFrame(TKS0LL, [v1_KS0, v2_KS0, v3_KS0,
                                          weights_init_KS0LL.name_weights,
                                          split_KS0LL.name_weights,
                                          "KS0_M", "Dplus_DTF_M_0", "Dplus_ID"])

    df_D_KS0DD = ROOT.RDataFrame(TKS0DD, [v1_KS0, v2_KS0, v3_KS0,
                                          weights_init_KS0DD.name_weights,
                                          split_KS0DD.name_weights,
                                          "KS0_M", "Dplus_DTF_M_0", "Dplus_ID"])

    df_D_phi = ROOT.RDataFrame(Tphi, [v1_phi, v2_phi, v3_phi,
                                      weights_init_phi.name_weights,
                                      split_phi.name_weights,
                                      "phi_M", "Dplus_DTFonlyPV_M_0", "Dplus_ID"])

    def fm_K0sLL(mk0):
        return fabs(mk0 - mK0s_PDG) < lim_mass_KS0LL

    def fm_K0sDD(mk0):
        return fabs(mk0 - mK0s_PDG) < lim_mass_KS0DD

    r4D_rew_phi = (df_D_phi
                   .Filter(f"fabs(phi_M - {mphi_PDG}) < {lim_mass_phi}")
                   .HistoND(model_h4D_phi,
                            [v1_phi, v2_phi, v3_phi,
                             split_phi.name_weights,
                             weights_init_phi.name_weights]))

    r4D_rew_KS0LL = (df_D_KS0LL
                     .Filter(fm_K0sLL, ["KS0_M"])
                     .HistoND(model_h4D_KS0LL,
                              [v1_KS0, v2_KS0, v3_KS0,
                               split_KS0LL.name_weights,
                               weights_init_KS0LL.name_weights]))

    r4D_rew_KS0DD = (df_D_KS0DD
                     .Filter(fm_K0sDD, ["KS0_M"])
                     .HistoND(model_h4D_KS0DD,
                              [v1_KS0, v2_KS0, v3_KS0,
                               split_KS0DD.name_weights,
                               weights_init_KS0DD.name_weights]))

    h4D_rew_KS0 = r4D_rew_KS0LL.GetPtr()
    h4D_rew_KS0DD = r4D_rew_KS0DD.GetPtr()
    h4D_rew_phi = r4D_rew_phi.GetPtr()

    # Sum LL and DD
    h4D_rew_KS0.Add(h4D_rew_KS0DD)

    h3D_KS0 = []
    h3D_phi = []

    for i in range(1, nbins_split + 1):
        h4D_bin_KS0 = h4D_rew_KS0.Clone(f"h4D_KS0_{i}")
        h4D_bin_KS0.GetAxis(3).SetRange(i, i)
        h3D_bin_KS0 = h4D_bin_KS0.Projection(0, 1, 2, f"h3D_bin_KS0{i}")
        h3D_KS0.append(h3D_bin_KS0)

        h4D_bin_phi = h4D_rew_phi.Clone(f"h4D_phi_{i}")
        h4D_bin_phi.GetAxis(3).SetRange(i, i)
        h3D_bin_phi = h4D_bin_phi.Projection(0, 1, 2, f"h3D_bin_phi{i}")
        h3D_phi.append(h3D_bin_phi)

        h4D_bin_KS0.Delete()
        h4D_bin_phi.Delete()

    vec_h3D_weights = []
    vec_h3D_accept = []

    for ibin in range(len(h3D_KS0)):
        if rew_id == "KS0->phi":
            h3D_weights = h3D_phi[ibin].Clone(f"h3D_weights{ibin+1}")
        elif rew_id == "phi->KS0":
            h3D_weights = h3D_KS0[ibin].Clone(f"h3D_weights{ibin+1}")
        else:
            print("Errore --- valore di rew_id non adeguato:", rew_id)
            return

        for i in range(1, h3D_weights.GetNcells() + 1):
            if (h3D_phi[ibin].GetBinContent(i) < min_bin_content_split or
                    h3D_KS0[ibin].GetBinContent(i) < min_bin_content_split):
                h3D_weights.SetBinContent(i, 0.0)

        h3D_phi[ibin].Scale(1.0 / h3D_phi[ibin].Integral())
        h3D_KS0[ibin].Scale(1.0 / h3D_KS0[ibin].Integral())
        h3D_weights.Scale(1.0 / h3D_weights.Integral())

        if rew_id == "KS0->phi":
            h3D_weights.Divide(h3D_KS0[ibin])
        elif rew_id == "phi->KS0":
            h3D_weights.Divide(h3D_phi[ibin])

        h3D_accept = h3D_weights.Clone(f"h3D_accept{ibin+1}")
        for i in range(1, h3D_weights.GetNcells() + 1):
            h3D_accept.SetBinContent(i, 1.0)
            if h3D_weights.GetBinContent(i) == 0.0:
                h3D_accept.SetBinContent(i, 0.0)
            if h3D_weights.GetBinContent(i) > 3.0:
                h3D_weights.SetBinContent(i, 0.0)
                h3D_accept.SetBinContent(i, 0.0)

        vec_h3D_weights.append(h3D_weights)
        vec_h3D_accept.append(h3D_accept)

    def fweights_phi(vx, vy, vz, weight_init, split):
        if split == 0:
            return 0.0
        if use_onlyLL and split > nbins_LL:
            return 0.0
        idx = split - 1
        if rew_id == "KS0->phi":
            bin_idx = vec_h3D_accept[idx].FindBin(vx, vy, vz)
            accept = vec_h3D_accept[idx].GetBinContent(bin_idx)
            return accept * weight_init
        elif rew_id == "phi->KS0":
            bin_idx = vec_h3D_weights[idx].FindBin(vx, vy, vz)
            weight = vec_h3D_weights[idx].GetBinContent(bin_idx)
            return weight * weight_init
        return 0.0

    def fweights_KS0(vx, vy, vz, weight_init, split):
        if split == 0:
            return 0.0
        if use_onlyLL and split > nbins_LL:
            return 0.0
        idx = split - 1
        if rew_id == "phi->KS0":
            bin_idx = vec_h3D_accept[idx].FindBin(vx, vy, vz)
            accept = vec_h3D_accept[idx].GetBinContent(bin_idx)
            return accept * weight_init
        elif rew_id == "KS0->phi":
            bin_idx = vec_h3D_weights[idx].FindBin(vx, vy, vz)
            weight = vec_h3D_weights[idx].GetBinContent(bin_idx)
            return weight * weight_init
        return 0.0

    df_D_phi_reweighted = df_D_phi.Define(
        weights_out_phi.name_weights + "_forfit",
        fweights_phi,
        [v1_phi, v2_phi, v3_phi,
         weights_init_phi.name_weights + "_forfit",
         split_phi.name_weights])

    df_D_KS0LL_reweighted = df_D_KS0LL.Define(
        weights_out_KS0LL.name_weights + "_forfit",
        fweights_KS0,
        [v1_KS0, v2_KS0, v3_KS0,
         weights_init_KS0LL.name_weights + "_forfit",
         split_KS0LL.name_weights])

    df_D_KS0DD_reweighted = df_D_KS0DD.Define(
        weights_out_KS0DD.name_weights + "_forfit",
        fweights_KS0,
        [v1_KS0, v2_KS0, v3_KS0,
         weights_init_KS0DD.name_weights + "_forfit",
         split_KS0DD.name_weights])

    # The rest of this function (sideband recomputation, stats, Snapshot)
    # should be ported following the C++ code. For space reasons, it is
    # abbreviated here.

    for i in range(len(vec_h3D_weights)):
        vec_h3D_weights[i].Delete()
        vec_h3D_accept[i].Delete()
        h3D_KS0[i].Delete()
        h3D_phi[i].Delete()
