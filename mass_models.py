import ROOT
from ROOT import (RooRealVar, RooGaussian, RooExponential, RooAddPdf, 
    RooDataHist, RooFit, RooPlot, TCanvas, RooArgList, TLegend, RooArgSet, RooFormulaVar,
    RooBifurGauss, RooJohnson, RooProdPdf, RooGenericPdf, RooGaussian,
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


def make_plus_minus(base, delta, name_base):
    plus  = RooFormulaVar(name_base+"_plus",  "@0+@1", RooArgList(base, delta))
    minus = RooFormulaVar(name_base+"_minus", "@0-@1", RooArgList(base, delta))
    return plus, minus

def johnson_exp(m):
    # Signal shape parameters
    mean          = RooRealVar("mean",          "Mean (Johnson)",          1869.66, 1849.66, 1889.66)
    Delta_mean    = RooRealVar("Delta_mean",    "Delta mean (Johnson)",    5., -15., 15.)
    sigma         = RooRealVar("sigma",         "Sigma (Johnson)",         15., 0.1, 100.)
    Delta_sigma   = RooRealVar("Delta_sigma",   "Delta sigma (Johnson)",   0., -5., 5.)
    Delta_sigma.setConstant(True)
    gamma_johnson         = RooRealVar("gamma_johnson",         "Gamma (Johnson)",         0.2, 0., 1.)
    Delta_gamma_johnson   = RooRealVar("Delta_gamma_johnson",   "Delta gamma (Johnson)",   0, -3., 3.)
    Delta_gamma_johnson.setConstant(True)
    delta_johnson         = RooRealVar("delta_johnson",         "Delta (Johnson)",         2., 0.0001, 30.)
    Delta_delta_johnson   = RooRealVar("Delta_delta_johnson",   "Delta delta (Johnson)",   0., -3., 3.)
    Delta_delta_johnson.setConstant(True)

    # Background shape parameters
    lambda_exp       = RooRealVar("lambda_exponential",       "Lambda (exponential)",       -6e-4, -1., 0.05)
    Delta_lambda_exp = RooRealVar("Delta_lambda_exponential", "Delta lambda (exponential)", 0., -1.,  1.)
    Delta_lambda_exp.setConstant(True)

    # Per-charge derived parameters
    mean_j_p,   mean_j_m   = make_plus_minus(mean,       Delta_mean,    "mean_johnson")
    sigma_j_p,  sigma_j_m  = make_plus_minus(sigma,      Delta_sigma,   "sigma_johnson")
    gamma_j_p,  gamma_j_m  = make_plus_minus(gamma_johnson,      Delta_gamma_johnson,   "gamma_johnson")
    delta_j_p,  delta_j_m  = make_plus_minus(delta_johnson,      Delta_delta_johnson,   "delta_johnson")
    lambda_p,   lambda_m   = make_plus_minus(lambda_exp,         Delta_lambda_exp,      "lambda_exp")

    # PDFs        
    johnson_plus   = RooJohnson("johnson_plus",  "Johnson+",  m, mean_j_p,  sigma_j_p,  gamma_j_p,  delta_j_p)
    johnson_minus  = RooJohnson("johnson_minus", "Johnson-",  m, mean_j_m,  sigma_j_m,  gamma_j_m,  delta_j_m)
    expo_plus      = RooExponential("expo_plus",  "Expo+",    m, lambda_p)
    expo_minus     = RooExponential("expo_minus", "Expo-",    m, lambda_m)

    signal_plus = johnson_plus
    signal_minus = johnson_minus
    background_plus = expo_plus
    background_minus = expo_minus


    sp, sm, bp, bm = signal_plus, signal_minus, background_plus, background_minus
    all_servers = RooArgSet()
    for pdf in (sp, sm, bp, bm):
        pdf.treeNodeServerList(all_servers)   # recursively collects every server node in the tree

    objects = list(all_servers)   # Python list holds a real reference to every node

    return {
        "sp": sp,
        "sm": sm,
        "bp": bp,
        "bm": bm,
        "objects": objects,
    }


def johnson_tail_expo(m):

    # Signal shape parameters
    mean        = RooRealVar("mean",          "Mean (sig)",          1869.66, 1849.66, 1889.66)
    Delta_mean    = RooRealVar("Delta_mean",    "Delta mean (sig)",    5., -15., 15.)
    sigma         = RooRealVar("sigma",         "Sigma (sig)",         15., 0.1, 100.)
    Delta_sigma   = RooRealVar("Delta_sigma",   "Delta sigma (sig)",   0., -5., 5.)
    Delta_sigma.setConstant(True)
    gamma_johnson         = RooRealVar("gamma_johnson",         "Gamma (Johnson)",         0, -1., 1.)
    gamma_johnson.setConstant(True)
    Delta_gamma_johnson   = RooRealVar("Delta_gamma_johnson",   "Delta gamma (Johnson)",   0., -3., 3.)
    Delta_gamma_johnson.setConstant(True)
    delta_johnson         = RooRealVar("delta_johnson",         "Delta (Johnson)",         2., 0.0001, 30.)
    Delta_delta_johnson   = RooRealVar("Delta_delta_johnson",   "Delta delta (Johnson)",   0., -3., 3.)
    Delta_delta_johnson.setConstant(True)
    b_tail                = RooRealVar("b_expo_tail", "b (expo tail)", 0.01, 1e-8,  1e2)
    Delta_b_tail          = RooRealVar("Delta_b_expo_tail", "Delta b (expo tail)", 0, -1e-7, 1e-7)
    Delta_b_tail.setConstant(True)
    f_john = ROOT.RooRealVar("f_john", "Johnson fraction", 0.9 , 0., 1.)


    # Background shape parameters
    lambda_exp       = RooRealVar("lambda_exponential",       "Lambda (exponential)",       -6e-4, -1., 0.05)
    Delta_lambda_exp = RooRealVar("Delta_lambda_exponential", "Delta lambda (exponential)", 0., -1.,  1.)
    Delta_lambda_exp.setConstant(True)


    # Per-charge derived parameters
    mean_sig_p,   mean_sig_m   = make_plus_minus(mean,       Delta_mean,    "mean_sig")
    sigma_sig_p,  sigma_sig_m  = make_plus_minus(sigma,      Delta_sigma,   "sigma_sig")
    gamma_j_p,  gamma_j_m  = make_plus_minus(gamma_johnson,      Delta_gamma_johnson,   "gamma_johnson")
    delta_j_p,  delta_j_m  = make_plus_minus(delta_johnson,      Delta_delta_johnson,   "delta_johnson")
    b_t_p,      b_t_m      = make_plus_minus(b_tail,             Delta_b_tail,          "b_expo_tail")
    lambda_p,   lambda_m   = make_plus_minus(lambda_exp,         Delta_lambda_exp,      "lambda_exp")


    # PDFs        
    johnson_plus   = RooJohnson("johnson_plus",  "Johnson+",  m, mean_sig_p,  sigma_sig_p,  gamma_j_p,  delta_j_p)
    johnson_minus  = RooJohnson("johnson_minus", "Johnson-",  m, mean_sig_m,  sigma_sig_m,  gamma_j_m,  delta_j_m)
    tail_plus      = ROOT.RooGenericPdf("tail_plus", "tail+", "exp(@1*(@0-@2))*TMath::Erfc((@0-@2)/@3)", ROOT.RooArgList(m, b_t_p, mean_sig_p, sigma_sig_p))
    tail_minus     = ROOT.RooGenericPdf("tail_minus", "tail-", "exp(@1*(@0-@2))*TMath::Erfc((@0-@2)/@3)", ROOT.RooArgList(m, b_t_m, mean_sig_m, sigma_sig_m))
    

    expo_plus      = RooExponential("expo_plus",  "Expo+",    m, lambda_p)
    expo_minus     = RooExponential("expo_minus", "Expo-",    m, lambda_m)

    signal_plus = ROOT.RooAddPdf("signal_plus", "Johnson+ + radiative tail+", ROOT.RooArgList(johnson_plus, tail_plus), ROOT.RooArgList(f_john))
    signal_minus = ROOT.RooAddPdf("signal_minus", "Johnson- + radiative tail-", ROOT.RooArgList(johnson_minus, tail_minus), ROOT.RooArgList(f_john))
    background_plus = expo_plus
    background_minus = expo_minus


    sp, sm, bp, bm = signal_plus, signal_minus, background_plus, background_minus
    all_servers = RooArgSet()
    for pdf in (sp, sm, bp, bm):
        pdf.treeNodeServerList(all_servers)   # recursively collects every server node in the tree

    objects = list(all_servers)   # Python list holds a real reference to every node

    return {
        "sp": sp,
        "sm": sm,
        "bp": bp,
        "bm": bm,
        "objects": objects,
    }

    


def johnson_gauss_exp(m):
    # Signal shape parameters
    mean          = RooRealVar("mean",          "Mean (sig)",          1869.66, 1849.66, 1889.66)
    Delta_mean    = RooRealVar("Delta_mean",    "Delta mean (sig)",    5., -15., 15.)
    sigma         = RooRealVar("sigma",         "Sigma (sig)",         15., 0.1, 100.)
    Delta_sigma   = RooRealVar("Delta_sigma",   "Delta sigma (sig)",   0., -5., 5.)
    Delta_sigma.setConstant(True)
    gamma_johnson         = RooRealVar("gamma_johnson",         "Gamma (Johnson)",         0.2, 0., 1.)
    Delta_gamma_johnson   = RooRealVar("Delta_gamma_johnson",   "Delta gamma (Johnson)",   0., -3., 3.)
    Delta_gamma_johnson.setConstant(True)
    delta_johnson         = RooRealVar("delta_johnson",         "Delta (Johnson)",         2., 0.0001, 30.)
    Delta_delta_johnson   = RooRealVar("Delta_delta_johnson",   "Delta delta (Johnson)",   0., -3., 3.)
    Delta_delta_johnson.setConstant(True)

    f_john = ROOT.RooRealVar("f_john", "Johnson fraction", 0.9 , 0., 1.)

    # Background shape parameters
    lambda_exp       = RooRealVar("lambda_exponential",       "Lambda (exponential)",       -6e-4, -1., 0.05)
    Delta_lambda_exp = RooRealVar("Delta_lambda_exponential", "Delta lambda (exponential)", 0., -1.,  1.)
    Delta_lambda_exp.setConstant(True)

    # Per-charge derived parameters
    mean_p,   mean_m   = make_plus_minus(mean,       Delta_mean,    "mean_johnson")
    sigma_p,  sigma_m  = make_plus_minus(sigma,      Delta_sigma,   "sigma_johnson")
    gamma_j_p,  gamma_j_m  = make_plus_minus(gamma_johnson,      Delta_gamma_johnson,   "gamma_johnson")
    delta_j_p,  delta_j_m  = make_plus_minus(delta_johnson,      Delta_delta_johnson,   "delta_johnson")
    lambda_p,   lambda_m   = make_plus_minus(lambda_exp,         Delta_lambda_exp,      "lambda_exp")

    # PDFs        
    johnson_plus   = RooJohnson("johnson_plus",  "Johnson+",  m, mean_p,  sigma_p,  gamma_j_p,  delta_j_p)
    johnson_minus  = RooJohnson("johnson_minus", "Johnson-",  m, mean_m,  sigma_m,  gamma_j_m,  delta_j_m)
    gaussian_plus  = RooGaussian("gaussian_plus", "Gaussian+", m, mean_p, sigma_p)
    gaussian_minus = RooGaussian("gaussian_minus", "Gaussian-", m, mean_m, sigma_m)
    expo_plus      = RooExponential("expo_plus",  "Expo+",    m, lambda_p)
    expo_minus     = RooExponential("expo_minus", "Expo-",    m, lambda_m)

    signal_plus = ROOT.RooAddPdf("signal_plus", "Johnson+ + Gaussian+", ROOT.RooArgList(johnson_plus, gaussian_plus), ROOT.RooArgList(f_john))
    signal_minus = ROOT.RooAddPdf("signal_minus", "Johnson- + Gaussian-", ROOT.RooArgList(johnson_minus, gaussian_minus), ROOT.RooArgList(f_john))

    background_plus = expo_plus
    background_minus = expo_minus



    sp, sm, bp, bm = signal_plus, signal_minus, background_plus, background_minus
    all_servers = RooArgSet()
    for pdf in (sp, sm, bp, bm):
        pdf.treeNodeServerList(all_servers)   # recursively collects every server node in the tree

    objects = list(all_servers)   # Python list holds a real reference to every node

    return {
        "sp": sp,
        "sm": sm,
        "bp": bp,
        "bm": bm,
        "objects": objects,
    }


    
# def johnson_free_gauss_exp(m):
#     # Signal shape parameters
#     mean          = RooRealVar("mean",          "Mean (sig)",          1869.66, 1849.66, 1889.66)
#     Delta_mean    = RooRealVar("Delta_mean",    "Delta mean (sig)",    5., -15., 15.)
#     sigma         = RooRealVar("sigma",         "Sigma (sig)",         15., 0.1, 100.)
#     Delta_sigma   = RooRealVar("Delta_sigma",   "Delta sigma (sig)",   0., -5., 5.)
#     Delta_sigma.setConstant(True)
#     gamma_johnson         = RooRealVar("gamma_johnson",         "Gamma (Johnson)",         0.2, 0., 1.)
#     Delta_gamma_johnson   = RooRealVar("Delta_gamma_johnson",   "Delta gamma (Johnson)",   0., -3., 3.)
#     Delta_gamma_johnson.setConstant(True)
#     delta_johnson         = RooRealVar("delta_johnson",         "Delta (Johnson)",         2., 0.0001, 30.)
#     Delta_delta_johnson   = RooRealVar("Delta_delta_johnson",   "Delta delta (Johnson)",   0., -3., 3.)
#     Delta_delta_johnson.setConstant(True)

#     mean_g          = RooRealVar("mean_gauss",          "Mean (gauss)",          1869.66, 1820, 1920)
#     Delta_mean_g    = RooRealVar("Delta_mean_gauss",    "Delta mean (gauss)",    5., -15., 15.)
#     sigma_g         = RooRealVar("sigma_gauss",         "Sigma (gauss)",         15., 0.1, 100.)
#     Delta_sigma_g   = RooRealVar("Delta_sigma_gauss",   "Delta sigma (gauss)",   0., -5., 5.)

#     f_john = ROOT.RooRealVar("f_john", "Johnson fraction", 0.9 , 0., 1.)

#     # Background shape parameters
#     lambda_exp       = RooRealVar("lambda_exponential",       "Lambda (exponential)",       -6e-4, -1., 0.05)
#     Delta_lambda_exp = RooRealVar("Delta_lambda_exponential", "Delta lambda (exponential)", 0., -1.,  1.)
#     Delta_lambda_exp.setConstant(True)

#     # Per-charge derived parameters
#     mean_p,   mean_m   = make_plus_minus(mean,       Delta_mean,    "mean_sig")
#     sigma_p,  sigma_m  = make_plus_minus(sigma,      Delta_sigma,   "sigma_sig")
#     mean_g_p,   mean_g_m   = make_plus_minus(mean_g,       Delta_mean_g,    "mean_gauss")
#     sigma_g_p,  sigma_g_m  = make_plus_minus(sigma_g,      Delta_sigma_g,   "sigma_gauss")
#     gamma_j_p,  gamma_j_m  = make_plus_minus(gamma_johnson,      Delta_gamma_johnson,   "gamma_johnson")
#     delta_j_p,  delta_j_m  = make_plus_minus(delta_johnson,      Delta_delta_johnson,   "delta_johnson")
#     lambda_p,   lambda_m   = make_plus_minus(lambda_exp,         Delta_lambda_exp,      "lambda_exp")

#     # PDFs        
#     johnson_plus   = RooJohnson("johnson_plus",  "Johnson+",  m, mean_p,  sigma_p,  gamma_j_p,  delta_j_p)
#     johnson_minus  = RooJohnson("johnson_minus", "Johnson-",  m, mean_m,  sigma_m,  gamma_j_m,  delta_j_m)
#     gaussian_plus  = RooGaussian("gaussian_plus", "Gaussian+", m, mean_g_p, sigma_g_p)
#     gaussian_minus = RooGaussian("gaussian_minus", "Gaussian-", m, mean_g_m, sigma_g_m)
#     expo_plus      = RooExponential("expo_plus",  "Expo+",    m, lambda_p)
#     expo_minus     = RooExponential("expo_minus", "Expo-",    m, lambda_m)

#     signal_plus = ROOT.RooAddPdf("signal_plus", "Johnson+ + Gaussian+", ROOT.RooArgList(johnson_plus, gaussian_plus), ROOT.RooArgList(f_john))
#     signal_minus = ROOT.RooAddPdf("signal_minus", "Johnson- + Gaussian-", ROOT.RooArgList(johnson_minus, gaussian_minus), ROOT.RooArgList(f_john))

#     background_plus = expo_plus
#     background_minus = expo_minus



#     sp, sm, bp, bm = signal_plus, signal_minus, background_plus, background_minus
#     all_servers = RooArgSet()
#     for pdf in (sp, sm, bp, bm):
#         pdf.treeNodeServerList(all_servers)   # recursively collects every server node in the tree

#     objects = list(all_servers)   # Python list holds a real reference to every node

#     return {
#         "sp": sp,
#         "sm": sm,
#         "bp": bp,
#         "bm": bm,
#         "objects": objects,
#     }




def johnson_exp1(m):
    # Signal shape parameters
    mean          = RooRealVar("mean",          "Mean (Johnson)",          1869.66, 1849.66, 1889.66)
    Delta_mean    = RooRealVar("Delta_mean",    "Delta mean (Johnson)",    5., -15., 15.)
    sigma         = RooRealVar("sigma",         "Sigma (Johnson)",         15., 0.1, 100.)
    Delta_sigma   = RooRealVar("Delta_sigma",   "Delta sigma (Johnson)",   0., -5., 5.)
    gamma_johnson         = RooRealVar("gamma_johnson",         "Gamma (Johnson)",         0.2, 0., 1.)
    Delta_gamma_johnson   = RooRealVar("Delta_gamma_johnson",   "Delta gamma (Johnson)",   0., -3., 3.)
    Delta_gamma_johnson.setConstant(True)
    delta_johnson         = RooRealVar("delta_johnson",         "Delta (Johnson)",         2., 0.0001, 30.)
    Delta_delta_johnson   = RooRealVar("Delta_delta_johnson",   "Delta delta (Johnson)",   0., -3., 3.)
    Delta_delta_johnson.setConstant(True)

    # Background shape parameters
    lambda_exp       = RooRealVar("lambda_exponential",       "Lambda (exponential)",       -6e-4, -1., 0.05)
    Delta_lambda_exp = RooRealVar("Delta_lambda_exponential", "Delta lambda (exponential)", 0., -1.,  1.)
    Delta_lambda_exp.setConstant(True)

    # Per-charge derived parameters
    mean_j_p,   mean_j_m   = make_plus_minus(mean,       Delta_mean,    "mean_johnson")
    sigma_j_p,  sigma_j_m  = make_plus_minus(sigma,      Delta_sigma,   "sigma_johnson")
    gamma_j_p,  gamma_j_m  = make_plus_minus(gamma_johnson,      Delta_gamma_johnson,   "gamma_johnson")
    delta_j_p,  delta_j_m  = make_plus_minus(delta_johnson,      Delta_delta_johnson,   "delta_johnson")
    lambda_p,   lambda_m   = make_plus_minus(lambda_exp,         Delta_lambda_exp,      "lambda_exp")

    # PDFs        
    johnson_plus   = RooJohnson("johnson_plus",  "Johnson+",  m, mean_j_p,  sigma_j_p,  gamma_j_p,  delta_j_p)
    johnson_minus  = RooJohnson("johnson_minus", "Johnson-",  m, mean_j_m,  sigma_j_m,  gamma_j_m,  delta_j_m)
    expo_plus      = RooExponential("expo_plus",  "Expo+",    m, lambda_p)
    expo_minus     = RooExponential("expo_minus", "Expo-",    m, lambda_m)

    signal_plus = johnson_plus
    signal_minus = johnson_minus
    background_plus = expo_plus
    background_minus = expo_minus



    sp, sm, bp, bm = signal_plus, signal_minus, background_plus, background_minus
    all_servers = RooArgSet()
    for pdf in (sp, sm, bp, bm):
        pdf.treeNodeServerList(all_servers)   # recursively collects every server node in the tree

    objects = list(all_servers)   # Python list holds a real reference to every node

    return {
        "sp": sp,
        "sm": sm,
        "bp": bp,
        "bm": bm,
        "objects": objects,
    }

