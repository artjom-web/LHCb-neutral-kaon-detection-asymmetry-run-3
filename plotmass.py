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
from pathlib import Path
from collections import defaultdict

  
folder = '/eos/user/a/ahulsber/scripts/data/08-19/dd_Pip_Hlt1TrackMVADecision_TOS_12hr54/dd_25c1_526/weighting/massfits/KS/KS_HLT2_ETA/bin12/'
ana = Analysis('ll')  
ana.plot_massfit(folder)
