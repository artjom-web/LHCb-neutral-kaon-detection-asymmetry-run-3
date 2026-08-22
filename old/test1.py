import numpy as np
import ROOT
from apd import AnalysisData
import numpy as np
import os




# datasetnumber = 0
# cset = 2

# datasets = AnalysisData("charm", "d_to_ksh")
# dataname = f"d_to_ksh_25c{cset}_magdown,_split_d2kspi_ll"

# result = datasets(
#     config="lhcb",
#     datatype="2025",
#     filetype="d2kspi_ll.root",
#     polarity="magdown",
#     eventtype="94000000",
#     name=dataname
# )
# f = ROOT.TFile.Open(result[datasetnumber])
# d = f.Get("D2KSpi_DD")
# tree = d.Get("DecayTree")
# rdf = ROOT.RDataFrame(tree)
# rdf1 = (
#     rdf
#     .Define("theta_x", "atan(Pip_PX/Pip_PZ)")
#     .Define("theta_y", "atan(Pip_PY/Pip_PZ)")
#     .Define("Pip_k", "1.0/sqrt(Pip_PX*Pip_PX*1e-6 + Pip_PZ*Pip_PZ*1e-6)")
#     .Define("Dp_k", "1.0/sqrt(Dp_PX*Dp_PX*1e-6 + Dp_PZ*Dp_PZ*1e-6)")
#     .Define("KS_k", "1.0/sqrt(KS_PX*KS_PX*1e-6 + KS_PZ*KS_PZ*1e-6)")
#     .Define(
#         "KS_FD",
#         "sqrt((KS_END_VX-KS_OWNPVX)*(KS_END_VX-KS_OWNPVX)"
#         " + (KS_END_VY-KS_OWNPVY)*(KS_END_VY-KS_OWNPVY)"
#         " + (KS_END_VZ-KS_OWNPVZ)*(KS_END_VZ-KS_OWNPVZ))"
#     )

#     # D+
#     .Filter("Dp_M > 1800", "Dp_M > 1800")
#     .Filter("Dp_M < 1950", "Dp_M < 1950")
#     .Filter("Dp_OWNPVIP < 1", "Dp IP")
#     .Filter("Dp_HLT2_ETA > 2.2", "Dp eta min")
#     .Filter("Dp_HLT2_ETA < 4.2", "Dp eta max")
#     .Filter("Dp_HLT2_PT > 2800", "Dp PT min")
#     .Filter("Dp_HLT2_PT < 12000", "Dp PT max")
#     .Filter("Dp_k > 0.01", "Dp k min")
#     .Filter("Dp_k < 0.12", "Dp k max")

#     # bachelor pion geometry
#     .Filter("Pip_k < (0.3 - abs(theta_x))", "Pip acceptance")
#     .Filter("pow(theta_x/0.027,2) + pow(theta_y/0.017,2) > 1", "Beam ellipse")
#     .Filter("abs(theta_y) > 0.001", "theta_y min")
#     .Filter(
#         "!(abs(theta_y) < 0.005 && abs(theta_x) > 0.06 && abs(theta_x) < 0.1)",
#         "Dead region"
#     )

#     # bachelor pion kinematics
#     .Filter("Pip_HLT2_PT > 1500", "Pip PT min")
#     .Filter("Pip_HLT2_PT < 6000", "Pip PT max")
#     .Filter("Pip_HLT2_ETA > 2.2", "Pip eta min")
#     .Filter("Pip_HLT2_ETA < 4.2", "Pip eta max")
#     .Filter("Pip_k > 0.005", "Pip k min")
#     .Filter("Pip_k < 0.06", "Pip k max")

#     # KS
#     .Filter("KS_HLT2_ETA > 2.2", "KS eta min")
#     .Filter("KS_HLT2_ETA < 4.2", "KS eta max")
#     .Filter("KS_OWNPVLTIME > 0", "KS lifetime min")
#     .Filter("KS_OWNPVLTIME < 0.4", "KS lifetime max")
#     .Filter("KS_FD > 20", "KS flight distance")
# )

# count = rdf1.Count()
# report = rdf1.Report()

# print("Events:", count.GetValue())
# report.Print()
# rdf = rdf.Filter(
#     f"Dp_M > 1700 && Dp_M < 2190"
#     " && KS_OWNPVLTIME > 0"
    # " && KSpim_PROBNN_GHOST < 0.5"
    # " && KSpip_PROBNN_GHOST < 0.5"
    # " && Pip_PROBNN_GHOST < 0.5"
    # " && Dp_MAXDOCA < 1"
    # " && KS_MAXDOCA < 1"
# Dp_OWNPVLTIME, > 0
# Dp_OWNPVDIRA < 10 mrad
# KS_OWNPVIP
# KSpim_OWNPVIP
# KSpip_OWNPVIP
# KSpim_OWNPVIP
# Pip_OWNPVIP
# )

# Dp_OWNPVIP < 1
# Dp_HLT2_ETA > 2.2
# Dp_HLT2_ETA < 4.2
# Dp_HLT2_PT > 2800
# Dp_HLT2_PT < 12000

# Dp_M [1789, 2049]


# def print_root_structure(rfile, indent=0):
#     """Recursively print all TDirectories and TTrees in a ROOT file."""
#     for key in rfile.GetListOfKeys():
#         name      = key.GetName()
#         classname = key.GetClassName()
#         cycle     = key.GetCycle()
#         prefix    = "  " * indent

#         if classname == "TTree":
#             obj = rfile.Get(f"{name};{cycle}")
#             print(f"{prefix}[TTree]      {name};{cycle}  ({obj.GetEntries()} entries)")

#         elif classname.startswith("TDirectory"):
#             print(f"{prefix}[TDirectory] {name}")
#             subdir = rfile.Get(name)
#             print_root_structure(subdir, indent + 1)

#         else:
#             print(f"{prefix}[{classname}] {name}")

# print_root_structure(f)

    # PX = rdf.AsNumpy([f'Dp_PX'])
    # PY = rdf.AsNumpy([f'Dp_PY'])
    # data = np.sqrt(PX**2 + PY**2)
# names = ['Dp_M']
# for i, name in enumerate(names):
#     c = ROOT.TCanvas(f"c_{i}", name, 800, 600)

#     h = rd1f.Histo1D((f"h_{i}", name, 100, 1700, 2200), name)

#     print(name, h.GetMean(), h.GetStdDev())

#     h.Draw()
#     c.SaveAs('./figures/test123.png')


# names = ['KSpim_PROBNN_GHOST', 'KSpim_PROBNN_PI', 'Pip_PROBNN_GHOST', 'Pip_PROBNN_PI', 'KSpip_PROBNN_PI', 'KSpip_PROBNN_GHOST', ]