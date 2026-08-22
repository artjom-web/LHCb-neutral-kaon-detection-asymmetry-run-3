import ROOT
from ROOT import (
    RooRealVar, RooFormulaVar, RooArgList, RooArgSet,
    RooDataHist, RooGaussian, RooExponential, RooBifurGauss,
    RooJohnson, RooAddPdf, RooProdPdf, RooGenericPdf,
    RooSimultaneous, RooCategory, RooFit, RooStats
)
import json, os


def mass_fit_simpl(
    h_plus,                          # TH1D: positive candidates
    h_minus,                         # TH1D: negative candidates
    name_observable,                 # str:  x-axis label
    params_file_basename,            # str:  base path for parameter files
    output_path,                     # str:  base path for output plots
    blinding_seed,                   # str:  seed string for blinding ("" to disable)
    first_time_runs=False,           # bool: if True, ignore existing param file
    range_min=-1,                    # float: fit range minimum (-1 = auto)
    range_max=-1,                    # float: fit range maximum (-1 = auto)
    mass_name="Dp_M",                # str:  name of the mass observable
    tag_name="hp_PARTICLE_ID",       # str:  name of the tag branch
):
    fit_converges = True

    h_plus.Sumw2()
    h_minus.Sumw2()

    params_file_in  = params_file_basename + "_in.txt"
    params_file_out = params_file_basename + "_out.txt"

    n_bins = h_plus.GetNbinsX()
    m_min  = h_plus.GetXaxis().GetXmin() if range_min == -1 else range_min
    m_max  = h_plus.GetXaxis().GetXmax() if range_max == -1 else range_max

    # ------------------------------------------------------------------
    # Blinding
    # ------------------------------------------------------------------
    rndm = ROOT.TRandom3(ROOT.TString(blinding_seed).Hash())
    rndm.SetSeed(ROOT.TString(blinding_seed).Hash())
    A_bias = RooRealVar("A_bias", "Bias of the signal asymmetry",
                        rndm.Uniform(-0.01, 0.01))
    if blinding_seed == "":
        A_bias.setVal(0)
        A_bias.setConstant(True)

    # ------------------------------------------------------------------
    # Observable
    # ------------------------------------------------------------------
    m = RooRealVar(mass_name, name_observable, m_min, m_max, "MeV/c^{2}")
    m.setBins(n_bins)

    data_p   = RooDataHist("data_p",   name_observable + " tag +", RooArgList(m), ROOT.RooFit.Import(h_plus))
    data_m   = RooDataHist("data_m",   name_observable + " tag -", RooArgList(m), ROOT.RooFit.Import(h_minus))
    h_tot    = h_plus.Clone("h_tot")
    h_tot.Add(h_minus)
    data_tot = RooDataHist("data_tot", name_observable,             RooArgList(m), ROOT.RooFit.Import(h_tot))

    N_plus  = int(h_plus.GetEntries()  * 1.5)
    N_minus = int(h_minus.GetEntries() * 1.5)

    # ------------------------------------------------------------------
    # Signal shape parameters — Johnson SU
    # ------------------------------------------------------------------
    mean_johnson          = RooRealVar("mean_johnson",          "Mean (Johnson)",          (m_min+m_max)*0.5, m_min, m_max)
    Delta_mean_johnson    = RooRealVar("Delta_mean_johnson",    "Delta mean (Johnson)",    0., -5., 5.)
    sigma_johnson         = RooRealVar("sigma_johnson",         "Sigma (Johnson)",         15., 0.1, 100.)
    Delta_sigma_johnson   = RooRealVar("Delta_sigma_johnson",   "Delta sigma (Johnson)",   0., -5., 5.)
    gamma_johnson         = RooRealVar("gamma_johnson",         "Gamma (Johnson)",         -0.19907, -3., 3.)
    Delta_gamma_johnson   = RooRealVar("Delta_gamma_johnson",   "Delta gamma (Johnson)",   0., -3., 3.)
    Delta_gamma_johnson.setConstant(True)
    delta_johnson         = RooRealVar("delta_johnson",         "Delta (Johnson)",         2., 1., 30.)
    Delta_delta_johnson   = RooRealVar("Delta_delta_johnson",   "Delta delta (Johnson)",   0., -3., 3.)
    Delta_delta_johnson.setConstant(True)

    # Signal shape parameters — Gaussian (disabled by default)
    mean_gaussian         = RooRealVar("mean_gaussian",         "Mean (Gaussian)",         (m_min+m_max)*0.5, m_min, m_max)
    Delta_mean_gaussian   = RooRealVar("Delta_mean_gaussian",   "Delta mean (Gaussian)",   0., -1., 1.)
    sigma_gaussian        = RooRealVar("sigma_gaussian",        "Sigma (Gaussian)",        3., 0.1, 10.)
    Delta_sigma_gaussian  = RooRealVar("Delta_sigma_gaussian",  "Delta sigma (Gaussian)",  0., -1., 1.)
    for v in [mean_gaussian, Delta_mean_gaussian, sigma_gaussian, Delta_sigma_gaussian]:
        v.setConstant(True)

    # Signal fraction (Johnson vs Gaussian) — fixed to 1 (pure Johnson)
    frac_sig              = RooRealVar("frac_sig_johnson_gaussian",       "Frac Johnson/Gaussian",       1., 0.85, 1.)
    Delta_frac_sig        = RooRealVar("Delta_frac_sig_johnson_gaussian", "Delta frac Johnson/Gaussian", 0., -1.,  1.)
    frac_sig.setConstant(True)
    Delta_frac_sig.setConstant(True)

    # ------------------------------------------------------------------
    # Background shape parameters
    # ------------------------------------------------------------------
    lambda_exp       = RooRealVar("lambda_exponential",       "Lambda (exponential)",       0., -10., 0.05)
    Delta_lambda_exp = RooRealVar("Delta_lambda_exponential", "Delta lambda (exponential)", 0., -1.,  1.)
    Delta_lambda_exp.setConstant(True)

    # Partially reconstructed background (bifurcated Gaussian) — fixed
    mean_pr    = RooRealVar("mean_gaussian_part_reco",      "Mean (part. reco)",        m_min + (m_max-m_min)*0.25, m_min, m_min + (m_max-m_min)*0.4)
    sigma_pr_L = RooRealVar("sigma_gaussian_part_reco_L",   "Sigma L (part. reco)",     10., 0.1, 30.)
    sigma_pr_R = RooRealVar("sigma_gaussian_part_reco_R",   "Sigma R (part. reco)",     10., 0.1, 30.)
    for v in [mean_pr, sigma_pr_L, sigma_pr_R]:
        v.setConstant(True)

    # Background fraction (exponential vs part. reco) — fixed to 1 (pure expo)
    frac_bkg       = RooRealVar("frac_bkg_expo_part_reco",       "Frac expo/part.reco",       1., 0.9, 1.)
    Delta_frac_bkg = RooRealVar("Delta_frac_bkg_expo_part_reco", "Delta frac expo/part.reco", 0., -1., 1.)
    frac_bkg.setConstant(True)
    Delta_frac_bkg.setConstant(True)

    # ------------------------------------------------------------------
    # Yields and asymmetries
    # ------------------------------------------------------------------
    N_sig       = RooRealVar("N_sig", "Total signal yield",     (N_plus+N_minus)*0.7, 0., N_plus+N_minus)
    A_sig_blind = RooRealVar("A_sig_blind", "Blinded asymmetry", 0., -1., 1.)
    A_sig       = RooFormulaVar("A_sig", "(@0+@1)", RooArgList(A_sig_blind, A_bias))

    N_bkg = RooRealVar("N_bkg", "Total background yield", (N_plus+N_minus)*0.2, 0., N_plus+N_minus)
    A_bkg = RooRealVar("A_bkg", "Background asymmetry",   0., -1., 1.)

    # ------------------------------------------------------------------
    # Per-charge derived parameters
    # ------------------------------------------------------------------
    def make_plus_minus(base, delta, name_base):
        plus  = RooFormulaVar(name_base+"_plus",  "@0+@1", RooArgList(base, delta))
        minus = RooFormulaVar(name_base+"_minus", "@0-@1", RooArgList(base, delta))
        return plus, minus

    mean_j_p,   mean_j_m   = make_plus_minus(mean_johnson,       Delta_mean_johnson,    "mean_johnson")
    sigma_j_p,  sigma_j_m  = make_plus_minus(sigma_johnson,      Delta_sigma_johnson,   "sigma_johnson")
    gamma_j_p,  gamma_j_m  = make_plus_minus(gamma_johnson,      Delta_gamma_johnson,   "gamma_johnson")
    delta_j_p,  delta_j_m  = make_plus_minus(delta_johnson,      Delta_delta_johnson,   "delta_johnson")
    mean_g_p,   mean_g_m   = make_plus_minus(mean_gaussian,      Delta_mean_gaussian,   "mean_gaussian")
    sigma_g_p,  sigma_g_m  = make_plus_minus(sigma_gaussian,     Delta_sigma_gaussian,  "sigma_gaussian")
    frac_s_p,   frac_s_m   = make_plus_minus(frac_sig,           Delta_frac_sig,        "frac_sig_jg")
    lambda_p,   lambda_m   = make_plus_minus(lambda_exp,         Delta_lambda_exp,      "lambda_exp")
    frac_b_p,   frac_b_m   = make_plus_minus(frac_bkg,           Delta_frac_bkg,        "frac_bkg")

    N_sig_plus  = RooFormulaVar("N_sig_plus",  "0.5*(1.0+@0)", RooArgList(A_sig))
    N_sig_minus = RooFormulaVar("N_sig_minus", "0.5*(1.0-@0)", RooArgList(A_sig))
    N_bkg_plus  = RooFormulaVar("N_bkg_plus",  "0.5*(1.0+@0)", RooArgList(A_bkg))
    N_bkg_minus = RooFormulaVar("N_bkg_minus", "0.5*(1.0-@0)", RooArgList(A_bkg))

    # ------------------------------------------------------------------
    # PDFs
    # ------------------------------------------------------------------
    johnson_plus   = RooJohnson("johnson_plus",  "Johnson+",  m, mean_j_p,  sigma_j_p,  gamma_j_p,  delta_j_p)
    johnson_minus  = RooJohnson("johnson_minus", "Johnson-",  m, mean_j_m,  sigma_j_m,  gamma_j_m,  delta_j_m)
    gauss_plus     = RooGaussian("gauss_plus",   "Gauss+",    m, mean_g_p,  sigma_g_p)
    gauss_minus    = RooGaussian("gauss_minus",  "Gauss-",    m, mean_g_m,  sigma_g_m)
    expo_plus      = RooExponential("expo_plus",  "Expo+",    m, lambda_p)
    expo_minus     = RooExponential("expo_minus", "Expo-",    m, lambda_m)
    bifur_plus     = RooBifurGauss("bifur_plus",  "BifurG+",  m, mean_pr, sigma_pr_L, sigma_pr_R)
    bifur_minus    = RooBifurGauss("bifur_minus", "BifurG-",  m, mean_pr, sigma_pr_L, sigma_pr_R)

    sig_plus  = RooAddPdf("signal_plus",  "Signal+",  RooArgList(johnson_plus,  gauss_plus),  RooArgList(frac_s_p))
    sig_minus = RooAddPdf("signal_minus", "Signal-",  RooArgList(johnson_minus, gauss_minus), RooArgList(frac_s_m))
    bkg_plus  = RooAddPdf("bkg_plus",     "Bkg+",     RooArgList(expo_plus,  bifur_plus),     RooArgList(frac_b_p))
    bkg_minus = RooAddPdf("bkg_minus",    "Bkg-",     RooArgList(expo_minus, bifur_minus),    RooArgList(frac_b_m))

    # Simultaneous PDF via RooCategory
    tag = RooCategory(tag_name, "Tag")
    tag.defineType("positive",  1)
    tag.defineType("negative", -1)

    obs = RooArgSet(m, tag)

    combData = RooDataHist("combData", "Combined data", RooArgList(m),
                           ROOT.RooFit.Index(tag),
                           ROOT.RooFit.Import("positive", data_p),
                           ROOT.RooFit.Import("negative", data_m))

    tag_plus_pdf  = RooGenericPdf("tag_plus",  "@0==1",  RooArgSet(tag))
    tag_minus_pdf = RooGenericPdf("tag_minus", "@0==-1", RooArgSet(tag))

    pdf_sig_plus  = RooProdPdf("pdf_sigD_plus",  "pdf_sigD_plus",  RooArgSet(tag_plus_pdf,  sig_plus))
    pdf_sig_minus = RooProdPdf("pdf_sigD_minus", "pdf_sigD_minus", RooArgSet(tag_minus_pdf, sig_minus))
    pdf_bkg_plus  = RooProdPdf("pdf_bkg_plus",   "pdf_bkg_plus",   RooArgSet(tag_plus_pdf,  bkg_plus))
    pdf_bkg_minus = RooProdPdf("pdf_bkg_minus",  "pdf_bkg_minus",  RooArgSet(tag_minus_pdf, bkg_minus))

    bkg_tot    = RooAddPdf("bkg_tot",    "Total bkg",    RooArgList(pdf_bkg_plus,  pdf_bkg_minus), RooArgList(N_bkg_plus, N_bkg_minus))
    signal_tot = RooAddPdf("signal_tot", "Total signal", RooArgList(pdf_sig_plus,  pdf_sig_minus), RooArgList(N_sig_plus, N_sig_minus))
    model_tot  = RooAddPdf("model_tot",  "Total model",  RooArgList(signal_tot,    bkg_tot),       RooArgList(N_sig,      N_bkg))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    fit_args = [
        RooFit.Extended(True),
        RooFit.Save(True),
        RooFit.SumW2Error(True),
        RooFit.EvalBackend("legacy"),
    ]

    params = model_tot.getParameters(obs)
    params.remove(A_bias)

    def do_fit(strategy=1):
        args = fit_args + ([RooFit.Strategy(strategy)] if strategy > 1 else [])
        return model_tot.fitTo(combData, *args)

    def fit_ok(res):
        return res.status() in (0, 1) and res.covQual() in (2, 3)

    if first_time_runs:
        for i, strat in enumerate([1, 1, 2, 2]):
            print(f"\nIteration {i+1}...")
            results = do_fit(strat)
            params.writeToFile(params_file_in if i < 3 else params_file_out)
    else:
        params.readFromFile(params_file_in)
        for attempt in range(5):
            results = do_fit(strategy=2)
            params.writeToFile(params_file_out)
            if fit_ok(results):
                print(f"Converged after {attempt+1} attempt(s).")
                break

    if not fit_ok(results):
        print(f"WARNING: fit status={results.status()}, covQual={results.covQual()}")
        fit_converges = False

    # Warn if parameters near boundaries
    it = params.createIterator()
    var = it.Next()
    while var:
        if isinstance(var, RooRealVar) and not var.isConstant():
            val, err = var.getVal(), var.getError()
            if err > 0:
                if abs(val - var.getMin()) / err < 3 or abs(val - var.getMax()) / err < 3:
                    print(f"WARNING: {var.GetName()} = {val:.4f} +/- {err:.4f} "
                          f"in [{var.getMin():.4f}, {var.getMax():.4f}] — close to boundary!")
        var = it.Next()

    print(f"\nBest possible precision = {(1.0/N_sig.getVal())**0.5:.6f}")
    print(f"Current precision       = {A_sig_blind.getError():.6f}\n")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    margin = 0.05

    def make_pads(canvas):
        upper = ROOT.TPad("upper", "", 0., 0.25, 1., 1.)
        lower = ROOT.TPad("lower", "", 0., 0.,   1., 0.23)
        for pad in [upper, lower]:
            pad.SetRightMargin(margin)
            pad.SetLeftMargin(3*margin)
        upper.SetTopMargin(margin)
        lower.SetBottomMargin(2*margin)
        canvas.cd()
        lower.Draw(); upper.Draw()
        return upper, lower

    def make_legend():
        leg = ROOT.TLegend(0.20, 0.65, 0.32, 0.90)
        leg.SetFillColor(ROOT.kWhite)
        leg.SetTextSize(0.055)
        leg.SetBorderSize(0)
        leg.SetTextFont(132)
        leg.SetHeader("LHCb")
        leg.GetListOfPrimitives().First().SetTextSize(0.07)
        return leg

    def draw_fit(canvas, data_hist, slice_label, model_name, bkg_name, save_path):
        upper, lower = make_pads(canvas)
        frame = m.frame(RooFit.Bins(n_bins))

        cut = f"{tag_name}=={tag_name}::{slice_label}" if slice_label else ""

        # --- plot data, always with Name("data_draw") ---
        if cut:
            combData.plotOn(
                frame,
                RooFit.Cut(cut),
                RooFit.Name("data_draw")
            )
        else:
            data_hist.plotOn(
                frame,
                RooFit.Name("data_draw")
            )

        # --- plot total model ---
        model_tot.plotOn(
            frame,
            RooFit.Precision(1e-6),
            RooFit.LineColor(ROOT.kRed),
            *([RooFit.Slice(tag, slice_label),
            RooFit.ProjWData(tag, combData)] if slice_label else []),
            RooFit.MoveToBack(),
            RooFit.Name("model_draw"),
        )

        # --- plot background component ---
        model_tot.plotOn(
            frame,
            RooFit.Components(bkg_name),
            RooFit.DrawOption("Fl"),
            RooFit.Precision(1e-6),
            RooFit.FillColor(ROOT.kBlue),
            RooFit.LineColor(ROOT.kBlue),
            *([RooFit.Slice(tag, slice_label),
            RooFit.ProjWData(tag, combData)] if slice_label else []),
            RooFit.MoveToBack(),
            RooFit.Name("bkg_draw"),
        )

        leg = make_legend()
        leg.AddEntry(frame.findObject("data_draw"),  "Data", "pe")
        leg.AddEntry(frame.findObject("model_draw"), "Fit",  "l")
        leg.AddEntry(frame.findObject("bkg_draw"),   "Bkg.", "f")

        pull_hist = frame.pullHist()
        pull_hist.SetLineColor(ROOT.kBlack)
        pull_hist.SetFillColor(ROOT.kBlue)
        pull_frame = m.frame(m_min, m_max, n_bins)
        pull_frame.addPlotable(pull_hist, "BX")

        upper.cd()
        frame.SetTitle("")
        frame.Draw()
        leg.Draw("same")

        lower.cd()
        pull_frame.SetTitle("")
        pull_frame.GetXaxis().SetLabelSize(0)
        pull_frame.GetXaxis().SetTitle("")
        pull_frame.GetYaxis().SetTitle("")
        pull_frame.GetYaxis().SetLabelSize(0.1)
        pull_frame.GetYaxis().SetRangeUser(-5, 5)
        pull_frame.Draw("B")

        canvas.SaveAs(save_path + ".png")

        return frame

    c_plus  = ROOT.TCanvas("c_plus",  "c_plus",  930, 700)
    c_minus = ROOT.TCanvas("c_minus", "c_minus", 930, 700)
    c_tot   = ROOT.TCanvas("c_tot",   "c_tot",   930, 700)

    draw_fit(c_plus,  data_p,   "positive", "model_plus_draw", "bkg_plus",  output_path + "_plus")
    draw_fit(c_minus, data_m,   "negative", "model_minus_draw","bkg_minus", output_path + "_minus")
    frame_tot = draw_fit(c_tot, data_tot, "",  "model_tot",       "bkg_tot",   output_path + "_tot")

    # Asymmetry plot
    c_asym   = ROOT.TCanvas("c_asym", "c_asym", 930, 700)
    upper_a  = ROOT.TPad("upper_a", "", 0., 0.25, 1., 1.)
    lower_a  = ROOT.TPad("lower_a", "", 0., 0.,   1., 0.23)
    for pad in [upper_a, lower_a]:
        pad.SetRightMargin(margin); pad.SetLeftMargin(3*margin)
    upper_a.SetTopMargin(margin); lower_a.SetBottomMargin(2*margin)
    c_asym.cd(); lower_a.Draw(); upper_a.Draw()

    # Retrieve fit curves for + and - from separate frames
    frame_p = m.frame(RooFit.Bins(n_bins))
    combData.plotOn(frame_p, RooFit.Cut(f"{tag_name}=={tag_name}::positive"))
    model_tot.plotOn(frame_p, RooFit.Precision(1e-6), RooFit.LineColor(ROOT.kRed),
                     RooFit.Slice(tag, "positive"), RooFit.ProjWData(tag, combData),
                     RooFit.MoveToBack(), RooFit.Name("curve_plus"))
    frame_m = m.frame(RooFit.Bins(n_bins))
    combData.plotOn(frame_m, RooFit.Cut(f"{tag_name}=={tag_name}::negative"))
    model_tot.plotOn(frame_m, RooFit.Precision(1e-6), RooFit.LineColor(ROOT.kRed),
                     RooFit.Slice(tag, "negative"), RooFit.ProjWData(tag, combData),
                     RooFit.MoveToBack(), RooFit.Name("curve_minus"))
    curve_plus  = frame_p.findObject("curve_plus")
    curve_minus = frame_m.findObject("curve_minus")

    h_asym       = h_plus.GetAsymmetry(h_minus)
    h_asym.GetXaxis().SetRangeUser(m_min, m_max)
    h_asym.SetXTitle(f"{name_observable} (MeV/c^{{2}})")
    h_asym.SetYTitle("Asymmetry")

    deltaX = (m_max - m_min) / n_bins
    pdf_asym  = ROOT.TH1D("pdf_asym",  "", n_bins, m_min, m_max)
    pull_asym = ROOT.TH1D("pull_asym", "", n_bins, m_min, m_max)
    chi2_asym = 0.

    import math
    for i in range(n_bins):
        X   = m_min + (i + 0.5) * deltaX
        fp  = curve_plus.Eval(X)
        fm  = curve_minus.Eval(X)
        denom = fp + fm
        asym_val = (fp - fm) / denom if denom != 0 else 0.
        pdf_asym.SetBinContent(i+1, asym_val)
        data_val = h_asym.GetBinContent(h_asym.FindBin(X))
        data_err = h_asym.GetBinError(h_asym.FindBin(X))
        pull_val = (data_val - asym_val) / data_err if data_err != 0 else 5.
        if math.isinf(pull_val): pull_val = 5.
        pull_asym.SetBinContent(i+1, pull_val)
        if data_err != 0:
            chi2_asym += ((data_val - asym_val) / data_err) ** 2

    upper_a.cd()
    h_asym.Draw("ep")
    pdf_asym.SetLineColor(ROOT.kRed); pdf_asym.SetMarkerColor(ROOT.kRed)
    pdf_asym.Draw("l same")
    leg_a = make_legend()
    leg_a.AddEntry(h_asym,   "Data", "pe")
    leg_a.AddEntry(pdf_asym, "Fit",  "l")
    leg_a.Draw("same")

    lower_a.cd()
    pull_asym.SetLineColor(ROOT.kBlue); pull_asym.SetFillColor(ROOT.kBlue)
    pull_asym.GetXaxis().SetLabelSize(0); pull_asym.GetXaxis().SetTitle("")
    pull_asym.GetYaxis().SetTitle(""); pull_asym.GetYaxis().SetLabelSize(0.1)
    pull_asym.GetYaxis().SetRangeUser(-5, 5)
    pull_asym.Draw("")
    c_asym.SaveAs(output_path + "_asym.png")
    return fit_converges
