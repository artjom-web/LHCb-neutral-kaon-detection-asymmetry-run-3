"""Mass fitting (RooFit).

``MassFitter`` owns the simultaneous D+/D- mass fit and its convergence
machinery.  Both large methods below are ported verbatim from
``Analysis.mass_fit`` / ``Analysis.plot_massfit``; the only edits are:

  * the retry budget and RNG seed come from :class:`MassModelConfig`
    instead of being magic numbers in the body;
  * a fresh output folder can warm-start from a seed file supplied by
    config, not only from a previous fit in the same folder;
  * ``plot_massfit``'s loop over mass bins uses the histogram's own bin
    count rather than the analysis-wide ``nbins_m``.  These were always
    equal for the fits actually performed, so results are unchanged, but
    the fitter no longer needs to know the global mass binning at all;
  * folders whose fit failed to converge are recorded in ``self.failed``
    so stage 3 can report them instead of the message scrolling past.

Everything else — the jitter/retry strategy, the tiered ``better_result``
comparison, the workspace contents, the fit report format — is unchanged.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np

import ROOT
from ROOT import (
    RooRealVar, RooGaussian, RooExponential, RooAddPdf, RooDataHist, RooFit,
    RooPlot, TCanvas, RooArgList, TLegend, RooArgSet, RooFormulaVar,
    RooBifurGauss, RooJohnson, RooProdPdf, RooGenericPdf, RooSimultaneous,
    RooCategory, RooStats,
)

from ..user_uploads import mass_models as mm

from ..config.massmodels import MassModelConfig


class MassFitter:
    """Fits the D+/D- mass spectrum and persists the result.

    One instance per stage; ``mfit_iter`` and ``objects_keepalive`` exist
    purely to stop Python garbage-collecting RooFit objects that ROOT still
    references, exactly as in the original class.
    """

    def __init__(self, cfg: Optional[MassModelConfig] = None):
        self.cfg = cfg or MassModelConfig()
        self.mfit_iter = 0
        self.objects_keepalive = {}
        #: folders whose fit did not reach covQual == 3
        self.failed: List[str] = []
        self.param_names{
            'johnson_exp':  ["mean", "Delta_mean", "sigma", "gamma_johnson", "delta_johnson", "lambda_exponential",],
            'johnson_exp1': ["mean", "Delta_mean", "sigma", "Delta_sigma", "gamma_johnson", "delta_johnson", "lambda_exponential",],
            'johnson_gauss_exp': ["mean", "Delta_mean", "sigma", "gamma_johnson", "delta_johnson", "f_john", "lambda_exponential",],
        }
    def fit(
        self,
        h_plus,                          # TH1D: positive candidates
        h_minus,                         # TH1D: negative candidates
        folder,                          # result folder
        model_name=None,                 # None -> config standard model
        warm_start_file='',                   # True: ignore existing warm-start file
    ):
        model_name = model_name or self.cfg.standard_model

        self.mfit_iter += 1
        params_file_in  = folder + "_in.txt"
        params_file_out = folder + "_out.txt"
        model_file_out = folder + "model.root"
        os.makedirs(folder, exist_ok=True)
        fit_converges = True

        xaxis = h_plus.GetXaxis()

        xmin = xaxis.GetXmin()
        xmax = xaxis.GetXmax()
        nbins = xaxis.GetNbins()

        m = RooRealVar("Dp_M", "Dp_M", xmin, xmax, "MeV/c^{2}")
        m.setBins(nbins)

        data_p   = RooDataHist("data_p",   "Dp_M tag +", RooArgList(m), ROOT.RooFit.Import(h_plus))
        data_m   = RooDataHist("data_m",   "Dp_M tag -", RooArgList(m), ROOT.RooFit.Import(h_minus))
        h_tot    = h_plus.Clone("h_tot")
        h_tot.Add(h_minus)
        data_tot = RooDataHist("data_tot", "Dp_M", RooArgList(m), ROOT.RooFit.Import(h_tot))
        N_tot = h_tot.Integral()



        model = getattr(mm, model_name)

        mm_result = model(m)

        signal_plus = mm_result['sp']
        signal_minus = mm_result['sm']
        background_plus = mm_result['bp']
        background_minus = mm_result['bm']
        self.objects_keepalive[self.mfit_iter] = mm_result['objects']

        # Yields and asymmetries
        N_tot = RooRealVar("N_tot", "Total events",     N_tot, 0., 1.5*(N_tot))
        N_tot.setConstant(True)
        f_sig = RooRealVar("f_sig", "Signal Fraction", 0.5, 0., 1.)
        N_sig = RooFormulaVar("N_sig", "@0*@1", RooArgList(N_tot, f_sig))
        N_bkg = RooFormulaVar("N_bkg", "@0*(1-@1)", RooArgList(N_tot, f_sig))
        A_sig = RooRealVar("A_sig", "Signal Asymmetry", 0., -1., 1.)
        A_bkg = RooRealVar("A_bkg", "Background asymmetry",   0., -1., 1.)

        Y_sig_plus  = RooFormulaVar("Y_sig_plus",  "@0*0.5*(1+@1)", RooArgList(N_sig, A_sig))
        Y_sig_minus = RooFormulaVar("Y_sig_minus", "@0*0.5*(1-@1)", RooArgList(N_sig, A_sig))
        Y_bkg_plus  = RooFormulaVar("Y_bkg_plus",  "@0*0.5*(1+@1)", RooArgList(N_bkg, A_bkg))
        Y_bkg_minus = RooFormulaVar("Y_bkg_minus", "@0*0.5*(1-@1)", RooArgList(N_bkg, A_bkg))

        # model 1
        pdf_positive = RooAddPdf(
            "pdf_positive", "pdf_positive",
            RooArgList(signal_plus, background_plus),
            RooArgList(Y_sig_plus, Y_bkg_plus)
        )
        pdf_negative = RooAddPdf(
            "pdf_negative", "pdf_negative",
            RooArgList(signal_minus, background_minus),
            RooArgList(Y_sig_minus, Y_bkg_minus)
        )


        tag = RooCategory("Dp_charge", "Tag")
        tag.defineType("positive",  1)
        tag.defineType("negative", -1)

        obs = RooArgSet(m, tag)

        combData = RooDataHist("combData", "Combined data", RooArgList(m),
                            ROOT.RooFit.Index(tag),
                            ROOT.RooFit.Import("positive", data_p),
                            ROOT.RooFit.Import("negative", data_m))


        model_tot = RooSimultaneous("model_tot", "model_tot", tag)
        model_tot.addPdf(pdf_positive, "positive")
        model_tot.addPdf(pdf_negative, "negative")

        # ------------------------------------------------------------------
        # Fit
        # ------------------------------------------------------------------


        def construct_plot_objects():
            f_sig_plus = RooFormulaVar("f_sig_plus", "0.5*(1+@0)", RooArgList(A_sig))
            f_bkg_plus = RooFormulaVar("f_bkg_plus", "0.5*(1+@0)", RooArgList(A_bkg))

            signal_tot = RooAddPdf(
                "signal_tot", "Total signal (for plotting)",
                RooArgList(signal_plus, signal_minus),
                RooArgList(f_sig_plus)
            )
            bkg_tot = RooAddPdf(
                "bkg_tot", "Total background (for plotting)",
                RooArgList(background_plus, background_minus),
                RooArgList(f_bkg_plus)
            )

            # NEW: plain extended total-model pdf, only for the unsliced "total" plot
            model_tot_plot = RooAddPdf(
                "model_tot_plot", "Total model (for plotting)",
                RooArgList(signal_tot, bkg_tot),
                RooArgList(N_sig, N_bkg)
            )

            signal_tot._f_sig_plus_keepalive = f_sig_plus
            bkg_tot._f_bkg_plus_keepalive = f_bkg_plus
            model_tot_plot._signal_tot_keepalive = signal_tot
            model_tot_plot._bkg_tot_keepalive = bkg_tot

            return signal_tot, bkg_tot, model_tot_plot



        def save_ws(results):
            signal_tot, bkg_tot, model_tot_plot = construct_plot_objects()
            ws = ROOT.RooWorkspace("ws")

            # result.SetName("fit_result")    
            # getattr(ws, "import")(fit_result)

            for pdf in (model_tot, bkg_tot, signal_tot, model_tot_plot):
                getattr(ws, "import")(pdf, ROOT.RooFit.RecycleConflictNodes())

            for data in (combData, data_p, data_m, data_tot):
                getattr(ws, "import")(data)

            ws.writeToFile(model_file_out)

            outfile = ROOT.TFile(model_file_out, "UPDATE")
            outfile.cd()
            h_plus.Write("h_plus")
            h_minus.Write("h_minus")

            # ---- fit report ----
            report_path = folder + "_fit_report.txt"

            params = results.floatParsFinal()
            names = [params.at(i).GetName() for i in range(params.getSize())]

            col_width = 25

            with open(report_path, "w") as f:

                # ============================================================
                # Fit summary
                # ============================================================
                f.write("============================================================\n")
                f.write("FIT RESULT\n")
                f.write("============================================================\n\n")

                f.write(f"Minimized FCN value: {results.minNll():.6f}\n")
                f.write(f"EDM:                 {results.edm():.6e}\n")
                f.write(f"Fit status:           {results.status()}\n")
                f.write(f"Covariance quality:   {results.covQual()}\n\n")
                f.write(f"Ntot:   {results.floatParsFinal().find("N_tot")}\n")
                f.write(f"Nsig:   {results.floatParsFinal().find("N_sig")}\n")


                # ============================================================
                # Floating parameters
                # ============================================================
                f.write("\n--- Floating Parameters ---\n\n")

                f.write(
                    f"{'Parameter':30s}"
                    f"{'Value':>18s}"
                    f"{'Error':>18s}\n"
                )

                f.write("-" * 70 + "\n")

                for i in range(params.getSize()):
                    par = params.at(i)

                    f.write(
                        f"{par.GetName():30s}"
                        f"{par.getVal():18.8e}"
                        f"{par.getError():18.8e}\n"
                    )

                # ============================================================
                # Parameter ordering
                # ============================================================
                f.write("\n--- Parameter ordering ---\n\n")

                for i, name in enumerate(names):
                    f.write(f"{i:3d}: {name}\n")

                # ============================================================
                # Covariance Matrix
                # ============================================================
                f.write("\n--- Covariance Matrix ---\n\n")

                f.write(f"{'':25s}")

                for name in names:
                    f.write(f"{name:{col_width}s}")

                f.write("\n")

                cov = results.covarianceMatrix()

                for i, name in enumerate(names):

                    f.write(f"{name:25s}")

                    for j in range(len(names)):
                        f.write(f"{cov[i][j]:{col_width}.6e}")

                    f.write("\n")

                # ============================================================
                # Correlation Matrix
                # ============================================================
                f.write("\n--- Correlation Matrix ---\n\n")

                f.write(f"{'':25s}")

                for name in names:
                    f.write(f"{name:{col_width}s}")
                f.write("\n")
                corr = results.correlationMatrix()
                for i, name in enumerate(names):
                    f.write(f"{name:25s}")

                    for j in range(len(names)):
                        f.write(f"{corr[i][j]:{col_width}.6f}")

                    f.write("\n")


            # ================================================================
            # Save the RooFitResult into the ROOT output file
            # ================================================================
            results.Write("fit_results")

            outfile.Close()


        def do_fit(strategy):
            args = fit_args + [ROOT.RooFit.Strategy(strategy)]
            return model_tot.fitTo(combData, *args)


        def is_good(res):
            """Full convergence: RooFit's strictest, most trustworthy tier."""
            return (res.status() == 0 and res.covQual() == 3
                    and np.isfinite(res.minNll()) and np.isfinite(res.edm()))


        def is_acceptable(res):
            """Usable but not ideal — accepted only as a fallback if nothing
            reaches is_good() in the attempt budget."""
            return (res.status() == 0 and res.covQual() >= 2
                    and np.isfinite(res.minNll()) and np.isfinite(res.edm()))


        def better_result(new, old):
            """Tiered comparison: status, then covQual, then NLL, then EDM.
            Used both to track the best-so-far attempt and to decide what to
            jitter from next."""
            if old is None:
                return True
            new_ok, old_ok = new.status() == 0, old.status() == 0
            if new_ok != old_ok:
                return new_ok
            if new.covQual() != old.covQual():
                return new.covQual() > old.covQual()
            new_nll, old_nll = new.minNll(), old.minNll()
            if np.isfinite(new_nll) and np.isfinite(old_nll) and not np.isclose(new_nll, old_nll):
                return new_nll < old_nll
            return new.edm() < old.edm()

        def jitter_floating_params(rng, sigma_mult, max_frac_of_range=0.15):
            for var in params:
                if not isinstance(var, RooRealVar) or var.isConstant():
                    continue
                lo, hi = var.getMin(), var.getMax()
                span = hi - lo
                sigma = var.getError()
                # Don't trust a missing OR already-inflated error as a jitter scale —
                # a degenerate fit's blown-up error would otherwise feed a runaway
                # jitter on the next attempt, walking the parameter to its boundary.
                if not np.isfinite(sigma) or sigma <= 0 or sigma > max_frac_of_range * span:
                    sigma = 0.05 * span
                step_sigma = min(sigma_mult * sigma, max_frac_of_range * span)
                new_val = var.getVal() + rng.normal(0, step_sigma)
                var.setVal(float(np.clip(new_val, lo, hi)))


        def make_gaussian_constraint(name, value, error):
            center = ROOT.RooRealVar(
                f"{name}_center",
                f"{name} constraint center",
                value
            )

            constraint = ROOT.RooGaussian(
                f"{name}_constraint",
                f"Gaussian constraint on {name}",
                param,
                center,
                ROOT.RooFit.RooConst(error)
            )

            return constraint


        MAX_ATTEMPTS = self.cfg.max_attempts
        seed = self.cfg.seed
        rng = np.random.default_rng(seed)
        params = model_tot.getParameters(obs)

        fit_args = [
            ROOT.RooFit.Extended(True),
            ROOT.RooFit.Save(True),
            ROOT.RooFit.EvalBackend("legacy"),
            ROOT.RooFit.ExternalConstraints(constraints),
            ROOT.RooFit.SumW2Error(True),
            ROOT.RooFit.Offset(True),   # NEW — improves numerical conditioning of the NLL for large N
        ]
            
        if os.path.exists(warm_start_file): 
            f = ROOT.TFile.Open(warm_start_file, "READ")
            ws_ws = f.Get("ws")
            ws_param_names = self.param_names[model_name]
            ws_result = ws_ws.obj("fit_result")

            if is_good(ws_result):
                constraints = ROOT.RooArgSet()

                for param_name in ws_param_names:
                    param = ws_result.floatParsFinal().find(param_name)
                    if param is None:
                        raise ValueError(f"{param_name} not found in fit result")

                    value = param.getVal()
                    error = param.getError()
                    constraint = make_gaussian_constraint(param_name, value, error)
                    constraints.add(constraint)
                    params.setRealValue(param_name, value)

                fit_args = fit_args + ROOT.RooFit.ExternalConstraints(constraints)



        # if warm_start_file != '':
        #     if os.path.exists(warm_start_file):
        #         warm_start = warm_start_file
        #     else:
        #         warm_start = self.cfg.seed_file(model_name)
        # if warm_start:
        #     params.readFromFile(warm_start)

        initial_params = params.snapshot()   # attempt 0's controlled starting point (warm-started, if available)

        best_results = None
        best_params = None

        for attempt in range(MAX_ATTEMPTS):
            if attempt == 0:
                params.assign(initial_params)
            elif best_params is not None:
                params.assign(best_params)
                sigma_mult = 1.0 + 0.75 * (attempt - 1)   # widen the jump the longer it takes
                jitter_floating_params(rng, sigma_mult)
            else:
                # nothing usable found yet even once — restart from the controlled
                # point with a mild jitter rather than compounding on a bad state
                params.assign(initial_params)
                jitter_floating_params(rng, 0.5)

            strategy = 1 if attempt < 2 else 2   # cheap first two tries, escalate only if needed
            results = do_fit(strategy)
            current_params = params.snapshot()

            print(f"Attempt {attempt + 1}/{MAX_ATTEMPTS}: status={results.status()}, "
                f"covQual={results.covQual()}, EDM={results.edm():.3g}, "
                f"NLL={results.minNll():.6g}, strategy={strategy}")

            if better_result(results, best_results):
                best_results = results
                best_params = current_params

            if is_good(results):
                print(f"Converged (covQual=3) after {attempt + 1} attempt(s).")
                break
        else:
            print(f"WARNING: did not reach full convergence in {MAX_ATTEMPTS} attempts; "
                f"using best result found (status={best_results.status()}, "
                f"covQual={best_results.covQual()}, edm={best_results.edm():.3g}).")

        results = best_results
        fit_converges = is_good(results)

        # restore the model's LIVE parameters to the best snapshot found — fitTo
        # leaves them at whatever the *last* attempt produced, which isn't
        # necessarily best_results if a later jitter made things worse. Everything
        # below (save_ws, boundary checks, N_sig.getVal(), A_sig.getError()) reads
        # live parameter state, so this must happen before any of that runs.
        params.assign(best_params)

        if fit_converges:
            params.writeToFile(params_file_out)   # only ever persist a fully-converged snapshot
        elif is_acceptable(results):
            print("WARNING: best result is only partially acceptable (covQual < 3); "
                "not persisting to warm-start file.")
        else:
            print("WARNING: no acceptable fit found in any attempt; not persisting to warm-start file.")

        print("h_tot entries:", h_tot.GetEntries())
        print("data_tot.sumEntries():", data_tot.sumEntries())
        print("combData.sumEntries():", combData.sumEntries())
        print("N_tot (fixed):", N_tot.getVal())
        print("N_sig fitted:", N_sig.getVal())
        print("N_bkg fitted:", N_bkg.getVal())
        print("N_sig + N_bkg:", N_sig.getVal() + N_bkg.getVal())
        save_ws(results)

        if not fit_converges and self.cfg.plot_on_failure:
            self.failed.append(folder)
            try:
                self.plot(folder)
            except Exception as exc:
                print(f'WARNING: could not draw failed fit in {folder}: {exc}')

        # Warn if parameters near boundaries
        for var in params:
            if isinstance(var, RooRealVar) and not var.isConstant():
                val, err = var.getVal(), var.getError()
                if err > 0:
                    if abs(val - var.getMin()) / err < 3 or abs(val - var.getMax()) / err < 3:
                        print(f"WARNING: {var.GetName()} = {val:.4f} +/- {err:.4f} "
                            f"in [{var.getMin():.4f}, {var.getMax():.4f}] — close to boundary!")

        Nsig = N_sig.getVal()
        if Nsig > 1e-7:
            print(f"1/sqrt(N) = {(1.0/Nsig)**0.5:.6f}")
        else:
            print(f"Warning: nsig too small: {Nsig}")
        print(f"A_sig_err = {A_sig.getError():.6f}\n")

        return fit_converges

    def plot(self, folder):

        model_file_out = folder + "model.root"
        output_path = folder + "figures/"
        os.makedirs(output_path, exist_ok=True)

        margin = 0.05

        # ------------------------------------------------------------------
        # Load file + workspace
        # ------------------------------------------------------------------
        f = ROOT.TFile.Open(model_file_out)

        ws = f.Get("ws")

        model_tot = ws.pdf("model_tot")
        model_tot_plot = ws.pdf("model_tot_plot")
        combData  = ws.data("combData")
        data_p    = ws.data("data_p")
        data_m    = ws.data("data_m")
        data_tot  = ws.data("data_tot")
        
        if not model_tot_plot:
            raise RuntimeError("model_tot_plot not found in workspace — was save_ws() updated to import it?")

        h_plus = f.Get("h_plus")
        h_minus = f.Get("h_minus")

        if not h_plus or not isinstance(h_plus, ROOT.TH1D):
            print(f"WARNING: h_plus not a TH1D in {model_file_out} (type={type(h_plus)}), skipping plot")
            f.Close()
            return
        if not h_minus or not isinstance(h_minus, ROOT.TH1D):
            print(f"WARNING: h_minus not a TH1D in {model_file_out} (type={type(h_minus)}), skipping plot")
            f.Close()
            return

        m   = ws.var("Dp_M")
        tag = ws.cat("Dp_charge")

        xaxis = h_plus.GetXaxis()
        xmin = xaxis.GetXmin()
        xmax = xaxis.GetXmax()
        nbins = xaxis.GetNbins()

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def make_pads(canvas):
            upper = ROOT.TPad("upper", "", 0., 0.25, 1., 1.)
            lower = ROOT.TPad("lower", "", 0., 0.,   1., 0.23)

            for pad in (upper, lower):
                pad.SetRightMargin(margin)
                pad.SetLeftMargin(3 * margin)

            upper.SetTopMargin(margin)
            lower.SetBottomMargin(2 * margin)

            canvas.cd()
            lower.Draw()
            upper.Draw()
            return upper, lower

        def make_legend():
            leg = ROOT.TLegend(0.20, 0.65, 0.32, 0.90)
            leg.SetFillColor(ROOT.kWhite)
            leg.SetTextSize(0.055)
            leg.SetBorderSize(0)
            leg.SetTextFont(132)
            return leg

        # ------------------------------------------------------------------
        # Core plotting function
        # ------------------------------------------------------------------

        def draw_fit(canvas, data, slice_label, comp_bkg, save_name):
            plot_pdf = model_tot if slice_label else model_tot_plot
            upper, lower = make_pads(canvas)
            frame = m.frame(ROOT.RooFit.Bins(nbins))



            # ---------------- full model ----------------
            model_args = [
                ROOT.RooFit.Precision(1e-6),
                ROOT.RooFit.LineColor(ROOT.kRed),
                ROOT.RooFit.Name("model"),
            ]

            # Only the total plot needs fit-range normalization
            if slice_label:
                model_args += [
                    ROOT.RooFit.Slice(tag, slice_label),
                    ROOT.RooFit.ProjWData(tag, combData),
                ]



            # ---------------- background ----------------
            bkg_args = [
                ROOT.RooFit.Components(comp_bkg),
                ROOT.RooFit.FillColor(ROOT.kBlue),
                ROOT.RooFit.LineColor(ROOT.kBlue),
                ROOT.RooFit.DrawOption("F"),
                ROOT.RooFit.Precision(1e-6),
                ROOT.RooFit.Name("bkg"),
            ]

            if slice_label:
                bkg_args += [
                    ROOT.RooFit.Slice(tag, slice_label),
                    ROOT.RooFit.ProjWData(tag, combData),
                ]
            plot_pdf.plotOn(frame, *bkg_args)

    
            # ---------------- data ----------------
            if slice_label == "positive":
                data_p.plotOn(frame, ROOT.RooFit.Name("data"))
            elif slice_label == "negative":
                data_m.plotOn(frame, ROOT.RooFit.Name("data"))
            else:
                data.plotOn(frame, ROOT.RooFit.Name("data"))

            plot_pdf.plotOn(frame, *model_args)



            # ---------------- legend ----------------
            leg = make_legend()
            leg.AddEntry(frame.findObject("data"), "Data", "pe")
            leg.AddEntry(frame.findObject("model"), "Fit", "l")
            leg.AddEntry(frame.findObject("bkg"), "Bkg", "f")

            # ---------------- pull ----------------
            pull = frame.pullHist("data", "model")
            pull.SetLineColor(ROOT.kBlack)

            pull_frame = m.frame(
                ROOT.RooFit.Range(xmin,xmax),
                ROOT.RooFit.Bins(nbins)
            )
            pull_frame.addPlotable(pull, "BX")

            # ---------------- draw ----------------
            upper.cd()
            frame.SetTitle("")
            frame.Draw()

            leg.Draw()

            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextSize(0.04)
            latex.DrawLatex(0.65, 0.82, f"Entries = {data.sumEntries():.2e}")

            lower.cd()
            pull_frame.SetTitle("")
            pull_frame.GetYaxis().SetRangeUser(-5, 5)
            pull_frame.Draw("B")

            canvas.SaveAs(save_name + ".pdf")

            return frame

        # ------------------------------------------------------------------
        # Make plots
        # ------------------------------------------------------------------
    

        c_plus  = ROOT.TCanvas("c_plus", "", 900, 700)
        c_minus = ROOT.TCanvas("c_minus", "", 900, 700)
        c_tot   = ROOT.TCanvas("c_tot", "", 900, 700)

        draw_fit(c_plus,  data_p,   "positive", "expo_plus",              output_path + "plus")
        draw_fit(c_minus, data_m,   "negative", "expo_minus",              output_path + "minus")
        draw_fit(c_tot, data_tot, "", "bkg_tot", output_path + "total")

        # ------------------------------------------------------------------
        # Asymmetry plot
        # ------------------------------------------------------------------
        c_asym = ROOT.TCanvas("c_asym", "", 900, 700)

        upper = ROOT.TPad("u", "", 0, 0.25, 1, 1)
        lower = ROOT.TPad("l", "", 0, 0, 1, 0.23)

        upper.Draw()
        lower.Draw()

        frame_p = m.frame(ROOT.RooFit.Bins(nbins))
        frame_m = m.frame(ROOT.RooFit.Bins(nbins))

        model_tot.plotOn(
            frame_p,
            ROOT.RooFit.Slice(tag, "positive"),
            ROOT.RooFit.ProjWData(tag, combData),
            ROOT.RooFit.LineColor(ROOT.kRed),
            ROOT.RooFit.Name("p"),
        )

        model_tot.plotOn(
            frame_m,
            ROOT.RooFit.Slice(tag, "negative"),
            ROOT.RooFit.ProjWData(tag, combData),
            ROOT.RooFit.LineColor(ROOT.kRed),
            ROOT.RooFit.Name("m"),
        )


        combData.plotOn(frame_p, ROOT.RooFit.Cut(f"Dp_charge==Dp_charge::positive"))
        combData.plotOn(frame_m, ROOT.RooFit.Cut(f"Dp_charge==Dp_charge::negative"))



        cp = frame_p.findObject("p")
        cm = frame_m.findObject("m")

        h_asym = h_plus.GetAsymmetry(h_minus)
        h_asym.SetXTitle("Dp_M")
        h_asym.SetYTitle("Asymmetry")

        pdf_asym = ROOT.TH1D("pdf_asym", "", nbins, xmin, xmax)

        for i in range(nbins):
            x = xmin + (i + 0.5) * (xmax - xmin) / nbins

            fp = cp.Eval(x)
            fm = cm.Eval(x)

            denom = fp + fm
            val = (fp - fm) / denom if denom != 0 else 0

            pdf_asym.SetBinContent(i + 1, val)

        upper.cd()
        h_asym.Draw("EP")
        pdf_asym.SetLineColor(ROOT.kRed)
        pdf_asym.Draw("L SAME")

        lower.cd()
        ROOT.gPad.SetGridy()

        c_asym.SaveAs(output_path + "asym.pdf")

        f.Close()


