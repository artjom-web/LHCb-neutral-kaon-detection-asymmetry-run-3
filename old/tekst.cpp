bool mass_fit(TH1D* h_plus, TH1D* h_minus, TString name_observable, TString decay, TString params_file_basename, TString output_path, TString blinding_seed, bool first_time_runs = false, double range_min = -1, double range_max = -1, int save_asymmetry = 0, bool make_sweights = false, TString mass_name = "Dp_M", TString tag_name = "hp_PARTICLE_ID", int compare_sw = 0, json* jsonInformation = nullptr)
{
    
    bool fit_converges = true;
    h_plus->Sumw2();
    h_minus->Sumw2();
    TString paramsFileName_in = params_file_basename + "_in.txt";
    TString paramsFileName_out = params_file_basename + "_out.txt";
    int n_bins = h_plus->GetNbinsX();
    auto m_max = h_plus->GetXaxis()->GetXmax();
    auto m_min = h_plus->GetXaxis()->GetXmin();
    if (range_min != -1)
    {
        m_min = range_min;
    }
    if (range_max != -1)
    {
        m_max = range_max;
    }

    //Blinding value
    TRandom3 *rndm = new TRandom3(blinding_seed.Hash());
    rndm->SetSeed(blinding_seed.Hash());
    RooRealVar A_bias("A_bias", "Bias of the signal asymmetry", rndm->Uniform(-0.01,0.01));
    if (blinding_seed == "")
    {
        A_bias.setVal(0);
        A_bias.setConstant(kTRUE);
    }

    //Observable
    RooRealVar m(mass_name, name_observable, m_min, m_max, "MeV/#it{c}^{2}");

    m.setBins(n_bins); // 68
    RooDataHist data_p("data_p", name_observable + " tag +", RooArgList(m), Import(*h_plus));
    RooDataHist data_m("data_m", name_observable + " tag +", RooArgList(m), Import(*h_minus));

    TH1D* h_tot = (TH1D*)h_plus->Clone("h_tot");
    h_tot->Add(h_minus);
    RooDataHist data_tot("data_tot", name_observable, RooArgList(m), Import(*h_tot));

    const int N_plus = h_plus->GetEntries() * 1.5;
    const int N_minus = h_minus->GetEntries() * 1.5;

    // Defining variables for the total sample and asymmetries
    // Johnson - signal
    RooRealVar mean_johnson("mean_johnson", "Mean of the Johnson", (m_min + m_max)*0.5, m_min, m_max);
    RooRealVar Delta_mean_johnson("Delta_mean_johnson", "Asymmetry of Mean of the Johnson", 0., -5, 5);
    RooRealVar sigma_johnson("sigma_johnson", "Sigma of the Johnson", 15.0, 0.1, 100.0);
    RooRealVar Delta_sigma_johnson("Delta_sigma_johnson", "Asymmetry of the Sigma of the Johnson", 0., -5.0, 5.0);
    RooRealVar gamma_johnson("gamma_johnson", "Shape parameter that distorts distribution to left/right for positive", -0.199070, -3., 3.);
    //RooRealVar A_gamma_johnson("A_gamma_johnson", "Asymmetry of the gamma parameter of the Johnson", 0., -1.0, 3);
    RooRealVar Delta_gamma_johnson("Delta_gamma_johnson", "Delta of the gamma parameter of the Johnson", 0., -5, 5);
    RooRealVar delta_johnson("delta_johnson", "Shape parameter that determines strength of Gaussian-like component", 2, 1., 30.);
    RooRealVar Delta_delta_johnson("Delta_delta_johnson", "Asymmetry of the delta parameter of the Johnson", 0., -5.0, 5.0);

    //A_gamma_johnson.setVal(0);
    //A_gamma_johnson.setError(0);
    //A_delta_johnson.setVal(0);
    //A_delta_johnson.setError(0);
    //
    //A_gamma_johnson.setConstant(kTRUE);
    //A_delta_johnson.setConstant(kTRUE);

    //Gaussian - signal
    RooRealVar mean_gaussian("mean_gaussian", "Mean of the Gaussian", (m_min + m_max)*0.5, m_min, m_max);
    //RooRealVar& mean_gaussian = mean_johnson;
    RooRealVar Delta_mean_gaussian("Delta_mean_gaussian", "Asymmetry of Mean of the Gaussian", 0., -1.0, 1.0);
    RooRealVar sigma_gaussian("sigma_gaussian", "Sigma of the Gaussian", 3.0, 0.1, 10.0);
    RooRealVar Delta_sigma_gaussian("Delta_sigma_gaussian", "Asymmetry of the Sigma of the Gaussian", 0., -1.0, 1.0);

    Delta_sigma_gaussian.setVal(0);
    Delta_sigma_gaussian.setError(0);
    Delta_sigma_gaussian.setConstant(kTRUE);

    mean_gaussian.setConstant(kTRUE);
    Delta_mean_gaussian.setConstant(kTRUE);
    sigma_gaussian.setConstant(kTRUE);

    //Fractions of signal
    RooRealVar frac_sig_johnson_gaussian("frac_sig_johnson_gaussian", "Fraction of Johnson over Gaussian component in the signal model", 0.95, 0.85, 1.0);
    RooRealVar Delta_frac_sig_johnson_gaussian("Delta_frac_sig_johnson_gaussian", "Asymmetry of the fraction of Johnson over Gaussian component in the signal model", 0., -1.0, 1.0);

    frac_sig_johnson_gaussian.setVal(1);
    frac_sig_johnson_gaussian.setError(0);
    frac_sig_johnson_gaussian.setConstant(kTRUE);

    Delta_frac_sig_johnson_gaussian.setVal(0);
    Delta_frac_sig_johnson_gaussian.setError(0);
    Delta_frac_sig_johnson_gaussian.setConstant(kTRUE);

    //Exponential - background
    RooRealVar lambda_exponential("lambda_exponential", "Lambda parameter of the decreasing exponential", 0., -10., 0.05);
    RooRealVar Delta_lambda_exponential("Delta_lambda_exponential", "Asymmetry of the lambda paramenter of the exponential", 0., -1.0, 1.0);

    //BifurGaussian - background
    RooRealVar mean_gaussian_part_reco("mean_gaussian_part_reco", "Mean of the bifurcated Gaussian for partially reconstructed background", m_min + (m_max - m_min)*0.25, m_min, m_min + (m_max - m_min)*0.4);
    RooRealVar sigma_gaussian_part_reco_L("sigma_gaussian_part_reco_L", "Left sigma paramenter of the Gaussian for partially reconstructed background", 10., 0.1, 30.);
    RooRealVar sigma_gaussian_part_reco_R("sigma_gaussian_part_reco_R", "Right sigma paramenter of the Gaussian for partially reconstructed background", 10., 0.1, 30.);
    //Assuming no asymmetry between positive and negative tagged particles

    mean_gaussian_part_reco.setConstant(kTRUE);
    sigma_gaussian_part_reco_L.setConstant(kTRUE);
    sigma_gaussian_part_reco_R.setConstant(kTRUE);

    //Fractions of background
    RooRealVar frac_bkg_expo_part_reco("frac_bkg_expo_part_reco", "Fraction of background composed of the decreasing exponential over the partially reconstructed bifurcated Gaussian", 0.95, 0.9, 1.);
    RooRealVar Delta_frac_bkg_expo_part_reco("Delta_frac_bkg_expo_part_reco", "Asymmetry of the fraction of background composed of the decreasing exponential over the partially reconstructed bifurcated Gaussian", 0., -1., 1.);

    frac_bkg_expo_part_reco.setVal(1);
    frac_bkg_expo_part_reco.setError(0);
    frac_bkg_expo_part_reco.setConstant(kTRUE);

    Delta_frac_bkg_expo_part_reco.setVal(0);
    Delta_frac_bkg_expo_part_reco.setError(0);
    Delta_frac_bkg_expo_part_reco.setConstant(kTRUE);

    //Yields
    RooRealVar N_sig("N_sig", "Total number of signal events", (N_plus + N_minus) * 0.7, 0., N_plus + N_minus);
    RooRealVar A_sig_blind("A_sig_blind", "Asymmetry of the signal blinded", 0., -1.0, 1.0);            
    RooFormulaVar A_sig("A_sig", "A_sig", "(@0+@1)", RooArgSet(A_sig_blind, A_bias));

    RooRealVar N_bkg("N_bkg", "Total number of background events", (N_plus + N_minus) * 0.2, 0., N_plus + N_minus);
    RooRealVar A_bkg("A_bkg", "Asymmetry of the background", 0., -1.0, 1.0);

    //Deriving variables for positive and negative candidates
    //Johnson
    RooFormulaVar mean_johnson_plus("mean_johnson_plus", "@0+@1", RooArgList(mean_johnson, Delta_mean_johnson));
    RooFormulaVar sigma_johnson_plus("sigma_johnson_plus", "@0+@1", RooArgList(sigma_johnson, Delta_sigma_johnson));
    //RooFormulaVar gamma_johnson_plus("gamma_johnson_plus", "@0*(1.+@1)", RooArgList(gamma_johnson, A_gamma_johnson));
    RooFormulaVar gamma_johnson_plus("gamma_johnson_plus", "@0+@1", RooArgList(gamma_johnson, Delta_gamma_johnson));
    RooFormulaVar delta_johnson_plus("delta_johnson_plus", "@0+@1", RooArgList(delta_johnson, Delta_delta_johnson));

    RooFormulaVar mean_johnson_minus("mean_johnson_minus", "@0-@1", RooArgList(mean_johnson, Delta_mean_johnson));
    RooFormulaVar sigma_johnson_minus("sigma_johnson_minus", "@0-@1", RooArgList(sigma_johnson, Delta_sigma_johnson));
    //RooFormulaVar gamma_johnson_minus("gamma_johnson_minus", "@0*(1.-@1)", RooArgList(gamma_johnson, A_gamma_johnson));
    RooFormulaVar gamma_johnson_minus("gamma_johnson_minus", "@0-@1", RooArgList(gamma_johnson, Delta_gamma_johnson));
    RooFormulaVar delta_johnson_minus("delta_johnson_minus", "@0-@1", RooArgList(delta_johnson, Delta_delta_johnson));
    //when the "total observable" is not the sum of the two positive and negative components, but somethig like an average, the 0.5 factor shall not be placed

    //Gaussian
    RooFormulaVar mean_gaussian_plus("mean_gaussian_plus", "@0+@1", RooArgList(mean_gaussian, Delta_mean_gaussian));
    RooFormulaVar sigma_gaussian_plus("sigma_gaussian_plus", "@0+@1", RooArgList(sigma_gaussian, Delta_sigma_gaussian));

    RooFormulaVar mean_gaussian_minus("mean_gaussian_minus", "@0-@1", RooArgList(mean_gaussian, Delta_mean_gaussian));
    RooFormulaVar sigma_gaussian_minus("sigma_gaussian_minus", "@0-@1", RooArgList(sigma_gaussian, Delta_sigma_gaussian));

    //Fraction signal
    RooFormulaVar frac_sig_johnson_gaussian_plus("frac_sig_johnson_gaussian_plus", "@0+@1", RooArgList(frac_sig_johnson_gaussian, Delta_frac_sig_johnson_gaussian));

    RooFormulaVar frac_sig_johnson_gaussian_minus("frac_sig_johnson_gaussian_minus", "@0-@1", RooArgList(frac_sig_johnson_gaussian, Delta_frac_sig_johnson_gaussian));

    //Exponential background
    RooFormulaVar lambda_exponential_plus("lambda_exponential_plus", "@0+@1", RooArgList(lambda_exponential, Delta_lambda_exponential));

    RooFormulaVar lambda_exponential_minus("lambda_exponential_minus", "@0-@1", RooArgList(lambda_exponential, Delta_lambda_exponential));

    //Fractions of background
    RooFormulaVar frac_bkg_expo_part_reco_plus("frac_bkg_expo_part_reco_plus", "@0+@1", RooArgList(frac_bkg_expo_part_reco, Delta_frac_bkg_expo_part_reco));
    
    RooFormulaVar frac_bkg_expo_part_reco_minus("frac_bkg_expo_part_reco_minus", "@0-@1", RooArgList(frac_bkg_expo_part_reco, Delta_frac_bkg_expo_part_reco));

    //Yields
    //RooFormulaVar N_sig_plus("N_sig_plus", "0.5*@0*(1.+@1)", RooArgList(N_sig, A_sig));
    //RooFormulaVar N_sig_minus("N_sig_minus", "0.5*@0*(1.0-@1)", RooArgList(N_sig, A_sig));
    //
    //RooFormulaVar N_bkg_plus("N_bkg_plus", "0.5*@0*(1.+@1)", RooArgList(N_bkg, A_bkg));
    //RooFormulaVar N_bkg_minus("N_bkg_minus", "0.5*@0*(1.0-@1)", RooArgList(N_bkg, A_bkg));

    RooFormulaVar N_sig_plus("N_sig_plus", "0.5*(1.0+@0)", RooArgList(A_sig));
    RooFormulaVar N_sig_minus("N_sig_minus", "0.5*(1.0-@0)", RooArgList(A_sig));
    
    RooFormulaVar N_bkg_plus("N_bkg_plus", "0.5*(1.0+@0)", RooArgList(A_bkg));
    RooFormulaVar N_bkg_minus("N_bkg_minus", "0.5*(1.0-@0)", RooArgList(A_bkg));

    //Models for positive and negative candidates
    RooJohnson johnson_plus("johnson_plus", "Signal Johnson for the positive candidates", m, mean_johnson_plus, sigma_johnson_plus, gamma_johnson_plus, delta_johnson_plus);
    RooGaussian gaussian_plus("gaussian_plus", "Signal Gaussian for the positive candidates", m, mean_gaussian_plus, sigma_gaussian_plus);
    RooBifurGauss bifurGauss_part_reco_plus("bifurGauss_part_reco", "Bifurcated Gaussian for partially reconstructed background for positively tagged candidates", m, mean_gaussian_part_reco, sigma_gaussian_part_reco_L, sigma_gaussian_part_reco_R);
    RooExponential expo_plus("expo_plus", "Decreasing exponential background for the positive candidates", m, lambda_exponential_plus);
    
    RooAddPdf bkg_plus("bkg_plus", "Positive candidates background PDF", RooArgList(expo_plus, bifurGauss_part_reco_plus), RooArgList(frac_bkg_expo_part_reco_plus));
    RooAddPdf signal_plus("signal_plus", "Positive candidates signal PDF", RooArgList(johnson_plus, gaussian_plus), RooArgList(frac_sig_johnson_gaussian_plus));
    RooAddPdf model_plus("model_plus", "Positive candidates model", RooArgList(signal_plus, bkg_plus), RooArgList(N_sig_plus, N_bkg_plus));

    RooJohnson johnson_minus("johnson_minus", "Signal Johnson for the negative candidates", m, mean_johnson_minus, sigma_johnson_minus, gamma_johnson_minus, delta_johnson_minus);
    RooGaussian gaussian_minus("gaussian_minus", "Signal Gaussian for the negative candidates", m, mean_gaussian_minus, sigma_gaussian_minus);
    RooBifurGauss bifurGauss_part_reco_minus("bifurGauss_part_reco_minus", "Bifurcated Gaussian for partially reconstructed background for negatively tagged candidates", m, mean_gaussian_part_reco, sigma_gaussian_part_reco_L, sigma_gaussian_part_reco_R);
    RooExponential expo_minus("expo_minus", "Decreasing exponential background for the negative candidates", m, lambda_exponential_minus);
    
    RooAddPdf bkg_minus("bkg_minus", "Negative candidates background PDF", RooArgList(expo_minus, bifurGauss_part_reco_minus), RooArgList(frac_bkg_expo_part_reco_minus));
    RooAddPdf signal_minus("signal_minus", "Negative candidates signal PDF", RooArgList(johnson_minus, gaussian_minus), RooArgList(frac_sig_johnson_gaussian_minus));
    RooAddPdf model_minus("model_minus", "Negative candidates model", RooArgList(signal_minus, bkg_minus), RooArgList(N_sig_minus, N_bkg_minus));

    ////Total model
    //RooAddPdf bkg_tot("bkg_tot", "Total background PDF", RooArgList(bkg_plus, bkg_minus), RooArgList(N_bkg_plus, N_bkg_minus));
    //RooAddPdf signal_tot("signal_tot", "Total signal PDF", RooArgList(signal_plus, signal_minus), RooArgList(N_sig_plus, N_sig_minus));
    //RooAddPdf model_tot("model_tot", "Total model", RooArgList(signal_tot, bkg_tot), RooArgList(N_sig, N_bkg));
    //
    ////Defining category of data
    //RooCategory tag("tag", "Tag for positive and negative candidates");
    //tag.defineType("positive");
    //tag.defineType("negative");
    //
    //RooDataHist combData("combData", "Combined data", RooArgList(m), Index(tag), Import("positive", data_p), Import("negative", data_m));
    //RooSimultaneous simPdf("simPdf", "Simultaneous pdf", tag);
    //simPdf.addPdf(model_plus, "positive");
    //simPdf.addPdf(model_minus, "negative");
    //
    //RooArgSet *obs = new RooArgSet();
    //obs->add(m);
    //obs->add(tag);

    // SERENA'S VERSION
    RooCategory tag(tag_name, "Tag for positive and negative candidates");
    tag.defineType("positive", 1);
    tag.defineType("negative", -1);

    RooArgSet *obs = new RooArgSet();
    obs->add(m);
    obs->add(tag);

    RooDataHist combData("combData", "Combined data", RooArgList(m), Index(tag), Import("positive", data_p), Import("negative", data_m));

    RooGenericPdf tag_plus("tag_plus", "tag_plus", "@0==1", RooArgSet(tag));
    RooGenericPdf tag_minus("tag_minus", "tag_minus", "@0==-1", RooArgSet(tag));
  
    RooProdPdf pdf_sigD_plus("pdf_sigD_plus", "pdf_sigD_plus", RooArgSet(tag_plus, signal_plus));
    RooProdPdf pdf_sigD_minus("pdf_sigD_minus", "pdf_sigD_minus", RooArgSet(tag_minus, signal_minus));
    RooProdPdf pdf_bkg_plus("pdf_bkg_plus", "pdf_bkg_plus", RooArgSet(tag_plus, bkg_plus));
    RooProdPdf pdf_bkg_minus("pdf_bkg_minus", "pdf_bkg_minus", RooArgSet(tag_minus, bkg_minus));
  
    //RooAddPdf signal_tot("signal_tot", "signal_tot", RooArgSet(*pdf_sigD_plus, *pdf_sigD_minus), RooArgSet(*N_sigD_plus, *N_sigD_minus));
    //RooAddPdf bkg_tot("bkg_tot","bkg_tot", RooArgSet(*pdf_bkg_plus, *pdf_bkg_minus), RooArgSet(*N_bkg_plus, *N_bkg_minus));

    //RooAddPdf totPDF("totPDF", "totPDF", RooArgSet(*pdf_sigD, *pdf_bkg), RooArgSet(*N_sigD, *N_bkg));

    RooAddPdf bkg_tot("bkg_tot", "Total background PDF", RooArgList(pdf_bkg_plus, pdf_bkg_minus), RooArgList(N_bkg_plus, N_bkg_minus));
    RooAddPdf signal_tot("signal_tot", "Total signal PDF", RooArgList(pdf_sigD_plus, pdf_sigD_minus), RooArgList(N_sig_plus, N_sig_minus));
    RooAddPdf model_tot("model_tot", "Total model", RooArgList(signal_tot, bkg_tot), RooArgList(N_sig, N_bkg));

    //RooNumIntConfig &conf=RooNumIntConfig::defaultConfig();
    //conf.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");
    //RooAbsReal::defaultIntegratorConfig()->setEpsAbs(1e-8) ;
    //RooAbsReal::defaultIntegratorConfig()->setEpsRel(1e-8) ;  
    //RooAbsReal::defaultIntegratorConfig()->Print("v");

    RooArgSet *params = model_tot.getParameters(*obs);
    params->remove(A_bias); // Hiding the random shift from parameter files
    RooFitResult *results = new RooFitResult();

    if (first_time_runs)
    {
        for (int f = 0; f < 4; ++f)
        {
            if (f == 0)
            {
                //params->readFromFile(paramsFileName_in);
                results = model_tot.fitTo(combData, Extended(true), Save(true), SumW2Error(true), EvalBackend("legacy"));
                params->writeToFile(paramsFileName_in);
            }
            if (f == 1)
            {
                cout << "\nSecond iteration!\n\n" << flush;
                results = model_tot.fitTo(combData, Extended(true), Save(true), SumW2Error(true), EvalBackend("legacy"));
                params->writeToFile(paramsFileName_in);
            }
            if (f == 2)
            {
                cout << "\nThird iteration!\n\n" << flush;
                results = model_tot.fitTo(combData, Extended(true), Save(true), SumW2Error(true), Strategy(2), EvalBackend("legacy"));
                params->writeToFile(paramsFileName_in);
            }
            if (f == 3)
            {
                cout << "\nFourth iteration!\n\n" << flush;
                results = model_tot.fitTo(combData, Extended(true), Save(true), SumW2Error(true), Strategy(2), EvalBackend("legacy"));
                params->writeToFile(paramsFileName_out);
            }
        }
        if ((results->status() != 0 && results->status() != 1) || (results->covQual() != 3 && results->covQual() != 2))
        {
            cout << "\n\n\nATTENTION!!!\tMINUIT STATUS CODE IS " << results->status() << endl
                 << endl;
            cout << "\tCovariance Matrix Quality IS " << results->covQual() << " for " << endl
                 << endl;
            fit_converges = false;
        }
    }
    else
    {
        params->readFromFile(paramsFileName_in);
        bool fit_not_ok = false; int contatore = 0;
        while(true && !fit_not_ok && contatore < 5)
        {
            results = model_tot.fitTo(combData, Extended(true), Save(true), SumW2Error(true), Strategy(2), EvalBackend("legacy"));
            params->writeToFile(paramsFileName_out);
            contatore++;

            if ((results->status() == 0 || results->status() == 1) && (results->covQual() == 3 || results->covQual() == 2))
            {
                fit_not_ok = true;
            }
        }

        cout << "Loop fit " << contatore << endl << flush;

        if ((results->status() != 0 && results->status() != 1) || (results->covQual() != 3 && results->covQual() != 2))
        {
            fit_converges = false;
        }
    }

    TCanvas * c_tot = new TCanvas("c_tot","c_tot",930,700);
    TCanvas * c_plus = new TCanvas("c_plus","c_plus",930,700);
    TCanvas * c_minus = new TCanvas("c_minus","c_minus",930,700);   
    TPad* upperPad = new TPad("upperPad", "upperPad",   0., .25, 1., 1.);
    TPad* lowerPad = new TPad("lowerPad", "lowerPad",   0., 0.,  1., .23);
    TPad* upperPad_plus = new TPad("upperPad_plus", "upperPad_plus",   0., .25, 1., 1.);
    TPad* lowerPad_plus = new TPad("lowerPad_plus", "lowerPad_plus",   0., 0.,  1., .23);
    TPad* upperPad_minus = new TPad("upperPad_minus", "upperPad_minus",   0., .25, 1., 1.);
    TPad* lowerPad_minus = new TPad("lowerPad_minus", "lowerPad_minus",   0., 0.,  1., .23);

    Double_t margin = 0.05;
    upperPad->SetRightMargin(margin);
    upperPad->SetLeftMargin(3*margin);
    upperPad->SetTopMargin(margin);
    lowerPad->SetRightMargin(margin);
    lowerPad->SetLeftMargin(3*margin);
    lowerPad->SetBottomMargin(2*margin); 

    upperPad_plus->SetRightMargin(margin);
    upperPad_plus->SetLeftMargin(3*margin);
    upperPad_plus->SetTopMargin(margin);
    lowerPad_plus->SetRightMargin(margin);
    lowerPad_plus->SetLeftMargin(3*margin);
    lowerPad_plus->SetBottomMargin(2*margin); 

    upperPad_minus->SetRightMargin(margin);
    upperPad_minus->SetLeftMargin(3*margin);
    upperPad_minus->SetTopMargin(margin);
    lowerPad_minus->SetRightMargin(margin);
    lowerPad_minus->SetLeftMargin(3*margin);
    lowerPad_minus->SetBottomMargin(2*margin);    

    TLegend * leg = new TLegend(0.20,0.65,0.32,0.90);
    leg->SetFillColor(kWhite);
    leg->SetTextSize(0.055);
    leg->SetBorderSize(0);
    leg->SetTextFont(132);
    leg->SetHeader("LHCb");
    TLegendEntry *header = (TLegendEntry*)leg->GetListOfPrimitives()->First();
    header->SetTextSize(.07);

    //Plotting positive
    RooPlot * plot_plus = m.frame(RooFit::Bins(n_bins));
    combData.plotOn(plot_plus, Cut(tag_name+"=="+tag_name+"::positive"), Name("positive_data_draw"));
    model_tot.plotOn(plot_plus, Precision(1e-6), LineColor(kRed), Slice(tag, "positive"), ProjWData(tag, combData), MoveToBack(), Name("model_plus_draw"));
    model_tot.plotOn(plot_plus, Components("bkg_plus"), DrawOption("Fl"), Precision(1e-6), FillColor(kBlue), LineColor(kBlue), Slice(tag,"positive"), ProjWData(tag, combData), MoveToBack(), Name("bkg_plus_draw"));
    //model_tot.plotOn(plot_plus, Components("bkg_plus"), Precision(1e-6), LineColor(kBlue), Slice(tag,"positive"), ProjWData(tag, combData), MoveToBack());
    leg->AddEntry(plot_plus->findObject("positive_data_draw"), "Data", "pe");
    leg->AddEntry(plot_plus->findObject("model_plus_draw"), "Fit", "l");
    leg->AddEntry(plot_plus->findObject("bkg_plus_draw"), "Bkg.", "f");
    RooHist* hpull_plus = plot_plus->pullHist();
    hpull_plus->SetLineColor(kBlack);
    hpull_plus->SetFillColor(kBlue);
    RooPlot* pulls_plus = m.frame(m_min, m_max, n_bins);
    pulls_plus->addPlotable(hpull_plus, "BX");
    c_plus->SetFillColor(kWhite);
    c_plus->cd();
    lowerPad_plus->Draw();
    upperPad_plus->Draw();
    upperPad_plus->cd();
    plot_plus->SetTitle("");
    //plot_plus->SetXTitle(XTitle);
    //plot_plus->SetYTitle(YTitle);
    plot_plus->Draw();
    leg->Draw("same");
    lowerPad_plus->cd();
    pulls_plus->SetTitle("");
    pulls_plus->GetXaxis()->SetLabelSize(0);
    pulls_plus->GetXaxis()->SetTitle("");
    pulls_plus->GetYaxis()->SetTitle("");
    pulls_plus->GetYaxis()->SetLabelSize(0.1);
    pulls_plus->GetYaxis()->SetRangeUser(-5,5);
    pulls_plus->Draw("B");
    RooCurve * totPDF_plus = (RooCurve*)upperPad_plus->FindObject("model_plus_draw");

    //Plotting negative
    RooPlot * plot_minus = m.frame(RooFit::Bins(n_bins));
    combData.plotOn(plot_minus, Cut(tag_name+"=="+tag_name+"::negative"), Name("negative_data"));
    model_tot.plotOn(plot_minus, Precision(1e-6), LineColor(kRed), Slice(tag, "negative"), ProjWData(tag, combData), MoveToBack(), Name("model_minus_draw"));
    model_tot.plotOn(plot_minus, Components("bkg_minus"), DrawOption("Fl"), Precision(1e-6), FillColor(kBlue), LineColor(kBlue), Slice(tag,"negative"), ProjWData(tag, combData), MoveToBack(), Name("bkg_minus_draw"));
    //model_tot.plotOn(plot_minus, Components("bkg_minus"), Precision(1e-6), LineColor(kBlue), Slice(tag,"negative"), ProjWData(tag, combData), MoveToBack());
    RooHist* hpull_minus = plot_minus->pullHist();
    hpull_minus->SetLineColor(kBlack);
    hpull_minus->SetFillColor(kBlue);
    RooPlot* pulls_minus = m.frame(m_min, m_max, n_bins);
    pulls_minus->addPlotable(hpull_minus, "BX");
    c_minus->SetFillColor(kWhite);
    c_minus->cd();
    lowerPad_minus->Draw();
    upperPad_minus->Draw();
    upperPad_minus->cd();
    plot_minus->SetTitle("");
    //plot_minus->SetXTitle(XTitle);
    //plot_minus->SetYTitle(YTitle);
    plot_minus->Draw();
    leg->Draw("same");
    lowerPad_minus->cd();
    pulls_minus->SetTitle("");
    pulls_minus->GetXaxis()->SetLabelSize(0);
    pulls_minus->GetXaxis()->SetTitle("");
    pulls_minus->GetYaxis()->SetTitle("");
    pulls_minus->GetYaxis()->SetLabelSize(0.1);
    pulls_minus->GetYaxis()->SetRangeUser(-5,5);
    pulls_minus->Draw("B");
    RooCurve * totPDF_minus = (RooCurve*)upperPad_minus->FindObject("model_minus_draw");

    //Total model
    RooPlot *plot = m.frame(RooFit::Bins(n_bins));
    data_tot.plotOn(plot); 
    model_tot.plotOn(plot, Precision(1e-6), LineColor(kRed), MoveToBack(), Name("model_tot"));
    model_tot.plotOn(plot, Components("bkg_tot"), DrawOption("Fl"), Precision(1e-6), FillColor(kBlue), LineColor(kBlue), MoveToBack(), Name("bkg_tot"));
    //model_tot.plotOn(plot, Components("bkg_tot"), Precision(1e-6), LineColor(kBlue), MoveToBack());
    RooHist* hpull = plot->pullHist();
    hpull->SetLineColor(kBlack);
    hpull->SetFillColor(kBlue);
    RooPlot* pulls = m.frame(m_min, m_max, n_bins);
    pulls->addPlotable(hpull, "BX");
    c_tot->SetFillColor(kWhite);
    c_tot->cd();
    lowerPad->Draw();
    upperPad->Draw();
    upperPad->cd();
    plot->SetTitle("");
    //plot->SetXTitle(XTitle);
    //plot->SetYTitle(YTitle);
    plot->Draw();
    leg->Draw("same");

    lowerPad->cd();
    pulls->SetTitle("");
    pulls->GetXaxis()->SetLabelSize(0);
    pulls->GetXaxis()->SetTitle("");
    pulls->GetYaxis()->SetTitle("");
    pulls->GetYaxis()->SetLabelSize(0.1);
    pulls->GetYaxis()->SetRangeUser(-5,5);
    pulls->Draw("");


    TCanvas * c_asym = new TCanvas("c_asym","c_asym",930,700);
    TPad*    upperPad_asym = new TPad("upperPad_asym", "upperPad_asym",   0., .25, 1., 1.);
    TPad*    lowerPad_asym = new TPad("lowerPad_asym", "lowerPad_asym",   0., 0.,  1., .23);

    TH1D *h_asymmetry = new TH1D("h_asymmetry", "h_asymmetry", n_bins, m_min, m_max);
    TH1D *totPDF_asymmetry = new TH1D("totPDF_asymmetry", "totPDF_asymmetry", n_bins, m_min, m_max);
    TH1D *pull_asymmetry = new TH1D("pull_asymmetry", "pull_asymmetry", n_bins, m_min, m_max);
    h_asymmetry = (TH1D*)h_plus->GetAsymmetry(h_minus);
    h_asymmetry->GetXaxis()->SetRange(m_min, m_max);
    h_asymmetry->GetXaxis()->SetRangeUser(m_min, m_max);
    h_asymmetry->SetXTitle(name_observable + " (MeV/#it{c}^{2})");
    h_asymmetry->SetYTitle("Asymmetry");

    Double_t deltaX = (m_max - m_min)/n_bins;
    Double_t X;
    Double_t chi2_asymmetry = 0;

    //for (Int_t i = 0; i < n_bins; i++)
    //{
    //    X = m_min + (i + 0.5) * deltaX;
    //    m.setVal(X);
    //    totPDF_asymmetry->SetBinContent(i + 1, ((N_bkg_plus.getVal() + N_sig_plus.getVal()) * model_plus.getVal(RooArgSet(m)) - (N_bkg_minus.getVal() + N_sig_minus.getVal()) * model_minus.getVal(RooArgSet(m))) / ((N_bkg_plus.getVal() + N_sig_plus.getVal()) * model_plus.getVal(RooArgSet(m)) + (N_bkg_minus.getVal() + N_sig_minus.getVal()) * model_minus.getVal(RooArgSet(m))));
    //    if(h_asymmetry->GetBinError(h_asymmetry->FindBin(X))) pull_asymmetry->SetBinContent(i + 1, (h_asymmetry->GetBinContent(h_asymmetry->FindBin(X)) - totPDF_asymmetry->GetBinContent(i + 1)) / h_asymmetry->GetBinError(h_asymmetry->FindBin(X)));
    //    else pull_asymmetry->SetBinContent(i + 1, 5);
    //    chi2_asymmetry += pow((h_asymmetry->GetBinContent(h_asymmetry->FindBin(X)) - totPDF_asymmetry->GetBinContent(i + 1)) / h_asymmetry->GetBinError(h_asymmetry->FindBin(X)), 2);
    //}
    
    for (Int_t i = 0; i < n_bins; i++)
    {
        X = m_min + (i + 0.5) * deltaX;
        totPDF_asymmetry->SetBinContent(i + 1, (totPDF_plus->Eval(X) - totPDF_minus->Eval(X)) / (totPDF_plus->Eval(X) + totPDF_minus->Eval(X)));
        pull_asymmetry->SetBinContent(i + 1, (h_asymmetry->GetBinContent(h_asymmetry->FindBin(X)) - totPDF_asymmetry->GetBinContent(i+1))/h_asymmetry->GetBinError(h_asymmetry->FindBin(X)));
        if(isinf(pull_asymmetry->GetBinContent(i + 1)))
          pull_asymmetry->SetBinContent(i + 1, 5);
        chi2_asymmetry += pow((h_asymmetry->GetBinContent(h_asymmetry->FindBin(X))-totPDF_asymmetry->GetBinContent(i+1))/h_asymmetry->GetBinError(h_asymmetry->FindBin(X)),2);
    }

    c_asym->cd();
    upperPad_asym->SetRightMargin(margin);
    upperPad_asym->SetLeftMargin(3*margin);
    upperPad_asym->SetTopMargin(margin);
    lowerPad_asym->SetRightMargin(margin);
    lowerPad_asym->SetLeftMargin(3*margin);
    lowerPad_asym->SetBottomMargin(2*margin);
    
    lowerPad_asym->Draw();
    upperPad_asym->Draw();
    upperPad_asym->cd();
    upperPad_asym->cd();
    h_asymmetry->Draw("ep");
    totPDF_asymmetry->SetMarkerColor(kRed);
    totPDF_asymmetry->SetLineColor(kRed);
    totPDF_asymmetry->Draw("lsame");

    TLegend * leg_asy = new TLegend(0.20,0.65,0.32,0.90); // 0.20,0.65,0.32,0.90
    leg_asy->SetFillColor(kWhite);
    leg_asy->SetTextSize(0.055);
    leg_asy->SetBorderSize(0);
    leg_asy->SetTextFont(132);
    leg_asy->SetHeader("LHCb");
    TLegendEntry *header_asy = (TLegendEntry*)leg_asy->GetListOfPrimitives()->First();
    header_asy->SetTextSize(.07);
    leg_asy->AddEntry(h_asymmetry, "Data", "pe");
    leg_asy->AddEntry(totPDF_asymmetry, "Fit", "l");
    leg_asy->Draw("same");

    lowerPad_asym->cd();
    pull_asymmetry->SetLineColor(kBlue);
    pull_asymmetry->SetFillColor(kBlue);
    pull_asymmetry->GetXaxis()->SetLabelSize(0);
    pull_asymmetry->GetXaxis()->SetTitle("");
    pull_asymmetry->GetYaxis()->SetTitle("");
    pull_asymmetry->GetYaxis()->SetLabelSize(0.1);
    pull_asymmetry->GetYaxis()->SetRangeUser(-5,5);
    pull_asymmetry->Draw("");

    c_tot->SaveAs(output_path+"_tot"+".pdf");
    c_tot->SaveAs(output_path+"_tot"+".C");
    c_plus->SaveAs(output_path+"_plus"+".pdf");
    c_plus->SaveAs(output_path+"_plus"+".C");
    c_minus->SaveAs(output_path+"_minus"+".pdf");
    c_minus->SaveAs(output_path+"_minus"+".C");
    c_asym->SaveAs(output_path+"_asym"+".pdf");
    c_asym->SaveAs(output_path+"_asym"+".C");
    
    for (const auto arg : *params)
    {
        RooRealVar *var_temp = dynamic_cast<RooRealVar *>(arg);
        if (!var_temp->isConstant())
        {
            //cout << "Checking parameter " << var_temp->GetName() << endl << flush;
            double min = var_temp->getMin(), max = var_temp->getMax();
            double central_value = var_temp->getVal(), error = var_temp->getError();
            double sigma_low = abs(central_value - min)/error, sigma_high = abs(central_value - max)/error;

            if (sigma_low < 3 || sigma_high < 3)
            {
                cerr << "\n\nAttention, paramenter " << var_temp->GetName() << " is close to its limit! Please check!\n" << flush;
                cerr << var_temp->GetName() << " : " << var_temp->getVal() << " +/- " << var_temp->getError() << " in range: (" << min << " - " << max << ")\n" << flush; 
            }
        }
    }

    cout << "\n-------------------------------------------------------\nBest precision possible = " << sqrt(1.0/N_sig.getVal()) << "\nCurrent precision = " << A_sig_blind.getError() << "\n-------------------------------------------------------\n\n" << flush;

    if (save_asymmetry)
    {
        if (!fit_converges)
        {
            cerr << "Not saving asymmetry !!! Fit status not ideal, force writing?\n";
        }
        else
        {
            cout << "Writing results...\n";
            json jsonData;
            TString file_name = "", key_label = "";
            
            json this_json = *jsonInformation;
            TString year = (TString)this_json["year"].get<std::string>();
            TString polarity = (TString)this_json["polarity"].get<std::string>();
            TString add_string = "";
            for(const auto& channel : this_json["channels"])
            {
                TString decay_iteration = (TString)channel["name"].get<std::string>();
                if (decay != decay_iteration) continue;
                add_string = (TString)channel["selection_name"].get<std::string>();
                if (decay == "KK")
                {
                    add_string = "";
                }
            }

            if (save_asymmetry == 1)
            {
                file_name = "asymmetries.json";
                key_label = decay;
            }
            if (save_asymmetry == 2)
            {
                file_name = "../asymmetries_before_weighting_"+year+"_"+polarity+".json";
                key_label = decay + add_string;
            }
            if (save_asymmetry == 3)
            {
                file_name = "asymmetries_random.json";
                key_label = decay;
            }

            std::ifstream in_file(file_name); // Open JSON file
            if (in_file)
            {
                in_file >> jsonData;
                in_file.close();
            }
            jsonData[key_label]["A_blind"] = A_sig_blind.getVal();
            jsonData[key_label]["A_err"] = A_sig_blind.getError();

            std::ofstream out_file(file_name); // Open file for writing
            out_file << jsonData.dump(4); // Pretty-print JSON with indentation
            out_file.close();
        }
    }

    if (make_sweights) 
    { 
        params->selectByName("*")->setAttribAll("Constant",kTRUE);
        params->selectByName("N_*")->setAttribAll("Constant",kFALSE);
        
        TTree::SetMaxTreeSize( 11000000000000LL );
        
        json this_json = *jsonInformation;

        TString year = (TString)this_json["year"].get<std::string>();
        TString polarity = (TString)this_json["polarity"].get<std::string>();
        TString path_to_data = (TString)this_json["path_to_output_files"].get<std::string>();
        TString test_name = (TString)this_json["name_of_the_test"].get<std::string>();
        TString name_tree = (TString)this_json["name_tree"].get<std::string>();
        TString sw_key_name = this_json.contains("sw_key_name") ? "_" + (TString)this_json["sw_key_name"].get<std::string>() : "";
        TString add_string = "";
        for(const auto& channel : this_json["channels"])
        {
            TString decay_iteration = (TString)channel["name"].get<std::string>();
            if (decay != decay_iteration) continue;
            add_string = (TString)channel["selection_name"].get<std::string>();
            if (decay == "KK")
            {
                add_string = "";
            }
        }
        
        TChain* tree = new TChain(name_tree);
        TString FileName = path_to_data + decay + "/" + decay + "_" + year + "_" + polarity + add_string + ".root";
        tree->Add(FileName);
        FileName.ReplaceAll(".root", "_1.root");
        int counter_files = 1;
        while (fs::exists(FileName.Data()))
        {
          tree->Add(FileName);
          ++counter_files;
          FileName.ReplaceAll("_" + (TString)std::to_string(counter_files - 1).c_str() + ".root", "_" + (TString)std::to_string(counter_files).c_str() + ".root");
        }

        tree->SetBranchStatus("*", 0);
        tree->SetBranchStatus(mass_name, 1);
        tree->SetBranchStatus(tag_name, 1);

        double N_entries = tree->GetEntries();
        
        TString outFileName = path_to_data + decay + "/" + decay + "_" + year + "_" + polarity + add_string + ".root";
        outFileName.ReplaceAll(".root","_sw_info.root");
        TString to_be_removed = outFileName;
        TFile * outFile = TFile::Open(outFileName, "RECREATE");
        cout << "Copying!\n" << flush;
        TTree * outTree= tree->CloneTree(-1,"fast");
        outTree->Print();
        RooDataSet* dataset = new RooDataSet ("dataset", "dataset", RooArgSet(m), Import(*outTree));
        RooStats::SPlot *mySPlot = new RooStats::SPlot("mySPlot", "mySPlot", *dataset, &model_tot, RooArgSet(N_sig, N_bkg)); 
        cout << "SWeights done!\n" << flush;
        outFileName = path_to_data + decay + "/" + decay + "_" + year + "_" + polarity + add_string + sw_key_name + ".root";
        outFileName.ReplaceAll(".root","_SPlot.root");
        TFile* new_file = new TFile(outFileName, "RECREATE");
        TTree* tree_sw = new TTree(name_tree, name_tree);
        Double_t N_sigD_sw;
        tree_sw->Branch("N_sigD_sw", &N_sigD_sw, "N_sigD_sw/D");
        
        for (int entry = 0; entry < N_entries; ++entry)
        {
          N_sigD_sw = mySPlot->GetSWeight(entry, "N_sig_sw");
          tree_sw->Fill();
          if (!(entry % 10000000))
            cout << entry << '/' << N_entries << " events processed ...\n" << flush;
        }
        cout << "Out of the loop!\n" << flush;
        tree_sw->Write("", TObject::kOverwrite);
        new_file->Close();
        cout << "Written!\n" << flush;
        outFile->Close();

        system("rm " + to_be_removed);
    }

    if (compare_sw)
    {
        cout << "Starting fit - sw asymmetry comparison ...\n" << flush;
        double fitted_asymmetry = A_sig_blind.getVal(); // Getting the blind asymmetry for comparison
        double fitted_asymmetry_error = A_sig_blind.getError();

        json this_json = *jsonInformation;

        TString year = (TString)this_json["year"].get<std::string>();
        TString polarity = (TString)this_json["polarity"].get<std::string>();
        TString path_to_data = (TString)this_json["path_to_output_files"].get<std::string>();
        TString test_name = (TString)this_json["name_of_the_test"].get<std::string>();
        TString name_tree = (TString)this_json["name_tree"].get<std::string>();
        TString sw_key_name = this_json.contains("sw_key_name") ? "_" + (TString)this_json["sw_key_name"].get<std::string>() : "";
        TString add_string = "";
        for(const auto& channel : this_json["channels"])
        {
            TString decay_iteration = (TString)channel["name"].get<std::string>();
            if (decay != decay_iteration) continue;
            add_string = (TString)channel["selection_name"].get<std::string>();
            if (decay == "KK")
            {
                add_string = "";
            }
        }

        TChain* tree = new TChain(name_tree);
        TString FileName = path_to_data + decay + "/" + decay + "_" + year + "_" + polarity + add_string + ".root";
        tree->Add(FileName);
        FileName.ReplaceAll(".root", "_1.root");
        int counter_files = 1;
        while (fs::exists(FileName.Data()))
        {
          tree->Add(FileName);
          ++counter_files;
          FileName.ReplaceAll("_" + (TString)std::to_string(counter_files - 1).c_str() + ".root", "_" + (TString)std::to_string(counter_files).c_str() + ".root");
        }

        FileName = path_to_data + decay + "/" + decay + "_" + year + "_" + polarity + add_string + sw_key_name + ".root";
        FileName.ReplaceAll(".root","_SPlot.root");
        tree->AddFriend(name_tree, FileName);

        TString path_to_weights = path_to_data + test_name + "/" + decay + "_" + year + "_" + polarity + add_string + "_rew_" + test_name + ".root";
        if (compare_sw == 2) tree->AddFriend(name_tree, path_to_weights);

        tree->SetBranchStatus("*",0);
        tree->SetBranchStatus("N_sigD_sw", 1);
        tree->SetBranchStatus(tag_name, 1);
        if (compare_sw == 2) tree->SetBranchStatus("Iw", 1);

        TTreeFormula* tag_form = new TTreeFormula("tag_form", tag_name, tree);
        TTreeFormula* sw_form = new TTreeFormula("sw_form", "N_sigD_sw", tree);
        TTreeFormula* Iw_form;
        if (compare_sw == 2) Iw_form = new TTreeFormula("Iw_form", "Iw", tree);

        TH1D* N_plus = new TH1D("N_plus", "", 1, 0, 1);
        N_plus->Sumw2();
        TH1D* N_minus = new TH1D("N_minus", "", 1, 0, 1);
        N_minus->Sumw2();

        double N_entries = tree->GetEntries();

        for (int entry = 0; entry < N_entries; ++entry)
        {
            tree->LoadTree(entry);
            tree->GetEntry(entry);

            sw_form->GetNdata();
            sw_form->UpdateFormulaLeaves();
            tag_form->GetNdata();
            tag_form->UpdateFormulaLeaves();

            double sw = sw_form->EvalInstance(), Iw = 1;
            if (compare_sw == 2)
            {
                Iw_form->GetNdata();
                Iw_form->UpdateFormulaLeaves();
                Iw = Iw_form->EvalInstance();
            }
            if (tag_form->EvalInstance() > 0)
            {
                N_plus->Fill(0., sw * Iw);
            }
            else
            {
                N_minus->Fill(0., sw * Iw);
            }
            if (!(entry % 10000000))
                cout << entry << '/' << N_entries << " events processed ...\n" << flush;
        }

        TH1D* asymmetry = (TH1D*)N_plus->GetAsymmetry(N_minus);
        double A_sw = asymmetry->GetBinContent(1) - A_bias.getVal();
        double A_sw_err = asymmetry->GetBinError(1);
        double difference = fitted_asymmetry - A_sw;
        double max_corr_err = sqrt(abs(pow(fitted_asymmetry_error, 2) - pow(A_sw_err, 2)));
        double chi2 = abs(difference / max_corr_err);

        cout << "The difference between the fitted asymmetry and the one computed with s-weights is " << difference * 100 << " %\nCorresponding to a compatibility chi2 of " << chi2 << " (1dof) -> " << TMath::Prob(chi2, 1) * 100 << " %\n" << flush;
        cout << "Fitted asymmetry (blind) (" << fitted_asymmetry * 100 << " +/- " << fitted_asymmetry_error * 100 << ") %\ns-weights asymmetry (blind) (" << A_sw * 100 << " +/- " << A_sw_err * 100 << ") %\n" << flush;
        
    }

    return fit_converges;
}

