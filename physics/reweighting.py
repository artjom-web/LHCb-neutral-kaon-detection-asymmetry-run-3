"""Iterative kinematic reweighting (stage 2).

``Reweighter`` performs the sideband-subtraction initial weight and then N
reweighting cycles, each of which builds a 4D (3 kinematic vars x lifetime)
template, normalises it per lifetime bin, and defines a new ``weight_<i>``
column on the RDataFrame.

Based on ``Analysis.init_massfit`` / ``init_weights`` / ``equal_bins`` /
``weighting_cycle`` / ``normalize_th4d`` / ``apply_weight_acc`` /
``ltbin_means``.  The numerics are untouched.  What changed:

  * mass-window and lifetime-binning constants come from a ``TrackConfig``
    rather than being set in a constructor branch on the track name;
  * mass fitting is delegated to a ``MassFitter`` instead of being a method
    on the same object;
  * the sideband ratio, previously only printed, is stored so stage 2 can
    write it into its metadata;
  * the weight-ratio clamp is a named configuration constant;
  * ``weight_columns()`` reports exactly which columns must be snapshotted,
    so stage 2 cannot forget one and silently break stage 4.

Note on lifetime binning: ``equal_bins`` derives quantile-equalised lifetime
edges from the data. It is not used by the nominal flow (which takes fixed
edges from config) but is kept because it is how those fixed edges were
originally obtained.
"""

from __future__ import annotations

import array
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import cppyy
import ROOT

from ..config.selection import TrackConfig, build_hist_params
from ..config.weighting import Cycle, WEIGHT_RATIO_MAX, WEIGHT_RATIO_MIN
from ..core.util import asymmetric_xerr, weight_key
from .massfit import MassFitter

_CPP_DECLARED = "_kspi_update_weight_3d_declared"


def declare_cpp_helpers() -> None:
    """JIT-declare the per-event weight update used inside RDataFrame.

    Declared once per process; ROOT has no way to redefine it, so the guard
    matters if several Reweighters are built in one job.
    """
    if getattr(ROOT, _CPP_DECLARED, False):
        return
    ROOT.gInterpreter.Declare(
        r"""
        double update_weight_3d(double x, double y, double z, double t, double w,
                                TH3D* h3, THnD* h4)
        {
            double num = h3->GetBinContent(h3->FindBin(x, y, z));
            Int_t bins[4];
            bins[0] = h4->GetAxis(0)->FindBin(x);
            bins[1] = h4->GetAxis(1)->FindBin(y);
            bins[2] = h4->GetAxis(2)->FindBin(z);
            bins[3] = h4->GetAxis(3)->FindBin(t);
            double den = h4->GetBinContent(bins);

            if (den <= 0.0) return 0.0;
            num = std::max(num, 0.0);
            double ratio = num / den;

            const double lo = RATIO_MIN , hi = RATIO_MAX ;
            if (ratio < lo) ratio = lo;
            if (ratio > hi) ratio = hi;

            return w * ratio;
        }
        """.replace("RATIO_MIN", repr(float(WEIGHT_RATIO_MIN)))
        .replace("RATIO_MAX", repr(float(WEIGHT_RATIO_MAX)))
    )
    setattr(ROOT, _CPP_DECLARED, True)
    ROOT.TH1.SetDefaultSumw2(True)


class Reweighter:
    _instance_counter = 0

    def __init__(self, rdf, cfg: TrackConfig, fitter: MassFitter,
                 hist_params: Optional[Dict[str, list]] = None):
        Reweighter._instance_counter += 1
        self._instance_id = Reweighter._instance_counter

        declare_cpp_helpers()

        self.rdf = rdf
        self.cfg = cfg
        self.fitter = fitter
        self.hist_params = hist_params or build_hist_params(cfg)
        self.lt_bin_edges = list(cfg.lt_bin_edges)

        self.iteration = 0
        self.N_total_events = rdf.Count().GetValue()
        self.sideband_ratio = None
        self.sideband_ratio_flat = None

        self.empty_bins: Dict[str, set] = {}
        self.N_sigw0: Dict[str, float] = {}
        self.hists: Dict[str, object] = {}
        self.hists_filled: Dict[str, object] = {}
        self.profiles: Dict[str, object] = {}
        self.profiles_filled: Dict[str, object] = {}
        self.bin_x: Dict[str, list] = {}
        self.bin_xerr: Dict[str, object] = {}
        self._pending_acc_sum = None
        self._pending_acc_iter = None
        self.accepted_fraction: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # public driver
    # ------------------------------------------------------------------
    def run(self, cycles: Sequence[Cycle], massfit_folder) -> None:
        """Initial weights, then every cycle, then the final weight column."""
        self.init_weights(massfit_folder)
        for cycle in cycles:
            self.weighting_cycle(*cycle.params)
        self.finalize()

    def finalize(self) -> None:
        """Define ``final_weight`` and drain the last pending acceptance sum."""
        self.rdf = self.rdf.Define(
            "final_weight", f"weight_{self.iteration} / weight_0"
        )
        self._drain_pending_acc()

    def _drain_pending_acc(self) -> None:
        if self._pending_acc_sum is not None:
            N = self._pending_acc_sum.GetValue()
            frac = N / self.N_total_events if self.N_total_events else float("nan")
            self.accepted_fraction[int(self._pending_acc_iter)] = frac
            print(
                f"[iter {self._pending_acc_iter}] accepted events = {N}, "
                f"fraction = {frac}"
            )
            self._pending_acc_sum = None

    # ------------------------------------------------------------------
    # what stage 2 must persist
    # ------------------------------------------------------------------
    def weight_columns(self) -> List[str]:
        """Every weight-related column that downstream stages need.

        Stage 4 needs weight_i / w2_i / weight_eff_i for *all* iterations, so
        they must all survive the snapshot; stage 3 needs only weight_0 and
        final_weight. Listing them here rather than in the script means adding
        an iteration cannot silently drop a column.
        """
        cols: List[str] = []
        for i in range(self.iteration + 1):
            cols += [f"weight_{i}", f"w2_{i}", f"weight_eff_{i}", f"weight_acc_{i}"]
        cols.append("final_weight")
        return cols


    def init_massfit(self, folder):
        keys = ["Dp_M_p", 'Dp_M_m', "Dp_M"]
        for key in keys:
            if key.endswith("_p"):
                column = key[:-2]
                rdf = self.rdf.Filter("Dp_charge > 0")
            elif key.endswith("_m"):
                column = key[:-2]
                rdf = self.rdf.Filter("Dp_charge < 0")
            else:
                column = key
                rdf = self.rdf

            self.hists[key] = rdf.Histo1D(
                (f"h_{key}", key, *self.hist_params[column]),
                column,
            )

        # fill the hists
        for key in keys:
            self.hists_filled[key] = self.hists[key].GetValue()

        h_p = self.hists_filled['Dp_M_p']
        h_m = self.hists_filled['Dp_M_m']

        # massfit
        self.fitter.fit(h_p, h_m, folder)


    def init_weights(self, massfit_folder):
        self.iteration = 0
        
        massfit_folder = str(massfit_folder).rstrip('/') + '/'
        os.makedirs(massfit_folder, exist_ok=True)
        self.init_massfit(massfit_folder)

        # finding the range
        f = ROOT.TFile.Open(massfit_folder + "model.root")
        ws = f.Get("ws")
        params = ws.allVars()
        m = ws.var("Dp_M")
        bkg_tot = ws.pdf("bkg_tot")
        m.setRange("leftSB",  self.cfg.min_m, self.cfg.l_edge_sig)
        m.setRange("rightSB", self.cfg.r_edge_sig, self.cfg.max_m)
        m.setRange("sig_region", self.cfg.l_edge_sig, self.cfg.r_edge_sig)

        # calculating the Nbkg_sb and Nbkg_sig
        N_bkg_sb = bkg_tot.createIntegral(ROOT.RooArgSet(m), ROOT.RooFit.Range("leftSB")).getVal() + bkg_tot.createIntegral(ROOT.RooArgSet(m), ROOT.RooFit.Range("rightSB")).getVal()
        N_bkg_sig = bkg_tot.createIntegral(ROOT.RooArgSet(m), ROOT.RooFit.Range("sig_region")).getVal()

        self.sideband_ratio = N_bkg_sig / N_bkg_sb
        self.sideband_ratio_flat = (
            (self.cfg.r_edge_sig - self.cfg.l_edge_sig)
            / ((self.cfg.l_edge_sig - self.cfg.min_m)
               + (self.cfg.max_m - self.cfg.r_edge_sig))
        )
        print(f'N_bkg_sig / N_bkg_sb from fit = {self.sideband_ratio}')
        print(f'N_bkg_sig / N_bkg_sb assuming constant bkg = {(self.cfg.r_edge_sig - self.cfg.l_edge_sig) / ((self.cfg.l_edge_sig - self.cfg.min_m) + (self.cfg.max_m - self.cfg.r_edge_sig))}')
      

        self.rdf = (self.rdf.Filter(f"Dp_M > {self.cfg.min_m}", f"Dp_M > {self.cfg.min_m}")
            .Filter(f"Dp_M < {self.cfg.max_m}", f"Dp_M < {self.cfg.max_m}"))

        self.rdf = (self.rdf
                .Define(
                    f"weight_hist_{self.iteration}",
                    f"""
                    ((Dp_M >= {self.cfg.l_edge_sig} && Dp_M <= {self.cfg.r_edge_sig}) ? 1.0 :
                    ((Dp_M >= {self.cfg.l_edge_sb} && Dp_M < {self.cfg.l_edge_sig}) ||
                    (Dp_M > {self.cfg.r_edge_sig} && Dp_M <= {self.cfg.r_edge_sb}))
                        ? {-N_bkg_sig / N_bkg_sb}
                        : 0.0)
                    """)
                .Define(f"weight_acc_{self.iteration}", "1")
                .Define(
                    f"weight_{self.iteration}",
                    f"weight_hist_{self.iteration} * weight_acc_{self.iteration}"
                )
                .Define(f"w2_{self.iteration}", f"weight_{self.iteration} * weight_{self.iteration}")
                .Define(f"weight_eff_{self.iteration}", f"weight_0 * weight_acc_{self.iteration}")
        )
        f.Close()
        self.fitter.plot(massfit_folder)
        return 

    def equal_bins(self, hist=None, bins=None):
        if hist is None:
            hist = self.rdf.Histo1D(
                ("tmp", "tmp", 1000,
                self.hist_params["KS_LT"][1],
                self.hist_params["KS_LT"][2]),
                "KS_LT",
                "weight_0"
            ).GetValue()

        if bins is None:
            bins = self.cfg.kslt_anabins
        probs = ROOT.std.vector("double")()
        for i in range(bins + 1):
            probs.push_back(i / bins)

        quant = ROOT.std.vector("double")(bins + 1)

        hist.GetQuantiles(
            bins + 1,
            quant.data(),
            probs.data()
        )

        bin_edges = [quant[i] for i in range(bins + 1)]
        return bin_edges

    def weighting_cycle(
        self,
        param1,
        param2,
        param3
    ):
        self.iteration += 1
        

        print(
            f"Starting iteration {self.iteration}: "
            f"{param1}, {param2}, {param3}"
        )

        def make_edges(params):
            nbins, lo, hi = params
            return array.array(
                "d",
                np.linspace(lo,hi,nbins+1)
            )

        name = ( f"h4d_{param1}_{param2}_{param3}_{self.iteration}")
        name2 = name + '_unw'


        xedges = make_edges(self.hist_params[param1])
        yedges = make_edges(self.hist_params[param2])
        zedges = make_edges(self.hist_params[param3])
        tedges = array.array("d", self.lt_bin_edges)

        model = ROOT.RDF.THnDModel(
            name,
            name,
            4,
            [
                len(xedges)-1,
                len(yedges)-1,
                len(zedges)-1,
                len(tedges)-1
            ],
            [
                xedges,
                yedges,
                zedges,
                tedges
            ]
        )
        model2 = ROOT.RDF.THnDModel(
            name2,
            name2,
            4,
            [
                len(xedges)-1,
                len(yedges)-1,
                len(zedges)-1,
                len(tedges)-1
            ],
            [
                xedges,
                yedges,
                zedges,
                tedges
            ]
        )

        result = self.rdf.HistoND(
            model,
            [
                param1,
                param2,
                param3,
                "KS_LT"
            ],
            f"weight_{self.iteration - 1}"
        )

        self.rdf = self.rdf.Define(f"test_weight_{self.iteration}", f"weight_acc_{self.iteration-1} * weight_0")        
        result2 = self.rdf.HistoND(
            model2,
            [
                param1,
                param2,
                param3,
                "KS_LT"
            ],
            f"test_weight_{self.iteration}" 

        )

        self.hists[name] = result
        self.hists[name2] = result2
        h4d = self.hists[name].GetValue()
        h4d_unw = self.hists[name2].GetValue()
        if self._pending_acc_sum is not None:
            N = self._pending_acc_sum.GetValue()
            print(f'[iter {self._pending_acc_iter}] accepted events = {N}, fraction = {N/self.N_total_events}')
            self._pending_acc_sum = None

        h4d = self.normalize_th4d(h4d, h4d_unw, param1, param2, param3)

        del h4d_unw                 # local, would fall out of scope anyway — explicit for clarity
        self._purge([name2])        # the *_unw RResultPtr — safe, nothing else points at it

        h3d = h4d.Projection(0, 1, 2)

        h3d.Scale(1/self.cfg.kslt_anabins)

        self.hists_filled[f"{name}"] = h4d
        self.hists_filled[f"{name}_h3d"] = h3d

        address3d = cppyy.addressof(h3d)
        address4d = cppyy.addressof(h4d)
        self.rdf = self.rdf.Define(
            f"weight_hist_{self.iteration}",

            f"""
            update_weight_3d(
                {param1},
                {param2},
                {param3},
                KS_LT,
                weight_{self.iteration-1},
                (TH3D*){address3d},
                (THnD*){address4d}
            )
            """
        )
        self.rdf = (self.rdf.Define(
                f"weight_{self.iteration}",
                f"weight_hist_{self.iteration} * weight_acc_{self.iteration}"
            )
            .Define(f"w2_{self.iteration}", f"weight_{self.iteration} * weight_{self.iteration}")
            .Define(f"weight_eff_{self.iteration}", f"weight_0 * weight_acc_{self.iteration}"))
        print(f"Column weight_{self.iteration} is created." )
        

    def normalize_th4d(self, h4d, h4d_unw, param1, param2, param3):
        h = h4d
        h2 = h4d_unw
        new_empty_bins = []
        key = f"{param1}_{param2}_{param3}"
        self.empty_bins.setdefault(key, set())

        axis0 = h.GetAxis(0)
        axis1 = h.GetAxis(1)
        axis2 = h.GetAxis(2)
        axis3 = h.GetAxis(3)

        n0 = axis0.GetNbins()
        n1 = axis1.GetNbins()
        n2 = axis2.GetNbins()
        n3 = axis3.GetNbins()
        n_bins = n0 * n1 * n2
        coord = array.array('i', [0, 0, 0, 0])
        axis3_tots = np.zeros(n3)
        axis3_totsw0 = np.zeros(n3)
        axis3_thresholds = np.zeros(n3)

        GetBinContent = h.GetBinContent
        SetBinContent = h.SetBinContent
        GetBinError = h.GetBinError
        SetBinError = h.SetBinError

        GetBinContent2 = h2.GetBinContent

        # # determine thresholds
        # for i3 in range(1, n3 + 1):
        #     axis3.SetRange(i3, i3)
        #     h3 = h.Projection(0, 1, 2)
        #     total = h3.Integral()
        #     axis3_thresholds[i3 - 1] = (total / n_bins) * self.cfg.threshold
        #     h3.Delete()

        # empty bins and find integral
        for i0 in range(1, n0 + 1):
            coord[0] = i0
            for i1 in range(1, n1 + 1):
                coord[1] = i1
                for i2 in range(1, n2 + 1):
                    coord[2] = i2
                    is_empty = False
                    for i3 in range(1, n3 + 1):
                        coord[3] = i3
                        content = GetBinContent2(coord)
                        if content <= self.cfg.threshold:
                            is_empty = True
                            break
                    if is_empty:
                        coord3d = (i0, i1, i2)
                        if coord3d not in self.empty_bins[f'{param1}_{param2}_{param3}']:
                            new_empty_bins.append(coord3d)
                            for i3 in range(1, n3 + 1):
                                coord[3] = i3
                                SetBinContent(coord, 0.)
                                SetBinError(coord, 0.)
                        continue
                    for i3 in range(1, n3 + 1):
                        coord[3] = i3
                        axis3_tots[i3 - 1] += GetBinContent(coord)
                        axis3_totsw0[i3-1] += GetBinContent2(coord)

        for i3 in range(1, n3 + 1):
            self.N_sigw0[f'{self.iteration}_{i3}'] = axis3_totsw0[i3-1]
    
        # normalize. note that the histogram is not perfectly normalized: 
        # To do this a new threshold should be calculated, 
        # but then also a new bin rejection should be applied and you end up in an infinite loop
        for i3 in range(1, n3 + 1):
            if axis3_tots[i3 - 1] > 1e-13:
                inv_total = 1.0 / axis3_tots[i3 - 1]
            else:
                print(f'WARNING: Integral too small')
                inv_total = 0
            print(f'Integral h4d ({param1}_{param2}_{param3}) ltbin {i3} = {axis3_tots[i3 - 1]}')
            for i0 in range(1, n0 + 1):
                coord[0] = i0
                for i1 in range(1, n1 + 1):
                    coord[1] = i1
                    for i2 in range(1, n2 + 1):
                        coord[2] = i2
                        coord[3] = i3
                        content = GetBinContent(coord)
                        error = GetBinError(coord)
                        SetBinContent(coord, content * inv_total)
                        SetBinError(coord, error * inv_total)

        axis3.SetRange()  # reset range back to full, otherwise it stays restricted
    
        self.empty_bins[f'{param1}_{param2}_{param3}'].update(new_empty_bins)
        self.apply_weight_acc(axis0, axis1, axis2, new_empty_bins, param1, param2, param3)

        print(f'emptied events in {len(new_empty_bins)} / {n0*n1*n2} bins')
        self._pending_acc_sum = self.rdf.Sum(f"weight_acc_{self.iteration}")   # booked, not triggered
        self._pending_acc_iter = self.iteration

        return h

    def apply_weight_acc(self, axis0, axis1, axis2, empty_bins, param1, param2, param3):

        n1 = axis1.GetNbins()
        n2 = axis2.GetNbins()

        ns = f"weightacc_{self._instance_id}_{self.iteration}"

        for name, ax in (("axis0", axis0), ("axis1", axis1), ("axis2", axis2)):
            try:
                _ = ax.GetNbins()
            except Exception as e:
                raise RuntimeError(
                    f"instance {self._instance_id}, iteration {self.iteration}: "
                    f"{name} appears invalid before assignment into {ns} — {e}"
                )
        ROOT.gInterpreter.Declare(f"""
        #include <unordered_set>
        namespace {ns} {{
            std::unordered_set<long long> empty_flat;
            TAxis* axis0 = nullptr;
            TAxis* axis1 = nullptr;
            TAxis* axis2 = nullptr;
            int n1 = 0, n2 = 0;

            inline double factor(double x, double y, double z) {{
                int i0 = axis0->FindBin(x);
                int i1 = axis1->FindBin(y);
                int i2 = axis2->FindBin(z);
                long long flat = ((long long)i0 * (n1 + 2) + i1) * (n2 + 2) + i2;
                return empty_flat.count(flat) ? 0.0 : 1.0;
            }}
        }}
        """)

        ns_obj = getattr(ROOT, ns)
        ns_obj.axis0 = axis0
        ns_obj.axis1 = axis1
        ns_obj.axis2 = axis2
        ns_obj.n1 = n1
        ns_obj.n2 = n2

        for i0, i1, i2 in empty_bins:
            flat = (i0 * (n1 + 2) + i1) * (n2 + 2) + i2
            ns_obj.empty_flat.insert(flat)

        # first call: initialize the column
        self.rdf = self.rdf.Define(
            f"weight_acc_{self.iteration}",
            f"weight_acc_{self.iteration - 1} * {ns}::factor({param1}, {param2}, {param3})"
        )


    def ltbin_means(self, weights=None):
        param = 'KS_LT'
        tedges = array.array('d', [float(e) for e in self.lt_bin_edges])

        if weights is None:
            weights = ['', 'weight_0', f'weight_{self.iteration}', 'final_weight']

        for w in weights:
            profile_model = ROOT.RDF.TProfile1DModel(f'ltmean_{weight_key(w)}', f'ltmean_{weight_key(w)}', len(tedges) - 1, tedges)
            if w == '':
                self.profiles[weight_key(w)] = self.rdf.Profile1D(profile_model, param, param)
            else:
                self.profiles[weight_key(w)] = self.rdf.Profile1D(
                    profile_model,
                    param,
                    param,
                    w
                )

        for w in weights:
            self.profiles_filled[weight_key(w)] = self.profiles[weight_key(w)].GetValue()

        for w in weights:
            prof = self.profiles_filled[weight_key(w)]
            centers = [prof.GetBinContent(i) for i in range(1, prof.GetNbinsX() + 1)]
            self.bin_x[f"{param}_{weight_key(w)}"] = centers
            self.bin_xerr[f"{param}_{weight_key(w)}"] = asymmetric_xerr(tedges, centers)
        
    def _purge(self, keys):
        for key in keys:
            self.hists.pop(key, None)
            self.hists_filled.pop(key, None)