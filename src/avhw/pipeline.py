"""End-to-end experiment steps. Each step writes its tables to results/tables."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from . import climate, crop, experiment as ex, forecast as fc, io, prices, simulate as sim, solar, water
from .canopy import Microclimate, cv_metrics
from .geometry import ArrayGeometry


class Runner:
    def __init__(self, cfg: dict, quick: bool = False):
        self.cfg = cfg
        self.quick = quick
        self.out = Path(cfg["_root"]) / cfg["results_dir"]
        (self.out / "tables").mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(parents=True, exist_ok=True)
        self.log_lines = []
        self.res = {}
        s = cfg["system"]
        self.tilts = np.arange(s["tilt_min"], s["tilt_max"] + 1e-9, s["tilt_step"])
        self.geo = ArrayGeometry(s["height_m"], s["pitch_m"], s["gcr"], self.tilts, s["n_ground_points"])
        self.main = cfg["main_site"]
        self.era5 = {k: io.load_era5(cfg, k) for k in cfg["sites"]}
        self.ecmwf = io.load_ecmwf(cfg, self.main)
        self.price = io.load_omie(cfg)
        if quick:
            cfg["forecast"]["n_synthetic"] = 3
            cfg["sensitivity"]["surrogate_runs"] = 12
            cfg["sensitivity"]["surrogate_holdout"] = 4

    # ------------------------------------------------------------------ utilities
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_lines.append(line)

    def save(self, df: pd.DataFrame, name: str, latex=True, index=False, floatfmt="%.3f"):
        df.to_csv(self.out / "tables" / f"{name}.csv", index=index)
        if latex:
            try:
                tex = df.to_latex(index=index, float_format=lambda x: floatfmt % x, na_rep="--")
                (self.out / "tables" / f"{name}.tex").write_text(tex)
            except Exception:
                pass

    def years(self):
        y0, y1 = self.cfg["years"]
        ys = list(range(y0, y1 + 1))
        return ys[-4:] if self.quick else ys

    # ================================================================== 1. prices
    def step_prices(self):
        scfg = self.cfg["sites"][self.main]
        tab, prof = prices.hot_day_price_analysis(self.era5[self.main], self.price, scfg["tz"],
                                                  self.cfg["crops"][self.cfg["main_crop"]]["stress_day_tmax_C"] + 4)
        self.save(tab, "T01_prices_hot_vs_normal")
        prof.to_csv(self.out / "tables" / "T01b_price_profile_hot_vs_normal.csv")
        self.res["prices"] = (tab, prof)
        self.log(f"prices: {len(tab)} year x day-type rows")

    # ================================================================== 2. forecasting
    def step_forecast(self):
        scfg = self.cfg["sites"][self.main]
        ccfg = self.cfg["crops"][self.cfg["main_crop"]]
        scores, preds = fc.evaluate_forecasters(self.era5[self.main], self.ecmwf, scfg["tz"], self.cfg,
                                                ccfg["stress_day_tmax_C"])
        self.save(scores, "T02_forecast_skill")
        preds.to_csv(self.out / "tables" / "T02b_forecast_predictions.csv")
        self.res["forecast"] = (scores, preds)
        # models used by the controllers, fitted on the whole archive
        obs = fc.daily_obs(self.era5[self.main], scfg["tz"])
        d = fc.add_features(fc.daily_fc(self.ecmwf, 1, scfg["tz"]), obs, 1).join(obs[["Tmax"]]).dropna()
        self.emos_all = fc.EMOS().fit(d, d["Tmax"].values)
        self.hcorr_all = fc.HourlyCorrector().fit(self.era5[self.main], self.ecmwf, 1)
        self.errlib = fc.ErrorLibrary(self.era5[self.main], self.ecmwf, 1)
        hh = pd.DataFrame({"hour_utc": range(24), "bias_C": self.hcorr_all.bias, "sd_C": self.hcorr_all.sd})
        self.save(hh, "T02c_hourly_bias_spread_lead1")
        self.log("forecast: " + scores[scores.lead_days == 1][["model", "CRPS_C", "Brier", "BSS_vs_clim"]]
                 .round(3).to_string(index=False).replace("\n", " | "))

    # ================================================================== 3. microclimate
    def _hourly_period(self, site, start, end):
        tz = self.cfg["sites"][site]["tz"]
        t0 = pd.Timestamp(start).tz_localize(tz).tz_convert("UTC")
        t1 = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
        return self.era5[site].loc[t0:t1]

    def _fsh(self, wx, site, tilt):
        sc = self.cfg["sites"][site]
        sp = solar.sun_position_hourly(wx.index, sc["lat"], sc["lon"])
        _, n, u = solar.sun_vector(sp)
        k = int(np.argmin(np.abs(self.tilts - tilt)))
        return self.geo.ground_shade_fraction(n, u)[:, k]

    def step_microclimate(self, tilt=None):
        fcfg = self.cfg["field"]
        tilt = fcfg["tilt_lettuce_deg"] if tilt is None else tilt
        obs = io.load_field_daily(self.cfg)
        tz = self.cfg["sites"][self.main]["tz"]
        # label every digitised day with the configured cycle it belongs to (+-7 days tolerance)
        lab = pd.Series(index=obs.index, dtype=object)
        for cyc, (a, b) in fcfg["cycles"].items():
            k = (obs.date >= pd.Timestamp(a) - pd.Timedelta(days=7)) & (obs.date <= pd.Timestamp(b) + pd.Timedelta(days=7))
            lab[k] = cyc
        obs = obs.assign(cycle=lab).dropna(subset=["cycle"])
        feats = []
        for cyc in fcfg["cycles"]:
            o = obs[obs.cycle == cyc]
            wx = self._hourly_period(self.main, o.date.min(), o.date.max())
            f = Microclimate.daily_features(wx, self._fsh(wx, self.main, tilt), tz)
            f["cycle"] = cyc
            feats.append(f)
        feats = pd.concat(feats)
        rows, preds = [], []
        cycles = list(fcfg["cycles"])
        for test in cycles + ["both"]:
            if test == "both":
                tr_f, te_f = feats, feats
            else:
                tr_f, te_f = feats[feats.cycle != test], feats[feats.cycle == test]
            tr_o = obs[obs.cycle.isin(tr_f.cycle.unique())]
            m = Microclimate().fit(tr_f, tr_o)
            p = m.predict_daily(te_f).join(obs.set_index("date")[["dT_day", "dT_night"]], how="inner")
            for part in ("day", "night"):
                r = cv_metrics(p[f"dT_{part}"], p[f"dT_{part}_hat"])
                r.update(test_cycle=test, period=part, train="all" if test == "both" else f"not {test}")
                rows.append(r)
            if test != "both":
                p["cycle"] = test
                preds.append(p)
            else:
                self.micro = m
        cvt = pd.DataFrame(rows)[["test_cycle", "train", "period", "n", "obs_mean", "pred_mean", "rmse", "bias", "r"]]
        # null model: predict the training-cycle mean (what a constant offset would give)
        null = []
        for test in cycles:
            tr = obs[obs.cycle != test]
            te = obs[obs.cycle == test]
            for part in ("day", "night"):
                r = cv_metrics(te[f"dT_{part}"], np.full(len(te), tr[f"dT_{part}"].mean()))
                null.append({"test_cycle": test, "period": part, "rmse_constant_model": r["rmse"]})
        cvt = cvt.merge(pd.DataFrame(null), on=["test_cycle", "period"], how="left")
        self.save(cvt, "T03_microclimate_cv")
        coefs = pd.DataFrame({"term": Microclimate.day_terms + [f"night_{t}" for t in Microclimate.night_terms],
                              "coef": np.r_[self.micro.beta_day, self.micro.beta_night]})
        self.save(coefs, "T03b_microclimate_coefficients")
        self.res["micro"] = (cvt, pd.concat(preds), coefs)
        self.log(f"microclimate (tilt {tilt}): beta_day={np.round(self.micro.beta_day, 3)}")
        return cvt

    # ================================================================== 4. crop + PV validation
    def _field_physics(self, cyc, tilt, crop_cfg):
        a, b = self.cfg["field"]["cycles"][cyc]
        wx = self._hourly_period(self.main, a, b)
        sc = self.cfg["sites"][self.main]
        ph = sim.compute_physics(wx, sc["lat"], sc["lon"], self.geo, self.micro, self.cfg, crop_cfg)
        k = int(np.argmin(np.abs(self.tilts - tilt)))
        return wx, ph, k

    def _field_ratios(self, tilt, ccfg):
        rows = []
        tz = self.cfg["sites"][self.main]["tz"]
        fcfg = self.cfg["field"]
        for cyc in fcfg["cycles"]:
            wx, ph, k = self._field_physics(cyc, tilt, ccfg)
            L_apv, L_open = ph.sw[:, k].sum(), ph.sw_open.sum()
            H_apv, H_open = ph.stress()[:, k].sum(), ph.stress_open().sum()
            y = fcfg["yield_t_ha"][cyc]
            dT = self.micro.predict_hourly(ph.fsh, ph.sw_open, wx["T"].values, wx["RH"].values, wx["u10"].values)[:, k]
            dates = fc.local_date(wx.index, tz)
            dli_c = pd.Series(ph.sw_open * crop.PAR_FACTOR, index=wx.index).groupby(dates).sum()
            dli_a = pd.Series(ph.sw[:, k] * crop.PAR_FACTOR, index=wx.index).groupby(dates).sum()
            rows.append({"cycle": cyc, "tilt_assumed": tilt, "light_ratio": L_apv / L_open, "H_open": H_open, "H_apv": H_apv,
                         "dH": H_apv - H_open, "yield_ctrl": y["control"], "yield_apv": y["apv"],
                         "yield_ratio_obs": y["apv"] / y["control"],
                         "DLI_ctrl": dli_c.mean(), "DLI_apv": dli_a.mean(), "DLI_reduction_pct": 100 * (1 - dli_a.mean() / dli_c.mean()),
                         "_tm_c": wx["T"].groupby(dates).mean().values, "_tm_a": (wx["T"] + dT).groupby(dates).mean().values,
                         "_dli_c": dli_c.mean(), "_dli_a": dli_a.mean()})
        return pd.DataFrame(rows)

    @staticmethod
    def _joint_alpha_lambda(t):
        """Solve ln(Y_apv/Y_ctrl) = alpha ln(L_apv/L_open) - lambda dH for the two cycles."""
        A = np.column_stack([np.log(t.light_ratio), -t.dH])
        b = np.log(t.yield_ratio_obs)
        al, lm = np.linalg.solve(A, b) if len(t) == 2 else np.linalg.lstsq(A, b, rcond=None)[0]
        if lm < 0:   # heat relief cannot lower yield: constrain lambda >= 0
            lm = 0.0
            al = float(np.sum(np.log(t.light_ratio) * b) / np.sum(np.log(t.light_ratio) ** 2))
        return float(al), float(lm)

    def step_crop_validation(self):
        ccfg = self.cfg["crops"]["lettuce"]
        fcfg = self.cfg["field"]
        tilt = fcfg["tilt_lettuce_deg"]
        t = self._field_ratios(tilt, ccfg)
        # (a) joint identification of light exponent and heat damage coefficient from the two cycles
        self.alpha, self.lam_field = self._joint_alpha_lambda(t)
        t["alpha_joint"], t["lambda_joint"] = self.alpha, self.lam_field
        # (b) out-of-sample check of the light response: alpha from one cycle predicts the other (lambda = 0
        #     and lambda = literature value)
        lam_lit = ccfg["lambda"]["high"]
        for lam, tag in ((0.0, "lam0"), (lam_lit, "lamlit")):
            al = [crop.calibrate_alpha(r.yield_ratio_obs, r.light_ratio, r.dH, lam) for r in t.itertuples()]
            t[f"alpha_cycle_{tag}"] = al
            for i in range(len(t)):
                a_o = np.mean([al[j] for j in range(len(t)) if j != i])
                pr = t.loc[i, "light_ratio"] ** a_o * np.exp(-lam * t.loc[i, "dH"])
                t.loc[i, f"ratio_pred_loco_{tag}"] = pr
                t.loc[i, f"rel_error_pct_loco_{tag}"] = 100 * (pr - t.loc[i, "yield_ratio_obs"]) / t.loc[i, "yield_ratio_obs"]
        # (c) identifiability: how alpha and lambda depend on the (unreported) experimental tilt
        grid = []
        for tl in (10, 20, 30, 40, 50, 60):
            tt = self._field_ratios(tl, ccfg)
            al, lm = self._joint_alpha_lambda(tt)
            grid.append({"tilt_assumed": tl, "light_ratio_fall": tt.light_ratio.iloc[0], "light_ratio_spring": tt.light_ratio.iloc[1],
                         "dH_spring": tt.dH.iloc[1], "alpha_joint": al, "lambda_joint": lm})
        self.save(pd.DataFrame(grid), "T04c_crop_identification_vs_tilt", floatfmt="%.5f")
        # independent check with the published GDD-DLI regression (Table S1): base temperature fitted on control
        dens = fcfg["plant_density_m2"]

        def fw_yield(tb, tm, dli):
            return crop.table_s1_fresh_weight(crop.gdd(tm, tb), dli) * dens * 0.01

        def obj(tb):
            return sum((fw_yield(tb, r["_tm_c"], r["_dli_c"]) - r["yield_ctrl"]) ** 2 for _, r in t.iterrows())

        tb = optimize.minimize_scalar(obj, bounds=(0, 20), method="bounded").x
        s1 = []
        for _, r in t.iterrows():
            for trt, tm, dli, yobs in (("control", r["_tm_c"], r["_dli_c"], r["yield_ctrl"]),
                                       ("apv", r["_tm_a"], r["_dli_a"], r["yield_apv"])):
                yp = fw_yield(tb, tm, dli)
                s1.append({"cycle": r["cycle"], "treatment": trt, "T_base_fitted_C": tb, "GDD": crop.gdd(tm, tb),
                           "DLI_mean": dli, "yield_obs_t_ha": yobs, "yield_TableS1_t_ha": yp,
                           "rel_error_pct": 100 * (yp - yobs) / yobs})
        t = t.drop(columns=[c for c in t.columns if c.startswith("_")])
        self.save(t, "T04_crop_light_heat_validation")
        self.save(pd.DataFrame(s1), "T04b_crop_TableS1_check")
        self.log(f"crop: joint alpha={self.alpha:.3f}, lambda={self.lam_field:.2e}; LOCO errors (lambda=0) "
                 f"{t.rel_error_pct_loco_lam0.round(1).tolist()} %")
        return t

    def step_water(self):
        """Calibrate the crop water-use factor on the spring control plots; validate on the other three totals."""
        ccfg = self.cfg["crops"]["lettuce"]
        fcfg = self.cfg["field"]
        r_eff = self.cfg["water"]["rain_efficiency"]
        tilt = fcfg["tilt_lettuce_deg"]
        rows = {}
        for cyc in fcfg["cycles"]:
            wx, ph, k = self._field_physics(cyc, tilt, ccfg)
            rows[cyc] = (ph.et0_open, ph.et0[:, k], ph.rain, ph.day_ids)
        cal = self.cfg["water"]["calibrate_on"]
        obs = fcfg["irrigation_mm"]
        eo, _, rain, dd = rows[cal]
        self.k_w = optimize.brentq(lambda kw_: water.season_irrigation_mm(kw_ * eo, rain, r_eff, dd) - obs[cal]["control"], 0.05, 5)
        out = []
        for cyc, (eo, ea, rain, dd) in rows.items():
            for trt, e in (("control", eo), ("apv", ea)):
                pred = water.season_irrigation_mm(self.k_w * e, rain, r_eff, dd)
                out.append({"cycle": cyc, "treatment": trt, "ET0_mm": e.sum(), "rain_mm": rain.sum(), "k_w": self.k_w,
                            "irrigation_obs_mm": obs[cyc][trt], "irrigation_pred_mm": pred,
                            "rel_error_pct": 100 * (pred - obs[cyc][trt]) / obs[cyc][trt],
                            "used_for": "calibration" if (cyc == cal and trt == "control") else "validation"})
        t = pd.DataFrame(out)
        for cyc in rows:
            g = t[t.cycle == cyc].set_index("treatment")
            t.loc[t.cycle == cyc, "saving_obs_pct"] = 100 * (1 - g.loc["apv", "irrigation_obs_mm"] / g.loc["control", "irrigation_obs_mm"])
            t.loc[t.cycle == cyc, "saving_pred_pct"] = 100 * (1 - g.loc["apv", "irrigation_pred_mm"] / g.loc["control", "irrigation_pred_mm"])
        self.save(t, "T05b_water_validation")
        self.log("water: k_w=%.3f; errors %s %%" % (self.k_w, t.rel_error_pct.round(1).tolist()))
        return t

    def step_pv_validation(self):
        ccfg = self.cfg["crops"]["lettuce"]
        rows = []
        for cyc in self.cfg["field"]["cycles"]:
            for tilt in (20, self.cfg["field"]["tilt_lettuce_deg"]):
                wx, ph, k = self._field_physics(cyc, tilt, ccfg)
                sc = self.cfg["sites"][self.main]
                sp = solar.sun_position_hourly(wx.index, sc["lat"], sc["lon"])
                b, d, dni = solar.split_beam_diffuse(wx.GHI.values, sp, wx.BHI.values, wx.DHI.values)
                _, n, u = solar.sun_vector(sp)
                from . import pv
                s = self.cfg["system"]
                poa = pv.poa_irradiance(self.geo, b, d, dni, b + d, n, u, s["albedo"], s["iam_b0"])[:, k]
                tm0 = pv.module_temperature(poa[:, None], wx["T"].values, wx["u10"].values, s["faiman_u0"], s["faiman_u1"], 0)[:, 0]
                tm1 = pv.module_temperature(poa[:, None], wx["T"].values, wx["u10"].values, s["faiman_u0"], s["faiman_u1"], s["canopy_cooling_C"])[:, 0]
                e25 = (poa / 1000 * s["inverter_eff"] * (1 - s["other_losses"])).sum()
                e0 = (poa / 1000 * (1 + s["gamma_pmp"] * (tm0 - 25)) * s["inverter_eff"] * (1 - s["other_losses"])).sum()
                e1 = (poa / 1000 * (1 + s["gamma_pmp"] * (tm1 - 25)) * s["inverter_eff"] * (1 - s["other_losses"])).sum()
                day = poa > 50
                rows.append({"cycle": cyc, "tilt_deg": tilt, "POA_kWh_m2": poa.sum() / 1000,
                             "PR_no_cooling": e0 / (poa.sum() / 1000), "PR_with_cooling": e1 / (poa.sum() / 1000),
                             "temp_loss_pct_no_cooling": 100 * (1 - e0 / e25),
                             "module_cooling_daytime_mean_C": float((tm0 - tm1)[day].mean()),
                             "module_cooling_max_C": float((tm0 - tm1).max()),
                             "energy_gain_cooling_pct": 100 * (e1 / e0 - 1)})
        t = pd.DataFrame(rows)
        self.save(t, "T05_pv_validation")
        self.log("pv: PR " + t.PR_no_cooling.round(3).astype(str).str.cat(sep=", "))
        return t

    # ================================================================== 5. context
    def context(self, site=None, crop_name=None, lam_key="central", **kw) -> ex.Context:
        site = site or self.main
        crop_name = crop_name or self.cfg["main_crop"]
        cfg = kw.pop("cfg", self.cfg)
        ccfg = cfg["crops"][crop_name]
        lv = ccfg["lambda"][lam_key]
        lam = kw.pop("lam", self.lam_field if lv == "calibrate" else float(lv))
        geo = kw.pop("geo", self.geo)
        ctx = ex.Context(cfg=cfg, site=site, era5=self.era5[site], price=self.price, geo=geo, micro=self.micro,
                         crop_name=crop_name, lam=lam, ecmwf=self.ecmwf if site == self.main else None,
                         errlib=self.errlib, emos_all=self.emos_all, hcorr_all=self.hcorr_all,
                         obs_daily=fc.daily_obs(self.era5[site], cfg["sites"][site]["tz"]), **kw)
        a = ccfg["alpha_light"]
        ctx.alpha = self.alpha if a == "calibrate" else float(a)
        ctx.lam = lam
        wcfg = cfg["water"]
        ctx.k_w = getattr(self, "k_w", 1.0) * ccfg.get("kc", 1.0) / cfg["crops"]["lettuce"].get("kc", 1.0)
        ctx.water_price = float(kw.get("water_price", cfg["sites"][site].get("water_price_eur_m3", wcfg["price_eur_m3"])))
        # persistence nowcast of next-hour temperature, from the site's own history
        w = ctx.era5
        st = (w["T"] - w["T"].shift(1, freq="h").reindex(w.index))
        step = st.groupby(w.index.hour).mean().reindex(range(24)).fillna(0).values
        err = st - step[w.index.hour]
        ctx.nowcast_step = step
        ctx.nowcast_sd = err.groupby(w.index.hour).std().reindex(range(24)).fillna(err.std()).values
        # anchor the potential yield on the reference season (open field = measured control yield)
        ref_year = ccfg["y_ref_year"]
        idx = ex.season_index(ctx, ref_year)
        ref_site = self.main
        ctx_ref = copy.copy(ctx)
        ctx_ref.site, ctx_ref.era5, ctx_ref.weather_override = ref_site, self.era5[ref_site], None
        ph_ref = ex.physics(ctx_ref, self.era5[ref_site].reindex(idx).interpolate(limit_direction="both"))
        # the heat-free potential is anchored with the reference damage coefficient (field-identified for
        # lettuce) and kept fixed when lambda is varied, so a hypothetical heat-sensitive crop loses yield
        # instead of gaining an inflated potential
        lc = ccfg["lambda"]["central"]
        lam_anchor = self.lam_field if lc == "calibrate" else float(lc)
        ctx.y_pot = ccfg["y_ref_t_ha"] * np.exp(lam_anchor * ph_ref.stress_open().sum())
        # reference light and yield for the expected-value controller: S3 over all years (climatology)
        Ls, Ys = [], []
        for y in self.years():
            ix = ex.season_index(ctx, y)
            ph = ex.physics(ctx, ex.weather(ctx).reindex(ix).interpolate(limit_direction="both"))
            i3 = sim.idx_light_sharing(ph, cfg["strategies"]["s3_fmax"])
            r = np.arange(len(ix))
            Ls.append(ph.sw[r, i3].sum())
            Ys.append(crop.yield_t_ha(ctx.y_pot, ph.sw[r, i3].sum() / ph.sw_open.sum(), ph.stress()[r, i3].sum(), ctx.alpha, lam))
        ctx.l_ref = float(np.mean(Ls))
        ctx.y_ref_kg = float(np.mean(Ys) * 1000)
        return ctx

    def run_years(self, ctx, years=None, **kw):
        years = years or self.years()
        return pd.concat([ex.run_season(ctx, y, **kw) for y in years], ignore_index=True)

    # ================================================================== 6. main comparison
    def step_main(self):
        ctx = self.context()
        self.ctx_main = ctx
        raw = self.run_years(ctx)
        raw.to_csv(self.out / "tables" / "T06a_strategies_raw.csv", index=False)
        summ = ex.summarise(raw)
        self.save(summ, "T06b_strategies_by_year")
        cols = ["energy_MWh_ha", "elec_rev_eur_ha", "imbalance_eur_ha", "capture_price", "light_rel", "heat_degh",
                "yield_t_ha", "crop_rev_eur_ha", "irrigation_mm", "water_cost_eur_ha", "value_eur_ha", "dV_vs_S3",
                "dV_vs_S2", "LER", "mean_fsh_day"]
        order = ["open_field"] + ex.STRATEGIES
        agg = summ.groupby("strategy")[cols].mean().reindex(order)
        sd = summ.groupby("strategy")["dV_vs_S3"].std().reindex(order)
        agg["dV_vs_S3_sd_years"] = sd
        self.save(agg.reset_index(), "T06_strategies_mean_2015_2025")
        voi = ex.voi_table(summ)
        voi_m = voi.drop(columns="year").agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
        self.save(voi, "T07_value_of_information_by_year")
        self.save(voi_m, "T07b_value_of_information_summary")
        # the real-forecast season (2025, archived ECMWF) on its own
        real = summ[summ.year == 2025]
        self.save(real, "T08_case_2025_real_forecasts")
        _, sched = ex.run_season(ctx, 2025, keep_schedules=True)
        self.res["main"] = (raw, summ, agg, voi, sched)
        self.log("main: mean dV vs S3 (EUR/ha): " + agg["dV_vs_S3"].round(0).to_string().replace("\n", " | "))
        self.log("VOI: " + voi_m.round(3).to_string(index=False).replace("\n", " | "))

    # ================================================================== 7. Pareto frontier
    def step_pareto(self):
        ctx = self.ctx_main
        rows = []
        ys = self.years()
        for w in self.cfg["strategies"]["crop_weight_grid"]:
            raw = pd.concat([ex.run_season(ctx, y, w_crop=w, n_real=3, strategies=["MPC_prob", "MPC_oracle", "EV_noinfo"])
                             for y in ys])
            m = raw.groupby("strategy")[["elec_rev_eur_ha", "crop_rev_eur_ha", "yield_t_ha", "energy_MWh_ha", "value_eur_ha"]].mean()
            for s, r in m.iterrows():
                if s in ("open_field", "pv_ref"):
                    continue
                rows.append({"family": s, "param": "crop_weight", "value": w, **r.to_dict()})
        for p in self.cfg["strategies"]["p_star_grid"]:
            raw = pd.concat([ex.run_season(ctx, y, p_star=p, n_real=3, strategies=["S4_rule"]) for y in ys])
            r = raw[raw.strategy == "S4_rule"][["elec_rev_eur_ha", "crop_rev_eur_ha", "yield_t_ha", "energy_MWh_ha", "value_eur_ha"]].mean()
            rows.append({"family": "S4_rule", "param": "p_star", "value": p, **r.to_dict()})
        t = pd.DataFrame(rows)
        self.save(t, "T09_pareto")
        self.res["pareto"] = t
        self.log(f"pareto: {len(t)} points")

    # ================================================================== 8. ablations
    def _quick_value(self, ctx, years=None, n_real=None, strategies=("S2_tracking", "EV_energy_light", "S3_light", "S4_rule", "S5_rule", "EV_noinfo",
                                                                      "EV_clim", "MPC_det", "MPC_prob", "MPC_oracle", "MPC_reactive")):
        years = years or self.years()
        n_real = n_real or max(3, self.cfg["forecast"]["n_synthetic"] // 4)
        raw = pd.concat([ex.run_season(ctx, y, n_real=n_real, strategies=list(strategies)) for y in years])
        summ = ex.summarise(raw)
        voi = ex.voi_table(summ)
        m = summ.groupby("strategy")[["value_eur_ha", "dV_vs_S3", "dV_vs_S2", "yield_t_ha", "elec_rev_eur_ha", "LER",
                                      "irrigation_mm", "water_cost_eur_ha"]].mean()
        return m, voi.drop(columns="year").mean()

    def step_ablation(self):
        """Remove one component at a time, at the field-identified and at a heat-sensitive damage level."""
        rows = []
        ccfg = self.cfg["crops"][self.cfg["main_crop"]]
        for lam_key in ("central", "high"):
            base = self.ctx_main if lam_key == "central" else self.context(lam_key="high")
            variants = {"full model": base}
            c = copy.copy(base); c.cooling = False; variants["no canopy cooling"] = c
            c = copy.copy(base); c.lam = 0.0; variants["no heat penalty (lambda=0)"] = c
            c = copy.copy(base); c.flat_price = True; variants["flat electricity price"] = c
            c = copy.copy(base); c.water_price = 0.0; variants["no water value"] = c
            cfg2 = copy.deepcopy(self.cfg); cfg2["crops"][self.cfg["main_crop"]]["stress_on"] = "air"
            variants["air instead of leaf temperature"] = self.context(cfg=cfg2, lam_key=lam_key)
            for name, ctx in variants.items():
                m, v = self._quick_value(ctx)
                rows.append({"lambda_level": f"{lam_key} ({ctx.lam if name != 'no heat penalty (lambda=0)' else base.lam:.2e})",
                             "variant": name, "dV_MPC_prob_vs_S3": m.loc["MPC_prob", "dV_vs_S3"],
                             "dV_S4_vs_S3": m.loc["S4_rule", "dV_vs_S3"],
                             "VOI_mpc": v["VOI_mpc_eur_ha"], "VOI_mpc_perfect": v["VOI_mpc_perfect_eur_ha"],
                             "eta_mpc": v["eta_mpc"], "prob_minus_det": v["prob_vs_det_eur_ha"],
                             "dayahead_minus_reactive": v["dayahead_vs_reactive_eur_ha"],
                             "controller_vs_tracking": v["controller_vs_tracking_eur_ha"],
                             "water_term": v["water_term_eur_ha"],
                             "LER_MPC_prob": m.loc["MPC_prob", "LER"]})
                self.log(f"ablation [{lam_key}] {name}: dV={rows[-1]['dV_MPC_prob_vs_S3']:.1f}, VOI={rows[-1]['VOI_mpc']:.1f}")
        t = pd.DataFrame(rows)
        self.save(t, "T10_ablation")
        self.res["ablation"] = t

    # ================================================================== 9. sensitivity + surrogate
    def step_sensitivity(self):
        base = self.ctx_main
        ccfg = self.cfg["crops"][self.cfg["main_crop"]]
        sc = self.cfg["sensitivity"]
        m0, v0 = self._quick_value(base)
        rows = [{"parameter": "baseline", "setting": "", "dV_MPC_prob": m0.loc["MPC_prob", "dV_vs_S3"],
                 "dV_vs_S2": m0.loc["MPC_prob", "dV_vs_S2"],
                 "VOI_mpc": v0["VOI_mpc_eur_ha"], "dV_S4": m0.loc["S4_rule", "dV_vs_S3"]}]

        def add(param, setting, ctx, years=None):
            m, v = self._quick_value(ctx, years)
            rows.append({"parameter": param, "setting": setting, "dV_MPC_prob": m.loc["MPC_prob", "dV_vs_S3"],
                         "dV_vs_S2": m.loc["MPC_prob", "dV_vs_S2"],
                         "VOI_mpc": v["VOI_mpc_eur_ha"], "dV_S4": m.loc["S4_rule", "dV_vs_S3"]})
            self.log(f"sensitivity {param}={setting}: dV={rows[-1]['dV_MPC_prob']:.1f}")

        for k in ("low", "high"):
            add("lambda", f"{k} ({ccfg['lambda'][k]})", self.context(lam_key=k))
        for f in (0.6, 1.5):
            c = copy.copy(base); c.alpha = base.alpha * f
            add("light exponent alpha", f"x{f} ({c.alpha:.2f})", c)
        for wp in (0.0, 0.53, 1.0):
            c = copy.copy(base); c.water_price = wp
            add("water price", f"{wp} EUR/m3", c)
        for d in sc["t_crit_delta"]:
            c = copy.copy(base); c.t_crit_shift = d
            add("T_crit", f"{d:+d} C", c)
        for f in sc["crop_price_factor"]:
            cfg2 = copy.deepcopy(self.cfg); cfg2["crops"][self.cfg["main_crop"]]["price_eur_kg"] *= f
            add("crop price", f"x{f}", self.context(cfg=cfg2))
        for name, (a, b) in sc["price_periods"].items():
            ys = [y for y in self.years() if a <= y <= b]
            if ys:
                add("price period", name, base, ys)
        for p in sc["pitch_m"]:
            cfg2 = copy.deepcopy(self.cfg); s = cfg2["system"]
            geo = ArrayGeometry(s["height_m"], p, s["gcr"] * s["pitch_m"] / p, self.tilts, s["n_ground_points"])
            cfg2["system"]["kwp_per_ha"] = s["kwp_per_ha"] * s["pitch_m"] / p
            add("row pitch", f"{p} m (GCR {s['gcr'] * s['pitch_m'] / p:.2f})", self.context(cfg=cfg2, geo=geo))
        for h in sc.get("height_m", []):   # invariant for infinite rows; kept as a check
            s = self.cfg["system"]
            geo = ArrayGeometry(h, s["pitch_m"], s["gcr"], self.tilts, s["n_ground_points"])
            add("panel height", f"{h} m", self.context(geo=geo))
        t = pd.DataFrame(rows)
        self.save(t, "T11_sensitivity")
        self.res["sensitivity"] = t

    def step_surrogate(self):
        """Gaussian-process surrogate of Delta V(MPC_prob) over the uncertain parameters, checked on held-out runs."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        sc = self.cfg["sensitivity"]
        rng = np.random.default_rng(self.cfg["seed"])
        n = sc["surrogate_runs"]
        ccfg = self.cfg["crops"][self.cfg["main_crop"]]
        X, Y = [], []
        years = self.years()[-3:]
        for i in range(n):
            lam = rng.uniform(0, 0.003)
            dtc = rng.uniform(-3, 3)
            pf = rng.uniform(0.4, 1.6)
            cfg2 = copy.deepcopy(self.cfg)
            cfg2["crops"][self.cfg["main_crop"]]["price_eur_kg"] = ccfg["price_eur_kg"] * pf
            ctx = self.context(cfg=cfg2, lam=lam)
            ctx.t_crit_shift = dtc
            m, _ = self._quick_value(ctx, years, n_real=2)
            X.append([lam * 1000, dtc, pf])
            Y.append(m.loc["MPC_prob", "dV_vs_S3"])
        X, Y = np.array(X), np.array(Y)
        ho = sc["surrogate_holdout"]
        k = ConstantKernel(1.0) * Matern(length_scale=[1, 2, 0.5], nu=2.5) + WhiteKernel(1.0)
        ys = Y.std() or 1.0
        gp = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0).fit(X[:-ho], Y[:-ho])
        pr = gp.predict(X[-ho:])
        r2 = 1 - np.sum((pr - Y[-ho:]) ** 2) / np.sum((Y[-ho:] - Y[-ho:].mean()) ** 2)
        t = pd.DataFrame(X, columns=["lambda_x1000", "dT_crit_C", "crop_price_factor"])
        t["dV_MPC_prob"] = Y
        t["set"] = ["train"] * (n - ho) + ["holdout"] * ho
        self.save(t, "T12_surrogate_runs")
        self.save(pd.DataFrame([{"holdout_R2": r2, "holdout_RMSE": float(np.sqrt(np.mean((pr - Y[-ho:]) ** 2))),
                                 "kernel": str(gp.kernel_)}]), "T12b_surrogate_check", latex=False)
        self.res["surrogate"] = (t, gp, r2)
        self.log(f"surrogate: hold-out R2 = {r2:.3f}")

    # ================================================================== 9b. decision map over the damage coefficient
    def step_lambda_sweep(self):
        """Value of forecast information versus the heat-damage coefficient, present and 2041-2050."""
        grid = [0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3]
        site = self.main
        tz = self.cfg["sites"][site]["tz"]
        cm = io.load_cmip6(self.cfg, self.cfg["sites"][site]["cmip6_prefix"])
        py = self.cfg["climate"]["price_years_for_future"]
        years = self.years()
        pmap = {y: py[i % len(py)] for i, y in enumerate(years)}
        climates = {"present": None}
        for model in self.cfg["climate"]["models"]:
            d = climate.deltas(cm, model)
            if d is not None:
                climates[f"2041-2050 {model}"] = climate.perturb(self.era5[site], d, tz)
        rows = []
        for lam in grid:
            for name, wx in climates.items():
                ctx = self.context(lam=lam)
                ctx.weather_override = wx
                ctx.price_year_map = pmap
                m, v = self._quick_value(ctx, years, n_real=2)
                rows.append({"lambda": lam, "climate": name, "VOI_mpc": v["VOI_mpc_eur_ha"],
                             "VOI_mpc_perfect": v["VOI_mpc_perfect_eur_ha"], "eta_mpc": v["eta_mpc"],
                             "VOI_rule": v["VOI_rule_eur_ha"], "dV_MPC_prob": m.loc["MPC_prob", "dV_vs_S3"],
                             "dayahead_minus_reactive": v["dayahead_vs_reactive_eur_ha"],
                             "controller_vs_tracking": v["controller_vs_tracking_eur_ha"],
                             "heat_degh_EV_noinfo": None})
            self.log(f"lambda sweep {lam:.0e}: " + ", ".join(f"{r['climate']}={r['VOI_mpc']:.0f}" for r in rows if r["lambda"] == lam))
        t = pd.DataFrame(rows).drop(columns="heat_degh_EV_noinfo")
        self.save(t, "T16_voi_vs_lambda", floatfmt="%.4g")
        self.res["lambda_sweep"] = t

    # ================================================================== 10. climate 2050
    def step_climate(self):
        ccfg = self.cfg["crops"][self.cfg["main_crop"]]
        rows, hot = [], []
        py = self.cfg["climate"]["price_years_for_future"]
        for site in self.cfg["sites"]:
            pre = self.cfg["sites"][site].get("cmip6_prefix")
            if not pre:
                continue
            cm = io.load_cmip6(self.cfg, pre)
            base_ctx = self.context(site=site)
            years = self.years()
            pmap = {y: py[i % len(py)] for i, y in enumerate(years)}
            for model in self.cfg["climate"]["models"]:
                h0, h1 = climate.hot_days_per_year(cm, model, ccfg["stress_day_tmax_C"], months=[5, 6])
                h40a, h40b = climate.hot_days_per_year(cm, model, 40.0)
                hot.append({"site": site, "model": model, "days_tmax_ge_event_MayJun_hist": h0,
                            "days_tmax_ge_event_MayJun_2041_2050": h1, "days_ge_40C_hist": h40a, "days_ge_40C_2041_2050": h40b})
                d = climate.deltas(cm, model)
                if d is None:
                    continue
                for period, wx in (("present (prices 2023-25)", None), ("2041-2050", climate.perturb(self.era5[site], d, self.cfg["sites"][site]["tz"]))):
                    if period.startswith("present") and model != self.cfg["climate"]["models"][0]:
                        continue
                    ctx = copy.copy(base_ctx)
                    ctx.weather_override = wx
                    ctx.price_year_map = pmap
                    m, v = self._quick_value(ctx, years)
                    rows.append({"site": site, "model": model if wx is not None else "ERA5", "period": period,
                                 "heat_degh_open": None, "dV_MPC_prob": m.loc["MPC_prob", "dV_vs_S3"],
                                 "dV_S4": m.loc["S4_rule", "dV_vs_S3"], "VOI_mpc": v["VOI_mpc_eur_ha"],
                                 "eta_mpc": v["eta_mpc"], "yield_S3": m.loc["S3_light", "yield_t_ha"],
                                 "yield_MPC_prob": m.loc["MPC_prob", "yield_t_ha"], "LER_MPC_prob": m.loc["MPC_prob", "LER"]})
                    # heat load in the open field
                    H = []
                    for y in years:
                        ix = ex.season_index(ctx, y)
                        ph = ex.physics(ctx, ex.weather(ctx).reindex(ix).interpolate(limit_direction="both"))
                        H.append(ph.stress_open().sum())
                    rows[-1]["heat_degh_open"] = float(np.mean(H))
                    self.log(f"climate {site} {rows[-1]['model']} {period}: dV={rows[-1]['dV_MPC_prob']:.1f}, H_open={rows[-1]['heat_degh_open']:.0f}")
        self.save(pd.DataFrame(hot), "T13_cmip6_hot_days")
        t = pd.DataFrame(rows)
        self.save(t, "T13b_climate_2050_value")
        self.res["climate"] = (pd.DataFrame(hot), t)

    # ================================================================== 11. transfer site(s)
    def step_transfer(self):
        rows = []
        for site in self.cfg["sites"]:
            if site == self.main:
                continue
            ctx = self.context(site=site)
            raw = self.run_years(ctx, n_real=max(3, self.cfg["forecast"]["n_synthetic"] // 4))
            summ = ex.summarise(raw)
            voi = ex.voi_table(summ).drop(columns="year").mean()
            m = summ.groupby("strategy")[["energy_MWh_ha", "yield_t_ha", "value_eur_ha", "dV_vs_S3", "LER", "heat_degh"]].mean()
            for s, r in m.iterrows():
                rows.append({"site": site, "strategy": s, **r.to_dict(), "VOI_mpc_site": voi["VOI_mpc_eur_ha"], "eta_mpc_site": voi["eta_mpc"]})
            self.log(f"transfer {site}: dV MPC_prob {m.loc['MPC_prob', 'dV_vs_S3']:.1f}")
        if rows:
            t = pd.DataFrame(rows)
            self.save(t, "T14_transfer_sites")
            self.res["transfer"] = t

    def step_transfer_lambda(self):
        """Value of information versus lambda at every transfer site (and with the Seville calendar at hot sites)."""
        grid = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3]
        rows = []
        for site in self.cfg["sites"]:
            if site == self.main:
                continue
            variants = {"site calendar": None}
            if self.cfg["sites"][site].get("season"):
                variants["Seville calendar (May-Jun)"] = self.cfg["crops"][self.cfg["main_crop"]]["season"]
            if self.cfg["sites"][site].get("price") == "flat":
                variants["site calendar, OMIE-shaped prices"] = "omie"
            for vname, v in variants.items():
                cfg2 = copy.deepcopy(self.cfg)
                if isinstance(v, list):
                    cfg2["sites"][site]["season"] = v
                if v == "omie":
                    cfg2["sites"][site]["price"] = "omie_shape"
                for lam in grid:
                    ctx = self.context(site=site, cfg=cfg2, lam=lam)
                    if v == "omie":
                        ctx.price_year_map = {}
                    m, vv = self._quick_value(ctx, n_real=3)
                    rows.append({"site": site, "variant": vname, "lambda": lam, "VOI_mpc": vv["VOI_mpc_eur_ha"],
                                 "VOI_mpc_perfect": vv["VOI_mpc_perfect_eur_ha"], "eta_mpc": vv["eta_mpc"],
                                 "dV_MPC_prob": m.loc["MPC_prob", "dV_vs_S3"], "yield_MPC": m.loc["MPC_prob", "yield_t_ha"],
                                 "LER_MPC": m.loc["MPC_prob", "LER"]})
                self.log(f"transfer lambda {site} [{vname}]: " + ", ".join(f"{r['lambda']:.0e}:{r['VOI_mpc']:.0f}" for r in rows if r["site"] == site and r["variant"] == vname))
        t = pd.DataFrame(rows)
        self.save(t, "T17_transfer_voi_vs_lambda", floatfmt="%.4g")
        self.res["transfer_lambda"] = t

    # ================================================================== 12. second crop (illustrative)
    def step_tomato(self):
        """Heat-sensitive case study: processing tomato in Seville (full strategy set, price variants, lambda sweep)."""
        if "tomato" not in self.cfg["crops"]:
            return
        tc = self.cfg["crops"]["tomato"]
        n_real = max(3, self.cfg["forecast"]["n_synthetic"] // 4)
        ctx = self.context(crop_name="tomato")
        raw = self.run_years(ctx, n_real=n_real)
        summ = ex.summarise(raw)
        cols = ["energy_MWh_ha", "yield_t_ha", "heat_degh", "irrigation_mm", "value_eur_ha", "dV_vs_S3", "dV_vs_S2", "LER"]
        m = summ.groupby("strategy")[cols].mean().reindex(["open_field"] + ex.STRATEGIES)
        self.save(m.reset_index(), "T15_tomato_strategies")
        rows = []
        variants = {"processing price, own-year prices": (tc["price_eur_kg"], {}),
                    "fresh-market price, own-year prices": (tc["price_fresh_eur_kg"], {}),
                    "processing price, 2023-2025 prices": (tc["price_eur_kg"], "recent"),
                    "fresh-market price, 2023-2025 prices": (tc["price_fresh_eur_kg"], "recent")}
        py = self.cfg["climate"]["price_years_for_future"]
        for name, (pc, pm) in variants.items():
            cfg2 = copy.deepcopy(self.cfg)
            cfg2["crops"]["tomato"]["price_eur_kg"] = pc
            for lam in [0.0, 2.5e-4, 5e-4, 1e-3, 2e-3, 3e-3]:
                c = self.context(crop_name="tomato", cfg=cfg2, lam=lam)
                if pm == "recent":
                    c.price_year_map = {y: py[i % len(py)] for i, y in enumerate(self.years())}
                mm, v = self._quick_value(c, n_real=2)
                rows.append({"variant": name, "lambda": lam, "VOI_mpc": v["VOI_mpc_eur_ha"],
                             "VOI_mpc_perfect": v["VOI_mpc_perfect_eur_ha"],
                             "dayahead_minus_reactive": v["dayahead_vs_reactive_eur_ha"],
                             "controller_vs_tracking": v["controller_vs_tracking_eur_ha"],
                             "water_term": v["water_term_eur_ha"],
                             "yield_MPC_t_ha": mm.loc["MPC_prob", "yield_t_ha"], "yield_S2_t_ha": mm.loc["S2_tracking", "yield_t_ha"],
                             "LER_MPC": mm.loc["MPC_prob", "LER"]})
            self.log(f"tomato [{name}]: " + ", ".join(f"{r['lambda']:.1e}:{r['VOI_mpc']:.0f}" for r in rows if r["variant"] == name))
        t = pd.DataFrame(rows)
        self.save(t, "T15b_tomato_voi_vs_lambda", floatfmt="%.4g")
        self.res["tomato"] = (m, t)
        self.log(f"tomato: dV MPC_prob vs S3 {m.loc['MPC_prob', 'dV_vs_S3']:.0f}, vs S2 {m.loc['MPC_prob', 'dV_vs_S2']:.0f}")

    def write_log(self):
        (self.out / "logs").mkdir(exist_ok=True)
        (self.out / "logs" / "run_log.txt").write_text("\n".join(self.log_lines))
