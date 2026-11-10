"""Probabilistic day-ahead forecasting of heat stress.

Target: daily maximum air temperature (local day) and the stress-day event
Tmax >= T_event. Hourly temperature distributions for the controllers are
obtained by an hour-of-day bias/spread correction of the NWP forecast.

Forecasters (all evaluated with month-blocked cross-validation on the ECMWF
archive, Sep 2024 - Jul 2025):
  climatology   ERA5 day-of-year window (2015-2023), empirical distribution
  persistence   observed Tmax `lead` days earlier, Gaussian spread
  raw_ecmwf     ECMWF IFS deterministic forecast (no post-processing)
  emos          Gaussian EMOS (Gneiting et al., 2005) fitted by minimum CRPS
  qgb           quantile gradient boosting (scikit-learn HistGradientBoosting)
An Informer / Transformer forecaster is provided separately in
`avhw/informer.py` (requires PyTorch) and plugs into the same interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor


# =============================================================================
# scores
def crps_gaussian(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_ensemble(samples, y):
    """CRPS for an empirical distribution (rows = cases)."""
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    t1 = np.mean(np.abs(s - y[:, None]), axis=1)
    i = np.arange(1, m + 1)
    t2 = np.sum((2 * i - m - 1) * s, axis=1) / m ** 2
    return t1 - t2


def crps_quantiles(qs, levels, y):
    """CRPS approximated by twice the mean pinball loss over the quantile levels."""
    levels = np.asarray(levels)
    d = y[:, None] - qs
    pin = np.maximum(levels * d, (levels - 1) * d)
    return 2 * pin.mean(axis=1)


def brier(p, o):
    return float(np.mean((np.asarray(p) - np.asarray(o)) ** 2))


def expected_exceedance(mu, sigma, c):
    """E[max(0, X - c)] for X ~ N(mu, sigma^2)."""
    sigma = np.maximum(sigma, 1e-6)
    z = (mu - c) / sigma
    return sigma * stats.norm.pdf(z) + (mu - c) * stats.norm.cdf(z)


# =============================================================================
# daily frames
def local_date(index_utc: pd.DatetimeIndex, tz: str) -> pd.Index:
    loc = (index_utc - pd.Timedelta(minutes=30)).tz_convert(tz)
    return loc.normalize().tz_localize(None)


def daily_obs(era5: pd.DataFrame, tz: str) -> pd.DataFrame:
    d = era5.assign(date=local_date(era5.index, tz))
    g = d.groupby("date")
    out = pd.DataFrame({"Tmax": g["T"].max(), "Tmin": g["T"].min(), "Tmean": g["T"].mean(),
                        "RH": g["RH"].mean(), "GHI": g["GHI"].sum() / 1000, "u": g["u10"].mean(),
                        "n": g["T"].size()})
    return out[out.n >= 22].drop(columns="n")


def daily_fc(ecmwf: pd.DataFrame, lead: int, tz: str) -> pd.DataFrame:
    d = ecmwf.assign(date=local_date(ecmwf.index, tz))
    g = d.groupby("date")
    out = pd.DataFrame({"Tmax_fc": g[f"T_l{lead}"].max(), "Tmin_fc": g[f"T_l{lead}"].min(),
                        "RH_fc": g[f"RH_l{lead}"].mean(), "GHI_fc": g[f"GHI_l{lead}"].sum() / 1000,
                        "u_fc": g[f"u10_l{lead}"].mean(), "n": g[f"T_l{lead}"].count()})
    return out[out.n >= 22].drop(columns="n")


def climatology_samples(obs: pd.DataFrame, dates, years, window) -> list[np.ndarray]:
    base = obs[(obs.index.year >= years[0]) & (obs.index.year <= years[1])]
    doy = base.index.dayofyear.values
    out = []
    for d in pd.DatetimeIndex(dates):
        dd = np.abs(((doy - d.dayofyear) + 182) % 365 - 182)
        out.append(base["Tmax"].values[dd <= window])
    return out


# =============================================================================
# EMOS
class EMOS:
    """mu = X @ a ; log(sigma) = Z @ b ; fitted by minimum mean CRPS."""

    def __init__(self):
        self.a = self.b = None

    @staticmethod
    def design(df):
        X = np.column_stack([np.ones(len(df)), df["Tmax_fc"], df["RH_fc"] / 100, df["GHI_fc"] / 10])
        Z = np.column_stack([np.ones(len(df)), (df["Tmax_fc"] - 30) / 10])
        return X, Z

    def fit(self, df, y):
        X, Z = self.design(df)
        a0 = np.linalg.lstsq(X, y, rcond=None)[0]
        s0 = np.log(np.std(y - X @ a0) + 1e-3)

        def loss(p):
            a, b = p[:X.shape[1]], p[X.shape[1]:]
            return crps_gaussian(X @ a, np.exp(Z @ b), y).mean()

        res = optimize.minimize(loss, np.r_[a0, s0, 0.0], method="L-BFGS-B")
        self.a, self.b = res.x[:X.shape[1]], res.x[X.shape[1]:]
        return self

    def predict(self, df):
        X, Z = self.design(df)
        return X @ self.a, np.exp(Z @ self.b)


class QGB:
    """Quantile gradient boosting with monotone re-sorting of the quantiles."""

    feats = ["Tmax_fc", "Tmin_fc", "RH_fc", "GHI_fc", "u_fc", "doy_s", "doy_c", "Tmax_persist"]

    def __init__(self, levels, seed=0):
        self.levels = list(levels)
        self.seed = seed
        self.models = {}

    def fit(self, df, y):
        X = df[self.feats].values
        for q in self.levels:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150, learning_rate=0.05,
                                              max_depth=3, min_samples_leaf=10, random_state=self.seed)
            self.models[q] = m.fit(X, y)
        return self

    def predict(self, df):
        X = df[self.feats].values
        qs = np.column_stack([self.models[q].predict(X) for q in self.levels])
        return np.sort(qs, axis=1)


def prob_exceed_from_quantiles(qs, levels, thr):
    """P(Y >= thr) from a set of quantiles (linear interpolation, flat tails)."""
    levels = np.asarray(levels)
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        if thr <= q[0]:
            out[i] = 1 - levels[0] * max(0.0, 1 - (q[0] - thr) / max(q[-1] - q[0], 1.0))
        elif thr >= q[-1]:
            out[i] = (1 - levels[-1]) * max(0.0, 1 - (thr - q[-1]) / max(q[-1] - q[0], 1.0))
        else:
            out[i] = 1 - np.interp(thr, q, levels)
    return np.clip(out, 0, 1)


# =============================================================================
def add_features(fc: pd.DataFrame, obs: pd.DataFrame, lead: int) -> pd.DataFrame:
    f = fc.copy()
    doy = f.index.dayofyear.values
    f["doy_s"], f["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["Tmax_persist"] = obs["Tmax"].reindex(f.index - pd.Timedelta(days=lead)).values
    return f


def evaluate_forecasters(era5, ecmwf, tz, cfg, t_event) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-blocked cross-validation of all forecasters for every lead time.

    Returns (scores, predictions); predictions hold mu/sigma/p per model.
    """
    fcfg = cfg["forecast"]
    levels = fcfg["quantiles"]
    obs = daily_obs(era5, tz)
    rows, preds = [], []
    for lead in fcfg["leads"]:
        fc = add_features(daily_fc(ecmwf, lead, tz), obs, lead)
        d = fc.join(obs[["Tmax"]], how="inner").dropna()
        months = d.index.to_period("M")
        out = pd.DataFrame(index=d.index)
        out["lead"], out["y"] = lead, d["Tmax"]
        out["event"] = (d["Tmax"] >= t_event).astype(int)
        for m in months.unique():
            te = months == m
            tr = ~te
            dtr, dte = d[tr], d[te]
            # EMOS
            em = EMOS().fit(dtr, dtr["Tmax"].values)
            mu, sg = em.predict(dte)
            out.loc[te, "emos_mu"], out.loc[te, "emos_sd"] = mu, sg
            # QGB
            qg = QGB(levels, cfg["seed"]).fit(dtr, dtr["Tmax"].values)
            qs = qg.predict(dte)
            for j, q in enumerate(levels):
                out.loc[te, f"qgb_q{q}"] = qs[:, j]
            # persistence spread from the training months
            res = (dtr["Tmax"] - dtr["Tmax_persist"]).values
            out.loc[te, "pers_mu"] = dte["Tmax_persist"].values
            out.loc[te, "pers_sd"] = np.std(res)
        out["raw_mu"] = d["Tmax_fc"]
        clim = climatology_samples(obs, d.index, fcfg["climatology_years"], fcfg["climatology_window_days"])
        out["clim_p"] = [np.mean(c >= t_event) for c in clim]
        out["clim_med"] = [np.median(c) for c in clim]
        y = out["y"].values
        o = out["event"].values
        qcols = [f"qgb_q{q}" for q in levels]
        scores = {
            "climatology": (crps_ensemble(np.array([np.sort(np.random.default_rng(0).choice(c, 200)) for c in clim]), y),
                            out["clim_med"].values, out["clim_p"].values),
            "persistence": (crps_gaussian(out.pers_mu.values, out.pers_sd.values, y), out.pers_mu.values,
                            1 - stats.norm.cdf(t_event, out.pers_mu, out.pers_sd)),
            "raw_ecmwf": (np.abs(out.raw_mu.values - y), out.raw_mu.values, (out.raw_mu.values >= t_event).astype(float)),
            "emos": (crps_gaussian(out.emos_mu.values, out.emos_sd.values, y), out.emos_mu.values,
                     1 - stats.norm.cdf(t_event, out.emos_mu, out.emos_sd)),
            "qgb": (crps_quantiles(out[qcols].values, levels, y), out[f"qgb_q0.5"].values,
                    prob_exceed_from_quantiles(out[qcols].values, levels, t_event)),
        }
        bs_clim = brier(out["clim_p"], o)
        for name, (crps, med, p) in scores.items():
            out[f"{name}_p"] = p
            pred_evt = np.asarray(p) >= cfg["strategies"]["p_star"]
            hits = np.sum(pred_evt & (o == 1))
            rows.append({"lead_days": lead, "model": name, "n_days": len(y), "n_events": int(o.sum()),
                         "CRPS_C": float(np.mean(crps)), "MAE_median_C": float(np.mean(np.abs(med - y))),
                         "Brier": brier(p, o), "BSS_vs_clim": 1 - brier(p, o) / bs_clim if bs_clim > 0 else np.nan,
                         "hit_rate": hits / max(o.sum(), 1),
                         "false_alarm_ratio": np.sum(pred_evt & (o == 0)) / max(pred_evt.sum(), 1)})
        # interval coverage (10-90 %) for the probabilistic models
        lo, hi = out["qgb_q0.1"].values, out["qgb_q0.9"].values
        cov_q = np.mean((y >= lo) & (y <= hi))
        z = stats.norm.ppf(0.9)
        cov_e = np.mean(np.abs(y - out.emos_mu) <= z * out.emos_sd)
        for r in rows:
            if r["lead_days"] == lead and r["model"] == "qgb":
                r["cov_10_90"] = cov_q
            if r["lead_days"] == lead and r["model"] == "emos":
                r["cov_10_90"] = cov_e
        preds.append(out)
    return pd.DataFrame(rows), pd.concat(preds)


# =============================================================================
# models used by the controllers
class HourlyCorrector:
    """Hour-of-day bias and spread of the lead-1 temperature forecast vs ERA5."""

    def fit(self, era5, ecmwf, lead=1, mask=None):
        j = ecmwf[[f"T_l{lead}"]].join(era5[["T"]], how="inner").dropna()
        if mask is not None:
            j = j[mask(j.index)]
        r = j["T"] - j[f"T_l{lead}"]
        h = j.index.hour
        self.bias = r.groupby(h).mean().reindex(range(24)).fillna(0).values
        self.sd = r.groupby(h).std().reindex(range(24)).fillna(r.std()).values
        return self

    def apply(self, t_fc, index_utc):
        h = index_utc.hour
        return t_fc + self.bias[h], self.sd[h]


class ErrorLibrary:
    """Archive of whole-day lead-1 forecast errors (forecast - ERA5) for T, RH, GHI, u10.

    Used to generate synthetic but realistic day-ahead forecasts for seasons
    without archived NWP: forecast = ERA5 + error-day drawn from the archive
    within +-window days of the same day of year (block bootstrap of complete
    days, so the diurnal and cross-variable error structure is preserved).
    """

    vars = ["T", "RH", "GHI", "u10"]

    def __init__(self, era5, ecmwf, lead=1):
        j = ecmwf[[f"{v}_l{lead}" for v in self.vars]].join(era5[self.vars], how="inner").dropna()
        days = {}
        for d, g in j.groupby(j.index.floor("D")):
            if len(g) == 24:
                days[d] = np.column_stack([g[f"{v}_l{lead}"].values - g[v].values for v in self.vars])
        self.dates = pd.DatetimeIndex(list(days.keys()))
        self.err = np.stack(list(days.values()))            # (days, 24, 4)
        self.doy = self.dates.dayofyear.values

    def synthetic(self, era5_season: pd.DataFrame, rng, window=30) -> pd.DataFrame:
        out = era5_season[self.vars].copy()
        for d, g in era5_season.groupby(era5_season.index.floor("D")):
            dd = np.abs(((self.doy - d.dayofyear) + 182) % 365 - 182)
            cand = np.where(dd <= window)[0]
            if len(cand) == 0:
                cand = np.arange(len(self.doy))
            e = self.err[rng.choice(cand)]
            hrs = g.index.hour.values
            out.loc[g.index] = g[self.vars].values + e[hrs]
        out["RH"] = out["RH"].clip(1, 100)
        out["GHI"] = out["GHI"].clip(lower=0).where(era5_season["GHI"] > 0, 0)
        out["u10"] = out["u10"].clip(lower=0.1)
        return out


def fc_frame_from_ecmwf(ecmwf, lead=1) -> pd.DataFrame:
    return pd.DataFrame({"T": ecmwf[f"T_l{lead}"], "RH": ecmwf[f"RH_l{lead}"].clip(1, 100),
                         "GHI": ecmwf[f"GHI_l{lead}"].clip(lower=0), "u10": ecmwf[f"u10_l{lead}"]})


def daily_fc_from_hourly(fc_hourly: pd.DataFrame, obs_daily: pd.DataFrame, tz: str, lead: int = 1) -> pd.DataFrame:
    """Daily EMOS/QGB predictors from an hourly forecast frame."""
    d = fc_hourly.assign(date=local_date(fc_hourly.index, tz))
    g = d.groupby("date")
    f = pd.DataFrame({"Tmax_fc": g["T"].max(), "Tmin_fc": g["T"].min(), "RH_fc": g["RH"].mean(),
                      "GHI_fc": g["GHI"].sum() / 1000, "u_fc": g["u10"].mean()})
    return add_features(f, obs_daily, lead)
