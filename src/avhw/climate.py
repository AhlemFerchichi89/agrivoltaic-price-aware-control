"""Mid-century (2041-2050) weather by quantile delta mapping of CMIP6 HighResMIP output.

For each model and calendar month, the change in daily maximum temperature is
computed per quantile (future minus historical, Cannon et al., 2015) and the
change in daily minimum temperature, relative humidity and shortwave
radiation as a mean. The hourly ERA5 series is then perturbed day by day:

  T'(h) = T(h) + dTmin + (dTmax(q) - dTmin) * (T(h) - Tmin) / (Tmax - Tmin)

where q is the quantile of that day's Tmax in the historical model
climate, so the hottest days warm by the change of the hottest model days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import local_date


def _col(df, var, model):
    c = f"{var}_{model}"
    return df[c] if c in df else None


def deltas(cm: dict, model: str, qs=np.linspace(0.02, 0.98, 25)) -> dict:
    h, f = cm["hist"], cm["future"]
    out = {"q": qs, "tmax": {}, "tmin": {}, "rh": {}, "sw": {}}
    for m in range(1, 13):
        hm, fm = h[h.index.month == m], f[f.index.month == m]
        th, tf = _col(hm, "temperature_2m_max", model).dropna(), _col(fm, "temperature_2m_max", model).dropna()
        if len(tf) < 30:
            return None
        out["tmax"][m] = (np.quantile(th, qs), np.quantile(tf, qs) - np.quantile(th, qs))
        out["tmin"][m] = _col(fm, "temperature_2m_min", model).mean() - _col(hm, "temperature_2m_min", model).mean()
        rh_h, rh_f = _col(hm, "relative_humidity_2m_mean", model), _col(fm, "relative_humidity_2m_mean", model)
        out["rh"][m] = (rh_f.mean() - rh_h.mean()) if rh_h is not None else 0.0
        sw_h, sw_f = _col(hm, "shortwave_radiation_sum", model), _col(fm, "shortwave_radiation_sum", model)
        ok = sw_h is not None and sw_h.notna().sum() > 30 and sw_f.notna().sum() > 30
        out["sw"][m] = (sw_f.mean() / sw_h.mean()) if ok else 1.0
    return out


def perturb(era5: pd.DataFrame, d: dict, tz: str) -> pd.DataFrame:
    w = era5.copy()
    dates = local_date(w.index, tz)
    g = w["T"].groupby(dates)
    tmax, tmin = g.transform("max").values, g.transform("min").values
    months = pd.DatetimeIndex(dates).month.values
    dTx = np.zeros(len(w))
    dTn = np.zeros(len(w))
    drh = np.zeros(len(w))
    fsw = np.ones(len(w))
    for m in np.unique(months):
        k = months == m
        qh, dq = d["tmax"][m]
        # map each day's (ERA5) Tmax to its quantile rank within ERA5 for that month
        era_m = tmax[k]
        ranks = np.searchsorted(np.sort(era_m), era_m, side="right") / len(era_m)
        dTx[k] = np.interp(ranks, d["q"], dq)
        dTn[k] = d["tmin"][m]
        drh[k] = d["rh"][m]
        fsw[k] = d["sw"][m]
    span = np.maximum(tmax - tmin, 1.0)
    w["T"] = w["T"].values + dTn + (dTx - dTn) * (w["T"].values - tmin) / span
    w["RH"] = np.clip(w["RH"].values + drh, 2, 100)
    for c in ("GHI", "BHI", "DHI"):
        if c in w:
            w[c] = w[c].values * fsw
    return w


def hot_days_per_year(cm: dict, model: str, thr: float, months=None) -> tuple[float, float]:
    out = []
    for p in ("hist", "future"):
        s = _col(cm[p], "temperature_2m_max", model).dropna()
        if months:
            s = s[s.index.month.isin(months)]
        out.append((s >= thr).groupby(s.index.year).sum().mean())
    return tuple(out)
