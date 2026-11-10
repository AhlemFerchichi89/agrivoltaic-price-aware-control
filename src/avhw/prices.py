"""Electricity prices on heat-stress days versus other days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import daily_obs, local_date


def hot_day_price_analysis(era5: pd.DataFrame, price: pd.Series, tz: str, t_hot: float,
                           months=(6, 7, 8)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare hourly price profiles and solar-weighted (capture) prices on hot vs normal days."""
    obs = daily_obs(era5, tz)
    j = era5[["T", "GHI"]].join(price, how="inner").dropna()
    j["date"] = local_date(j.index, tz)
    j["hour"] = (j.index - pd.Timedelta(minutes=30)).tz_convert(tz).hour
    j = j[pd.DatetimeIndex(j["date"]).month.isin(months)]
    j["hot"] = j["date"].map(obs["Tmax"] >= t_hot).astype(bool)
    j["year"] = pd.DatetimeIndex(j["date"]).year
    prof = j.groupby(["hot", "hour"])["price"].mean().unstack(0)
    rows = []
    for (y, hot), g in j.groupby(["year", "hot"]):
        mid = g[(g.hour >= 11) & (g.hour < 17)]
        rows.append({"year": y, "hot_day": hot, "n_days": g["date"].nunique(),
                     "mean_price": g["price"].mean(), "midday_price_11_17": mid["price"].mean(),
                     "solar_weighted_price": np.sum(g.price * g.GHI) / max(g.GHI.sum(), 1),
                     "share_hours_le_0": np.mean(g.price <= 0)})
    tab = pd.DataFrame(rows)
    return tab, prof
