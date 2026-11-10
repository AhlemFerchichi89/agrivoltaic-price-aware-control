"""Microclimate under the panels and crop leaf temperature.

Two pieces:
1. `Microclimate`: an empirical model of the air-temperature difference
   between agrivoltaic and open-field plots,
       dT = b0 + b1 * f_sh * G + b2 * VPD + b3 * u           (day)
       dT = c0 + c2 * VPD + c3 * u                            (night)
   calibrated on the daily day/night differences measured in Seville
   (Fig. S3 of Fernandez Solas et al., 2026) with cross-validation between
   the two crop cycles.
2. `leaf_temperature`: a linearised leaf energy balance (Campbell & Norman,
   1998, Eq. 14.6) that turns shortwave load, air temperature, humidity and
   wind into a sunlit-canopy leaf temperature. Shading changes the absorbed
   shortwave and the longwave exchange (the panel replaces part of the cold
   sky), so the protective effect of a tilt is resolved physically, not
   assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA = 5.670e-8
CP = 29.3          # J mol-1 K-1
GAMMA = 6.66e-4    # psychrometric constant / pressure, 1/K


def es_kpa(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def vpd_kpa(t, rh):
    return es_kpa(t) * (1 - np.clip(rh, 0, 100) / 100)


def wind_at(u10, z, z0):
    return np.maximum(u10, 0.2) * np.log(z / z0) / np.log(10 / z0)


def sky_emissivity(t, rh):
    """Brutsaert (1975) clear-sky emissivity."""
    ea_hpa = 10 * es_kpa(t) * np.clip(rh, 0, 100) / 100
    return 1.24 * (ea_hpa / (t + 273.15)) ** (1 / 7)


def leaf_temperature(t_air, rh, u10, sw_on_crop, svf, p_kpa, leaf: dict):
    """Leaf temperature (degC) of the upper canopy.

    All inputs broadcast; typical shapes are (hours, 1) for weather and
    (hours, tilts) for shortwave and (1, tilts) for the sky-view factor.
    """
    tk = t_air + 273.15
    u = wind_at(u10, max(leaf["crop_height_m"], 0.1) + 1.0, leaf["z0_m"])  # ~1 m above crop
    d = 0.72 * leaf["width_m"]
    g_ha = 1.4 * 0.135 * np.sqrt(u / d)                     # boundary-layer heat conductance
    g_va = 1.4 * 0.147 * np.sqrt(u / d)
    vpd = vpd_kpa(t_air, rh)
    gs = leaf["gs_max_mol"] / (1 + vpd / leaf["d0_kpa"])    # stomatal closure with VPD
    gs = np.where(sw_on_crop > 5, gs, 0.02)                 # stomata near-closed at night
    g_v = 1 / (1 / gs + 1 / g_va)
    eps = leaf["emissivity"]
    g_r = 4 * eps * SIGMA * tk ** 3 / CP
    g_hr = g_ha + g_r
    eps_sky = sky_emissivity(t_air, rh)
    # longwave in: half hemisphere from sky (partly replaced by panels at ~air temperature), half from ground
    lw_down = (svf * eps_sky + (1 - svf)) * SIGMA * tk ** 4
    lw_up = SIGMA * tk ** 4
    # absorbed radiation averaged over both leaf sides (C&N Eq. 11.14): the upper side
    # sees the incoming shortwave, the lower side the ground-reflected part (albedo 0.2)
    r_abs = leaf["abs_sw"] * 0.5 * sw_on_crop * (1 + 0.2) + eps * 0.5 * (lw_down + lw_up)
    r_ni = r_abs - eps * SIGMA * tk ** 4                    # isothermal net radiation
    s = 4098 * es_kpa(t_air) / (t_air + 237.3) ** 2 / p_kpa  # slope / pressure, 1/K
    gstar = GAMMA * g_hr / g_v
    dt = gstar / (s + gstar) * (r_ni / (CP * g_hr) - vpd / (p_kpa * gstar))
    return t_air + dt


# =============================================================================
class Microclimate:
    """Linear APV - open field air-temperature model (day / night)."""

    day_terms = ["const", "fshG", "vpd", "u"]
    night_terms = ["const", "vpd", "u"]

    def __init__(self, beta_day=None, beta_night=None):
        self.beta_day = None if beta_day is None else np.asarray(beta_day, float)
        self.beta_night = None if beta_night is None else np.asarray(beta_night, float)

    @staticmethod
    def daily_features(hourly: pd.DataFrame, fsh_field: np.ndarray, tz: str) -> pd.DataFrame:
        """Day- and night-time daily means of the predictors.

        `hourly` holds T, RH, GHI, u10 (UTC hour-ending), `fsh_field` the beam
        shading fraction at the experimental tilt.
        """
        h = hourly.copy()
        h["fshG"] = fsh_field * h["GHI"].values / 1000.0
        h["vpd"] = vpd_kpa(h["T"].values, h["RH"].values)
        h["u"] = h["u10"].values
        loc = h.index.tz_convert(tz) - pd.Timedelta(minutes=30)
        h["date"] = loc.normalize().tz_localize(None)
        h["is_day"] = h["GHI"] > 20
        day = h[h.is_day].groupby("date")[["fshG", "vpd", "u"]].mean().add_suffix("_day")
        night = h[~h.is_day].groupby("date")[["vpd", "u"]].mean().add_suffix("_night")
        return day.join(night, how="outer")

    @staticmethod
    def _design(df, terms, suffix):
        cols = [np.ones(len(df))] + [df[f"{t}{suffix}"].values for t in terms[1:]]
        return np.column_stack(cols)

    def fit(self, feats: pd.DataFrame, obs: pd.DataFrame):
        d = feats.join(obs.set_index("date")[["dT_day", "dT_night"]], how="inner")
        dd = d.dropna(subset=["dT_day", "fshG_day", "vpd_day", "u_day"])
        X = self._design(dd, self.day_terms, "_day")
        self.beta_day = np.linalg.lstsq(X, dd["dT_day"].values, rcond=None)[0]
        dn = d.dropna(subset=["dT_night", "vpd_night", "u_night"])
        Xn = self._design(dn, self.night_terms, "_night")
        self.beta_night = np.linalg.lstsq(Xn, dn["dT_night"].values, rcond=None)[0]
        return self

    def predict_daily(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=feats.index)
        out["dT_day_hat"] = self._design(feats, self.day_terms, "_day") @ self.beta_day
        out["dT_night_hat"] = self._design(feats, self.night_terms, "_night") @ self.beta_night
        return out

    def predict_hourly(self, fsh, ghi, t, rh, u10):
        """Hourly dT, broadcasting over tilts (fsh has shape (hours, tilts))."""
        vpd = vpd_kpa(t, rh)
        g = ghi / 1000.0
        bd, bn = self.beta_day, self.beta_night
        day = bd[0] + bd[1] * fsh * g[:, None] + bd[2] * vpd[:, None] + bd[3] * u10[:, None]
        night = bn[0] + bn[1] * vpd + bn[2] * u10
        return np.where((ghi > 20)[:, None], day, night[:, None])


def cv_metrics(obs, pred) -> dict:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    m = ~(np.isnan(obs) | np.isnan(pred))
    o, p = obs[m], pred[m]
    r = np.corrcoef(o, p)[0, 1] if len(o) > 2 and np.std(p) > 0 else np.nan
    return {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean((o - p) ** 2))),
            "bias": float(np.mean(p - o)), "r": float(r),
            "obs_mean": float(o.mean()), "pred_mean": float(p.mean())}
