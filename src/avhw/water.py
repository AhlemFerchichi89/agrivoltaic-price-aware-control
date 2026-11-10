"""Crop water use under the array (FAO-56 hourly Penman-Monteith).

Hourly reference evapotranspiration (Allen et al., 1998, Eq. 53) is computed
with the net radiation the crop actually receives: the shortwave reaching the
crop S_h(theta) and a net longwave loss reduced by the panels, which replace
part of the cold sky. Crop water use is

    ETc_h(theta) = k_w * ET0_h(theta)

where the single factor k_w (crop coefficient / irrigation efficiency) is
calibrated on the measured irrigation of the open-field control in one
season, and the other three irrigation totals of the Seville experiment
(control and agrivoltaic plots, two cycles) are used for validation.
Irrigation is a daily balance, sum over days of max(0, ETc_d - r_eff * rain_d).
"""
from __future__ import annotations

import numpy as np

from .canopy import es_kpa

SIGMA_MJ = 2.043e-10        # Stefan-Boltzmann, MJ m-2 h-1 K-4


def et0_hourly(t, rh, u10, rs_wm2, svf, rso_ratio, p_kpa=101.3):
    """FAO-56 hourly reference ET (mm/h) for given shortwave on the surface.

    rs_wm2   shortwave reaching the crop (W m-2)
    svf      sky-view factor (1 in the open field)
    rso_ratio  Rs/Rso clearness of the hour (open-field), for the longwave term
    """
    rs = np.asarray(rs_wm2) * 0.0036                     # MJ m-2 h-1
    u2 = np.maximum(np.asarray(u10), 0.1) * 4.87 / np.log(67.8 * 10 - 5.42)
    ea = es_kpa(t) * np.clip(rh, 0, 100) / 100
    rns = 0.77 * rs
    tk = np.asarray(t) + 273.16
    f_cloud = 1.35 * np.clip(rso_ratio, 0.3, 1.0) - 0.35
    rnl_open = SIGMA_MJ * tk ** 4 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.01))) * f_cloud
    rnl = rnl_open * svf                                  # panels replace part of the sky at ~air temperature
    rn = rns - rnl
    g = np.where(rs > 0.01, 0.1 * rn, 0.5 * rn)
    delta = 4098 * es_kpa(t) / (np.asarray(t) + 237.3) ** 2
    gamma = 0.000665 * p_kpa
    vpd = es_kpa(t) - ea
    num = 0.408 * delta * (rn - g) + gamma * 37 / tk * u2 * vpd
    den = delta + gamma * (1 + 0.34 * u2)
    return np.clip(num / den, 0, None)


def season_irrigation_mm(etc_mm, rain_mm, r_eff=0.8, day_ids=None):
    """Seasonal irrigation with a daily balance: rain offsets only the same day's crop water use."""
    etc_mm, rain_mm = np.asarray(etc_mm, float), np.asarray(rain_mm, float)
    if day_ids is None:
        return max(0.0, float(etc_mm.sum() - r_eff * rain_mm.sum()))
    _, inv = np.unique(day_ids, return_inverse=True)
    e = np.bincount(inv, etc_mm)
    r = np.bincount(inv, rain_mm)
    return float(np.clip(e - r_eff * r, 0, None).sum())
