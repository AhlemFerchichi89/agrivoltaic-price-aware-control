"""Hourly simulation of one crop season under every operating strategy.

For every hour h and every candidate tilt k the physics is evaluated once
(`Physics`): beam shading of the crop, light on the crop, canopy temperature
and heat stress, and PV power. Strategies then only choose a tilt index per
hour, and the season is scored on the true (ERA5) weather.

Strategies
----------
open_field   no panels (crop reference)
pv_ref       ground-mounted PV reference for the LER (fixed tilt, no crop)
S1_fixed     fixed tilt (as operated at the site)
S2_tracking  maximise PV power
S3_light     maximise PV power subject to beam shading <= f_max
S4_rule      protective tilt in the afternoon window of forecast stress days (p_hat >= p*), S3 otherwise
S5_rule      as S4 with the observed stress days (perfect foresight)
EV_noinfo    hourly expected-value controller without heat information (economic light sharing)
EV_clim      expected-value controller with climatological heat expectations
MPC_reactive expected-value controller with a persistence nowcast of the last observed hour
MPC_det      expected-value controller with the deterministic (median) day-ahead forecast
MPC_prob     expected-value controller with the probabilistic day-ahead forecast (proposed)
MPC_oracle   expected-value controller with perfect knowledge of the weather (upper bound)

The expected-value controller maximises, hour by hour, the linearised
contribution of the tilt to the season value:
  V_h(k) = pi_h E_h(k) + w * P_c * Y_ref * (alpha * S_h(k) / L_ref - lambda * E[H_h(k)])
where E[H_h(k)] is the expected heat-stress degree-hours under the forecast
distribution of air temperature (closed form for a Gaussian).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import canopy, crop, pv, solar, water
from .forecast import expected_exceedance


# =============================================================================
@dataclass
class Physics:
    index: pd.DatetimeIndex
    fsh: np.ndarray        # (H, K) beam shading of the crop
    sw: np.ndarray         # (H, K) shortwave on the crop, W m-2
    sw_open: np.ndarray    # (H,)
    power: np.ndarray      # (H, K) kW per kWp
    t_sun: np.ndarray      # (H, K) canopy temperature, sunlit patch
    t_sh: np.ndarray       # (H, K) canopy temperature, shaded patch
    t_open: np.ndarray     # (H,)   canopy temperature, open field
    t_crit: float
    daytime: np.ndarray    # (H,) bool
    et0: np.ndarray = None       # (H, K) reference ET with the crop's radiation, mm/h
    et0_open: np.ndarray = None  # (H,)
    rain: np.ndarray = None      # (H,) mm

    @property
    def day_ids(self):
        return (self.index - pd.Timedelta(minutes=30)).floor("D").asi8

    def stress(self, sigma=None):
        """Expected heat-stress degree-hours per hour and tilt (area-weighted)."""
        if sigma is None:
            hs = np.clip(self.t_sun - self.t_crit, 0, None)
            hh = np.clip(self.t_sh - self.t_crit, 0, None)
        else:
            s = np.asarray(sigma)[:, None]
            hs = expected_exceedance(self.t_sun, s, self.t_crit)
            hh = expected_exceedance(self.t_sh, s, self.t_crit)
        return (1 - self.fsh) * hs + self.fsh * hh

    def stress_open(self, sigma=None):
        if sigma is None:
            return np.clip(self.t_open - self.t_crit, 0, None)
        return expected_exceedance(self.t_open, np.asarray(sigma), self.t_crit)


def compute_physics(wx: pd.DataFrame, lat, lon, geo, micro, cfg, crop_cfg, cooling=True,
                    t_crit_shift=0.0) -> Physics:
    """Evaluate the physics on a weather frame (truth or forecast)."""
    syscfg, leaf = cfg["system"], cfg["leaf"]
    sp = solar.sun_position_hourly(wx.index, lat, lon)
    bhi = wx["BHI"].values if "BHI" in wx else None
    dhi = wx["DHI"].values if "DHI" in wx else None
    ghi = wx["GHI"].values
    bhi, dhi, dni = solar.split_beam_diffuse(ghi, sp, bhi, dhi)
    ghi = bhi + dhi
    _, sn, su = solar.sun_vector(sp)
    fsh = geo.ground_shade_fraction(sn, su)
    svf = geo.svf[None, :]
    sw = bhi[:, None] * (1 - fsh) + dhi[:, None] * svf
    t, rh, u = wx["T"].values, wx["RH"].values, wx["u10"].values
    pk = wx["P_kpa"].values if "P_kpa" in wx else np.full(len(wx), 101.3)
    dT = micro.predict_hourly(fsh, ghi, t, rh, u)
    t_apv = t[:, None] + dT
    sw_sun = bhi[:, None] + dhi[:, None] * svf
    sw_sh = dhi[:, None] * svf + 0 * fsh
    c2 = lambda x: x[:, None]
    if crop_cfg["stress_on"] == "leaf":
        t_sun = canopy.leaf_temperature(t_apv, c2(rh), c2(u), sw_sun, svf, c2(pk), leaf)
        t_sh = canopy.leaf_temperature(t_apv, c2(rh), c2(u), sw_sh, svf, c2(pk), leaf)
        t_open = canopy.leaf_temperature(t, rh, u, ghi, 1.0, pk, leaf)
    else:
        t_sun = t_sh = t_apv
        t_open = t
    poa = pv.poa_irradiance(geo, bhi, dhi, dni, ghi, sn, su, syscfg["albedo"], syscfg["iam_b0"])
    tm = pv.module_temperature(poa, t, u, syscfg["faiman_u0"], syscfg["faiman_u1"],
                               syscfg["canopy_cooling_C"] if cooling else 0.0)
    pw = pv.pv_power_per_kwp(poa, tm, syscfg["gamma_pmp"], syscfg["inverter_eff"], syscfg["other_losses"])
    rso = np.where(sp["I0h"].values > 10, ghi / np.maximum(0.75 * sp["I0h"].values, 1), 0.6)
    et0 = water.et0_hourly(t_apv, c2(rh), c2(u), sw, svf, c2(rso), c2(pk))
    et0_open = water.et0_hourly(t, rh, u, ghi, 1.0, rso, pk)
    rain = wx["rain"].values if "rain" in wx else np.zeros(len(wx))
    return Physics(wx.index, fsh, sw, ghi, pw, t_sun, t_sh, t_open,
                   crop_cfg["t_crit_C"] + t_crit_shift, ghi > 20, et0, et0_open, rain)


# =============================================================================
# tilt-selection rules
def idx_fixed(n, tilts, tilt):
    return np.full(n, int(np.argmin(np.abs(tilts - tilt))))


def idx_tracking(ph: Physics):
    return np.argmax(ph.power, axis=1)


def idx_light_sharing(ph: Physics, fmax):
    p = np.where(ph.fsh <= fmax, ph.power, -np.inf)
    feasible = np.isfinite(p).any(axis=1)
    return np.where(feasible, np.argmax(p, axis=1), np.argmin(ph.fsh, axis=1))


def idx_protect(ph: Physics):
    return np.argmax(ph.fsh, axis=1)


def protect_mask(index, tz, window, daytime, day_flags: pd.Series):
    loc = (index - pd.Timedelta(minutes=30)).tz_convert(tz)
    d = loc.normalize().tz_localize(None)
    flag = day_flags.reindex(d).fillna(False).values.astype(bool)
    in_win = (loc.hour >= window[0]) & (loc.hour < window[1])
    return flag & in_win & daytime


def idx_ev(ph: Physics, price, kwp, w_crop, pc_eur_kg, y_ref_kg, alpha, lam, l_ref, sigma=None,
           water_eur_m3=0.0, k_w=1.0):
    """Hourly expected-value tilt choice (see module docstring)."""
    e_rev = price[:, None] * ph.power * kwp / 1000.0           # EUR/ha in the hour
    light = alpha * ph.sw / l_ref
    heat = lam * ph.stress(sigma) if lam > 0 else 0.0
    w_cost = water_eur_m3 * 10.0 * k_w * ph.et0 if water_eur_m3 > 0 else 0.0   # 1 mm = 10 m3/ha
    v = e_rev + w_crop * pc_eur_kg * y_ref_kg * (light - heat) - w_cost
    return np.argmax(v, axis=1)


# =============================================================================
def score(ph_true: Physics, idx, price, kwp, crop_cfg, econ, alpha, lam, y_pot,
          ph_bid: Physics | None = None, idx_bid=None, kwp_ref=None, water_eur_m3=0.0, k_w=1.0, r_eff=0.8):
    """Evaluate a tilt schedule on the true weather."""
    r = np.arange(len(idx))
    e = ph_true.power[r, idx] * kwp                                 # kWh/ha per hour
    rev = np.nansum(price * e) / 1000.0
    imb = 0.0
    if ph_bid is not None:
        ib = idx if idx_bid is None else idx_bid
        e_bid = ph_bid.power[r, ib] * kwp
        imb = econ["imbalance_eur_mwh"] * np.sum(np.abs(e - e_bid)) / 1000.0
    L = ph_true.sw[r, idx].sum()
    L_open = ph_true.sw_open.sum()
    H = ph_true.stress()[r, idx].sum()
    Y = crop.yield_t_ha(y_pot, L / L_open, H, alpha, lam)
    crop_rev = Y * 1000 * crop_cfg["price_eur_kg"]
    irr = water.season_irrigation_mm(k_w * ph_true.et0[r, idx], ph_true.rain, r_eff, ph_true.day_ids)
    w_cost = water_eur_m3 * 10.0 * irr
    return {"energy_MWh_ha": e.sum() / 1000, "elec_rev_eur_ha": rev, "imbalance_eur_ha": imb,
            "light_rel": L / L_open, "heat_degh": H, "yield_t_ha": Y, "crop_rev_eur_ha": crop_rev,
            "irrigation_mm": irr, "water_cost_eur_ha": w_cost,
            "value_eur_ha": rev - imb + crop_rev - w_cost,
            "capture_price": rev / max(e.sum() / 1000, 1e-9),
            "mean_fsh_day": float(ph_true.fsh[r, idx][ph_true.daytime].mean())}


def open_field_score(ph_true: Physics, crop_cfg, lam, y_pot, water_eur_m3=0.0, k_w=1.0, r_eff=0.8):
    H = ph_true.stress_open().sum()
    Y = crop.yield_t_ha(y_pot, 1.0, H, 0.0, lam)
    irr = water.season_irrigation_mm(k_w * ph_true.et0_open, ph_true.rain, r_eff, ph_true.day_ids)
    w_cost = water_eur_m3 * 10.0 * irr
    return {"energy_MWh_ha": 0.0, "elec_rev_eur_ha": 0.0, "imbalance_eur_ha": 0.0, "light_rel": 1.0,
            "heat_degh": H, "yield_t_ha": Y, "crop_rev_eur_ha": Y * 1000 * crop_cfg["price_eur_kg"],
            "irrigation_mm": irr, "water_cost_eur_ha": w_cost,
            "value_eur_ha": Y * 1000 * crop_cfg["price_eur_kg"] - w_cost, "capture_price": np.nan}
