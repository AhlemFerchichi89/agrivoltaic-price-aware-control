"""Crop response: light and heat.

Seasonal yield (t/ha) of a crop grown under the array:

    Y = Y_pot * (L / L_open) ** alpha * exp(-lambda * H)

L      seasonal shortwave reaching the crop (MJ m-2), L_open the open-field value
alpha  light-response exponent (0 = shade tolerant, 1 = proportional to light)
H      heat-stress degree hours, sum over hours of max(0, T_canopy - T_crit)
lambda yield loss per degree-hour of heat stress

Y_pot is anchored so the open field reproduces the measured control yield in
the reference season. alpha is calibrated on the measured APV / control yield
ratio (Table 5 of D2.2), with the two cycles used for leave-one-cycle-out
validation. The published GDD-DLI regressions (Table S1 of Fernandez Solas et
al., 2026) are also evaluated as an independent check.
"""
from __future__ import annotations

import numpy as np

PAR_FACTOR = 0.45 * 4.57e-6 * 3600   # W m-2 for one hour -> mol PAR m-2


def heat_degree_hours(t_canopy, t_crit):
    return np.clip(t_canopy - t_crit, 0, None)


def yield_t_ha(y_pot, light_rel, H, alpha, lam):
    return y_pot * np.power(np.clip(light_rel, 1e-6, None), alpha) * np.exp(-lam * H)


def calibrate_alpha(yield_ratio, light_ratio, dH, lam):
    """alpha such that (L_apv/L_ctrl)^alpha * exp(-lam * dH) = Y_apv / Y_ctrl."""
    return float(np.log(yield_ratio * np.exp(lam * dH)) / np.log(light_ratio))


# Table S1 regressions (Fernandez Solas et al., 2026); GDD in degC.day, DLI in mol m-2 d-1
def table_s1_fresh_weight(gdd, dli):
    return -226.573 - 0.0430 * dli + 1.331 * gdd


def table_s1_dry_weight(gdd, dli):
    return -9.856 + 0.00556 * dli + 0.0396 * gdd


def gdd(t_daily_mean, t_base):
    return np.clip(np.asarray(t_daily_mean) - t_base, 0, None).sum()
