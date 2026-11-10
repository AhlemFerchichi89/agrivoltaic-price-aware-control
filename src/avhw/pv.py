"""Plane-of-array irradiance and PV power for every candidate tilt."""
from __future__ import annotations

import numpy as np


def poa_irradiance(geo, bhi, dhi, dni, ghi, sun_n, sun_up, albedo, iam_b0):
    """Effective plane-of-array irradiance (W m-2), shape (hours, tilts).

    Beam: DNI * cos(AOI) * IAM * (1 - inter-row shading).
    Diffuse: isotropic sky (1 + cos th)/2; ground-reflected: albedo * GHI (1 - cos th)/2.
    """
    cosa = geo.cos_aoi(sun_n, sun_up)
    cpos = np.clip(cosa, 0, 1)
    iam = np.where(cpos > 0.05, 1 - iam_b0 * (1 / np.maximum(cpos, 0.05) - 1), 0)
    iam = np.clip(iam, 0, 1)
    rs = geo.row_shade_fraction(sun_n, sun_up)
    beam = dni[:, None] * cpos * iam * (1 - rs)
    ct = np.cos(geo.th)[None, :]
    sky = dhi[:, None] * (1 + ct) / 2
    grd = albedo * ghi[:, None] * (1 - ct) / 2
    return beam + sky + grd


def module_temperature(poa, t_air, u10, u0, u1, cooling_C=0.0):
    """Faiman (2008) module temperature with an optional canopy-cooling offset.

    The canopy-cooling term (Hosseini et al., 2026) lowers module temperature
    over a transpiring crop. It is applied in proportion to irradiance so that
    it vanishes at night.
    """
    u = np.maximum(u10, 0.5)[:, None]
    tm = t_air[:, None] + poa / (u0 + u1 * u)
    if np.ndim(cooling_C) == 0:
        cool = cooling_C * np.clip(poa / 1000.0, 0, 1)
    else:
        cool = np.asarray(cooling_C)[:, None] * np.clip(poa / 1000.0, 0, 1)
    return tm - cool


def pv_power_per_kwp(poa, t_mod, gamma, inv_eff, losses):
    """AC power per installed kWp (kW/kWp)."""
    p = poa / 1000.0 * (1 + gamma * (t_mod - 25.0)) * inv_eff * (1 - losses)
    return np.clip(p, 0, None)
