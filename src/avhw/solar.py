"""Solar position (NOAA algorithm), beam/diffuse handling and transposition.

The NOAA general solar position equations (Spencer series) are accurate to
about 0.01-0.1 deg in elevation, well below the hourly resolution used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def solar_position(times_utc: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """Solar elevation and azimuth (deg, azimuth clockwise from north)."""
    t = times_utc.tz_convert("UTC") if times_utc.tz is not None else times_utc.tz_localize("UTC")
    doy = t.dayofyear.values
    hour = t.hour.values + t.minute.values / 60 + t.second.values / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = np.deg2rad(tst / 4 - 180)
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    zen = np.arccos(cosz)
    # azimuth clockwise from north
    sin_az = -np.sin(ha) * np.cos(decl) / np.maximum(np.sin(zen), 1e-9)
    cos_az = (np.sin(decl) - np.sin(phi) * cosz) / np.maximum(np.cos(phi) * np.sin(zen), 1e-9)
    az = np.rad2deg(np.arctan2(sin_az, cos_az)) % 360
    el = 90 - np.rad2deg(zen)
    # extraterrestrial horizontal irradiance for clearness index
    e0 = 1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g) + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g)
    i0h = 1361.0 * e0 * np.maximum(cosz, 0)
    return pd.DataFrame({"elev": el, "azim": az, "I0h": i0h}, index=times_utc)


def sun_position_hourly(index_hour_ending: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """Sun position at the middle of each hour (radiation is the preceding-hour mean)."""
    mid = index_hour_ending - pd.Timedelta(minutes=30)
    sp = solar_position(mid, lat, lon)
    sp.index = index_hour_ending
    return sp


def erbs_diffuse_fraction(kt: np.ndarray) -> np.ndarray:
    """Erbs et al. (1982) diffuse fraction as a function of clearness index."""
    kt = np.clip(kt, 0, 1)
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.80,
                           0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                           0.165))
    return kd


def split_beam_diffuse(ghi: np.ndarray, sp: pd.DataFrame, bhi=None, dhi=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (BHI, DHI, DNI). Uses the given split when available, Erbs otherwise."""
    ghi = np.asarray(ghi, float)
    el = sp["elev"].values
    sin_el = np.sin(np.deg2rad(np.maximum(el, 0)))
    if bhi is None or np.all(np.isnan(bhi)):
        kt = np.where(sp["I0h"].values > 10, ghi / np.maximum(sp["I0h"].values, 1), 0)
        kd = erbs_diffuse_fraction(kt)
        dhi = ghi * kd
        bhi = ghi - dhi
    bhi = np.asarray(bhi, float).copy()
    dhi = np.asarray(dhi, float).copy()
    low = el < 3  # beam is ill-defined near the horizon: treat it as diffuse
    dhi[low] += bhi[low]
    bhi[low] = 0
    dni = np.where(sin_el > 0, bhi / np.maximum(sin_el, 0.05), 0)
    dni = np.clip(dni, 0, 1100)
    bhi = dni * sin_el
    dhi = np.clip(ghi - bhi, 0, None)
    return bhi, dhi, dni


def sun_vector(sp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vector toward the sun: (east, north, up)."""
    el = np.deg2rad(sp["elev"].values)
    az = np.deg2rad(sp["azim"].values)
    return np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)
