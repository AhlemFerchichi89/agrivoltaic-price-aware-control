"""Unit tests of the physical and statistical building blocks.

Run with `pytest -q` or, without pytest, `python tests/test_core.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avhw import canopy, crop, forecast as fc, geometry, solar  # noqa: E402

LEAF = {"width_m": 0.1, "gs_max_mol": 0.4, "d0_kpa": 1.5, "abs_sw": 0.5, "emissivity": 0.97,
        "crop_height_m": 0.3, "z0_m": 0.03}


def test_solar_noon_elevation():
    # summer solstice, Seville: noon elevation = 90 - 37.39 + 23.44
    t = pd.date_range("2025-06-21 11:00", "2025-06-21 15:00", freq="1min", tz="UTC")
    sp = solar.solar_position(t, 37.39, -5.98)
    assert abs(sp.elev.max() - (90 - 37.39 + 23.44)) < 0.2
    assert abs(sp.azim.iloc[sp.elev.values.argmax()] - 180) < 2


def test_shade_fraction_bounds_and_sun_facing_tilt():
    g = geometry.ArrayGeometry(3.0, 4.5, 0.32, np.arange(0, 66), n_ground=20)
    el = np.deg2rad(np.array([30.0, 50.0, 70.0]))
    n, up = -np.cos(el), np.sin(el)                    # sun due south
    f = g.ground_shade_fraction(n, up)
    assert np.all((f >= 0) & (f <= 1))
    # the tilt normal to the sun casts the largest shadow and receives the most beam
    best = g.crossing_tilt(n, up)
    assert np.all(np.abs(g.tilts[f.argmax(1)] - np.round(best)) <= 1)
    assert np.all(np.abs(g.tilts[g.cos_aoi(n, up).argmax(1)] - np.round(best)) <= 1)
    # a strip of width w normal to the sun casts a ground shadow of length w / sin(elevation)
    w, P = 0.32 * 4.5, 4.5
    expected = np.minimum(w / np.sin(el), P) / P
    best_tilt_shade = f[np.arange(3), np.round(best).astype(int)]
    assert np.allclose(best_tilt_shade, expected, atol=0.01)


def test_svf_independent_of_height_for_infinite_rows():
    a = geometry.ArrayGeometry(2.0, 4.5, 0.32, np.array([0, 30, 60]), n_ground=60).svf
    b = geometry.ArrayGeometry(5.0, 4.5, 0.32, np.array([0, 30, 60]), n_ground=60).svf
    assert np.allclose(a, b, atol=0.01)
    # flat panels block exactly the ground-coverage fraction of isotropic diffuse light
    assert abs(a[0] - (1 - 0.32)) < 0.01


def test_leaf_temperature_shade_cooler_than_sun():
    t = np.array([38.0])
    sun = canopy.leaf_temperature(t, 25.0, 2.0, np.array([900.0]), 1.0, 101.3, LEAF)
    sh = canopy.leaf_temperature(t, 25.0, 2.0, np.array([100.0]), 0.7, 101.3, LEAF)
    assert sun[0] > sh[0] + 2


def test_expected_exceedance_matches_monte_carlo():
    rng = np.random.default_rng(0)
    x = rng.normal(31, 2, 400000)
    assert abs(fc.expected_exceedance(31.0, 2.0, 32.0) - np.mean(np.clip(x - 32, 0, None))) < 0.01


def test_crps_gaussian_matches_ensemble():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, (1, 20000))
    assert abs(fc.crps_gaussian(0.0, 1.0, 0.5) - fc.crps_ensemble(s, np.array([0.5]))[0]) < 0.01


def test_emos_recovers_parameters():
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({"Tmax_fc": rng.uniform(20, 42, n), "RH_fc": rng.uniform(20, 80, n), "GHI_fc": rng.uniform(5, 9, n)})
    y = 1.0 + 0.95 * df.Tmax_fc + rng.normal(0, 0.8, n)
    m = fc.EMOS().fit(df, y.values)
    mu, sd = m.predict(df)
    assert np.mean(np.abs(mu - (1.0 + 0.95 * df.Tmax_fc))) < 0.2
    assert abs(sd.mean() - 0.8) < 0.15


def test_alpha_calibration_roundtrip():
    a = crop.calibrate_alpha(0.8 ** 0.6 * np.exp(-0.001 * -50), 0.8, -50, 0.001)
    assert abs(a - 0.6) < 1e-9


def test_omie_hours_map_to_utc_hour_ending():
    # 10 May (CEST = UTC+2): OMIE hour 13 is 12:00-13:00 local = 10:00-11:00 UTC -> stamp 11:00 UTC
    start = pd.Timestamp("2025-05-10").tz_localize("Europe/Madrid").tz_convert("UTC")
    assert start + pd.Timedelta(hours=13) == pd.Timestamp("2025-05-10 11:00", tz="UTC")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
