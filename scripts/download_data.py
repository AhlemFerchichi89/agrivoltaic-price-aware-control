#!/usr/bin/env python
"""Download every open input dataset used by the pipeline into data/raw/.

    pip install requests pandas openpyxl OMIEData
    python scripts/download_data.py --site seville          # ERA5, ECMWF, CMIP6
    python scripts/download_data.py --site hail --lat 27.52 --lon 41.69 --tz Asia/Riyadh --no-ecmwf
    python scripts/download_data.py --omie                   # Spanish day-ahead prices

Sources (all open, no account needed):
  ERA5 hourly reanalysis       Open-Meteo historical weather API (https://open-meteo.com/en/docs/historical-weather-api)
  ECMWF IFS past forecasts     Open-Meteo previous-runs API (https://open-meteo.com/en/docs/previous-runs-api)
  CMIP6 HighResMIP daily       Open-Meteo climate API (https://open-meteo.com/en/docs/climate-api)
  Day-ahead prices (Spain)     OMIE (https://www.omie.es), via the OMIEData package
The field data of the Seville experiment are taken from the open-access
supplementary material of Fernandez Solas et al. (2026); see data/field/README.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SITES = {"seville": (37.39, -5.98, "Europe/Madrid"), "almeria": (36.84, -2.46, "Europe/Madrid"),
         "hail": (27.52, 41.69, "Asia/Riyadh")}


def get(url, params, tries=6):
    for k in range(tries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(30 * (k + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"too many retries for {url}")


def era5(name, lat, lon, y0=2015, y1=2025):
    v = "temperature_2m,relative_humidity_2m,dew_point_2m,shortwave_radiation,direct_radiation,diffuse_radiation,wind_speed_10m,precipitation,surface_pressure"
    frames = []
    for y in range(y0, y1 + 1):
        js = get("https://archive-api.open-meteo.com/v1/archive",
                 dict(latitude=lat, longitude=lon, start_date=f"{y}-01-01", end_date=f"{y}-12-31", hourly=v,
                      models="era5", timezone="GMT"))
        frames.append(pd.DataFrame(js["hourly"]))
        print("ERA5", name, y)
        time.sleep(2)
    f = RAW / f"era5_{name}_hourly_{y0}_{y1}.csv"
    pd.concat(frames).to_csv(f, index=False)
    print("saved", f)


def ecmwf(name, lat, lon, start="2024-09-01", end="2025-07-31"):
    base = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation", "wind_speed_10m"]
    hv = [f"{b}_previous_day{d}" for b in base for d in (1, 2, 3)] + base
    js = get("https://previous-runs-api.open-meteo.com/v1/forecast",
             dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(hv),
                  models="ecmwf_ifs025", timezone="GMT"))
    f = RAW / f"ECMWF_previous_runs_{name}_{start[:7]}_{end[:7]}.xlsx"
    pd.DataFrame(js["hourly"]).to_excel(f, index=False)
    print("saved", f)


def cmip6(names):
    models = ["EC_Earth3P_HR", "MPI_ESM1_2_XR", "CMCC_CM2_VHR4"]
    daily = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,shortwave_radiation_sum,wind_speed_10m_mean,precipitation_sum"
    f = RAW / "CMIP6_Seville_Almeria_2005-2024_2041-2050.xlsx"
    with pd.ExcelWriter(f) as xw:
        for name in names:
            lat, lon, _ = SITES[name]
            for tag, (a, b) in {"hist_2005_2024": ("2005-01-01", "2024-12-31"), "future_2041_2050": ("2041-01-01", "2050-12-31")}.items():
                df = None
                for m in models:
                    js = get("https://climate-api.open-meteo.com/v1/climate",
                             dict(latitude=lat, longitude=lon, start_date=a, end_date=b, models=m, daily=daily))
                    d = pd.DataFrame(js["daily"]).set_index("time")
                    d.columns = [f"{c}_{m}" if not c.endswith(m) else c for c in d.columns]
                    df = d if df is None else df.join(d)
                    time.sleep(10)
                df.reset_index().to_excel(xw, sheet_name=f"{name}_{tag}"[:31], index=False)
                print("CMIP6", name, tag)
    print("saved", f)


def omie(y0=2015, y1=2025):
    from OMIEData.DataImport.omie_marginalprice_importer import OMIEMarginalPriceFileImporter
    frames = []
    for y in range(y0, y1 + 1):
        df = OMIEMarginalPriceFileImporter(date_ini=dt.datetime(y, 5, 1), date_end=dt.datetime(y, 9, 30)).read_to_dataframe(verbose=False)
        frames.append(df)
        print("OMIE", y)
    p = pd.concat(frames)
    sp = p[p["CONCEPT"].astype(str).str.contains("SP")]
    hcols = [c for c in sp.columns if str(c).startswith("H")]
    long = sp.melt(id_vars=["DATE"], value_vars=hcols, var_name="Hour", value_name="Price (EUR/MWh)").dropna()
    long["Hour"] = long["Hour"].str.replace("H", "").astype(int)
    long = long.rename(columns={"DATE": "Date"}).sort_values(["Date", "Hour"])
    f = RAW / f"OMIE_Spain_hourly_prices_May-Sep_{y0}-{y1}.xlsx"
    with pd.ExcelWriter(f) as xw:
        long.to_excel(xw, sheet_name="Hourly (long)", index=False)
    print("saved", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITES))
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--tz")
    ap.add_argument("--no-ecmwf", action="store_true")
    ap.add_argument("--cmip6", nargs="*", help="sites for CMIP6, e.g. seville almeria")
    ap.add_argument("--omie", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.site:
        lat, lon, _ = SITES[a.site]
        lat, lon = a.lat or lat, a.lon or lon
        era5(a.site, lat, lon)
        if not a.no_ecmwf:
            ecmwf(a.site, lat, lon)
    if a.cmip6:
        cmip6(a.cmip6)
    if a.omie:
        omie()
