"""Data loaders. Every loader returns tidy pandas objects with explicit units.

Conventions
-----------
* Weather series are indexed by the UTC timestamp at the END of each hour
  (Open-Meteo/ERA5 convention: radiation is the mean of the preceding hour,
  temperature, humidity and wind are instantaneous at that time).
* Wind is converted to m s-1, radiation is W m-2, temperature degC, RH %.
* Prices are EUR/MWh keyed to the same UTC hour-ending index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(Path(path).resolve().parent)
    return cfg


def _p(cfg: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def _wind_factor(cfg: dict) -> float:
    return 1 / 3.6 if cfg["data"].get("wind_units", "kmh") == "kmh" else 1.0


# --------------------------------------------------------------------------- ERA5
def load_era5(cfg: dict, site: str) -> pd.DataFrame:
    """Hourly ERA5 (Open-Meteo archive export) for a site, UTC hour-ending index."""
    f = _p(cfg, cfg["sites"][site]["era5"])
    df = pd.read_csv(f)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated()]
    out = pd.DataFrame(index=df.index)
    out["T"] = df["temperature_2m"].astype(float)
    out["RH"] = df["relative_humidity_2m"].astype(float).clip(1, 100)
    out["GHI"] = df["shortwave_radiation"].astype(float).clip(lower=0)
    if "direct_radiation" in df and "diffuse_radiation" in df:
        out["BHI"] = df["direct_radiation"].astype(float).clip(lower=0)
        out["DHI"] = df["diffuse_radiation"].astype(float).clip(lower=0)
    else:
        out["BHI"] = np.nan
        out["DHI"] = np.nan
    out["u10"] = df["wind_speed_10m"].astype(float) * _wind_factor(cfg)
    out["rain"] = df["precipitation"].astype(float).clip(lower=0) if "precipitation" in df else 0.0
    out["P_kpa"] = df.get("surface_pressure", pd.Series(1013.0, index=df.index)).astype(float) / 10
    out = out.interpolate(limit=3)
    return out


# --------------------------------------------------------------------------- ECMWF archive
def load_ecmwf(cfg: dict, site: str) -> pd.DataFrame | None:
    """ECMWF IFS forecasts issued 1, 2, 3 days earlier (Open-Meteo previous-runs API)."""
    rel = cfg["sites"][site].get("ecmwf")
    if not rel:
        return None
    df = pd.read_excel(_p(cfg, rel))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    wf = _wind_factor(cfg)
    out = {}
    for lead in cfg["forecast"]["leads"]:
        s = f"_previous_day{lead}"
        out[f"T_l{lead}"] = df[f"temperature_2m{s}"].astype(float)
        out[f"RH_l{lead}"] = df[f"relative_humidity_2m{s}"].astype(float)
        out[f"GHI_l{lead}"] = df[f"shortwave_radiation{s}"].astype(float)
        out[f"u10_l{lead}"] = df[f"wind_speed_10m{s}"].astype(float) * wf
    return pd.DataFrame(out).interpolate(limit=3)


# --------------------------------------------------------------------------- OMIE prices
def load_omie(cfg: dict, tz: str = "Europe/Madrid") -> pd.Series:
    """OMIE Spanish day-ahead price mapped to the UTC hour-ending index.

    OMIE hour k (1..24/25) is the local delivery interval [k-1, k). The UTC
    hour-ending stamp of that interval is local_start + 1 h converted to UTC.
    """
    df = pd.read_excel(_p(cfg, cfg["data"]["omie_prices"]), sheet_name="Hourly (long)")
    df.columns = ["date", "hour", "price"]
    df["date"] = pd.to_datetime(df["date"])
    out = []
    for d, g in df.groupby("date"):
        start = pd.Timestamp(d).tz_localize(tz)
        # use UTC arithmetic so 23 / 25-hour days are handled correctly
        t0 = start.tz_convert("UTC")
        idx = t0 + pd.to_timedelta(g["hour"].values, unit="h")
        out.append(pd.Series(g["price"].values.astype(float), index=idx))
    s = pd.concat(out).sort_index()
    s = s[~s.index.duplicated()]
    s.name = "price"
    return s


# --------------------------------------------------------------------------- CMIP6
def load_cmip6(cfg: dict, prefix: str) -> dict[str, pd.DataFrame]:
    f = _p(cfg, cfg["data"]["cmip6"])
    xl = pd.ExcelFile(f)
    out = {}
    for period in ("hist", "future"):
        name = [s for s in xl.sheet_names if s.startswith(f"{prefix}_{period}")]
        if not name:
            continue
        d = xl.parse(name[0])
        d["time"] = pd.to_datetime(d["time"])
        out[period] = d.set_index("time")
    return out


# --------------------------------------------------------------------------- field (digitised)
def load_field_daily(cfg: dict) -> pd.DataFrame:
    """Daily APV - control differences digitised from Fig. S3 of Fernandez Solas et al. (2026)."""
    d = pd.read_excel(_p(cfg, cfg["data"]["field_digitized"]), sheet_name="Fig S3 digitized")
    d = d.rename(columns={
        "dT_day (°C)": "dT_day", "dT_night (°C)": "dT_night",
        "dRH_day (pp)": "dRH_day", "dRH_night (pp)": "dRH_night",
        "dea_day (kPa)": "dea_day", "dea_night (kPa)": "dea_night"})
    d["date"] = pd.to_datetime(d["date"])
    return d
