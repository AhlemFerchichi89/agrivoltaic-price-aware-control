"""Season-level experiments: context building, strategy runs and value of information."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import crop, forecast as fc, simulate as sim
from .canopy import Microclimate
from .geometry import ArrayGeometry

STRATEGIES = ["S1_fixed", "S2_tracking", "S3_light", "S4_rule", "S5_rule", "EV_energy_light", "EV_noinfo", "EV_clim",
              "MPC_reactive", "MPC_det", "MPC_prob", "MPC_oracle"]


@dataclass
class Context:
    cfg: dict
    site: str
    era5: pd.DataFrame
    price: pd.Series
    geo: ArrayGeometry
    micro: Microclimate
    crop_name: str
    lam: float
    alpha: float = 0.5
    y_pot: float = 20.0
    l_ref: float = 1.0
    y_ref_kg: float = 20000.0
    ecmwf: pd.DataFrame | None = None
    errlib: fc.ErrorLibrary | None = None
    emos_all: fc.EMOS | None = None
    hcorr_all: fc.HourlyCorrector | None = None
    obs_daily: pd.DataFrame | None = None
    nowcast_sd: np.ndarray | None = None
    nowcast_step: np.ndarray | None = None
    cooling: bool = True
    flat_price: bool = False
    t_crit_shift: float = 0.0
    weather_override: pd.DataFrame | None = None   # e.g. future climate
    k_w: float = 1.0                # crop water-use factor (calibrated on field irrigation)
    water_price: float = 0.0        # EUR per m3 of irrigation water
    price_year_map: dict = field(default_factory=dict)

    @property
    def scfg(self):
        return self.cfg["sites"][self.site]

    @property
    def ccfg(self):
        return self.cfg["crops"][self.crop_name]


# =============================================================================
def season_index(ctx: Context, year: int) -> pd.DatetimeIndex:
    s, e = ctx.scfg.get("season") or ctx.ccfg["season"]
    tz = ctx.scfg["tz"]
    t0 = pd.Timestamp(f"{year}-{s}").tz_localize(tz).tz_convert("UTC") + pd.Timedelta(hours=1)
    t1 = pd.Timestamp(f"{year}-{e}").tz_localize(tz).tz_convert("UTC") + pd.Timedelta(hours=24)
    return pd.date_range(t0, t1, freq="h")


def weather(ctx: Context) -> pd.DataFrame:
    return ctx.weather_override if ctx.weather_override is not None else ctx.era5


def season_prices(ctx: Context, idx: pd.DatetimeIndex, year: int) -> np.ndarray:
    py = ctx.price_year_map.get(year, year)
    if ctx.scfg.get("price", "omie") == "flat":
        return np.full(len(idx), float(ctx.scfg.get("flat_price_eur_mwh", 45.0)))
    if ctx.scfg.get("price") == "omie_shape":
        # mean OMIE hourly profile of the 2023-2025 summers (by local clock hour), rescaled to the tariff level;
        # represents a solar-saturated market with cheap midday hours
        loc = idx.tz_convert(ctx.scfg["tz"])
        pr = ctx.price[ctx.price.index.year >= 2023]
        pl = (pr.index - pd.Timedelta(minutes=30)).tz_convert("Europe/Madrid")
        prof = pr.groupby(pl.hour).mean().reindex(range(24)).values
        p = prof[(loc - pd.Timedelta(minutes=30)).hour]
        return p / np.nanmean(p) * float(ctx.scfg.get("flat_price_eur_mwh", 45.0))
    shift = pd.DateOffset(years=py - year)
    src = (idx.tz_convert(ctx.scfg["tz"]) + shift).tz_convert("UTC")
    p = ctx.price.reindex(src).values.astype(float)
    if np.isnan(p).any():
        s = pd.Series(p).interpolate(limit_direction="both").values
        p = s
    if ctx.flat_price:
        p = np.full_like(p, np.nanmean(p))
    return p


def physics(ctx, wx):
    return sim.compute_physics(wx, ctx.scfg["lat"], ctx.scfg["lon"], ctx.geo, ctx.micro, ctx.cfg, ctx.ccfg,
                               cooling=ctx.cooling, t_crit_shift=ctx.t_crit_shift)


def climatology_wx(ctx: Context, idx: pd.DatetimeIndex, year: int, window=3):
    """Mean weather for the same day of year and hour over the other years (no information)."""
    w = weather(ctx)
    w = w[(w.index.year != year)]
    doy, hr = w.index.dayofyear.values, w.index.hour.values
    out = np.zeros((len(idx), 6))
    sd = np.zeros(len(idx))
    cols = ["T", "RH", "GHI", "BHI", "DHI", "u10"]
    vals = w[cols].values
    tdoy, thr = idx.dayofyear.values, idx.hour.values
    for d in np.unique(tdoy):
        dd = np.abs(((doy - d) + 182) % 365 - 182) <= window
        for h in range(24):
            m = dd & (hr == h)
            k = (tdoy == d) & (thr == h)
            out[k] = vals[m].mean(axis=0)
            sd[k] = vals[m, 0].std()
    c = pd.DataFrame(out, index=idx, columns=cols)
    c["P_kpa"] = 101.3
    return c, sd


def nowcast_wx(ctx: Context, truth: pd.DataFrame):
    """Persistence nowcast of the next hour's air temperature from the last observation."""
    w = weather(ctx)
    prev = w["T"].reindex(truth.index - pd.Timedelta(hours=1)).values
    h = truth.index.hour.values
    now = truth.copy()
    now["T"] = prev + ctx.nowcast_step[h]
    return now, ctx.nowcast_sd[h]


# =============================================================================
def forecast_frames(ctx: Context, idx, year, rng, n_real):
    """Day-ahead forecast frames: archived ECMWF when available, otherwise synthetic."""
    truth = weather(ctx).reindex(idx)
    if ctx.ecmwf is not None and ctx.weather_override is None and idx[0] >= ctx.ecmwf.index[0] and idx[-1] <= ctx.ecmwf.index[-1]:
        f = fc.fc_frame_from_ecmwf(ctx.ecmwf.reindex(idx).interpolate(limit_direction="both"))
        # models fitted without the months of this season (no information leakage)
        months = idx.tz_convert(ctx.scfg["tz"]).month.unique()
        mask = lambda ix: ~ix.month.isin(months)
        hc = fc.HourlyCorrector().fit(ctx.era5, ctx.ecmwf, 1, mask)
        obs = ctx.obs_daily
        dfc = fc.add_features(fc.daily_fc(ctx.ecmwf, 1, ctx.scfg["tz"]), obs, 1).join(obs[["Tmax"]]).dropna()
        dfc = dfc[~dfc.index.month.isin(months)]
        em = fc.EMOS().fit(dfc, dfc["Tmax"].values)
        return [(f, hc, em, "ecmwf")]
    frames = []
    for _ in range(n_real):
        f = ctx.errlib.synthetic(truth, rng, ctx.cfg["forecast"]["synthetic_window_days"])
        frames.append((f, ctx.hcorr_all, ctx.emos_all, "synthetic"))
    return frames


def run_season(ctx: Context, year: int, rng=None, n_real=None, w_crop=1.0, p_star=None,
               strategies=STRATEGIES, keep_schedules=False):
    """All strategies for one season. Returns a long DataFrame (one row per strategy x realisation)."""
    cfg, scfg, ccfg = ctx.cfg, ctx.scfg, ctx.ccfg
    rng = rng or np.random.default_rng(cfg["seed"] + year)
    n_real = n_real or cfg["forecast"]["n_synthetic"]
    p_star = cfg["strategies"]["p_star"] if p_star is None else p_star
    tz = scfg["tz"]
    idx = season_index(ctx, year)
    truth = weather(ctx).reindex(idx).interpolate(limit_direction="both")
    price = season_prices(ctx, idx, year)
    kwp = cfg["system"]["kwp_per_ha"]
    econ = cfg["economics"]
    ph = physics(ctx, truth)
    tilts = ctx.geo.tilts
    ev_kw = dict(price=price, kwp=kwp, w_crop=w_crop, pc_eur_kg=ccfg["price_eur_kg"], y_ref_kg=ctx.y_ref_kg,
                 alpha=ctx.alpha, lam=ctx.lam, l_ref=ctx.l_ref, water_eur_m3=ctx.water_price, k_w=ctx.k_w)
    wkw = dict(water_eur_m3=ctx.water_price, k_w=ctx.k_w)

    # forecast-independent schedules
    base = {}
    base["S1_fixed"] = sim.idx_fixed(len(idx), tilts, cfg["strategies"]["s1_tilt_deg"])
    base["S2_tracking"] = sim.idx_tracking(ph)
    i3 = sim.idx_light_sharing(ph, cfg["strategies"]["s3_fmax"])
    base["S3_light"] = i3
    iprot = sim.idx_protect(ph)
    t_event = ccfg["stress_day_tmax_C"] + ctx.t_crit_shift
    obs_d = fc.daily_obs(truth, tz)
    flags_obs = obs_d["Tmax"] >= t_event
    m5 = sim.protect_mask(idx, tz, cfg["strategies"]["protect_window_local"], ph.daytime, flags_obs)
    base["S5_rule"] = np.where(m5, iprot, i3)
    base["EV_noinfo"] = sim.idx_ev(ph, **{**ev_kw, "lam": 0.0})
    # economic controller that values energy and crop light only (no water, no heat): isolates the water term
    base["EV_energy_light"] = sim.idx_ev(ph, **{**ev_kw, "lam": 0.0, "water_eur_m3": 0.0})
    cw, csd = climatology_wx(ctx, idx, year)
    base["EV_clim"] = sim.idx_ev(physics(ctx, cw), sigma=csd, **ev_kw)
    nw, nsd = nowcast_wx(ctx, truth)
    ph_now = physics(ctx, nw)
    base["MPC_reactive"] = sim.idx_ev(ph_now, sigma=nsd, **ev_kw)
    base["MPC_oracle"] = sim.idx_ev(ph, **ev_kw)

    rows, sched = [], {}
    frames = forecast_frames(ctx, idx, year, rng, n_real)
    for r, (f, hc, em, kind) in enumerate(frames):
        mu, sd = hc.apply(f["T"].values, idx)
        fcw = f.copy()
        fcw["T"] = mu
        ph_fc = physics(ctx, fcw)
        dfc = fc.daily_fc_from_hourly(fcw.assign(T=f["T"].values), ctx.obs_daily, tz, 1)
        m_, s_ = em.predict(dfc.fillna(dfc.mean()))
        p_hat = pd.Series(1 - fc.stats.norm.cdf(t_event, m_, s_), index=dfc.index)
        sch = dict(base)
        m4 = sim.protect_mask(idx, tz, cfg["strategies"]["protect_window_local"], ph.daytime, p_hat >= p_star)
        sch["S4_rule"] = np.where(m4, iprot, i3)
        sch["MPC_det"] = sim.idx_ev(ph_fc, **ev_kw)
        sch["MPC_prob"] = sim.idx_ev(ph_fc, sigma=sd, **ev_kw)
        for s in strategies:
            ib = sch["EV_noinfo"] if s == "MPC_reactive" else None
            # every strategy bids its own tilt schedule with the same day-ahead weather forecast, so the
            # perfect-foresight bound measures the value of information for tilt decisions only
            bid = ph_fc
            m = sim.score(ph, sch[s], price, kwp, ccfg, econ, ctx.alpha, ctx.lam, ctx.y_pot, bid, ib, **wkw)
            m.update(strategy=s, year=year, real=r, forecast=kind)
            if s in ("S4_rule", "S5_rule"):
                mm = m4 if s == "S4_rule" else m5
                m["protect_hours"] = int(mm.sum())
            rows.append(m)
        o = sim.open_field_score(ph, ccfg, ctx.lam, ctx.y_pot, **wkw)
        o.update(strategy="open_field", year=year, real=r, forecast=kind)
        rows.append(o)
        # PV reference for the LER: conventional ground-mounted array, no crop, fixed tilt
        iref = sim.idx_fixed(len(idx), tilts, econ["pv_reference_tilt_deg"])
        e_ref = (ph.power[np.arange(len(idx)), iref] * econ["pv_reference_kwp_per_ha"]).sum() / 1000
        rows.append({"strategy": "pv_ref", "year": year, "real": r, "forecast": kind, "energy_MWh_ha": e_ref,
                     "elec_rev_eur_ha": np.nansum(price * ph.power[np.arange(len(idx)), iref] * econ["pv_reference_kwp_per_ha"]) / 1000})
        if keep_schedules and r == 0:
            sched = {"index": idx, "price": price, "truth": truth, "tilts": {k: tilts[v] for k, v in sch.items()},
                     "p_hat": p_hat, "physics": ph}
    df = pd.DataFrame(rows)
    return (df, sched) if keep_schedules else df


# =============================================================================
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per year: average over forecast realisations, then add Delta V, VOI, eta, LER, break-even price."""
    num = df.select_dtypes("number").columns.drop(["year", "real"], errors="ignore")
    m = df.groupby(["year", "strategy"])[list(num)].mean().reset_index()
    out = []
    for y, g in m.groupby("year"):
        g = g.set_index("strategy")
        v3 = g.loc["S3_light", "value_eur_ha"]
        y_open = g.loc["open_field", "yield_t_ha"]
        e_ref = g.loc["pv_ref", "energy_MWh_ha"]
        for s in [k for k in g.index if k not in ("open_field", "pv_ref")]:
            r = g.loc[s].to_dict()
            v2 = g.loc["S2_tracking", "value_eur_ha"] if "S2_tracking" in g.index else np.nan
            r.update(year=y, strategy=s, dV_vs_S3=r["value_eur_ha"] - v3, dV_vs_S2=r["value_eur_ha"] - v2,
                     LER=r["yield_t_ha"] / y_open + r["energy_MWh_ha"] / e_ref)
            dy = (r["yield_t_ha"] - g.loc["S3_light", "yield_t_ha"]) * 1000
            de = (g.loc["S3_light", "elec_rev_eur_ha"] - g.loc["S3_light", "imbalance_eur_ha"]) - \
                 (r["elec_rev_eur_ha"] - r["imbalance_eur_ha"])
            r["breakeven_crop_price_eur_kg"] = de / dy if dy > 1e-6 else np.nan
            out.append(r)
        r = g.loc["open_field"].to_dict()
        r.update(year=y, strategy="open_field", dV_vs_S3=r["value_eur_ha"] - v3,
                 dV_vs_S2=r["value_eur_ha"] - (g.loc["S2_tracking", "value_eur_ha"] if "S2_tracking" in g.index else np.nan),
                 LER=1.0)
        out.append(r)
    return pd.DataFrame(out)


def voi_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in summary.groupby("year"):
        v = g.set_index("strategy")["value_eur_ha"].reindex(STRATEGIES + ["open_field"])
        noinfo = max(v["EV_noinfo"], v["EV_clim"])
        rows.append({"year": y,
                     "VOI_rule_eur_ha": v["S4_rule"] - v["S3_light"],
                     "VOI_rule_perfect_eur_ha": v["S5_rule"] - v["S3_light"],
                     "eta_rule": _eta(v["S4_rule"], v["S3_light"], v["S5_rule"]),
                     "VOI_mpc_eur_ha": v["MPC_prob"] - noinfo,
                     "VOI_mpc_perfect_eur_ha": v["MPC_oracle"] - noinfo,
                     "eta_mpc": _eta(v["MPC_prob"], noinfo, v["MPC_oracle"]),
                     "dayahead_vs_reactive_eur_ha": v["MPC_prob"] - v["MPC_reactive"],
                     "controller_vs_tracking_eur_ha": v["MPC_prob"] - v["S2_tracking"],
                     "water_term_eur_ha": v["EV_noinfo"] - v["EV_energy_light"],
                     "prob_vs_det_eur_ha": v["MPC_prob"] - v["MPC_det"]})
    t = pd.DataFrame(rows)
    return t


def _eta(v, v0, v1):
    den = v1 - v0
    return (v - v0) / den if abs(den) > 1.0 else np.nan
