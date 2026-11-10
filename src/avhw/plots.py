"""Figures of the paper. All figures are drawn from the result tables, so they can be
re-made with `python run_all.py --steps figures` after a run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import experiment as ex, forecast as fc

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
SHOW = ["S1_fixed", "S2_tracking", "S3_light", "S4_rule", "EV_noinfo", "MPC_reactive", "MPC_prob", "MPC_oracle"]  # max 8 colours
LABEL = {"S1_fixed": "S1 fixed 20°", "S2_tracking": "S2 sun tracking", "S3_light": "S3 light sharing",
         "S4_rule": "S4 forecast rule", "S5_rule": "S5 rule, perfect", "EV_noinfo": "Economic light sharing",
         "EV_clim": "EV climatology", "MPC_reactive": "MPC reactive", "MPC_det": "MPC deterministic",
         "MPC_prob": "MPC probabilistic", "MPC_oracle": "MPC perfect foresight", "open_field": "Open field"}
COL = dict(zip(SHOW, C))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 110, "lines.linewidth": 1.6})


def _t(run, name):
    p = run.out / "tables" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _save(fig, run, name):
    for ext in ("pdf", "png"):
        fig.savefig(run.out / "figures" / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _tag(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, color=INK)


# =============================================================================
def fig_forecast(run):
    sc, pr = _t(run, "T02_forecast_skill"), _t(run, "T02b_forecast_predictions")
    if sc is None:
        return
    models = ["climatology", "persistence", "raw_ecmwf", "emos", "qgb"]
    names = {"climatology": "Climatology", "persistence": "Persistence", "raw_ecmwf": "Raw ECMWF", "emos": "EMOS", "qgb": "Quantile GB"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    for i, m in enumerate(models):
        g = sc[sc.model == m]
        ax[0].plot(g.lead_days, g.CRPS_C, marker="o", ms=4, color=C[i], label=names[m])
        ax[1].plot(g.lead_days, g.BSS_vs_clim, marker="o", ms=4, color=C[i])
    ax[0].set(xlabel="Lead time (days)", ylabel="CRPS of daily Tmax (°C)", xticks=[1, 2, 3])
    ax[1].set(xlabel="Lead time (days)", ylabel="Brier skill vs climatology", xticks=[1, 2, 3])
    ax[0].legend(loc="upper left", ncol=1, fontsize=6.8)
    l1 = pr[pr.lead == 1]
    bins = np.linspace(0, 1, 6)
    for i, m in ((3, "emos"), (4, "qgb")):
        p, o = l1[f"{m}_p"].values, l1["event"].values
        b = np.digitize(p, bins[1:-1])
        x = [p[b == k].mean() for k in range(5) if (b == k).sum() >= 3]
        y = [o[b == k].mean() for k in range(5) if (b == k).sum() >= 3]
        ax[2].plot(x, y, marker="o", ms=4, color=C[i], label=names[m])
    ax[2].plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
    ax[2].set(xlabel="Forecast probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax[2].legend(loc="upper left")
    ev = int(sc.n_events.iloc[0])
    ax[2].set_title(f"Stress days, lead 1 ({ev} events)", fontsize=8)
    for a, s in zip(ax, "abc"):
        _tag(a, s)
    _save(fig, run, "Fig4_forecast_skill")


def fig_validation(run):
    if "micro" not in run.res:
        return
    _, preds, _ = run.res["micro"]
    crop_t = _t(run, "T04_crop_light_heat_validation")
    grid = _t(run, "T04c_crop_identification_vs_tilt")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    for i, (cyc, g) in enumerate(preds.groupby("cycle")):
        ax[0].scatter(g.dT_day, g.dT_day_hat, s=10, color=C[i], label=cyc.replace("_", " "), edgecolor="white", lw=0.4)
    lim = [min(preds.dT_day.min(), preds.dT_day_hat.min()) - 0.1, max(preds.dT_day.max(), preds.dT_day_hat.max()) + 0.1]
    ax[0].plot(lim, lim, color=GREY, ls="--", lw=0.8)
    ax[0].set(xlabel="Observed day ΔT (°C)", ylabel="Predicted, other cycle (°C)", xlim=lim, ylim=lim)
    ax[0].legend(loc="upper left")
    x = np.arange(len(crop_t))
    ax[1].bar(x - 0.2, crop_t.yield_ratio_obs, 0.36, color=C[0], label="Observed")
    ax[1].bar(x + 0.2, crop_t.ratio_pred_loco_lam0, 0.36, color=C[1], label="Predicted (other cycle)")
    for xi, (o, p) in enumerate(zip(crop_t.yield_ratio_obs, crop_t.ratio_pred_loco_lam0)):
        ax[1].text(xi + 0.2, p + 0.02, f"{100 * (p - o) / o:+.0f}%", ha="center", fontsize=7, color=MUTED)
    ax[1].set(xticks=x, xticklabels=[c.replace("_", " ") for c in crop_t.cycle], ylabel="APV / control yield", ylim=(0, 1.35))
    ax[1].legend(loc="upper left", fontsize=6.5, ncol=1)
    ax[2].plot(grid.tilt_assumed, grid.lambda_joint * 1e4, marker="o", ms=4, color=C[6])
    for _, r in grid.iterrows():
        ax[2].annotate(f"α={r.alpha_joint:.2f}", (r.tilt_assumed, r.lambda_joint * 1e4), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=6, color=MUTED)
    ax[2].set(xlabel="Assumed experimental tilt (°)", ylabel="Identified λ (10⁻⁴ per °C·h)",
              ylim=(0, grid.lambda_joint.max() * 1e4 * 1.35))
    for a, s in zip(ax, "abc"):
        _tag(a, s)
    _save(fig, run, "Fig5_validation")


def fig_heatwave_day(run):
    if "main" not in run.res:
        return
    sched = run.res["main"][4]
    idx, truth = sched["index"], sched["truth"]
    tz = run.cfg["sites"][run.main]["tz"]
    dates = fc.local_date(idx, tz)
    tmax = truth["T"].groupby(dates).max()
    day = tmax.idxmax()
    k = np.asarray(dates == day)
    loc = (idx[k] - pd.Timedelta(minutes=30)).tz_convert(tz)
    hrs = loc.hour + 0.5
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3), constrained_layout=True)
    ax[0].plot(hrs, truth["T"].values[k], color=C[7])
    ax[0].set(xlabel="Local time (h)", ylabel="Air temperature (°C)", xlim=(5, 22))
    ax[0].text(0.97, 0.05, f"{pd.Timestamp(day).date()}\nTmax {tmax.max():.1f} °C", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax[1].step(hrs, sched["price"][k], where="mid", color=C[0])
    ax[1].set(xlabel="Local time (h)", ylabel="Day-ahead price (€/MWh)", xlim=(5, 22))
    ax = [ax[0], ax[2], ax[3], ax[1]]
    day_k = np.asarray(sched["truth"]["GHI"].values[k] > 20)
    for s in ["S2_tracking", "S3_light", "EV_noinfo", "MPC_prob"]:
        ls = (0, (3, 2)) if s == "MPC_prob" else "-"
        ax[1].plot(hrs, np.where(day_k, sched["tilts"][s][k], np.nan), color=COL[s], label=LABEL[s],
                   drawstyle="steps-mid", ls=ls)
    ax[1].set(xlabel="Local time (h)", ylabel="Tilt (°)", xlim=(5, 22), ylim=(-2, 95))
    ax[1].legend(loc="upper center", fontsize=5.8, ncol=1, handlelength=1.5)
    ph = sched["physics"]
    tilts = run.tilts
    r = np.where(k)[0]
    for s in ["S2_tracking", "S3_light", "MPC_prob"]:
        ti = np.searchsorted(tilts, sched["tilts"][s][k])
        ax[2].plot(hrs, np.where(day_k, ph.fsh[r, ti], np.nan), color=COL[s], drawstyle="steps-mid", label=LABEL[s])
    ax[2].set(xlabel="Local time (h)", ylabel="Crop area in beam shade (–)", xlim=(5, 22), ylim=(0, 1))
    for a, s in zip([ax[0], ax[3], ax[1], ax[2]], "abcd"):
        _tag(a, s)
    _save(fig, run, "Fig6_heatwave_day")


def fig_strategies(run):
    s = _t(run, "T06b_strategies_by_year")
    if s is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    for i, st in enumerate(SHOW):
        g = s[s.strategy == st]
        ax[0].scatter(np.full(len(g), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(g)), g.dV_vs_S3 / 1000,
                      s=7, color=COL[st], alpha=0.55, lw=0)
        ax[0].scatter(i, g.dV_vs_S3.mean() / 1000, s=34, color=COL[st], edgecolor="white", lw=0.8, zorder=3)
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].set_xticks(range(len(SHOW)), [LABEL[x] for x in SHOW], rotation=40, ha="right", fontsize=7)
    ax[0].set_ylabel("Value vs S3 (k€/ha per season)")
    m = s.groupby("strategy")[["energy_MWh_ha", "yield_t_ha"]].mean()
    for st in SHOW:
        ax[1].scatter(m.loc[st, "energy_MWh_ha"], m.loc[st, "yield_t_ha"], s=34, color=COL[st], edgecolor="white", lw=0.8, label=LABEL[st])
    if "open_field" in m.index:
        ax[1].scatter(0, m.loc["open_field", "yield_t_ha"], s=34, color=GREY, label="Open field")
    ax[1].set(xlabel="Electricity (MWh/ha per season)", ylabel="Lettuce yield (t/ha)")
    ax[1].legend(fontsize=6.3, loc="lower left", ncol=1)
    for a, t in zip(ax, "ab"):
        _tag(a, t)
    _save(fig, run, "Fig7_strategy_comparison")


def fig_pareto(run):
    t = _t(run, "T09_pareto")
    if t is None:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    fam = {"MPC_prob": ("MPC probabilistic, crop weight sweep", COL["MPC_prob"]),
           "MPC_oracle": ("MPC perfect foresight", COL["MPC_oracle"]),
           "EV_noinfo": ("Economic light sharing", COL["EV_noinfo"]),
           "S4_rule": ("S4 rule, p* sweep", COL["S4_rule"])}
    for f, (lab, c) in fam.items():
        g = t[t.family == f].sort_values("elec_rev_eur_ha")
        ax.plot(g.elec_rev_eur_ha / 1000, g.crop_rev_eur_ha / 1000, marker="o", ms=3.5, color=c, label=lab, lw=1.2)
    s = _t(run, "T06_strategies_mean_2015_2025")
    if s is not None:
        for st in ["S1_fixed", "S2_tracking", "S3_light"]:
            r = s[s.strategy == st].iloc[0]
            ax.scatter(r.elec_rev_eur_ha / 1000, r.crop_rev_eur_ha / 1000, marker="D", s=28, color=COL[st], zorder=3, label=LABEL[st])
    ax.set(xlabel="Electricity revenue (k€/ha per season)", ylabel="Crop revenue (k€/ha per season)")
    ax.legend(fontsize=6.2, loc="lower left")
    _save(fig, run, "Fig8_pareto")


def fig_lambda(run):
    t = _t(run, "T16_voi_vs_lambda")
    if t is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    clim = list(t.climate.unique())
    for i, c in enumerate(clim):
        g = t[t.climate == c].sort_values("lambda")
        col = INK if c == "present" else C[i]
        lab = "Present climate, 2023–25 prices" if c == "present" else c.replace("2041-2050 ", "2041–50 ").replace("_", "-")
        ax[0].plot(g["lambda"] * 1e3, g.VOI_mpc, marker="o", ms=3.5, color=col, label=lab)
        ax[1].plot(g["lambda"] * 1e3, g.VOI_mpc_perfect, marker="o", ms=3.5, color=col, label=lab)
    lf = getattr(run, "lam_field", None)
    for a in ax:
        a.axhline(0, color=GREY, lw=0.8)
        if lf is not None:
            a.axvline(lf * 1e3, color=GREY, ls="--", lw=0.8)
            a.text(lf * 1e3, a.get_ylim()[1] * 0.75, " field\n estimate", fontsize=6.5, color=MUTED)
        a.set_xlabel("Heat-damage coefficient λ (10⁻³ per °C·h)")
    ax[0].set_ylabel("Value of day-ahead forecast (€/ha)")
    ax[1].set_ylabel("Value of perfect foresight (€/ha)")
    ax[0].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.45, 1.0))
    for a, s in zip(ax, "ab"):
        _tag(a, s)
    _save(fig, run, "Fig9_value_vs_damage")


def fig_sensitivity(run):
    t, ab = _t(run, "T11_sensitivity"), _t(run, "T10_ablation")
    if t is None or ab is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    base = t[t.parameter == "baseline"].dV_MPC_prob.iloc[0]
    tt = t[t.parameter != "baseline"].copy()
    tt["lab"] = tt.parameter + ": " + tt.setting.astype(str)
    tt["d"] = tt.dV_MPC_prob - base
    tt = tt.reindex(tt.d.abs().sort_values().index)
    ax[0].barh(tt.lab, tt.d, color=[C[0] if v >= 0 else C[1] for v in tt.d], height=0.6)
    ax[0].axvline(0, color=GREY, lw=0.8)
    ax[0].set_xlabel("Change in MPC value vs S3 (€/ha)")
    ax[0].tick_params(axis="y", labelsize=6.3)
    ax[0].set_title(f"Baseline {base:,.0f} €/ha", fontsize=8)
    ab = ab[ab.lambda_level.str.startswith("high")].reset_index(drop=True)
    y = np.arange(len(ab))
    ax[1].barh(y + 0.2, ab.VOI_mpc, 0.38, color=C[6], label="Value of day-ahead forecast")
    ax[1].barh(y - 0.2, ab.dayahead_minus_reactive, 0.38, color=C[2], label="Day-ahead minus reactive")
    ax[1].set_yticks(y, ab.variant, fontsize=6.8)
    ax[1].set_title(f"Heat-sensitive crop, λ = {ab.lambda_level.iloc[0].split('(')[1].rstrip(')')}", fontsize=8)
    ax[1].axvline(0, color=GREY, lw=0.8)
    ax[1].set_xlabel("€/ha per season")
    ax[1].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
    for a, s in zip(ax, "ab"):
        _tag(a, s)
    _save(fig, run, "Fig10_sensitivity_ablation")


def fig_climate(run):
    hot, val = _t(run, "T13_cmip6_hot_days"), _t(run, "T13b_climate_2050_value")
    if hot is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), constrained_layout=True)
    h = hot[hot.site == run.main].reset_index(drop=True)
    x = np.arange(len(h))
    ax[0].bar(x - 0.2, h.days_ge_40C_hist, 0.38, color=GREY, label="2005–2024")
    ax[0].bar(x + 0.2, h.days_ge_40C_2041_2050, 0.38, color=C[7], label="2041–2050")
    ax[0].set_xticks(x, [m.split("_")[0] for m in h.model], fontsize=7)
    ax[0].set_ylabel("Days per year with Tmax ≥ 40 °C")
    ax[0].legend(loc="upper left")
    if val is not None:
        v = val[val.site == run.main].reset_index(drop=True)
        x = np.arange(len(v))
        ax[1].bar(x, v.heat_degh_open, color=[GREY if p.startswith("present") else C[7] for p in v.period], width=0.6)
        ax[1].set_xticks(x, [m.split("_")[0] if m != "ERA5" else "Present" for m in v.model], fontsize=7)
        ax[1].set_ylabel("Open-field leaf heat stress (°C·h)")
    for a, s in zip(ax, "ab"):
        _tag(a, s)
    _save(fig, run, "Fig11_climate_2050")


def fig_prices(run):
    prof = run.out / "tables" / "T01b_price_profile_hot_vs_normal.csv"
    tab = _t(run, "T01_prices_hot_vs_normal")
    if tab is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True)
    era5 = run.era5[run.main]
    tz = run.cfg["sites"][run.main]["tz"]
    from .prices import hot_day_price_analysis
    thr = run.cfg["crops"][run.cfg["main_crop"]]["stress_day_tmax_C"] + 4
    recent = run.price[run.price.index.year >= 2023]
    _, pr = hot_day_price_analysis(era5, recent, tz, thr)
    ax[0].plot(pr.index + 0.5, pr[False], color=GREY, label="Other summer days")
    ax[0].plot(pr.index + 0.5, pr[True], color=C[7], label=f"Days with Tmax ≥ {thr:.0f} °C")
    ax[0].set(xlabel="Local hour", ylabel="Mean day-ahead price (€/MWh)", xlim=(0, 24), title="Jun–Aug 2023–2025")
    ax[0].legend(loc="upper center")
    g = tab.pivot(index="year", columns="hot_day", values="solar_weighted_price")
    ax[1].plot(g.index, g[False], marker="o", ms=3.5, color=GREY, label="Other days")
    if True in g:
        ax[1].plot(g.index, g[True], marker="o", ms=3.5, color=C[7], label="Hot days")
    ax[1].set(xlabel="Year", ylabel="Solar-weighted price (€/MWh)")
    ax[1].legend()
    for a, s in zip(ax, "ab"):
        _tag(a, s)
    _save(fig, run, "Fig12_prices")


def fig_tomato(run):
    t = _t(run, "T15b_tomato_voi_vs_lambda")
    if t is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    for i, (v, g) in enumerate(t.groupby("variant", sort=False)):
        g = g.sort_values("lambda")
        ax[0].plot(g["lambda"] * 1e3, g.VOI_mpc, marker="o", ms=3.5, color=C[i], label=v)
        ax[1].plot(g["lambda"] * 1e3, g.dayahead_minus_reactive, marker="o", ms=3.5, color=C[i], label=v)
    for a in ax:
        a.axhline(0, color=GREY, lw=0.8)
        a.set_xlabel("Heat-damage coefficient λ (10⁻³ per °C·h)")
    ax[0].set_ylabel("Value of day-ahead forecast (€/ha)")
    ax[1].set_ylabel("Day-ahead minus reactive (€/ha)")
    ax[0].legend(fontsize=6.2, loc="upper left")
    for a, s in zip(ax, "ab"):
        _tag(a, s)
    _save(fig, run, "Fig13_tomato_case")


def fig_regime(run):
    v = _t(run, "T07_value_of_information_by_year")
    if v is None or "controller_vs_tracking_eur_ha" not in v:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    ax.bar(v.year, v.controller_vs_tracking_eur_ha, color=C[6], width=0.7)
    for x, y in zip(v.year, v.controller_vs_tracking_eur_ha):
        if y > 30:
            ax.text(x, y + 10, f"{y:.0f}", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set(xlabel="Season", ylabel="Price-aware control minus\nsun tracking (€/ha)")
    ax.set_xticks(v.year, [str(y) for y in v.year], fontsize=6.5, rotation=45)
    _save(fig, run, "Fig14_gain_over_tracking_by_year")


def make_all(run):
    for f in (fig_forecast, fig_validation, fig_heatwave_day, fig_strategies, fig_pareto, fig_lambda,
              fig_sensitivity, fig_climate, fig_prices, fig_tomato, fig_regime):
        try:
            f(run)
        except Exception as e:  # a missing table must not stop the other figures
            run.log(f"figure {f.__name__} skipped: {e!r}")
