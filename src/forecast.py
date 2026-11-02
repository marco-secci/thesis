# ============= #
# FORECAST FILE #
# ============= #
"""The 2026-2031 forecast, with the model that won the comparison.

The selected model is the **damped trend**, on the aggregate annual series.
It is not the most sophisticated one in the comparison: it is the one that errs
least (`reports/backtest_modelli.csv`). The value of phi is 0.7, but the choice
matters little: the whole 0.3-0.9 range lies between 3.8% and 4.9% MAPE, while
not damping at all gives 6.2%.

**The intervals are empirical.** They do not come from a formula on the model's
standard errors, but from the distribution of the errors that model actually
made in the backtest, horizon by horizon. The reason lies in the results
themselves: the 95% intervals built by the per-donor simulation contain the
true value between 8% and 25% of the time
(`reports/backtest_modelli_copertura.csv`), because they capture only the
variability between donors and not the shocks that hit everyone in the same
year. Empirical intervals already contain those shocks, because they are
measured on real forecasts.

The limitation must be stated: at horizon h the backtest has between 9 and 4
observations. These are indicative intervals, not quantile estimates.

The forecast of the **per-donor model** is also computed, not because it is
more accurate (it is not), but because it is the only one that produces both
donations and donors, and so allows the coherence check the original work could
not do: the implied donations per donor must stay within the range observed
over seventeen years, 1.84-1.97.

Functions
---------
empirical_errors
    Relative backtest errors of the selected model, per horizon. Used only
    inside this module, by :func:`main`.
aggregate_forecast
    Damped-trend point forecast with empirical intervals. Used only inside
    this module, by :func:`main`.
per_donor_forecast
    Forecast of the per-donor Poisson model by simulation. Used only inside
    this module, by :func:`main`.
forecast_by_type
    Split of the forecast total by donation type. Used only inside this
    module, by :func:`main`.
figure
    Observed series plus forecast and interval. Used only inside this
    module, by :func:`main`.
main
    Entry point. Run via ``python -m forecast`` from the ``Makefile`` and
    ``scripts/run_pipeline.py``, and called by :mod:`figures_tesi` with the
    thesis image directory. Its CSV is read by :mod:`export_legacy`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import viz
from baselines import annual_series, damped_trend
from config import DONATIONS, DONOR_YEAR, PANEL_BURN_IN_YEAR, REPORTS, YEAR_MAX
from models import PoissonGLM
from simulate import forecast_entrants, simulate
from viz import SERIES, plt

PHI = 0.7
HORIZON = 6
N_SIMS = 200


# ======================= #
# EMPIRICAL_ERRORS METHOD #
# ======================= #
def empirical_errors(model: str = f"trend_smorzato_{PHI}") -> pd.DataFrame:
    """Relative errors observed in the backtest, per horizon.

    They are the errors of the model actually used for the point forecast:
    `PHI` is among the annual baselines precisely so that the interval does
    not rest on the errors of a different phi.

    Parameters
    ----------
    model : str, default ``"trend_smorzato_<PHI>"``
        Name of the model in ``reports/backtest_annuale_dettaglio.csv``
        (column ``modello``).

    Returns
    -------
    pandas.DataFrame
        One row per horizon, with columns ``orizzonte``, ``n`` (number of
        backtest errors), ``err_medio`` (mean relative error) and
        ``err_p10`` / ``err_p90`` (10th and 90th percentiles). The relative
        error is ``previsto / osservato - 1``.

    Raises
    ------
    SystemExit
        If the backtest file is missing (run ``make baselines`` first).
    """
    path = REPORTS / "backtest_annuale_dettaglio.csv"
    if not path.exists():
        raise SystemExit("manca reports/backtest_annuale_dettaglio.csv: esegui `make baselines`")
    bt = pd.read_csv(path)
    bt = bt[bt.modello == model].copy()
    bt["err_rel"] = bt.previsto / bt.osservato - 1
    return (bt.groupby("orizzonte")
              .agg(n=("err_rel", "size"),
                   err_medio=("err_rel", "mean"),
                   err_p10=("err_rel", lambda s: s.quantile(0.10)),
                   err_p90=("err_rel", lambda s: s.quantile(0.90)))
              .reset_index())


# ========================= #
# AGGREGATE_FORECAST METHOD #
# ========================= #
def aggregate_forecast(series: pd.Series, errors: pd.DataFrame) -> pd.DataFrame:
    """Damped-trend point forecast with empirical intervals.

    The interval inverts the backtest errors: ``lo = point / (1 + err_p90)``
    and ``hi = point / (1 + err_p10)``.

    Parameters
    ----------
    series : pandas.Series
        Observed annual donations, indexed by year.
    errors : pandas.DataFrame
        Errors per horizon, as returned by :func:`empirical_errors`.

    Returns
    -------
    pandas.DataFrame
        ``HORIZON`` rows with columns ``anno``, ``donazioni_previste``
        (rounded point forecast), ``lo`` and ``hi`` (interval bounds, as
        integers).
    """
    point = damped_trend(PHI)(series, HORIZON)
    years = series.index[-1] + np.arange(1, HORIZON + 1)
    e = errors.set_index("orizzonte")
    # The interval is built by inverting the error: if the model tends to
    # overestimate by 5%, the truth lies below the point, not above.
    return pd.DataFrame({
        "anno": years,
        "donazioni_previste": np.round(point).astype(int),
        "lo": np.round(point / (1 + e.err_p90.reindex(range(1, HORIZON + 1)).to_numpy())).astype(int),
        "hi": np.round(point / (1 + e.err_p10.reindex(range(1, HORIZON + 1)).to_numpy())).astype(int),
    })


# ========================= #
# PER_DONOR_FORECAST METHOD #
# ========================= #
def per_donor_forecast(panel: pd.DataFrame) -> pd.DataFrame:
    """Donations and new donors forecast by the per-donor model.

    Fits a :class:`models.PoissonGLM` on the panel from
    ``PANEL_BURN_IN_YEAR``, forecasts the entrants with
    :func:`simulate.forecast_entrants` and runs :func:`simulate.simulate`
    with ``N_SIMS`` paths (seed 7) from ``YEAR_MAX``.

    Parameters
    ----------
    panel : pandas.DataFrame
        Full donor x year panel up to ``YEAR_MAX``.

    Returns
    -------
    pandas.DataFrame
        ``HORIZON`` rows with columns ``anno``,
        ``donazioni_modello_donatore`` (mean simulated total, rounded) and
        ``nuovi_donatori_previsti`` (forecast entrants, rounded).
    """
    train = panel[panel.year >= PANEL_BURN_IN_YEAR]
    m = PoissonGLM().fit(train)
    ent = forecast_entrants(panel, YEAR_MAX, HORIZON)
    tot = simulate(m, panel, YEAR_MAX, HORIZON, ent, n_sims=N_SIMS, seed=7)
    years = YEAR_MAX + np.arange(1, HORIZON + 1)
    return pd.DataFrame({
        "anno": years,
        "donazioni_modello_donatore": np.round(tot.mean(axis=1)).astype(int),
        "nuovi_donatori_previsti": np.round(ent).astype(int),
    })


#: Donation types projected individually; the others go into "ALTRO".
TYPES = ["SANGUE", "PLASMAF", "AFPLTPLASM"]

#: Years on which the trend of the shares is estimated: the last decade.
SHARE_YEARS = range(YEAR_MAX - 9, YEAR_MAX + 1)


# ======================= #
# FORECAST_BY_TYPE METHOD #
# ======================= #
def forecast_by_type(don: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Forecast donations by type: extrapolated share x forecast total.

    For each type a line is fitted to its annual share over the last decade
    and extended; the projected shares are renormalised to sum to 1 and
    applied to the damped-trend total. The total therefore does not change:
    the projection only says how it is split.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donation_year`` and ``donation_type``.
    prev : pandas.DataFrame
        Aggregate forecast, as returned by :func:`aggregate_forecast`.

    Returns
    -------
    pandas.DataFrame
        Long table with columns ``anno``, ``tipologia`` (one of ``TYPES`` or
        ``"ALTRO"``), ``quota_pct`` and ``donazioni``: observed values for
        ``YEAR_MAX`` followed by the projected ones for every forecast year.
    """
    d = don[don.donation_year.isin(SHARE_YEARS)]
    dtype = d.donation_type.where(d.donation_type.isin(TYPES), "ALTRO")
    shares = pd.crosstab(d.donation_year, dtype, normalize="index")[TYPES + ["ALTRO"]]
    years = prev.anno.to_numpy()
    proj = {}
    for c in shares.columns:
        b, a = np.polyfit(shares.index.to_numpy(), shares[c].to_numpy(), 1)
        proj[c] = np.clip(a + b * years, 0, None)
    proj = pd.DataFrame(proj, index=years)
    proj = proj.div(proj.sum(axis=1), axis=0)

    rows = [{"anno": YEAR_MAX, "tipologia": c, "quota_pct": 100 * shares.loc[YEAR_MAX, c],
             "donazioni": int((dtype[d.donation_year == YEAR_MAX] == c).sum())}
            for c in shares.columns]
    tot = prev.set_index("anno").donazioni_previste
    rows += [{"anno": int(y), "tipologia": c, "quota_pct": 100 * proj.loc[y, c],
              "donazioni": int(round(proj.loc[y, c] * tot[y]))}
             for y in years for c in proj.columns]
    return pd.DataFrame(rows)


# ============= #
# FIGURE METHOD #
# ============= #
def figure(series: pd.Series, prev: pd.DataFrame, path) -> None:
    """Plot the observed annual series and the forecast with its interval.

    Parameters
    ----------
    series : pandas.Series
        Observed annual donations, indexed by year.
    prev : pandas.DataFrame
        Aggregate forecast, as returned by :func:`aggregate_forecast`.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(series.index, series.values, color=SERIES[0], marker="o", label="osservato")
    bridge_x = [series.index[-1], *prev.anno]
    bridge_y = [series.iloc[-1], *prev.donazioni_previste]
    ax.plot(bridge_x, bridge_y, color=SERIES[1], marker="o",
            linestyle="--", label=f"previsto (trend smorzato φ={PHI})")
    ax.fill_between(prev.anno, prev.lo, prev.hi, color=SERIES[1], alpha=0.15,
                    label="intervallo empirico (10°-90° pct degli errori di backtest)")
    ax.axvline(series.index[-1] + 0.5, color=viz.INK_SOFT, linestyle=":", linewidth=1)
    ax.set_ylim(0, max(series.max(), prev.hi.max()) * 1.1)
    ax.legend(loc="lower left", fontsize=8.5)
    viz.finish(ax, "Donazioni: osservato 2009-2025, previsto 2026-2031",
               None, "donazioni")
    viz.save(fig, path)


# =========== #
# MAIN METHOD #
# =========== #
def main(img_dir=None) -> None:
    """Entry point: compute and save the 2026-2031 forecast.

    Writes ``reports/forecast_2026_2031.csv`` (aggregate and per-donor
    forecast, with their gap in percent),
    ``reports/forecast_tipologia_2026_2031.csv``,
    ``reports/forecast_errori_empirici.csv`` and the figure
    ``fig_previsione_2026_2031.png``, and prints a summary including the
    observed range of donations per active donor.

    Parameters
    ----------
    img_dir : pathlib.Path, optional
        Directory for the figure. Defaults to ``REPORTS``; :mod:`figures_tesi`
        passes the thesis image directory.

    Returns
    -------
    None
    """
    img = img_dir or REPORTS
    don = pd.read_parquet(DONATIONS, columns=["donation_year", "donation_month",
                                              "donation_type"])
    panel = pd.read_parquet(DONOR_YEAR)
    series = annual_series(don)

    errors = empirical_errors()
    print("errori relativi del modello nel backtest, per orizzonte:")
    print(errors.round(4).to_string(index=False))

    prev = aggregate_forecast(series, errors)
    per_donor = per_donor_forecast(panel)
    out = prev.merge(per_donor, on="anno")

    # ── coherence check ──────────────────────────────────────────────────────
    # The per-donor model also produces active donors: the ratio between the
    # two forecasts must stay within the band observed over seventeen years.
    obs = (don.groupby("donation_year").size()
           / panel[panel.active == 1].groupby("year").size())
    lo, hi = obs.min(), obs.max()
    gap = out.donazioni_modello_donatore / out.donazioni_previste
    out["scarto_vs_aggregato_%"] = ((gap - 1) * 100).round(1)

    out.to_csv(REPORTS / "forecast_2026_2031.csv", index=False)
    types = forecast_by_type(don, prev)
    types.to_csv(REPORTS / "forecast_tipologia_2026_2031.csv", index=False)
    errors.to_csv(REPORTS / "forecast_errori_empirici.csv", index=False)
    # A single copy, with a name that says what it contains: the series is
    # annual, not monthly, and the old "fig_forecast_monthly" said otherwise.
    figure(series, prev, img / "fig_previsione_2026_2031.png")

    print("\nPrevisione 2026-2031:")
    print(out.to_string(index=False))
    print("\nQuote per tipologia (%), 2025 osservato e 2026-2031 proiettato:")
    print(types.pivot(index="anno", columns="tipologia", values="quota_pct")
              .round(2).to_string())
    print(f"\nDonazioni per donatore attivo, osservate 2009-2025: {lo:.3f}-{hi:.3f}")
    print("I due modelli distano al massimo "
          f"{out['scarto_vs_aggregato_%'].abs().max():.1f}% l'uno dall'altro.")
    print("\nNota: gli intervalli poggiano su 9 osservazioni a un anno e 4 a sei anni.")


if __name__ == "__main__":
    main()
