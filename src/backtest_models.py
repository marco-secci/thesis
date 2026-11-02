# ==================== #
# BACKTEST_MODELS FILE #
# ==================== #
"""The comparison: all candidates, on the same origins, on the same years.

Three families pitted against each other on the same evaluation grid:

* **baselines** — the rows to beat (`baselines`);
* **aggregate series** — monthly quasi-Poisson GLM with trend and seasonality,
  i.e. the form published in Chapter 4 of the thesis, summed by year;
* **per donor** — the models in `models`, simulated forward and summed over all
  donors (`simulate`).

The first two forecast the total; the third builds it. They are comparable
because the target is the same: the donations observed in each year.

The training panel at each origin is obtained by **filtering** the full panel
to years <= O. That this is equivalent to rebuilding it from scratch with only
the data up to O is verified by `tests/test_panel.py::test_covariates_are_causal`,
and it is that test that makes this cut legitimate.

Functions
---------
result_rows
    Generator turning a model's forecasts into one row per horizon. Used
    only inside this module, by :func:`main`.
main
    Command-line entry point. Run via ``python -m backtest_models`` from
    the ``Makefile`` and ``scripts/run_pipeline.py``. Its
    ``backtest_modelli*.csv`` outputs back the model choice made in
    :mod:`forecast`.
"""

from __future__ import annotations

import argparse
import time
import zlib

import numpy as np
import pandas as pd

import aggregate_models as agg
from baselines import ANNUAL_MODELS, annual_series, monthly_series
from config import DONATIONS, DONOR_YEAR, PANEL_BURN_IN_YEAR, REPORTS
from models import catalog
from simulate import forecast_entrants, simulate
from validation import metrics, rolling_origins, summarise

# Minimum number of years of history required before an origin can be
# evaluated, and number of future years (horizon) each origin must forecast.
MIN_TRAIN, HORIZON = 8, 3
# How many Monte Carlo paths to simulate for each per-donor model, by default.
N_SIMS = 20

# Models working on the aggregate monthly series (not per individual donor):
# the forecast is made on the monthly total and then summed by year.
AGGREGATE_MODELS = {
    "agg_glm_mensile_lineare": agg.monthly_glm_linear,
    "agg_glm_mensile_spline (forma Cap.4)": agg.monthly_glm_spline,
}


# ================== #
# RESULT_ROWS METHOD #
# ================== #
def result_rows(model_name, origin_year, target_years,
                predicted_values, observed_values, extra_fields=None):
    """Turn a model's forecasts into "long" rows, one per horizon.

    The rows are ready to go into the final results DataFrame. Each row
    means: "model X, starting from origin O, at horizon h (i.e. for the
    corresponding target year) predicted P against an observed value
    O_actual".

    Parameters
    ----------
    model_name : str
        Name of the model, stored in ``modello``.
    origin_year : int
        Forecast origin, stored in ``origine``.
    target_years : sequence of int
        Target years, in horizon order.
    predicted_values : sequence of float
        Forecasts, aligned with `target_years`.
    observed_values : sequence of float
        Observed values, aligned with `target_years`.
    extra_fields : dict, optional
        Additional key/value pairs added to every row.

    Yields
    ------
    dict
        One row per horizon with keys ``modello``, ``origine``,
        ``orizzonte`` (1-based), ``periodo``, ``osservato``, ``previsto``
        and those of `extra_fields`.
    """
    for horizon, (target_year, predicted, observed) in enumerate(
            zip(target_years, predicted_values, observed_values), start=1):
        yield {
            "modello": model_name,
            "origine": origin_year,
            "orizzonte": horizon,
            "periodo": target_year,
            "osservato": float(observed),
            "previsto": float(predicted),
            **(extra_fields or {}),
        }


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: run the full model comparison.

    Command-line options: ``--sims N`` simulated paths per per-donor model
    and origin (default ``N_SIMS``); ``--solo NAME ...`` restricts the
    per-donor models to those named; ``--origini N`` uses only the first N
    origins (quick trials). For every rolling origin it evaluates the annual
    baselines, the aggregate monthly GLMs (summed by year with
    :func:`aggregate_models.annual_forecast_from_monthly`) and every
    per-donor model of :func:`models.catalog` simulated forward with
    :func:`simulate.simulate`.

    Writes ``backtest_modelli_dettaglio.csv``, ``backtest_modelli_tempi.csv``,
    ``backtest_modelli.csv``, ``backtest_modelli_per_orizzonte.csv`` and
    ``backtest_modelli_copertura.csv`` (coverage of the 95% intervals of
    the per-donor models) to ``REPORTS``.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=N_SIMS)
    parser.add_argument("--solo", nargs="*", default=None,
                         help="limita ai modelli per donatore indicati")
    parser.add_argument("--origini", type=int, default=None,
                         help="usa solo le prime N origini (per prove rapide)")
    args = parser.parse_args()

    # Raw donation data (one row per donation) and donor-year panel (one row
    # per donor per year, with the covariates already computed).
    donations = pd.read_parquet(DONATIONS, columns=["donation_year", "donation_month"])
    donor_panel = pd.read_parquet(DONOR_YEAR)

    # Aggregate time series: total donations per year and per month.
    annual = annual_series(donations)
    monthly = monthly_series(donations)

    # The "origins" are the points in time from which forecasting starts: for
    # each origin only data up to that year are used, and the forecast is
    # evaluated on the following HORIZON years (the "targets").
    origins = rolling_origins(annual.index, MIN_TRAIN, HORIZON)
    if args.origini:
        origins = origins[:args.origini]
    print(f"{len(origins)} origini: {[o.origin for o in origins]}")
    print(f"{args.sims} percorsi simulati per modello per origine\n")

    # Catalogue of per-donor models (each builds its own forecasts donor by
    # donor, then they are summed). `--solo` restricts the comparison to a
    # subset, useful for quick trials.
    model_builders = catalog()
    if args.solo:
        model_builders = {name: builder
                          for name, builder in model_builders.items()
                          if name in args.solo}

    result_records, run_times = [], []
    for origin_info in origins:
        origin_year, target_years = origin_info.origin, list(origin_info.targets)
        # Actually observed values for the target years: the "target" against
        # which all forecasts are compared, whichever model produces them.
        observed_values = [annual.loc[t] for t in target_years]
        print(f"origine {origin_year} -> {target_years}")

        # ── baselines on the annual series ───────────────────────────────────
        # Annual history available up to (and including) the origin: it is all
        # and only what the baseline models can see to forecast the future.
        annual_history = annual.loc[:origin_year]
        for model_name, model_fn in ANNUAL_MODELS.items():
            forecasts = model_fn(annual_history, len(target_years))
            result_records += list(result_rows(
                model_name, origin_year, target_years, forecasts, observed_values))

        # ── models on the aggregate monthly series ───────────────────────────
        for model_name, model_fn in AGGREGATE_MODELS.items():
            forecasts = agg.annual_forecast_from_monthly(
                model_fn, monthly, origin_year, len(target_years))
            result_records += list(result_rows(
                model_name, origin_year, target_years, forecasts, observed_values))

        # ── per-donor models ─────────────────────────────────────────────────
        # Full panel history up to the origin: it serves as the initial state
        # of the simulation (e.g. how many years each donor has been
        # donating), even though model estimation will use only part of these
        # years.
        full_panel_history = donor_panel[donor_panel.year <= origin_year]
        # The model is estimated only on the years in which the panel
        # composition is "steady", i.e. after the initial burn-in period (see
        # `config.PANEL_BURN_IN_YEAR`): before then the panel is not
        # representative and would bias the estimate.
        estimation_panel = full_panel_history[
            full_panel_history.year >= PANEL_BURN_IN_YEAR]
        # New donors (who will appear in future years) do not exist yet in the
        # panel: their number is forecast separately, not by a per-donor model
        # (see `simulate.forecast_entrants`).
        new_entrants_forecast = forecast_entrants(
            full_panel_history, origin_year, len(target_years))

        for model_name, model_builder in model_builders.items():
            start_time = time.time()
            # Build a new model instance and fit it on the filtered training
            # panel.
            fitted_model = model_builder().fit(estimation_panel)
            # Simulate forward `args.sims` Monte Carlo paths of future
            # donations, donor by donor, also adding the new entrants
            # forecast above. The seed depends on (model name, origin) so as
            # to be reproducible but different for each combination. crc32
            # and not hash(): hash() on strings changes in every process
            # (PYTHONHASHSEED), and with it the backtest numbers.
            simulated_totals = simulate(
                fitted_model, full_panel_history, origin_year,
                len(target_years), new_entrants_forecast,
                n_sims=args.sims, seed=zlib.crc32(f"{model_name}|{origin_year}".encode()))

            # Point forecast = mean of the simulated paths; 95% interval =
            # 2.5 and 97.5 percentiles across simulated paths, for each
            # horizon (column) separately.
            forecasts = simulated_totals.mean(axis=1)
            interval_low, interval_high = np.percentile(
                simulated_totals, [2.5, 97.5], axis=1)

            for horizon, (target_year, predicted, observed,
                          low, high) in enumerate(
                    zip(target_years, forecasts, observed_values,
                        interval_low, interval_high), start=1):
                result_records.append({
                    "modello": model_name,
                    "origine": origin_year,
                    "orizzonte": horizon,
                    "periodo": target_year,
                    "osservato": float(observed),
                    "previsto": float(predicted),
                    "lo95": float(low),
                    "hi95": float(high),
                })

            elapsed = time.time() - start_time
            run_times.append({
                "modello": model_name,
                "origine": origin_year,
                "secondi": round(elapsed, 1),
            })
            print(f"    {model_name:22s} {elapsed:5.1f}s  "
                  f"prev={np.round(forecasts).astype(int)}  oss={observed_values}")

    # All collected forecasts, one row per (model, origin, horizon).
    detailed_results = pd.DataFrame(result_records)
    detailed_results.to_csv(REPORTS / "backtest_modelli_dettaglio.csv", index=False)
    pd.DataFrame(run_times).to_csv(
        REPORTS / "backtest_modelli_tempi.csv", index=False)

    # Summary error metrics: overall and per horizon, to see both the general
    # picture and how accuracy degrades as the forecast reaches further ahead.
    overall_summary = summarise(detailed_results)
    summary_by_horizon = summarise(detailed_results, ["modello", "orizzonte"])
    overall_summary.to_csv(REPORTS / "backtest_modelli.csv", index=False)
    summary_by_horizon.to_csv(
        REPORTS / "backtest_modelli_per_orizzonte.csv", index=False)

    print("\n" + "=" * 78)
    print("CONFRONTO COMPLESSIVO (orizzonti 1-3 insieme)")
    print("=" * 78)
    print(overall_summary.round(3).to_string(index=False))

    # Pivot table: one row per model, one column per horizon, to compare at a
    # glance how the MAPE (mean absolute percentage error) varies as the
    # forecast horizon grows.
    mape_by_horizon = summary_by_horizon.pivot(
        index="modello", columns="orizzonte", values="mape").round(2)
    print("\nMAPE per orizzonte:")
    print(mape_by_horizon.sort_values(1).to_string())

    # Coverage of the 95% prediction intervals, computable only for the
    # per-donor models (the only ones producing a lo95/hi95 interval).
    # Ideally the observed value should fall inside the interval about 95% of
    # the time, if the estimated uncertainty is well calibrated.
    rows_with_interval = (detailed_results.dropna(subset=["lo95"])
                          if "lo95" in detailed_results
                          else detailed_results.iloc[0:0])
    if len(rows_with_interval):
        interval_coverage = (
            rows_with_interval
            .assign(dentro=lambda d: (d.osservato >= d.lo95) & (d.osservato <= d.hi95))
            .groupby("modello").dentro.mean().mul(100).round(1)
            .rename("copertura_95_%")
        )
        interval_coverage.to_csv(REPORTS / "backtest_modelli_copertura.csv")
        print("\nCopertura degli intervalli al 95% (nominale: 95%):")
        print(interval_coverage.to_string())


if __name__ == "__main__":
    main()
