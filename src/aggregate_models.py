# ===================== #
# AGGREGATE_MODELS FILE #
# ===================== #
"""Models working directly on the aggregate series.

They serve two purposes: representing the approach of Chapter 4 of the thesis
in the comparison — a quasi-Poisson GLM on the monthly series with trend and
seasonality — and checking whether forecasting the total directly beats the
per-donor model.

Quasi-Poisson (`scale="X2"`) corrects the standard errors for overdispersion;
it does not affect the point forecast. Here the point is what matters, but the
variant is kept because it is the published one.

Functions
---------
_frame
    Private helper: builds the design-matrix columns (time index, centred
    year, month factor) for a monthly period index. Used only inside this
    module, by :func:`_monthly_glm`.
_monthly_glm
    Private helper: fits the quasi-Poisson monthly GLM with a chosen trend
    form and forecasts ahead. Used only inside this module, by
    :func:`monthly_glm_linear` and :func:`monthly_glm_spline`.
monthly_glm_linear
    Monthly GLM with a linear yearly trend. Used by
    :mod:`backtest_models` (registered in ``AGGREGATE_MODELS`` as
    ``"agg_glm_mensile_lineare"``).
monthly_glm_spline
    Monthly GLM with the spline trend published in Chapter 4. Used by
    :mod:`backtest_models` (registered in ``AGGREGATE_MODELS`` as
    ``"agg_glm_mensile_spline (forma Cap.4)"``).
annual_forecast_from_monthly
    Turns any monthly forecaster into annual forecasts by summing the
    forecast months per calendar year. Used by :mod:`backtest_models` in the
    backtest loop over ``AGGREGATE_MODELS``.
"""

# Imports:
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ============= #
# _FRAME METHOD #
# ============= #
def _frame(index: pd.PeriodIndex) -> pd.DataFrame:
    """Build the regressor columns for a monthly period index.

    Parameters
    ----------
    index : pandas.PeriodIndex
        Monthly periods (``freq="M"``) for which the regressors are built,
        either the observed history or the future months to forecast.

    Returns
    -------
    pandas.DataFrame
        One row per period in `index`, with a default ``RangeIndex`` and
        columns:

        ``t`` : int
            Position of the period within `index` (0, 1, 2, ...). Note it is
            relative to `index` itself, so it restarts from 0 on a future
            frame; it is not used by the formulas in :func:`_monthly_glm`.
        ``year_c`` : int
            Calendar year centred on 2017 (``year - 2017``).
        ``mese`` : pandas.Categorical
            Calendar month (1–12) as a categorical with all 12 categories
            declared, so a month absent from `index` still maps to the same
            dummy coding at prediction time.
    """
    return pd.DataFrame({
        "t": np.arange(len(index)),
        "year_c": index.year - 2017,
        "mese": pd.Categorical(index.month, categories=range(1, 13)),
    })

# =================== #
# _MONTHLY_GLM METHOD #
# =================== #
def _monthly_glm(history: pd.Series, h: int, trend: str) -> np.ndarray:
    """Fit a quasi-Poisson GLM on a monthly series and forecast `h` months.

    The model is ``y ~ <trend term> + C(mese)``: a Poisson log-link GLM with
    a yearly trend plus month fixed effects, fitted with the Pearson
    chi-squared scale (``scale="X2"``, quasi-Poisson). The scale only changes
    the standard errors, not the point forecast returned here.

    Parameters
    ----------
    history : pandas.Series
        Observed monthly counts, indexed by a monthly ``PeriodIndex`` in
        chronological order. The last period is the forecast origin.
    h : int
        Number of months to forecast after the last period of `history`.
    trend : {"lineare", "spline"}
        Form of the yearly trend:

        - ``"lineare"``: linear in the centred year (``year_c``);
        - ``"spline"``: cubic regression spline of ``year_c`` with 4 degrees
          of freedom and centring constraint
          (``cr(year_c, df=4, constraints="center")``).

    Returns
    -------
    numpy.ndarray
        Array of length `h` with the expected counts for the `h` months
        following the end of `history`.

    Raises
    ------
    KeyError
        If `trend` is not one of the accepted values.
    """
    d = _frame(history.index)
    d["y"] = history.to_numpy(dtype=float)
    term = {"lineare": "year_c",
            "spline": 'cr(year_c, df=4, constraints="center")'}[trend]
    res = smf.glm(f"y ~ {term} + C(mese)", data=d,
                  family=sm.families.Poisson()).fit(scale="X2")

    future = history.index[-1] + np.arange(1, h + 1)
    return np.asarray(res.predict(_frame(pd.PeriodIndex(future, freq="M"))))


# ========================= #
# MONTHLY_GLM_LINEAR METHOD #
# ========================= #
def monthly_glm_linear(history: pd.Series, h: int) -> np.ndarray:
    """Forecast a monthly series with a linear yearly trend plus month effects.

    Thin wrapper around :func:`_monthly_glm` with ``trend="lineare"``.

    Parameters
    ----------
    history : pandas.Series
        Observed monthly counts, indexed by a monthly ``PeriodIndex`` in
        chronological order.
    h : int
        Number of months to forecast after the last period of `history`.

    Returns
    -------
    numpy.ndarray
        Array of length `h` with the forecast monthly counts.
    """
    return _monthly_glm(history, h, "lineare")


# ========================= #
# MONTHLY_GLM_SPLINE METHOD #
# ========================= #
def monthly_glm_spline(history: pd.Series, h: int) -> np.ndarray:
    """Forecast a monthly series with the form published in Chapter 4.

    Natural cubic spline with 4 df on the year plus month effects. Thin
    wrapper around :func:`_monthly_glm` with ``trend="spline"``.

    Parameters
    ----------
    history : pandas.Series
        Observed monthly counts, indexed by a monthly ``PeriodIndex`` in
        chronological order.
    h : int
        Number of months to forecast after the last period of `history`.

    Returns
    -------
    numpy.ndarray
        Array of length `h` with the forecast monthly counts.
    """
    return _monthly_glm(history, h, "spline")


# =================================== #
# ANNUAL_FORECAST_FROM_MONTHLY METHOD #
# =================================== #
def annual_forecast_from_monthly(fn, monthly: pd.Series, origin_year: int,
                                 years: int) -> np.ndarray:
    """Forecast the required months and sum them by calendar year.

    The sum by year is done over the forecast months, not over a separate
    annual forecast: it is the same coherence logic by which the per-donor
    model produces the aggregate by summation.

    The monthly history is truncated at the end of `origin_year`. If that
    year is incomplete, the missing months are forecast too (but not
    included in the output); then ``12 * years`` further months are forecast
    and summed per calendar year.

    Parameters
    ----------
    fn : callable
        Monthly forecaster with signature ``fn(history, h) -> numpy.ndarray``,
        e.g. :func:`monthly_glm_linear` or :func:`monthly_glm_spline`. It
        must return `h` values for the months following `history`.
    monthly : pandas.Series
        Full observed monthly series, indexed by a monthly ``PeriodIndex``.
        Only the periods up to and including `origin_year` are passed to
        `fn`.
    origin_year : int
        Forecast origin: last calendar year considered observed.
    years : int
        Number of calendar years to forecast after `origin_year`.

    Returns
    -------
    numpy.ndarray
        Array of length `years`; element ``k`` is the sum of the forecast
        monthly values for calendar year ``origin_year + k + 1``.
    """
    history = monthly[monthly.index.year <= origin_year]
    last = history.index[-1]
    # months missing to close the origin year (normally none) + the years ahead
    to_forecast = (12 - last.month) + 12 * years
    pred = fn(history, to_forecast)
    idx = pd.PeriodIndex(last + np.arange(1, to_forecast + 1), freq="M")
    s = pd.Series(pred, index=idx)
    return np.array([s[s.index.year == origin_year + k + 1].sum()
                     for k in range(years)])
