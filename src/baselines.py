# ============== #
# BASELINES FILE #
# ============== #
"""The baselines every model must beat to deserve to exist.

None of these has parameters estimated in an interesting way; they all cost one
line. They set the bar *before* building complicated models: the original
repository had no baseline at all, so a GLM with splines and confidence
intervals looked like a result even when a mean of the last twelve months beat
it.

They are evaluated on two scales, because they answer two different questions:

* **annual** — it is the scale of the question ("how many donations in 2026?")
  and the one the donor-level model works on, producing the aggregate by
  summation;
* **monthly** — it has 204 observations instead of 17, so it discriminates
  between models much better, and seasonality is strong.

Every model follows the signature ``fn(history, h) -> numpy.ndarray`` of
:data:`validation.ModelFn`; the parametric ones are factories returning such
a function. Results are written to ``reports/backtest_*.csv``.

Functions
---------
naive
    Repeats the last observed value. Used in ``ANNUAL_MODELS`` and
    ``MONTHLY_MODELS``, and by ``tests/test_validation.py``.
mean_last_k
    Factory: mean of the last k periods. Used in the model dictionaries and
    by ``tests/test_validation.py``.
seasonal_naive
    Factory: repeats the last seasonal cycle. Used in ``MONTHLY_MODELS`` and
    by ``tests/test_validation.py``.
linear_trend
    Factory: least-squares line, extrapolated. Used in the model
    dictionaries and in the phi sweep of :func:`main`.
damped_trend
    Factory: linear trend whose slope decays as phi^i. Used in the model
    dictionaries, by :mod:`forecast`, :mod:`simulate` (entrants forecast)
    and ``tests/test_validation.py``.
seasonal_plus_damped
    Factory: last-year seasonal profile times a damped-trend level. Used in
    ``MONTHLY_MODELS``.
combination
    Factory: mean of several models. Used in ``ANNUAL_MODELS``.
annual_series
    Number of donations per year. Used by :func:`main`, :mod:`forecast` and
    :mod:`backtest_models`.
monthly_series
    Number of donations per month. Used by :func:`main` and
    :mod:`backtest_models`.
_run
    Private helper: backtests a model dictionary on one series and writes
    the CSVs. Used only inside this module, by :func:`main`.
main
    Command-line entry point. Run via ``python -m baselines`` from the
    ``Makefile`` and ``scripts/run_pipeline.py``.

``ANNUAL_MODELS`` is also imported by :mod:`backtest_models`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DONATIONS, REPORTS
from validation import backtest_series, rolling_origins, summarise

# ── baselines ────────────────────────────────────────────────────────────────
# Common signature: (observed history, horizon) -> array of length horizon.


# ============ #
# NAIVE METHOD #
# ============ #
def naive(history: pd.Series, h: int) -> np.ndarray:
    """The last observed value, repeated.

    Parameters
    ----------
    history : pandas.Series
        Observed series up to the forecast origin, in chronological order.
    h : int
        Number of periods to forecast.

    Returns
    -------
    numpy.ndarray
        Array of length `h`, every element equal to ``history.iloc[-1]``.
    """
    return np.repeat(history.iloc[-1], h)


# ================== #
# MEAN_LAST_K METHOD #
# ================== #
def mean_last_k(k: int):
    """The mean of the last k periods, repeated. Flatter than naive.

    Parameters
    ----------
    k : int
        Number of most recent periods to average.

    Returns
    -------
    callable
        Model ``fn(history, h) -> numpy.ndarray`` returning `h` copies of
        the mean of the last `k` values of `history`. Its ``__name__`` is
        ``"media_ultimi_<k>"``.
    """
    def fn(history: pd.Series, h: int) -> np.ndarray:
        return np.repeat(history.iloc[-k:].mean(), h)
    fn.__name__ = f"media_ultimi_{k}"
    return fn


# ===================== #
# SEASONAL_NAIVE METHOD #
# ===================== #
def seasonal_naive(period: int):
    """This period next year resembles the same period last year.

    Parameters
    ----------
    period : int
        Length of the seasonal cycle (12 for monthly data).

    Returns
    -------
    callable
        Model ``fn(history, h) -> numpy.ndarray`` that repeats the last
        `period` values of `history` cyclically for `h` steps. Its
        ``__name__`` is ``"naive_stagionale_<period>"``.
    """
    def fn(history: pd.Series, h: int) -> np.ndarray:
        last = history.iloc[-period:].to_numpy()
        return np.array([last[i % period] for i in range(h)])
    fn.__name__ = f"naive_stagionale_{period}"
    return fn


# =================== #
# LINEAR_TREND METHOD #
# =================== #
def linear_trend(k: int | None = None):
    """Least-squares line on the last k periods, extrapolated.

    Parameters
    ----------
    k : int, optional
        Number of most recent periods used to fit the line. If ``None`` (or
        0), the whole history is used.

    Returns
    -------
    callable
        Model ``fn(history, h) -> numpy.ndarray`` returning the fitted line
        evaluated at the `h` periods after the last observation. Its
        ``__name__`` is ``"trend_lineare_<k>"`` or ``"trend_lineare_tutto"``.
    """
    def fn(history: pd.Series, h: int) -> np.ndarray:
        y = history.to_numpy(dtype=float)
        if k:
            y = y[-k:]
        x = np.arange(len(y))
        b, a = np.polyfit(x, y, 1)
        return a + b * (len(y) - 1 + np.arange(1, h + 1))
    fn.__name__ = f"trend_lineare_{k or 'tutto'}"
    return fn


# =================== #
# DAMPED_TREND METHOD #
# =================== #
def damped_trend(phi: float = 0.8, k: int | None = None):
    """Damped linear trend: the slope dies out as phi^i.

    It is the correction that a linear extrapolation over a long horizon almost
    always needs. A line estimated on fifteen years of decline keeps going down
    forever; damping it acknowledges that real series flatten out.

    The forecast at step ``i`` is ``level + b * (phi + phi^2 + ... + phi^i)``,
    where ``level`` and ``b`` are the fitted value at the last observation and
    the slope of a least-squares line.

    Parameters
    ----------
    phi : float, default 0.8
        Damping factor; 1 gives the undamped linear trend, smaller values
        flatten faster.
    k : int, optional
        Number of most recent periods used to fit the line. If ``None`` (or
        0), the whole history is used.

    Returns
    -------
    callable
        Model ``fn(history, h) -> numpy.ndarray`` returning `h` damped-trend
        forecasts. Its ``__name__`` is ``"trend_smorzato_phi<phi>"``.
    """
    def fn(history: pd.Series, h: int) -> np.ndarray:
        y = history.to_numpy(dtype=float)
        if k:
            y = y[-k:]
        x = np.arange(len(y))
        b, a = np.polyfit(x, y, 1)
        level = a + b * (len(y) - 1)
        # partial sum of phi: a slope that runs out instead of continuing
        steps = np.cumsum(phi ** np.arange(1, h + 1))
        return level + b * steps
    fn.__name__ = f"trend_smorzato_phi{phi}"
    return fn


# =========================== #
# SEASONAL_PLUS_DAMPED METHOD #
# =========================== #
def seasonal_plus_damped(period: int, phi: float = 0.8):
    """Seasonal profile of the last year plus a damped-trend level.

    On the monthly scale seasonality is the dominant component: a model that
    ignores it loses a lot even if it gets the level right.

    The level is a damped trend fitted on the rolling `period`-mean of the
    history; it is multiplied by the seasonal factors of the last cycle
    (each value divided by the cycle mean).

    Parameters
    ----------
    period : int
        Length of the seasonal cycle (12 for monthly data).
    phi : float, default 0.8
        Damping factor of the level's slope (see :func:`damped_trend`).

    Returns
    -------
    callable
        Model ``fn(history, h) -> numpy.ndarray`` returning `h` forecasts.
        Its ``__name__`` is ``"stagionale+smorzato_phi<phi>"``.
    """
    def fn(history: pd.Series, h: int) -> np.ndarray:
        y = history.to_numpy(dtype=float)
        annual = pd.Series(y).rolling(period).mean().dropna().to_numpy()
        x = np.arange(len(annual))
        b, a = np.polyfit(x, annual, 1)
        level = a + b * (len(annual) - 1)
        steps = np.cumsum(phi ** np.arange(1, h + 1))
        levels = level + b * steps
        last_year = y[-period:]
        factors = last_year / last_year.mean()
        return np.array([levels[i] * factors[i % period] for i in range(h)])
    fn.__name__ = f"stagionale+smorzato_phi{phi}"
    return fn


# ================== #
# COMBINATION METHOD #
# ================== #
def combination(*models):
    """Mean of several forecasts.

    It has no parameters to choose, and is almost always more stable than a
    single model: that is why it deserves a place in the comparison next to
    the candidates it tries to beat.

    Parameters
    ----------
    *models : callable
        Models with signature ``fn(history, h) -> numpy.ndarray``.

    Returns
    -------
    callable
        Model ``fn(history, h) -> numpy.ndarray`` returning the element-wise
        mean of the forecasts of `models`.
    """
    def fn(history: pd.Series, h: int) -> np.ndarray:
        return np.mean([m(history, h) for m in models], axis=0)
    return fn


ANNUAL_MODELS = {
    "naive": naive,
    "media_ultimi_3": mean_last_k(3),
    "media_ultimi_5": mean_last_k(5),
    "trend_lineare_tutto": linear_trend(),
    "trend_lineare_8": linear_trend(8),
    "trend_smorzato_0.8": damped_trend(0.8),
    # phi=0.7 is the model adopted in Ch. 4: it must be here, because the
    # empirical intervals of the final forecast are built from its backtest
    # errors and not from those of a nearby phi.
    "trend_smorzato_0.7": damped_trend(0.7),
    "trend_smorzato_0.6": damped_trend(0.6),
    "combinazione(smorzato+media3)": combination(damped_trend(0.6), mean_last_k(3)),
}

#: Sensitivity to the damping parameter. It tells how much of the result
#: depends on the choice of phi: if the best one beats the others only because
#: phi was chosen by looking at the same backtest, its MAPE is optimistic and
#: should be read as the minimum of a range, not as an estimate.
PHI_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MONTHLY_MODELS = {
    "naive": naive,
    "media_ultimi_12": mean_last_k(12),
    "naive_stagionale_12": seasonal_naive(12),
    "trend_lineare_tutto": linear_trend(),
    "trend_smorzato_0.8": damped_trend(0.8),
    "stagionale+smorzato_0.8": seasonal_plus_damped(12, 0.8),
}

# How many periods to observe before the first forecast, and how far to forecast.
# Horizon 6: it is that of the real question (forecast up to 2031), and it gives
# empirical coverage to the intervals of the final forecast.
ANNUAL_MIN_TRAIN, ANNUAL_HORIZON = 8, 6
MONTHLY_MIN_TRAIN, MONTHLY_HORIZON, MONTHLY_STEP = 96, 36, 6


# ==================== #
# ANNUAL_SERIES METHOD #
# ==================== #
def annual_series(df: pd.DataFrame) -> pd.Series:
    """Count donations per calendar year.

    Parameters
    ----------
    df : pandas.DataFrame
        Donation events, one row per donation, with a ``donation_year``
        column.

    Returns
    -------
    pandas.Series
        Number of donations, indexed by year in increasing order.
    """
    return df.groupby("donation_year").size().sort_index()


# ===================== #
# MONTHLY_SERIES METHOD #
# ===================== #
def monthly_series(df: pd.DataFrame) -> pd.Series:
    """Count donations per calendar month.

    Parameters
    ----------
    df : pandas.DataFrame
        Donation events, one row per donation, with ``donation_year`` and
        ``donation_month`` (1–12) columns.

    Returns
    -------
    pandas.Series
        Number of donations, indexed by a monthly ``PeriodIndex``
        (``freq="M"``) in chronological order. Months with no donations are
        absent, not zero.
    """
    s = df.groupby([df.donation_year, df.donation_month]).size()
    s.index = pd.to_datetime(
        [f"{y}-{m:02d}-01" for y, m in s.index]).to_period("M")
    return s.sort_index()


# =========== #
# _RUN METHOD #
# =========== #
def _run(name, models, series, min_train, horizon, step):
    """Backtest a set of models on one series, save and print the results.

    Writes ``reports/backtest_<name>_dettaglio.csv`` (one row per model,
    origin and horizon), ``reports/backtest_<name>.csv`` (overall summary)
    and ``reports/backtest_<name>_per_orizzonte.csv`` (summary by horizon),
    and prints the summary and the MAPE of a few selected horizons.

    Parameters
    ----------
    name : str
        Label of the scale, used in the file names (e.g. ``"annuale"``).
    models : mapping of str to callable
        Model name -> ``fn(history, h) -> numpy.ndarray``.
    series : pandas.Series
        Observed series, indexed by period.
    min_train : int
        Minimum number of periods before the first origin.
    horizon : int
        Maximum forecast horizon.
    step : int
        Spacing between origins.

    Returns
    -------
    pandas.DataFrame
        Overall summary per model, as returned by
        :func:`validation.summarise`.
    """
    origins = rolling_origins(series.index, min_train, horizon, step)
    bt = backtest_series(models, series, origins)
    bt.to_csv(REPORTS / f"backtest_{name}_dettaglio.csv", index=False)

    overall = summarise(bt)
    by_h = summarise(bt, ["modello", "orizzonte"])
    overall.to_csv(REPORTS / f"backtest_{name}.csv", index=False)
    by_h.to_csv(REPORTS / f"backtest_{name}_per_orizzonte.csv", index=False)

    print(f"\n=== scala {name}: {len(origins)} origini, "
          f"{len(bt)} valutazioni ===")
    print(overall.round(3).to_string(index=False))
    piv = by_h.pivot(index="modello", columns="orizzonte", values="mape").round(2)
    # The full table is in the CSV; on screen a few horizons are enough,
    # otherwise on the monthly scale there are 36 unreadable columns.
    shown = [h for h in (1, 3, 6, 12, 24, 36) if h in piv.columns] or list(piv.columns)
    print(f"\nMAPE per orizzonte ({name}, colonne = orizzonti mostrati):")
    print(piv[shown].sort_values(shown[0]).to_string())
    return overall


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: run all baseline backtests.

    Reads the donations, then runs :func:`_run` three times: the annual
    models (``"annuale"``), the sensitivity sweep over ``PHI_GRID``
    (``"sensitivita_phi"``) and the monthly models (``"mensile"``).

    Returns
    -------
    None
    """
    df = pd.read_parquet(DONATIONS, columns=["donation_year", "donation_month"])

    a = annual_series(df)
    print(f"serie annuale: {len(a)} anni, {a.index.min()}-{a.index.max()}")
    _run("annuale", ANNUAL_MODELS, a, ANNUAL_MIN_TRAIN, ANNUAL_HORIZON, 1)

    sweep = {f"phi={phi}": damped_trend(phi) for phi in PHI_GRID}
    sweep["nessuno smorzamento (trend lineare)"] = linear_trend()
    _run("sensitivita_phi", sweep, a, ANNUAL_MIN_TRAIN, ANNUAL_HORIZON, 1)

    m = monthly_series(df)
    print(f"\nserie mensile: {len(m)} mesi")
    _run("mensile", MONTHLY_MODELS, m, MONTHLY_MIN_TRAIN, MONTHLY_HORIZON,
         MONTHLY_STEP)

    print("\nL'asticella è la riga migliore di ciascuna tabella: un modello più "
          "complicato che non la batte non va adottato.")


if __name__ == "__main__":
    main()
