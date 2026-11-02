# ==================== #
# TEST_VALIDATION FILE #
# ==================== #
"""Tests of the validation harness: fake data is fine here, because the object
under test is the mechanics of the split, not the data.

Functions
---------
Tests collected by pytest; no other module uses them. They check
:mod:`validation` and the baselines of :mod:`baselines`.

series
    Fixture: synthetic annual series.
test_origins_do_not_look_ahead
    Targets come after their origin.
test_first_origin_respects_minimum
    The first origin respects ``min_train``.
test_backtest_uses_only_history
    Models only see the history up to the origin.
test_metrics_zero_on_perfect_forecast
    Perfect forecasts give zero error.
test_bias_has_sign
    Bias is signed.
test_naive_repeats_last
    Naive repeats the last value.
test_mean_last_k
    Mean of the last k values.
test_seasonal_naive_cycles
    Seasonal naive repeats the cycle.
test_damped_trend_converges
    The damped trend flattens out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baselines import damped_trend, mean_last_k, naive, seasonal_naive
from validation import backtest_series, metrics, rolling_origins


# ============= #
# SERIES METHOD #
# ============= #
@pytest.fixture
def series():
    """Synthetic annual series: 10, 11, ..., 29 for the years 2000-2019.

    Returns
    -------
    pandas.Series
        Float series indexed by year.
    """
    return pd.Series(np.arange(10, 30, dtype=float), index=range(2000, 2020))


# ===================================== #
# TEST_ORIGINS_DO_NOT_LOOK_AHEAD METHOD #
# ===================================== #
def test_origins_do_not_look_ahead(series):
    """Every target period must be later than its own origin.

    Parameters
    ----------
    series : pandas.Series
        Fixture: synthetic annual series.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    for org in rolling_origins(series.index, min_train=5, max_horizon=3):
        assert all(t > org.origin for t in org.targets)


# ========================================= #
# TEST_FIRST_ORIGIN_RESPECTS_MINIMUM METHOD #
# ========================================= #
def test_first_origin_respects_minimum(series):
    """The first origin is the ``min_train``-th period.

    Parameters
    ----------
    series : pandas.Series
        Fixture: synthetic annual series.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    origins = rolling_origins(series.index, min_train=5, max_horizon=3)
    assert origins[0].origin == series.index[4]


# ====================================== #
# TEST_BACKTEST_USES_ONLY_HISTORY METHOD #
# ====================================== #
def test_backtest_uses_only_history(series):
    """The model receives a history that ends at the origin, never beyond.

    Parameters
    ----------
    series : pandas.Series
        Fixture: synthetic annual series.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    seen = []

    def spy(history, h):
        seen.append(history.index[-1])
        return np.repeat(history.iloc[-1], h)

    origins = rolling_origins(series.index, min_train=5, max_horizon=2)
    backtest_series({"spy": spy}, series, origins)
    assert seen == [o.origin for o in origins]


# ============================================ #
# TEST_METRICS_ZERO_ON_PERFECT_FORECAST METHOD #
# ============================================ #
def test_metrics_zero_on_perfect_forecast():
    """MAE, MAPE and bias are zero when forecast equals observed.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    m = metrics([100.0, 200.0], [100.0, 200.0])
    assert m["mae"] == 0 and m["mape"] == 0 and m["bias_pct"] == 0


# ========================= #
# TEST_BIAS_HAS_SIGN METHOD #
# ========================= #
def test_bias_has_sign():
    """Overestimating gives positive bias, underestimating negative.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert metrics([100.0], [110.0])["bias_pct"] == pytest.approx(10.0)
    assert metrics([100.0], [90.0])["bias_pct"] == pytest.approx(-10.0)


# ============================== #
# TEST_NAIVE_REPEATS_LAST METHOD #
# ============================== #
def test_naive_repeats_last(series):
    """The naive model repeats the last observed value.

    Parameters
    ----------
    series : pandas.Series
        Fixture: synthetic annual series.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert list(naive(series, 3)) == [series.iloc[-1]] * 3


# ======================= #
# TEST_MEAN_LAST_K METHOD #
# ======================= #
def test_mean_last_k(series):
    """``mean_last_k(4)`` forecasts the mean of the last four values.

    Parameters
    ----------
    series : pandas.Series
        Fixture: synthetic annual series.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert mean_last_k(4)(series, 2)[0] == pytest.approx(series.iloc[-4:].mean())


# ================================= #
# TEST_SEASONAL_NAIVE_CYCLES METHOD #
# ================================= #
def test_seasonal_naive_cycles():
    """The seasonal naive model repeats the last cycle.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    s = pd.Series([1.0, 2.0, 3.0, 4.0] * 3)
    assert list(seasonal_naive(4)(s, 6)) == [1, 2, 3, 4, 1, 2]


# ================================== #
# TEST_DAMPED_TREND_CONVERGES METHOD #
# ================================== #
def test_damped_trend_converges():
    """With damping the forecast must stabilise, not diverge.

    After 40 steps with phi = 0.5 the step size is negligible.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    s = pd.Series(np.arange(100, 0, -5, dtype=float))
    p = damped_trend(0.5)(s, 40)
    steps = np.abs(np.diff(p))
    assert steps[-1] < steps[0] * 1e-3
