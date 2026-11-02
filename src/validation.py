# =============== #
# VALIDATION FILE #
# =============== #
"""Rolling-origin temporal validation.

**Why not a random split.** In the original repository the only holdout was
`np.random.rand(len(df)) < 0.8`: a 2011 row could end up in test and a 2024 one
in training, and the same donor appeared in both. A model evaluated this way
measures how well it interpolates within a period it has already seen, not how
well it forecasts. Here the only allowed split cuts through time.

**Why rolling origin and not a single split.** A single split is a single draw.
The one used in the thesis — training 2009-2019, test 2020-2025 — contains the
whole pandemic in the test set: any ranking of models that comes out of it is
dominated by an unrepeatable event. Sliding the origin gives several
independent evaluations of the same horizon, and shows whether a model really
wins or just got lucky once.

**Expanding window.** At each origin O the model sees the whole history up to
and including O and nothing after. This is the real situation of whoever has to
make the forecast.

Classes and functions
---------------------
Origin
    Frozen dataclass describing one forecast origin and its target periods.
    Produced by :func:`rolling_origins` and consumed by
    :func:`backtest_series` and :mod:`backtest_models`.
rolling_origins
    Builds the expanding-window origins. Used by :mod:`baselines`,
    :mod:`backtest_models` and ``tests/test_validation.py``.
metrics
    MAE, RMSE, MAPE and percentage bias of a forecast. Used inside this
    module by :func:`summarise`, and by :mod:`backtest_models` and
    ``tests/test_validation.py``.
backtest_series
    Runs every model on every origin, one row per (model, origin, horizon).
    Used by :mod:`baselines` and ``tests/test_validation.py``.
summarise
    Aggregates a backtest table into metrics per model (and optionally per
    horizon). Used by :mod:`baselines` and :mod:`backtest_models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

#: A model is a function: observed history -> forecast for the next `horizon`
#: periods. Nothing else. This allows comparing trivial baselines, GLMs and
#: time-series models on the same grid.
ModelFn = Callable[[pd.Series, int], np.ndarray]


# ============ #
# ORIGIN CLASS #
# ============ #
@dataclass(frozen=True)
class Origin:
    """A single forecast origin.

    Attributes
    ----------
    origin : object
        Last observed period (e.g. a year or a ``pandas.Period``): the model
        sees the history up to and including it.
    targets : tuple
        Periods to forecast, in chronological order; ``targets[h-1]`` is the
        period at horizon ``h``.
    """
    origin: object                 # last observed period
    targets: tuple                 # periods to forecast, in order

    # ============== #
    # __LEN__ METHOD #
    # ============== #
    def __len__(self) -> int:
        """Number of target periods, i.e. the maximum horizon of this origin.

        Returns
        -------
        int
            ``len(self.targets)``.
        """
        return len(self.targets)


# ====================== #
# ROLLING_ORIGINS METHOD #
# ====================== #
def rolling_origins(index: Iterable, min_train: int, max_horizon: int,
                    step: int = 1) -> list[Origin]:
    """Expanding-window origins over an ordered index of periods.

    The first origin is the ``min_train``-th period; the last is the
    second-to-last period, so every origin has at least one target. Near the
    end of the index the targets are truncated, so later origins can have
    fewer than `max_horizon` targets.

    Parameters
    ----------
    index : iterable
        Ordered periods of the series (e.g. ``series.index``).
    min_train : int
        Minimum number of observed periods before the first forecast can be
        made.
    max_horizon : int
        Maximum number of periods to forecast from each origin.
    step : int, default 1
        Spacing between consecutive origins (1 = every period).

    Returns
    -------
    list of Origin
        Origins in chronological order.
    """
    idx = list(index)
    out = []
    for i in range(min_train - 1, len(idx) - 1, step):
        targets = tuple(idx[i + 1: i + 1 + max_horizon])
        if targets:
            out.append(Origin(origin=idx[i], targets=targets))
    return out


# ============== #
# METRICS METHOD #
# ============== #
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Absolute, squared and percentage error, and systematic bias.

    Bias is kept separate from error because it answers a different question:
    MAPE says how much a model is wrong, bias says whether it is always wrong
    on the same side. A model with low MAPE and strong bias systematically
    underestimates (or overestimates), and over a long horizon the error
    accumulates.

    Parameters
    ----------
    y_true : array_like
        Observed values. Must be non-zero, since percentage errors divide by
        them.
    y_pred : array_like
        Forecast values, same length as `y_true`.

    Returns
    -------
    dict
        ``"mae"`` : float
            Mean absolute error.
        ``"rmse"`` : float
            Root mean squared error.
        ``"mape"`` : float
            Mean absolute percentage error, in percent.
        ``"bias_pct"`` : float
            Mean signed percentage error, in percent (positive =
            overestimation).
        ``"n"`` : int
            Number of observations.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100),
        "bias_pct": float(np.mean(err / y_true) * 100),
        "n": int(len(y_true)),
    }


# ====================== #
# BACKTEST_SERIES METHOD #
# ====================== #
def backtest_series(models: Mapping[str, ModelFn], series: pd.Series,
                    origins: list[Origin]) -> pd.DataFrame:
    """Evaluate every model on every origin, horizon by horizon.

    Returns one row per (model, origin, horizon) with observed and predicted
    value: every later aggregation — by horizon, by model, by year — is done
    from here, so all models are compared on exactly the same target periods.

    Parameters
    ----------
    models : mapping of str to ModelFn
        Model name -> function ``fn(history, horizon) -> numpy.ndarray``
        returning `horizon` forecasts.
    series : pandas.Series
        Full observed series, indexed by period. Each model only receives
        ``series.loc[:origin]``.
    origins : list of Origin
        Forecast origins, e.g. from :func:`rolling_origins`.

    Returns
    -------
    pandas.DataFrame
        Columns ``modello``, ``origine``, ``orizzonte`` (1-based horizon),
        ``periodo`` (target period), ``osservato`` and ``previsto``.

    Raises
    ------
    ValueError
        If a model returns a number of values different from the number of
        targets of an origin.
    """
    rows = []
    for org in origins:
        history = series.loc[:org.origin]
        for name, fn in models.items():
            pred = np.asarray(fn(history, len(org.targets)), dtype=float)
            if len(pred) != len(org.targets):
                raise ValueError(
                    f"{name}: attesi {len(org.targets)} valori, ricevuti {len(pred)}")
            for h, (target, yhat) in enumerate(zip(org.targets, pred), start=1):
                rows.append({"modello": name, "origine": org.origin,
                             "orizzonte": h, "periodo": target,
                             "osservato": float(series.loc[target]),
                             "previsto": float(yhat)})
    return pd.DataFrame(rows)


# ================ #
# SUMMARISE METHOD #
# ================ #
def summarise(bt: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Summarise a backtest with the metrics, per model (and optionally horizon).

    Parameters
    ----------
    bt : pandas.DataFrame
        Backtest table with at least the columns ``osservato`` and
        ``previsto`` plus the grouping columns, as returned by
        :func:`backtest_series`.
    by : list of str, optional
        Grouping columns. Defaults to ``["modello"]``; pass e.g.
        ``["modello", "orizzonte"]`` for a per-horizon summary.

    Returns
    -------
    pandas.DataFrame
        One row per group, with the grouping columns and the keys of
        :func:`metrics` as columns, sorted by increasing ``mape``.
    """
    by = by or ["modello"]
    out = (bt.groupby(by)
             .apply(lambda g: pd.Series(metrics(g.osservato, g.previsto)),
                    include_groups=False)
             .reset_index())
    out["n"] = out["n"].astype(int)
    return out.sort_values("mape").reset_index(drop=True)
