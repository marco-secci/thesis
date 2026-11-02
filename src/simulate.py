# ============= #
# SIMULATE FILE #
# ============= #
"""From the per-donor forecast to the annual total, by simulation.

**Why summing expected values is not enough.** One year ahead it works: the
covariates for year O+1 are all determined by the data up to O. From two years
on they are not: `recency`, `prev_count` and the cumulative counts depend on how
much the donor *actually* donated in O+1. Propagating the means would give a
donor who donates 0.7 times a year and never drops out — i.e. exactly the
behaviour the model should predict would vanish from the forecast.

So one path per donor is drawn, the state is updated with the draw, and the
step is repeated. Many paths give the distribution of the total, and with it
intervals that arise from the mechanism rather than from a formula.

**The total has two components.** Donors already known at the origin, and those
who will enter later. The second is not a detail: ~2,500 new donors a year, who
in their first year donate as much as an average donor. A model that forecasts
only the known cohort underestimates structurally, and the underestimate grows
with the horizon.

Classes and functions
---------------------
_recency_class
    Private helper: recency coding of :func:`panel.build_panel` applied to
    simulated state. Used only inside this module, by :meth:`State.rows`.
State
    Dataclass with static donor attributes and per-path dynamic state; its
    methods are :attr:`State.n`, :attr:`State.n_sims`, :meth:`State.rows`,
    :meth:`State.update` and :meth:`State.extend`. Used inside this module.
initial_state
    State of the known donors at the origin. Used by :func:`simulate` and
    ``tests/test_simulate.py``.
_entrants_state
    Private helper: state block for new donors. Used only inside this
    module, by :func:`simulate`.
entrants_pool
    Attributes of recent entrants, resampled for new donors. Used only
    inside this module, by :func:`simulate`.
entrants_series
    Number of new donors per year. Used by :func:`forecast_entrants` and
    ``tests/test_simulate.py``.
forecast_entrants
    Damped-trend forecast of the number of new donors. Used by
    :mod:`forecast`, :mod:`backtest_models` and ``tests/test_simulate.py``.
simulate
    Monte Carlo simulation of the annual total. Used by :mod:`forecast`,
    :mod:`backtest_models` and ``tests/test_simulate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from panel import RECENCY_CAP

#: Donor attributes that do not change over time.
STATIC = ["birth_year", "sex", "donor_class_at_entry", "entry_year",
          "career_start", "left_censored"]


# ===================== #
# _RECENCY_CLASS METHOD #
# ===================== #
def _recency_class(recency: np.ndarray, is_entry: np.ndarray) -> np.ndarray:
    """Same coding used by `panel.build_panel`, applied to simulated state.

    Parameters
    ----------
    recency : numpy.ndarray
        Years since the last active year, one value per (donor, path).
    is_entry : numpy.ndarray of bool
        Whether the row is the donor's entry year, same shape as `recency`.

    Returns
    -------
    numpy.ndarray of str
        ``"entry"`` where `is_entry`, otherwise ``"1"``..``"5"`` (up to
        ``panel.RECENCY_CAP``) or ``"6+"``.
    """
    out = np.where(recency <= RECENCY_CAP, recency.astype(int).astype(str), "6+")
    return np.where(is_entry, "entry", out)


# =========== #
# STATE CLASS #
# =========== #
@dataclass
class State:
    """State of the donors at the start of a year to simulate.

    Static attributes have shape (n,), dynamic ones (n, n_sims): every
    simulated path has its own history.

    Attributes
    ----------
    birth_year, sex, donor_class_at_entry, entry_year : numpy.ndarray
        Static donor attributes, shape ``(n,)``.
    career_start : numpy.ndarray
        First year of the donor's career, from the raw data (entry year if
        unknown), shape ``(n,)``.
    left_censored : numpy.ndarray
        1 if the career started before the observation window, shape
        ``(n,)``.
    last_active_year : numpy.ndarray
        Last year with at least one donation, shape ``(n, n_sims)``.
    cum_count : numpy.ndarray
        Donations made so far, shape ``(n, n_sims)``.
    cum_active_years : numpy.ndarray
        Years with at least one donation so far, shape ``(n, n_sims)``.
    prev_count : numpy.ndarray
        Donations in the previous year, shape ``(n, n_sims)``.
    prev_active : numpy.ndarray
        1 if active in the previous year, shape ``(n, n_sims)``.
    """
    birth_year: np.ndarray
    sex: np.ndarray
    donor_class_at_entry: np.ndarray
    entry_year: np.ndarray
    career_start: np.ndarray          # first career year, from the raw data
    left_censored: np.ndarray
    last_active_year: np.ndarray      # (n, n_sims)
    cum_count: np.ndarray             # (n, n_sims)
    cum_active_years: np.ndarray      # (n, n_sims)
    prev_count: np.ndarray            # (n, n_sims)
    prev_active: np.ndarray           # (n, n_sims)

    # ======== #
    # N METHOD #
    # ======== #
    @property
    def n(self) -> int:
        """Number of donors in the state.

        Returns
        -------
        int
            Length of the static attributes.
        """
        return len(self.birth_year)

    # ============= #
    # N_SIMS METHOD #
    # ============= #
    @property
    def n_sims(self) -> int:
        """Number of simulated paths.

        Returns
        -------
        int
            Second dimension of the dynamic attributes.
        """
        return self.last_active_year.shape[1]

    # =========== #
    # ROWS METHOD #
    # =========== #
    def rows(self, year: int) -> pd.DataFrame:
        """The panel rows for year `year`, one per (donor, path).

        Parameters
        ----------
        year : int
            Calendar year being simulated.

        Returns
        -------
        pandas.DataFrame
            ``n * n_sims`` rows in donor-major order (all paths of donor 0,
            then donor 1, ...), with the covariates expected by
            :class:`features.DesignSpec`: ``year``, ``prev_count``,
            ``log1p_cum``, ``cum_active_years``, ``career_years``,
            ``left_censored``, ``age``, ``recency_class``, ``sex`` and
            ``donor_class_at_entry``.
        """
        s = self.n_sims
        rep = lambda a: np.repeat(a, s)                      # static -> (n*s,)
        flat = lambda a: a.reshape(-1)                        # dynamic -> (n*s,)

        recency = flat(year - self.last_active_year)
        is_entry = rep(self.entry_year) == year
        cum = flat(self.cum_count)
        return pd.DataFrame({
            "year": year,
            "prev_count": flat(self.prev_count),
            "log1p_cum": np.log1p(cum),
            "cum_active_years": flat(self.cum_active_years),
            "career_years": np.maximum(year - rep(self.career_start), 0),
            "left_censored": rep(self.left_censored),
            "age": year - rep(self.birth_year),
            "recency_class": _recency_class(recency, is_entry),
            "sex": rep(self.sex),
            "donor_class_at_entry": rep(self.donor_class_at_entry),
        })

    # ============= #
    # UPDATE METHOD #
    # ============= #
    def update(self, year: int, y: np.ndarray) -> None:
        """Apply the outcome of the year: y has shape (n, n_sims).

        Parameters
        ----------
        year : int
            Calendar year just simulated.
        y : numpy.ndarray
            Simulated donations per donor and path, shape ``(n, n_sims)``.

        Returns
        -------
        None
            The dynamic attributes are updated in place.
        """
        active = (y > 0)
        self.last_active_year = np.where(active, year, self.last_active_year)
        self.cum_count = self.cum_count + y
        self.cum_active_years = self.cum_active_years + active
        self.prev_count = y
        self.prev_active = active.astype(np.int64)

    # ============= #
    # EXTEND METHOD #
    # ============= #
    def extend(self, other: "State") -> None:
        """Append a block of new donors to the current state.

        Parameters
        ----------
        other : State
            State of the donors to append; must have the same ``n_sims``.

        Returns
        -------
        None
            Static attributes are concatenated and dynamic ones stacked
            row-wise, in place.
        """
        for c in STATIC:
            setattr(self, c, np.concatenate([getattr(self, c), getattr(other, c)]))
        for c in ("last_active_year", "cum_count", "cum_active_years",
                  "prev_count", "prev_active"):
            setattr(self, c, np.vstack([getattr(self, c), getattr(other, c)]))


# ==================== #
# INITIAL_STATE METHOD #
# ==================== #
def initial_state(panel: pd.DataFrame, origin: int, n_sims: int) -> State:
    """State of the donors known at the end of year `origin`.

    Each donor has, by construction of the grid, exactly one row in year
    `origin`: the state is read from that row.

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel (see :func:`panel.build_panel`) covering at least
        year `origin`.
    origin : int
        Last observed year.
    n_sims : int
        Number of simulated paths; the dynamic state is replicated across
        them.

    Returns
    -------
    State
        One donor per row of `panel` in year `origin`, sorted by
        ``donor_id``, with the outcome of year `origin` already included in
        the cumulative counts.
    """
    r = panel[panel.year == origin].sort_values("donor_id").reset_index(drop=True)
    n = len(r)
    tile = lambda a: np.tile(np.asarray(a).reshape(-1, 1), (1, n_sims))

    # last active year: `origin` if they donated, otherwise origin - recency
    last = np.where(r.active.to_numpy() == 1, origin,
                    origin - r.recency.fillna(0).to_numpy())
    return State(
        birth_year=r.birth_year.to_numpy(),
        sex=r.sex.astype(str).to_numpy(),
        donor_class_at_entry=r.donor_class_at_entry.astype(str).to_numpy(),
        entry_year=r.entry_year.to_numpy(),
        career_start=(r.first_activity_year.fillna(r.entry_year).to_numpy()),
        left_censored=r.left_censored.to_numpy(),
        last_active_year=tile(last),
        cum_count=tile(r.cum_count.to_numpy() + r.y.to_numpy()),
        cum_active_years=tile(r.cum_active_years.to_numpy() + r.active.to_numpy()),
        prev_count=tile(r.y.to_numpy()),
        prev_active=tile(r.active.to_numpy()),
    )


# ====================== #
# _ENTRANTS_STATE METHOD #
# ====================== #
def _entrants_state(pool: pd.DataFrame, count: int, year: int, n_sims: int,
                    rng: np.random.Generator) -> State:
    """Block of new donors: attributes drawn from recent entrants.

    Newcomers have no history, so all dynamic state starts from zero and the
    recency class is "entry" — the same one the model saw in training for
    first donations.

    Parameters
    ----------
    pool : pandas.DataFrame
        Attributes of recent entrants, from :func:`entrants_pool`; rows are
        resampled with replacement.
    count : int
        Number of new donors to create.
    year : int
        Entry year of the new donors.
    n_sims : int
        Number of simulated paths.
    rng : numpy.random.Generator
        Random generator used for the resampling.

    Returns
    -------
    State
        `count` donors entering in `year`, with zero history; the birth
        year is ``year - age_at_entry``.
    """
    idx = rng.integers(0, len(pool), size=count)
    p = pool.iloc[idx]
    z = np.zeros((count, n_sims), dtype=np.int64)
    return State(
        birth_year=(year - p.age_at_entry.to_numpy()),
        sex=p.sex.astype(str).to_numpy(),
        donor_class_at_entry=p.donor_class_at_entry.astype(str).to_numpy(),
        entry_year=np.full(count, year),
        career_start=np.full(count, year, dtype=float),
        left_censored=np.zeros(count, dtype=np.int8),
        last_active_year=np.full((count, n_sims), year),
        cum_count=z.copy(), cum_active_years=z.copy(),
        prev_count=z.copy(), prev_active=z.copy(),
    )


# ==================== #
# ENTRANTS_POOL METHOD #
# ==================== #
def entrants_pool(panel: pd.DataFrame, origin: int, years: int = 5) -> pd.DataFrame:
    """Attributes of the donors who entered in the last `years` before the origin.

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel.
    origin : int
        Last observed year.
    years : int, default 5
        Width of the window: entry years ``origin - years + 1 .. origin``.

    Returns
    -------
    pandas.DataFrame
        One row per recent entrant, with columns ``age_at_entry``, ``sex``
        and ``donor_class_at_entry`` (as strings).
    """
    e = panel[(panel.year == panel.entry_year)
              & (panel.entry_year > origin - years)
              & (panel.entry_year <= origin)]
    return pd.DataFrame({
        "age_at_entry": e.age.to_numpy(),
        "sex": e.sex.astype(str).to_numpy(),
        "donor_class_at_entry": e.donor_class_at_entry.astype(str).to_numpy(),
    })


# ====================== #
# ENTRANTS_SERIES METHOD #
# ====================== #
def entrants_series(panel: pd.DataFrame) -> pd.Series:
    """Number of new donors per year.

    Beware of the first year: in 2009 all 12,957 active donors "enter",
    because there is no observation before. It is not a record year for
    entries, it is the edge of the window. Anyone estimating a trend on this
    series must exclude it — see `forecast_entrants`.

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel with ``year`` and ``entry_year``.

    Returns
    -------
    pandas.Series
        Number of donors whose entry year is each year, indexed by year in
        increasing order.
    """
    return (panel[panel.year == panel.entry_year]
            .groupby("year").size().sort_index())


# ======================== #
# FORECAST_ENTRANTS METHOD #
# ======================== #
def forecast_entrants(panel: pd.DataFrame, origin: int, horizon: int,
                      since: int | None = None) -> np.ndarray:
    """How many new donors to expect in the years after the origin.

    A per-donor model cannot forecast donors who do not exist yet: their
    number is a separate piece of the total, and not a small one — 1,200-2,000
    people a year who at their first donation make on average 1.3, i.e. about
    an eighth of the annual donations.

    The trend is estimated **excluding the edge years**: including 2009 the
    line has a strongly negative slope and the forecast comes out negative,
    i.e. an impossible number of new donors. The result is clipped at zero
    anyway, but that clipping is a safety net, not the mechanism.

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel up to (at least) `origin`.
    origin : int
        Last observed year.
    horizon : int
        Number of years to forecast.
    since : int, optional
        First year of the entrants series used for the trend. Defaults to
        ``config.PANEL_BURN_IN_YEAR``.

    Returns
    -------
    numpy.ndarray
        Array of length `horizon` with the expected number of new donors per
        year, never negative. With fewer than 3 years available, the last
        observed value is repeated (0 if there is none); otherwise a damped
        trend with ``phi = 0.6`` (:func:`baselines.damped_trend`) is used.
    """
    from baselines import damped_trend
    from config import PANEL_BURN_IN_YEAR

    since = PANEL_BURN_IN_YEAR if since is None else since
    series = entrants_series(panel).loc[since:origin]
    if len(series) < 3:
        return np.full(horizon, float(series.iloc[-1]) if len(series) else 0.0)
    return np.clip(damped_trend(0.6)(series, horizon), 0.0, None)


# =============== #
# SIMULATE METHOD #
# =============== #
def simulate(model, panel: pd.DataFrame, origin: int, horizon: int,
             expected_entrants: np.ndarray, n_sims: int = 20,
             seed: int = 0) -> np.ndarray:
    """Simulated annual total: matrix (horizon, n_sims).

    `expected_entrants` is the number of new donors expected for each of the
    years to forecast — it comes from a separate model on the entries series,
    because the per-donor model by definition cannot forecast donors who do
    not exist yet.

    Each year, the new donors (rounded count) are added to the state, every
    donor-path is drawn from ``model.sample`` and the state is updated with
    the draw.

    Parameters
    ----------
    model : models.Model
        Fitted per-donor model exposing ``sample(rows, rng)``.
    panel : pandas.DataFrame
        Donor x year panel up to (at least) `origin`, used for the initial
        state and the pool of entrant attributes.
    origin : int
        Last observed year.
    horizon : int
        Number of years to simulate.
    expected_entrants : array_like
        Expected new donors for each of the `horizon` years, e.g. from
        :func:`forecast_entrants`.
    n_sims : int, default 20
        Number of simulated paths.
    seed : int, default 0
        Seed of the random generator, for reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(horizon, n_sims)``: total donations of each
        simulated year (row) in each path (column).

    Raises
    ------
    ValueError
        If any element of `expected_entrants` is negative.
    """
    expected_entrants = np.asarray(expected_entrants, dtype=float)
    if (expected_entrants < 0).any():
        raise ValueError(
            f"numero di nuovi donatori negativo: {expected_entrants}. "
            "Usare `forecast_entrants`, che esclude gli anni di bordo.")

    rng = np.random.default_rng(seed)
    state = initial_state(panel, origin, n_sims)
    pool = entrants_pool(panel, origin)

    totals = np.zeros((horizon, n_sims))
    for h in range(horizon):
        year = origin + 1 + h
        count = int(round(expected_entrants[h]))
        if count > 0 and len(pool):
            state.extend(_entrants_state(pool, count, year, n_sims, rng))

        rows = state.rows(year)
        y = model.sample(rows, rng).reshape(state.n, n_sims)
        state.update(year, y)
        totals[h] = y.sum(axis=0)
    return totals
