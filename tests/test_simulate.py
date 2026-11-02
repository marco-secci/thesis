# ================== #
# TEST_SIMULATE FILE #
# ================== #
"""The simulation must rebuild exactly the state the panel observes.

It is the check that holds the two halves of the model together: if the
simulated state diverged from the real one, forecasts beyond one year would
measure the simulator's defect instead of the model's quality.

Functions
---------
Tests collected by pytest; no other module uses them. They check
:mod:`simulate`. The small dummy model classes defined inside some tests
are documented by the test that defines them.

panel
    Fixture: donor x year panel (module-scoped, overrides the one in
    ``conftest.py``).
test_simulated_state_matches_real_panel
    The rebuilt state matches the panel covariates.
test_entrants_enter_the_total
    New donors raise the simulated total.
test_result_shape
    Shape of the simulation output.
test_entrants_series_sums_to_donors
    Entrants sum to the distinct donors.
test_entrants_forecast_never_negative
    The entrants forecast is non-negative and plausible.
test_simulate_rejects_negative_entrants
    Negative entrants are rejected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DONOR_YEAR
from simulate import entrants_series, initial_state, simulate


# ============ #
# PANEL METHOD #
# ============ #
@pytest.fixture(scope="module")
def panel():
    """Module fixture: the donor x year panel, or skip if not yet produced.

    Returns
    -------
    pandas.DataFrame
        Contents of ``DONOR_YEAR``.
    """
    if not DONOR_YEAR.exists():
        pytest.skip("panel non ancora prodotto: esegui `make panel`")
    return pd.read_parquet(DONOR_YEAR)


# ============================================== #
# TEST_SIMULATED_STATE_MATCHES_REAL_PANEL METHOD #
# ============================================== #
def test_simulated_state_matches_real_panel(panel):
    """The covariates of year O+1 rebuilt from the state at O equal the panel's.

    They must be identical to those the panel computes from the observed
    data (checked for O = 2018 on donors already present at O).

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    O = 2018
    history = panel[panel.year <= O]
    st = initial_state(history, O, n_sims=1)
    sim = st.rows(O + 1)
    sim["donor_id"] = np.sort(history[history.year == O].donor_id.unique())

    real = panel[(panel.year == O + 1) & (panel.entry_year <= O)]
    j = real.merge(sim, on="donor_id", suffixes=("_v", "_s"))
    assert len(j) == len(real) > 10_000

    for c in ["prev_count", "log1p_cum", "cum_active_years", "career_years", "age"]:
        assert np.allclose(j[f"{c}_v"].astype(float), j[f"{c}_s"].astype(float)), c
    assert (j["recency_class_v"].astype(str) == j["recency_class_s"].astype(str)).all()


# ==================================== #
# TEST_ENTRANTS_ENTER_THE_TOTAL METHOD #
# ==================================== #
def test_entrants_enter_the_total(panel):
    """Without new donors the simulated total must be lower.

    Uses a dummy model with constant mean 0.5.

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    class Constant:
        mu_max = 1.0
        def predict_mu(self, rows):
            return np.full(len(rows), 0.5)
        def sample(self, rows, rng):
            return rng.poisson(self.predict_mu(rows))

    O, H = 2020, 2
    history = panel[panel.year <= O]
    without = simulate(Constant(), history, O, H, np.zeros(H), n_sims=4, seed=1)
    with_ = simulate(Constant(), history, O, H, np.full(H, 3000), n_sims=4, seed=1)
    assert (with_.mean(axis=1) > without.mean(axis=1)).all()


# ======================== #
# TEST_RESULT_SHAPE METHOD #
# ======================== #
def test_result_shape(panel):
    """The simulation returns a (horizon, n_sims) matrix; a zero model sums to 0.

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    class Zero:
        mu_max = 0.0
        def predict_mu(self, rows):
            return np.zeros(len(rows))
        def sample(self, rows, rng):
            return np.zeros(len(rows), dtype=int)

    tot = simulate(Zero(), panel[panel.year <= 2020], 2020, 3, np.zeros(3), n_sims=5)
    assert tot.shape == (3, 5)
    assert tot.sum() == 0


# ========================================== #
# TEST_ENTRANTS_SERIES_SUMS_TO_DONORS METHOD #
# ========================================== #
def test_entrants_series_sums_to_donors(panel):
    """Summing new donors over the years gives the number of distinct donors.

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert entrants_series(panel).sum() == panel.donor_id.nunique()


# ============================================ #
# TEST_ENTRANTS_FORECAST_NEVER_NEGATIVE METHOD #
# ============================================ #
def test_entrants_forecast_never_negative(panel):
    """The entrants forecast is non-negative and below 6,000 for several origins.

    2009 acts as an edge: if it enters the trend estimate the forecast comes
    out negative, and that is how a component worth ~12% of the total used
    to vanish.

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    from simulate import forecast_entrants
    for O in (2016, 2018, 2020, 2022):
        p = forecast_entrants(panel, O, 3)
        assert (p >= 0).all()
        assert p.max() < 6000, "ordine di grandezza implausibile"


# ============================================== #
# TEST_SIMULATE_REJECTS_NEGATIVE_ENTRANTS METHOD #
# ============================================== #
def test_simulate_rejects_negative_entrants(panel):
    """A negative expected number of new donors raises ``ValueError``.

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    class Zero:
        mu_max = 0.0
        def predict_mu(self, r): return np.zeros(len(r))
        def sample(self, r, rng): return np.zeros(len(r), dtype=int)
    with pytest.raises(ValueError, match="negativo"):
        simulate(Zero(), panel[panel.year <= 2020], 2020, 2, np.array([-5.0, 3.0]))
