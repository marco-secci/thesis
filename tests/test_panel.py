# =============== #
# TEST_PANEL FILE #
# =============== #
"""Invariants of the donor x year panel, temporal leakage included.

Functions
---------
Tests collected by pytest; no other module uses them. They check
:mod:`panel`.

test_panel_accounts_for_every_donation
    The panel sums to the number of donations.
test_grid_is_dense_and_contiguous
    No gaps between entry and last year.
test_one_row_per_donor_year
    No duplicate (donor, year).
test_zeros_materialised
    Zero years are present and coded as inactive.
test_entry_year_is_always_active
    The entry row is active.
test_covariates_are_causal
    Rebuilding up to 2020 matches the full panel cut at 2020.
"""

from __future__ import annotations

import pandas as pd

from config import DONATIONS
from panel import build_panel, load_donations


# ============================================= #
# TEST_PANEL_ACCOUNTS_FOR_EVERY_DONATION METHOD #
# ============================================= #
def test_panel_accounts_for_every_donation(panel, donations):
    """The sum of y over the panel is the number of donations.

    Without this, going from individual to aggregate is not a sum but an
    approximation.

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.
    donations : pandas.DataFrame
        Fixture: cleaned donations (``DONATIONS``).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert int(panel.y.sum()) == len(donations)


# ======================================== #
# TEST_GRID_IS_DENSE_AND_CONTIGUOUS METHOD #
# ======================================== #
def test_grid_is_dense_and_contiguous(panel):
    """Every donor has one row for each year from their entry onwards, without gaps.

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
    g = panel.groupby("donor_id").year
    expected = (g.max() - g.min() + 1)
    assert (g.size() == expected).all()


# ================================== #
# TEST_ONE_ROW_PER_DONOR_YEAR METHOD #
# ================================== #
def test_one_row_per_donor_year(panel):
    """No (donor, year) pair appears twice.

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
    assert not panel.duplicated(subset=["donor_id", "year"]).any()


# ============================== #
# TEST_ZEROS_MATERIALISED METHOD #
# ============================== #
def test_zeros_materialised(panel):
    """The panel must contain zeros: if it does not, the truncation is still there.

    Also checks that ``active`` is exactly ``y > 0``.

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
    assert (panel.y == 0).any()
    assert panel.active.eq(panel.y > 0).all()


# ======================================= #
# TEST_ENTRY_YEAR_IS_ALWAYS_ACTIVE METHOD #
# ======================================= #
def test_entry_year_is_always_active(panel):
    """A donor enters the panel in the year of their first donation.

    On the entry row y is positive and ``is_entry_year`` is 1.

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
    entry = panel[panel.year == panel.entry_year]
    assert (entry.y > 0).all()
    assert entry.is_entry_year.eq(1).all()


# ================================= #
# TEST_COVARIATES_ARE_CAUSAL METHOD #
# ================================= #
def test_covariates_are_causal():
    """The panel truncated at O must equal the full panel restricted to O.

    It is the anti-leakage test: if a covariate used information later than
    the row's year — the last observed donation, the career total, the exit
    date — building the panel with data up to 2020 would give values different
    from the panel built on everything and then cut at 2020. If they match,
    every row contains only what was known at the end of its year.

    Rebuilds both panels from ``DONATIONS``, so it needs the cleaned data.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    ORIGIN = 2020
    full = build_panel(load_donations(2025), 2025)
    truncated = build_panel(load_donations(ORIGIN), ORIGIN)

    expected = (full[full.year <= ORIGIN]
                .sort_values(["donor_id", "year"]).reset_index(drop=True))
    obtained = truncated.sort_values(["donor_id", "year"]).reset_index(drop=True)

    # The truncated panel does not know donors who entered after the origin.
    expected = expected[expected.donor_id.isin(set(obtained.donor_id))].reset_index(drop=True)

    # Without this guard the comparison would also pass on empty frames.
    assert len(obtained) > 100_000, f"confronto su troppe poche righe: {len(obtained)}"
    assert len(expected) == len(obtained)

    causal = ["donor_id", "year", "y", "active", "n_types", "recency",
               "recency_class", "is_entry_year", "prev_count", "prev_active",
               "prev_n_types", "cum_count", "cum_active_years", "tenure",
               "career_years", "left_censored",
               "age", "birth_year", "sex", "entry_year"]
    pd.testing.assert_frame_equal(expected[causal], obtained[causal],
                                  check_dtype=False)
