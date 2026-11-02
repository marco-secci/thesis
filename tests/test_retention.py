# =================== #
# TEST_RETENTION FILE #
# =================== #
"""Invariants of retention and of the entry-cohort correction.

Functions
---------
Tests collected by pytest; no other module uses them. They check
:mod:`retention`.

test_one_row_per_donor
    One row per donor in the donor table.
test_entry_year_never_after_first_event
    The real entry year is never later than the observed one.
test_observation_truncated
    Observation and cohort windows are consistent.
test_genuine_entries_are_a_subset
    The corrected cohort is a strict subset.
test_no_genuine_entry_is_truncated
    Genuine entries are never left-truncated.
test_retention_non_negative
    Retention is never negative.
test_distribution_sums_to_hundred
    The distribution sums to 100%.
test_survival_shares_well_formed
    Survival shares are well formed.
test_2009_cohort_is_half_veterans
    About half the observed 2009 cohort are veterans.
"""

from __future__ import annotations

import pandas as pd

import retention as R


# ============================= #
# TEST_ONE_ROW_PER_DONOR METHOD #
# ============================= #
def test_one_row_per_donor(donations):
    """The donor table has one row per donor.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    assert not t.donor_id.duplicated().any()


# ============================================== #
# TEST_ENTRY_YEAR_NEVER_AFTER_FIRST_EVENT METHOD #
# ============================================== #
def test_entry_year_never_after_first_event(donations):
    """The real entry year cannot be later than the first observed event.

    If the activity start date were, the data is inconsistent and must be
    carried back to the first event.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    assert (t.anno_reale <= t.anno_osservato).all()


# ================================= #
# TEST_OBSERVATION_TRUNCATED METHOD #
# ================================= #
def test_observation_truncated(donations):
    """Measures stop at OBSERVATION_MAX and cohorts end two years earlier.

    Every cohort must have following years in which to return, otherwise
    the last one would be 'one-and-done' by construction.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    assert t.ultimo_anno.max() <= R.OBSERVATION_MAX
    assert R.COHORT_MAX <= R.OBSERVATION_MAX - 2


# ======================================== #
# TEST_GENUINE_ENTRIES_ARE_A_SUBSET METHOD #
# ======================================== #
def test_genuine_entries_are_a_subset(donations):
    """The corrected definition drops donors, it does not add any.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    obs = set(R.young_cohort(t, "osservata").donor_id)
    ent = set(R.young_cohort(t, "ingresso").donor_id)
    assert ent < obs


# ========================================= #
# TEST_NO_GENUINE_ENTRY_IS_TRUNCATED METHOD #
# ========================================= #
def test_no_genuine_entry_is_truncated(donations):
    """Corrected young cohort: all have a start date, none is left-truncated.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    young = R.young_cohort(t, "ingresso")
    assert (young.troncato_a_sinistra == 0).all()
    assert young.ha_data_inizio.all()


# ================================== #
# TEST_RETENTION_NON_NEGATIVE METHOD #
# ================================== #
def test_retention_non_negative(donations):
    """Retention r is never negative in the corrected young cohort.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    young = R.young_cohort(t, "ingresso")
    assert (young.r >= 0).all()


# ======================================== #
# TEST_DISTRIBUTION_SUMS_TO_HUNDRED METHOD #
# ======================================== #
def test_distribution_sums_to_hundred(donations):
    """The retention classes sum to 100% and cover every young donor.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    d = R.distribution(R.young_cohort(t, "ingresso"))
    assert abs(d.quota_pct.sum() - 100) < 1e-6
    assert d.donatori.sum() == len(R.young_cohort(t, "ingresso"))


# ======================================= #
# TEST_SURVIVAL_SHARES_WELL_FORMED METHOD #
# ======================================= #
def test_survival_shares_well_formed(donations):
    """Survival shares lie in [0, 100] with a constant cohort size.

    The share of active donors at k years is not a monotone function by
    definition (a donor can skip a year and come back), but the reference
    cohort must stay constant and the shares must lie between 0 and 100.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    s = R.survival(donations, R.young_cohort(t, "ingresso"))
    assert s.quota_pct.between(0, 100).all()
    assert (s.groupby("coorte").n_coorte.nunique() == 1).all()


# ======================================== #
# TEST_2009_COHORT_IS_HALF_VETERANS METHOD #
# ======================================== #
def test_2009_cohort_is_half_veterans(donations):
    """Between 45% and 55% of the observed 2009 young cohort started before 2009.

    It is the number the correction in sec:ret_def rests on. If it changes,
    the chapter text must be updated.

    Parameters
    ----------
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
    t = R.donor_table(donations)
    comp = R.cohort_composition(R.young_cohort(t, "osservata"))
    q = comp.loc[comp.coorte == 2009, "quota_pct"].iloc[0]
    assert 45 < q < 55
