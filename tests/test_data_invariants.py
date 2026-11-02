# ========================= #
# TEST_DATA_INVARIANTS FILE #
# ========================= #
"""Data invariants: properties that must hold, not expected numbers.

A test that pins "353,671 rows" would fail at the first legitimate fix to the
pipeline and would be updated without thinking. These check structural
properties, which signal a real bug if they break.

Functions
---------
Tests collected by pytest; no other module uses them. They check the output
of :mod:`ingest` and :mod:`clean`.

test_event_key_is_unique
    The event key identifies one row.
test_all_years_present
    Every year of the window is present.
test_no_year_anomalous_in_size
    No year is suspiciously small.
test_required_fields_not_null
    Key fields are never missing.
test_age_in_range
    Ages lie in the admissible range.
test_donation_date_consistent_with_year
    Date and year agree.
test_birth_precedes_donation
    Birth comes before donation.
test_age_consistent_with_birth_year
    Age agrees with the birth year.
test_leakage_columns_declared
    The leaky columns exist and are declared.
"""

from __future__ import annotations

import pandas as pd

from config import (AGE_MAX, AGE_MIN, EVENT_KEY, LEAKY_COLUMNS,
                             YEAR_MAX, YEAR_MIN)


# =============================== #
# TEST_EVENT_KEY_IS_UNIQUE METHOD #
# =============================== #
def test_event_key_is_unique(events):
    """A donor cannot donate the same type twice at the same instant.

    It is the invariant that makes merging the two 2023 files legitimate: if
    it broke, the merge would be collapsing distinct donations.

    Parameters
    ----------
    events : pandas.DataFrame
        Fixture: raw events (``EVENTS_RAW``).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert not events.duplicated(subset=EVENT_KEY).any()


# ============================= #
# TEST_ALL_YEARS_PRESENT METHOD #
# ============================= #
def test_all_years_present(donations):
    """No year of the period may be missing: a lost year is an unread file.

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
    years = set(donations.donation_year.unique())
    assert years == set(range(YEAR_MIN, YEAR_MAX + 1))


# ===================================== #
# TEST_NO_YEAR_ANOMALOUS_IN_SIZE METHOD #
# ===================================== #
def test_no_year_anomalous_in_size(donations):
    """No year may be worth less than half the median of the others.

    Catches the case in which a file is read only partially — a wrong sheet,
    a shifted header — without pinning expected counts.

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
    per_year = donations.groupby("donation_year").size()
    assert (per_year > 0.5 * per_year.median()).all(), per_year.to_dict()


# ==================================== #
# TEST_REQUIRED_FIELDS_NOT_NULL METHOD #
# ==================================== #
def test_required_fields_not_null(donations):
    """Identifier, date, type, age and year are never missing.

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
    for col in ("donor_id", "donation_datetime", "donation_type", "age", "donation_year"):
        assert donations[col].notna().all(), f"{col} contiene nulli"


# ======================== #
# TEST_AGE_IN_RANGE METHOD #
# ======================== #
def test_age_in_range(donations):
    """Every age lies within ``AGE_MIN``-``AGE_MAX``.

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
    assert donations.age.between(AGE_MIN, AGE_MAX).all()


# ============================================== #
# TEST_DONATION_DATE_CONSISTENT_WITH_YEAR METHOD #
# ============================================== #
def test_donation_date_consistent_with_year(donations):
    """The year of ``donation_datetime`` equals ``donation_year``.

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
    assert (donations.donation_datetime.dt.year == donations.donation_year).all()


# =================================== #
# TEST_BIRTH_PRECEDES_DONATION METHOD #
# =================================== #
def test_birth_precedes_donation(donations):
    """The birth year is always earlier than the donation year.

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
    assert (donations.birth_year < donations.donation_year).all()


# ========================================== #
# TEST_AGE_CONSISTENT_WITH_BIRTH_YEAR METHOD #
# ========================================== #
def test_age_consistent_with_birth_year(donations):
    """Age equals year - birth, or one less if the birthday has not come yet.

    Age is computed on the exact date, so no other difference is possible.

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
    diff = donations.donation_year - donations.birth_year - donations.age
    assert diff.isin([0, 1]).all()


# ==================================== #
# TEST_LEAKAGE_COLUMNS_DECLARED METHOD #
# ==================================== #
def test_leakage_columns_declared(donations):
    """The columns that rewrite the target exist and are the declared ones.

    They remain in the dataset as a check, but their name must appear in
    LEAKY_COLUMNS: that is the list the model tests use to verify they do not
    end up in a design matrix.

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
    for col in LEAKY_COLUMNS:
        assert col in donations.columns
