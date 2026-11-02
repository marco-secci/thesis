# ======================= #
# TEST_GLM_INFERENCE FILE #
# ======================= #
"""Invariants of the sample and of the specification of the inferential GLM.

Functions
---------
Tests collected by pytest; no other module uses them. They check
:mod:`glm_inference`.

sample
    Fixture: estimation sample from the real data.
test_sample_starts_at_burn_in
    No year before the burn-in.
test_zeros_are_present
    The sample keeps the zero years.
test_no_leaky_column_in_formula
    The formulas contain no leaky column.
test_donation_type_is_lagged
    The type covariate is lagged.
test_explicit_references_in_formula
    Reference categories are explicit.
test_cohort_not_in_main_model
    The cohort is only in the secondary model.
test_dispersion_larger_with_zeros
    Overdispersion appears only with the zeros.
test_alpha_positive
    The method-of-moments alpha is positive.
"""

from __future__ import annotations

import pandas as pd
import pytest

import glm_inference as G
from config import LEAKY_COLUMNS, PANEL_BURN_IN_YEAR


# ============= #
# SAMPLE METHOD #
# ============= #
@pytest.fixture(scope="module")
def sample(panel, donations):
    """Module fixture: the GLM estimation sample built from the real data.

    Parameters
    ----------
    panel : pandas.DataFrame
        Fixture: donor x year panel.
    donations : pandas.DataFrame
        Fixture: cleaned donations.

    Returns
    -------
    pandas.DataFrame
        Output of :func:`glm_inference.sample`.
    """
    return G.sample(panel, donations)


# ==================================== #
# TEST_SAMPLE_STARTS_AT_BURN_IN METHOD #
# ==================================== #
def test_sample_starts_at_burn_in(sample):
    """The sample contains no year before ``PANEL_BURN_IN_YEAR``.

    Parameters
    ----------
    sample : pandas.DataFrame
        Fixture: GLM estimation sample.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert sample.year.min() >= PANEL_BURN_IN_YEAR


# ============================= #
# TEST_ZEROS_ARE_PRESENT METHOD #
# ============================= #
def test_zeros_are_present(sample):
    """More than half of the sample rows have y = 0.

    Without zero years dropout is unobservable and the dispersion diagnosis
    is flawed: it is the point of sec:glm_unit.

    Parameters
    ----------
    sample : pandas.DataFrame
        Fixture: GLM estimation sample.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert (sample.y == 0).mean() > 0.5


# ====================================== #
# TEST_NO_LEAKY_COLUMN_IN_FORMULA METHOD #
# ====================================== #
def test_no_leaky_column_in_formula():
    """No column of ``LEAKY_COLUMNS`` appears in either GLM formula.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    for col in LEAKY_COLUMNS:
        assert col not in G.FORMULA
        assert col not in G.FORMULA_COHORT


# =================================== #
# TEST_DONATION_TYPE_IS_LAGGED METHOD #
# =================================== #
def test_donation_type_is_lagged(panel, donations):
    """The donation type covariate of year t comes only from years before t.

    Using the current one would be conditioning on the response.

    The `panel` fixture is requested but not used directly.

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
    tp = G.lagged_type(donations)
    counts = (donations.groupby(["donor_id", "donation_year", "donation_type"],
                                observed=True).size().reset_index())
    first = counts.groupby("donor_id", observed=True).donation_year.min()
    m = tp.merge(first.rename("first_year"), on="donor_id")
    assert (m.year > m.first_year).all()


# ========================================== #
# TEST_EXPLICIT_REFERENCES_IN_FORMULA METHOD #
# ========================================== #
def test_explicit_references_in_formula():
    """The formula names its reference categories explicitly.

    The previous version inherited alphabetical ordering and ended up with a
    reference category made of a single observation.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert f"reference='{G.REF_CLASS}'" in G.FORMULA
    assert f"reference='{G.REF_TYPE}'" in G.FORMULA


# ==================================== #
# TEST_COHORT_NOT_IN_MAIN_MODEL METHOD #
# ==================================== #
def test_cohort_not_in_main_model():
    """The cohort is only in the secondary formula.

    Age, period and cohort are not jointly identifiable.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    assert "coorte_ingresso" not in G.FORMULA
    assert "coorte_ingresso" in G.FORMULA_COHORT


# ======================================== #
# TEST_DISPERSION_LARGER_WITH_ZEROS METHOD #
# ======================================== #
def test_dispersion_larger_with_zeros(panel):
    """Var/mean is below 1.5 on truncated data and above it with the zeros.

    The ratio on the truncated data clearly underestimates the real
    overdispersion. It is the number that refutes the previous sec:5.3.

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
    d = G.dispersion(panel).set_index("grano").var_su_media
    truncated, complete = d.iloc[0], d.iloc[1]
    assert truncated < 1.5 < complete


# ========================== #
# TEST_ALPHA_POSITIVE METHOD #
# ========================== #
def test_alpha_positive(sample):
    """The method-of-moments alpha on a 50,000-row subsample is positive.

    Parameters
    ----------
    sample : pandas.DataFrame
        Fixture: GLM estimation sample.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    a = G.alpha_mm(sample.sample(50_000, random_state=0))
    assert a > 0
