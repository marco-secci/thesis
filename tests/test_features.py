# ================== #
# TEST_FEATURES FILE #
# ================== #
"""The design matrix must reject wrong combinations on its own.

These are the safeguards that replace the attention of whoever writes the
model: in the original repository nothing prevented `quantity_total_ml` from
entering a formula, and the exact collinearity between age, birth year and
donation year was reported only as a high condition number.

Functions
---------
Tests collected by pytest; no other module uses them. They check
:class:`features.DesignSpec` and :func:`features.design`.

fake
    Fixture: synthetic panel with the default covariates.
test_rejects_leakage_columns
    Leaky columns are rejected.
test_rejects_collinear_triple
    Age + birth year + year are rejected.
test_rejects_duplicate_recoding
    ``prev_active`` + ``recency_class`` are rejected.
test_matrix_has_full_rank
    The default matrix has full rank.
test_same_columns_in_estimation_and_prediction
    Columns are stable across fit and prediction.
test_covariates_are_clipped_to_support
    Numeric covariates are clipped to the training range.
test_trend_is_not_clipped
    The time term is not clipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import LEAKY_COLUMNS
from features import DesignSpec, design


# =========== #
# FAKE METHOD #
# =========== #
@pytest.fixture
def fake():
    """Synthetic panel with every default covariate of :class:`features.DesignSpec`.

    Returns
    -------
    pandas.DataFrame
        200 rows drawn with a fixed seed: ``y``, ``year`` (2015-2020) and the
        default numeric and categorical covariates.
    """
    n = 200
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "y": rng.poisson(1.0, n),
        "year": rng.integers(2015, 2021, n),
        "prev_count": rng.integers(0, 4, n),
        "log1p_cum": rng.random(n),
        "cum_active_years": rng.integers(0, 6, n),
        "career_years": rng.integers(0, 30, n),
        "left_censored": rng.integers(0, 2, n),
        "age": rng.integers(18, 70, n),
        "recency_class": rng.choice(["entry", "1", "2", "6+"], n),
        "sex": rng.choice(["M", "F"], n),
        "donor_class_at_entry": rng.choice(["P", "O"], n),
    })


# =================================== #
# TEST_REJECTS_LEAKAGE_COLUMNS METHOD #
# =================================== #
def test_rejects_leakage_columns(fake):
    """A spec containing a column of ``LEAKY_COLUMNS`` raises ``ValueError``.

    Parameters
    ----------
    fake : pandas.DataFrame
        Fixture: synthetic 200-row panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    spec = DesignSpec(numeric=["age", LEAKY_COLUMNS[0]])
    with pytest.raises(ValueError, match="riscrivono il target"):
        spec.validate()


# ==================================== #
# TEST_REJECTS_COLLINEAR_TRIPLE METHOD #
# ==================================== #
def test_rejects_collinear_triple(fake):
    """Age, birth year and year (through the trend) together raise ``ValueError``.

    Parameters
    ----------
    fake : pandas.DataFrame
        Fixture: synthetic 200-row panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    spec = DesignSpec(numeric=["age", "birth_year"], trend=True)   # + year
    with pytest.raises(ValueError, match="collineari"):
        spec.validate()


# ====================================== #
# TEST_REJECTS_DUPLICATE_RECODING METHOD #
# ====================================== #
def test_rejects_duplicate_recoding(fake):
    """``prev_active`` together with ``recency_class`` raises ``ValueError``.

    Parameters
    ----------
    fake : pandas.DataFrame
        Fixture: synthetic 200-row panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    spec = DesignSpec(numeric=["age", "prev_active"],
                      categorical=["recency_class"])
    with pytest.raises(ValueError, match="ricodifica esatta"):
        spec.validate()


# ================================ #
# TEST_MATRIX_HAS_FULL_RANK METHOD #
# ================================ #
def test_matrix_has_full_rank(fake):
    """The default design matrix has full column rank.

    Parameters
    ----------
    fake : pandas.DataFrame
        Fixture: synthetic 200-row panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    X, _, _ = design(fake)
    assert np.linalg.matrix_rank(X.to_numpy()) == X.shape[1]


# ===================================================== #
# TEST_SAME_COLUMNS_IN_ESTIMATION_AND_PREDICTION METHOD #
# ===================================================== #
def test_same_columns_in_estimation_and_prediction(fake):
    """Prediction rows missing some levels still get the training columns.

    Parameters
    ----------
    fake : pandas.DataFrame
        Fixture: synthetic 200-row panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    X, _, spec = design(fake)
    # a subset that happens not to contain all the levels
    X2 = spec.build(fake[fake.recency_class.isin(["1", "2"])])
    assert list(X.columns) == list(X2.columns)


# ============================================= #
# TEST_COVARIATES_ARE_CLIPPED_TO_SUPPORT METHOD #
# ============================================= #
def test_covariates_are_clipped_to_support(fake):
    """A value outside the range seen during estimation is clipped to the edge.

    Parameters
    ----------
    fake : pandas.DataFrame
        Fixture: synthetic 200-row panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    _, _, spec = design(fake)
    outside = fake.head(1).copy()
    outside["prev_count"] = 10_000
    assert spec.build(outside)["prev_count"].iloc[0] == fake.prev_count.max()


# ================================ #
# TEST_TREND_IS_NOT_CLIPPED METHOD #
# ================================ #
def test_trend_is_not_clipped(fake):
    """Extrapolating in time is the job of the trend term, not an error.

    A year beyond the training range keeps its own ``year_c``.

    Parameters
    ----------
    fake : pandas.DataFrame
        Fixture: synthetic 200-row panel.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the invariant does not hold.
    """
    _, _, spec = design(fake)
    future = fake.head(1).copy()
    future["year"] = 2031
    assert spec.build(future)["year_c"].iloc[0] == 2031 - 2017
