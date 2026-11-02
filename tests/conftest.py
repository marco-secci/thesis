# ============= #
# CONFTEST FILE #
# ============= #
"""Shared fixtures.

The tests run on the real data produced by the pipeline: they are invariant
tests, not unit tests on fake data. If the pipeline has not been run yet they
are skipped with an explicit message instead of failing obscurely.

Functions
---------
pytest discovers this file automatically; the fixtures are injected by
name into every test module under ``tests/`` that requests them.

_load
    Private helper: reads a parquet or skips the test. Used only inside
    this module, by the fixtures.
events
    Raw events. Used by ``test_data_invariants.py``.
donations
    Cleaned donations. Used by ``test_data_invariants.py``,
    ``test_panel.py``, ``test_retention.py`` and ``test_glm_inference.py``.
panel
    Donor x year panel. Used by ``test_panel.py`` and
    ``test_glm_inference.py`` (``test_simulate.py`` defines its own).
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import DONATIONS, DONOR_YEAR, EVENTS_RAW


# ============ #
# _LOAD METHOD #
# ============ #
def _load(path, name):
    """Read a parquet produced by the pipeline, or skip the test.

    Parameters
    ----------
    path : pathlib.Path
        Parquet file to read.
    name : str
        ``make`` target that produces it, quoted in the skip message.

    Returns
    -------
    pandas.DataFrame
        Contents of `path`.

    Raises
    ------
    pytest.skip.Exception
        If `path` does not exist yet.
    """
    if not path.exists():
        pytest.skip(f"{name} non ancora prodotto: esegui `make {name}`")
    return pd.read_parquet(path)


# ============= #
# EVENTS METHOD #
# ============= #
@pytest.fixture(scope="session")
def events():
    """Session fixture: the raw events table (``EVENTS_RAW``).

    Returns
    -------
    pandas.DataFrame
        Events produced by :mod:`ingest`; the test is skipped if missing.
    """
    return _load(EVENTS_RAW, "ingest")


# ================ #
# DONATIONS METHOD #
# ================ #
@pytest.fixture(scope="session")
def donations():
    """Session fixture: the cleaned donations (``DONATIONS``).

    Returns
    -------
    pandas.DataFrame
        Donations produced by :mod:`clean`; the test is skipped if missing.
    """
    return _load(DONATIONS, "clean-data")


# ============ #
# PANEL METHOD #
# ============ #
@pytest.fixture(scope="session")
def panel():
    """Session fixture: the donor x year panel (``DONOR_YEAR``).

    Returns
    -------
    pandas.DataFrame
        Panel produced by :mod:`panel`; the test is skipped if missing.
    """
    return _load(DONOR_YEAR, "panel")
