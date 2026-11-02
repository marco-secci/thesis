# ============= #
# FEATURES FILE #
# ============= #
"""Design matrix built from the donor x year panel.

A single place where what goes into a model is decided, for three reasons:

* columns that rewrite the target (`config.LEAKY_COLUMNS`) are rejected here,
  with an exception, instead of depending on the attention of whoever writes
  the model;
* `age`, `birth_year` and `year` are exactly collinear (age = year -
  birth_year): at most two of the three go in, otherwise the matrix is
  singular. The check is explicit;
* categorical levels are fixed on the training set and reused identically at
  prediction time, so an unseen level does not silently shift the columns.

Classes and functions
---------------------
DesignSpec
    Dataclass holding the covariate choice and, after fitting, the frozen
    categorical levels and numeric bounds. Its methods are
    :meth:`DesignSpec.validate`, :meth:`DesignSpec.fit_levels` and
    :meth:`DesignSpec.build`. Used by :mod:`models` (every model owns a
    spec) and ``tests/test_features.py``.
design
    Fits a spec on a training panel and returns matrix, response and spec.
    Used by :mod:`models` and ``tests/test_features.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import LEAKY_COLUMNS

#: Year on which the time term is centred. Centring does not change the fit but
#: keeps the numbers in a range where the optimiser does not struggle.
YEAR_CENTER = 2017

# `prev_n_types` is deliberately left out: using it for prediction would require
# also simulating the number of donation types, i.e. introducing a second
# model only to feed a covariate of the first. It costs more than it is worth.
# `prev_active` is out because it is exactly the level `recency_class == "1"`:
# having donated last year and having recency 1 are the same thing. Keeping
# both makes the matrix rank-deficient, and statsmodels reports it with an
# easy-to-miss warning rather than an error.
# `tenure` is out in favour of `career_years`: it counts years from the true
# start of the career rather than from the start of the observation window.
# See `panel.build_panel` for why.
NUMERIC = ["prev_count", "log1p_cum", "cum_active_years", "career_years",
           "left_censored", "age"]
CATEGORICAL = ["recency_class", "sex", "donor_class_at_entry"]

#: Triples of columns that are linearly dependent by construction. If a whole
#: triple goes in the matrix is singular, and statsmodels only reports it as
#: a high condition number — i.e. it often goes unnoticed.
COLLINEAR_TRIPLES = [("age", "birth_year", "year")]

#: Pairs in which one column is the exact recoding of the other.
DUPLICATE_PAIRS = [("prev_active", "recency_class")]


# ================ #
# DESIGNSPEC CLASS #
# ================ #
@dataclass
class DesignSpec:
    """What goes into the model. Frozen at fit time and reused for prediction.

    Attributes
    ----------
    numeric : list of str
        Numeric covariates, used as they are (after clipping). Defaults to
        ``NUMERIC``.
    categorical : list of str
        Categorical covariates, dummy-coded with the first sorted level as
        reference. Defaults to ``CATEGORICAL``.
    trend : bool, default True
        Whether to add the linear time term ``year_c = year - YEAR_CENTER``.
    levels : dict of str to list of str
        Sorted levels of each categorical column seen in training. Filled by
        :meth:`fit_levels`.
    bounds : dict of str to tuple of float
        ``(min, max)`` of each numeric column in training. Filled by
        :meth:`fit_levels`; used to clip covariates in :meth:`build`.
    """
    numeric: list[str] = field(default_factory=lambda: list(NUMERIC))
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL))
    trend: bool = True                 # linear term on the year
    levels: dict = field(default_factory=dict)   # filled by the fit
    bounds: dict = field(default_factory=dict)   # filled by the fit

    # =============== #
    # VALIDATE METHOD #
    # =============== #
    def validate(self) -> None:
        """Reject column choices that would leak the target or be singular.

        The checked columns are ``numeric``, ``categorical`` and ``year``
        when ``trend`` is true.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a column of ``config.LEAKY_COLUMNS`` is included, if a whole
            triple of ``COLLINEAR_TRIPLES`` is included, or if both columns
            of a pair in ``DUPLICATE_PAIRS`` are included.
        """
        cols = set(self.numeric) | set(self.categorical) | ({"year"} if self.trend else set())
        leaky = cols & set(LEAKY_COLUMNS)
        if leaky:
            raise ValueError(
                f"colonne che riscrivono il target nella matrice di progetto: {sorted(leaky)}. "
                "Vedi docs/DATA_NOTES.md §1-2.")
        for triple in COLLINEAR_TRIPLES:
            if set(triple) <= cols:
                raise ValueError(
                    f"{triple} sono esattamente collineari: la matrice sarebbe singolare. "
                    "Toglierne una.")
        for a, b in DUPLICATE_PAIRS:
            if {a, b} <= cols:
                raise ValueError(
                    f"'{a}' è la ricodifica esatta di '{b}': la matrice sarebbe "
                    "rank-deficient. Tenerne una sola.")

    # ================= #
    # FIT_LEVELS METHOD #
    # ================= #
    def fit_levels(self, panel: pd.DataFrame) -> "DesignSpec":
        """Validate the spec and freeze levels and bounds on the training panel.

        Parameters
        ----------
        panel : pandas.DataFrame
            Training panel containing every column of ``numeric`` and
            ``categorical``.

        Returns
        -------
        DesignSpec
            ``self``, with ``levels`` and ``bounds`` filled, to allow
            chaining.

        Raises
        ------
        ValueError
            If :meth:`validate` rejects the column choice.
        """
        self.validate()
        self.levels = {c: sorted(panel[c].dropna().astype(str).unique())
                       for c in self.categorical}
        self.bounds = {c: (float(panel[c].min()), float(panel[c].max()))
                       for c in self.numeric}
        return self

    # ============ #
    # BUILD METHOD #
    # ============ #
    def build(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Numeric matrix, columns in the same order as at fit time.

        Numeric covariates are clipped to the training ``bounds``; the time
        term is not clipped; categorical columns are dummy-coded against the
        training ``levels`` (unseen levels get all-zero dummies).

        Parameters
        ----------
        panel : pandas.DataFrame
            Panel to encode (training or prediction rows). Must contain the
            spec columns and, if ``trend`` is true, ``year``.

        Returns
        -------
        pandas.DataFrame
            Design matrix with a default ``RangeIndex`` and columns
            ``const``, the numeric covariates, ``year_c`` (if ``trend``) and
            one ``"<column>=<level>"`` dummy per non-reference level.

        Raises
        ------
        RuntimeError
            If :meth:`fit_levels` has not been called yet.
        """
        if not self.levels:
            raise RuntimeError("chiamare fit_levels() sul training prima di build()")
        # State covariates are clipped back into the support observed during
        # estimation. This is needed by the recursive simulation: a year with
        # many donations raises `prev_count`, which with a log link raises the
        # expected value of the next year, which raises `prev_count` again.
        # Without a bound the recursion diverges — not because the donor does,
        # but because the model is queried where it has never seen any data.
        num = panel[self.numeric].astype(float).reset_index(drop=True)
        for c, (lo, hi) in self.bounds.items():
            num[c] = num[c].clip(lo, hi)
        parts = [num]
        # `year_c` is not bounded: extrapolating in time is exactly what it is
        # asked to do, and it is an explicit choice of the spec (`trend`).
        if self.trend:
            parts.append(pd.Series(panel["year"].to_numpy() - YEAR_CENTER,
                                   name="year_c").astype(float).reset_index(drop=True))
        for c in self.categorical:
            v = panel[c].astype(str).reset_index(drop=True)
            # first category omitted: it is the reference, avoids collinearity
            # with the intercept
            for lev in self.levels[c][1:]:
                parts.append((v == lev).astype(float).rename(f"{c}={lev}"))
        X = pd.concat(parts, axis=1)
        X.insert(0, "const", 1.0)
        return X


# ============= #
# DESIGN METHOD #
# ============= #
def design(panel: pd.DataFrame, spec: DesignSpec | None = None
           ) -> tuple[pd.DataFrame, pd.Series, DesignSpec]:
    """Matrix, response and frozen spec, from the training panel.

    Parameters
    ----------
    panel : pandas.DataFrame
        Training panel with the spec columns and the response ``y``.
    spec : DesignSpec, optional
        Covariate choice. If ``None``, a default :class:`DesignSpec` is
        used. It is modified in place by :meth:`DesignSpec.fit_levels`.

    Returns
    -------
    X : pandas.DataFrame
        Design matrix, as returned by :meth:`DesignSpec.build`.
    y : pandas.Series
        Response ``panel["y"]`` as float, with a default ``RangeIndex``
        aligned with `X`.
    spec : DesignSpec
        The fitted spec, to reuse at prediction time.
    """
    spec = (spec or DesignSpec()).fit_levels(panel)
    return spec.build(panel), panel["y"].astype(float).reset_index(drop=True), spec
