# =========== #
# MODELS FILE #
# =========== #
"""Candidate models at the donor-year level.

Common interface, so the backtest treats them all the same way:

    m = Model().fit(panel_train)
    mu = m.predict_mu(panel_rows)          # expected value of y
    y  = m.sample(panel_rows, rng)         # draw, for the simulation

`sample` is needed because forecasting beyond one year requires propagating the
donor's state (recency, cumulative counts), and the state depends on how much
they actually donated, not on their expected value. Propagating the means would
give a donor who "donates 0.7 times" and never drops out.

The candidates cover three different ideas about how a zero arises:

* **Poisson / Negative Binomial** — a zero is a low count like any other.
* **Hurdle (two parts)** — whether to donate is one decision, how many times is
  another. It is the hypothesis closest to the phenomenon: the 63.9% of zeros
  in the panel do not come from donors who "donated little", but from donors
  who did not show up that year.
* **Gradient boosting** — no imposed functional form; it measures how much is
  lost with the linear structure of the others.

Classes and functions
---------------------
Model
    Base class: frozen :class:`features.DesignSpec`, cap on the expected
    value, default Poisson sampling. Not used directly outside this module.
PoissonGLM
    Poisson GLM. Used by :mod:`forecast` (final per-donor forecast) and,
    through :func:`catalog`, by :mod:`backtest_models`.
NegBinGLM
    Negative binomial GLM with alpha by moments. Used through
    :func:`catalog`.
Hurdle
    Logit for being active plus Poisson on ``y - 1`` for active rows. Used
    through :func:`catalog`.
LightGBMPoisson
    Gradient boosting with Poisson objective (requires ``lightgbm``). Used
    through :func:`catalog`.
catalog
    Name -> zero-argument constructor of every candidate. Used by
    :mod:`backtest_models`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import DesignSpec, design


# =========== #
# MODEL CLASS #
# =========== #
class Model:
    """Base: holds the frozen spec and requires implementing mu and sample.

    Subclasses must implement :meth:`fit` and :meth:`predict_mu`; they may
    override :meth:`sample` when the predictive distribution is not Poisson.

    Attributes
    ----------
    name : str
        Short identifier of the model family.
    spec : features.DesignSpec
        Covariate choice; frozen on the training panel by :meth:`fit`.
    """

    name = "base"

    # =============== #
    # __INIT__ METHOD #
    # =============== #
    def __init__(self, spec: DesignSpec | None = None):
        """Store the design spec and initialise the cap.

        Parameters
        ----------
        spec : features.DesignSpec, optional
            Covariate choice. If ``None``, a default
            :class:`features.DesignSpec` is used.
        """
        self.spec = spec or DesignSpec()
        #: Cap on the expected value, set to the maximum observed during
        #: estimation. It is a safety net for the simulation: no observed
        #: donor-year exceeds 21 donations, so a forecast that exceeds it is a
        #: numerical defect, not information.
        self.mu_max = np.inf

    # ================== #
    # _RECORD_CAP METHOD #
    # ================== #
    def _record_cap(self, y) -> None:
        """Set ``mu_max`` to the largest response seen in training.

        Parameters
        ----------
        y : array_like
            Training response. If empty, the cap stays ``inf``.

        Returns
        -------
        None
        """
        self.mu_max = float(np.max(y)) if len(y) else np.inf

    # ============ #
    # _CLIP METHOD #
    # ============ #
    def _clip(self, mu: np.ndarray) -> np.ndarray:
        """Clip expected values to ``[0, mu_max]``.

        Parameters
        ----------
        mu : numpy.ndarray
            Expected values predicted by the model.

        Returns
        -------
        numpy.ndarray
            `mu` clipped element-wise to ``[0, mu_max]``.
        """
        return np.clip(mu, 0.0, self.mu_max)

    # ========== #
    # FIT METHOD #
    # ========== #
    def fit(self, panel: pd.DataFrame):
        """Estimate the model on a training panel. Abstract.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year training panel with the spec covariates and ``y``.

        Returns
        -------
        Model
            ``self``, fitted (in subclasses).

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    # ================= #
    # PREDICT_MU METHOD #
    # ================= #
    def predict_mu(self, panel: pd.DataFrame) -> np.ndarray:
        """Expected number of donations for each row. Abstract.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to predict, with the spec covariates.

        Returns
        -------
        numpy.ndarray
            Expected value of ``y`` per row (in subclasses).

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    # ============= #
    # SAMPLE METHOD #
    # ============= #
    def sample(self, panel: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Draw from the predictive distribution. Default: Poisson around mu.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to simulate, with the spec covariates.
        rng : numpy.random.Generator
            Random generator used for the draws.

        Returns
        -------
        numpy.ndarray
            One simulated count of donations per row.
        """
        return rng.poisson(self.predict_mu(panel))


# ================ #
# POISSONGLM CLASS #
# ================ #
class PoissonGLM(Model):
    """Poisson GLM with log link on the design matrix of the spec.

    Sampling uses the default Poisson draw of :meth:`Model.sample`.

    Attributes
    ----------
    res : statsmodels GLMResults
        Fitted Poisson GLM, available after :meth:`fit`.
    """

    name = "poisson"

    # ========== #
    # FIT METHOD #
    # ========== #
    def fit(self, panel):
        """Fit the Poisson GLM on the training panel.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year training panel with the spec covariates and ``y``.

        Returns
        -------
        PoissonGLM
            ``self``, fitted.
        """
        X, y, self.spec = design(panel, self.spec)
        self._record_cap(y)
        self.res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        return self

    # ================= #
    # PREDICT_MU METHOD #
    # ================= #
    def predict_mu(self, panel):
        """Expected number of donations per row, clipped to ``[0, mu_max]``.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to predict, with the spec covariates.

        Returns
        -------
        numpy.ndarray
            Expected value of ``y`` per row.
        """
        return self._clip(np.asarray(self.res.predict(self.spec.build(panel))))


# =============== #
# NEGBINGLM CLASS #
# =============== #
class NegBinGLM(Model):
    """Negative Binomial with alpha estimated by moments from a first Poisson.

    Estimating alpha by maximum likelihood on half a million rows is costly
    and changes the expected value little: what matters here is mainly the
    dispersion, which the simulation draws need.

    Attributes
    ----------
    alpha : float
        NB2 dispersion (``Var = mu + alpha * mu^2``), at least ``1e-6``.
    res : statsmodels GLMResults
        Fitted negative binomial GLM with fixed `alpha`.
    """

    name = "negbin"

    # ========== #
    # FIT METHOD #
    # ========== #
    def fit(self, panel):
        """Estimate alpha by moments from a Poisson fit, then fit the NB GLM.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year training panel with the spec covariates and ``y``.

        Returns
        -------
        NegBinGLM
            ``self``, fitted.
        """
        X, y, self.spec = design(panel, self.spec)
        self._record_cap(y)
        pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = np.asarray(pois.predict(X))
        # alpha by moments: Var = mu + alpha*mu^2
        num = np.sum((y - mu) ** 2 - mu)
        den = np.sum(mu ** 2)
        self.alpha = float(max(num / den, 1e-6))
        self.res = sm.GLM(y, X,
                          family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        return self

    # ================= #
    # PREDICT_MU METHOD #
    # ================= #
    def predict_mu(self, panel):
        """Expected number of donations per row, clipped to ``[0, mu_max]``.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to predict, with the spec covariates.

        Returns
        -------
        numpy.ndarray
            Expected value of ``y`` per row.
        """
        return self._clip(np.asarray(self.res.predict(self.spec.build(panel))))

    # ============= #
    # SAMPLE METHOD #
    # ============= #
    def sample(self, panel, rng):
        """Draw from the negative binomial with mean mu and dispersion alpha.

        Uses NumPy's ``(n, p)`` parametrisation with ``n = 1 / alpha`` and
        ``p = n / (n + mu)``.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to simulate, with the spec covariates.
        rng : numpy.random.Generator
            Random generator used for the draws.

        Returns
        -------
        numpy.ndarray
            One simulated count of donations per row.
        """
        mu = self.predict_mu(panel)
        r = 1.0 / self.alpha
        p = r / (r + mu)
        return rng.negative_binomial(r, p)


# ============ #
# HURDLE CLASS #
# ============ #
class Hurdle(Model):
    """Two parts: probability of being active, and how many donations if active.

    The second part is estimated on active rows only, where the response is
    truncated at y>=1: `y-1` is modelled with a Poisson, i.e. how many
    donations beyond the first. It is the simplest way to respect the
    truncation without a dedicated likelihood.

    Attributes
    ----------
    active_part : statsmodels GLMResults
        Binomial (logit) GLM for ``P(y > 0)``.
    count_part : statsmodels GLMResults
        Poisson GLM for ``y - 1`` on the active rows.
    """

    name = "hurdle"

    # ========== #
    # FIT METHOD #
    # ========== #
    def fit(self, panel):
        """Fit the activity logit on all rows and the count part on active rows.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year training panel with the spec covariates and ``y``.

        Returns
        -------
        Hurdle
            ``self``, fitted.
        """
        X, y, self.spec = design(panel, self.spec)
        self._record_cap(y)
        active = (y > 0).astype(float)
        self.active_part = sm.GLM(active, X, family=sm.families.Binomial()).fit()

        act = panel[panel.y > 0]
        Xa = self.spec.build(act)
        ya = act["y"].astype(float).reset_index(drop=True) - 1.0
        self.count_part = sm.GLM(ya, Xa, family=sm.families.Poisson()).fit()
        return self

    # ============= #
    # _PARTS METHOD #
    # ============= #
    def _parts(self, panel):
        """Predict both parts of the hurdle for each row.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows, with the spec covariates.

        Returns
        -------
        p : numpy.ndarray
            Probability of being active (``y > 0``) per row.
        lam : numpy.ndarray
            Expected donations beyond the first, if active, clipped to
            ``[0, max(mu_max - 1, 0)]``.
        """
        X = self.spec.build(panel)
        p = np.asarray(self.active_part.predict(X))
        lam = np.clip(np.asarray(self.count_part.predict(X)),
                      0.0, max(self.mu_max - 1.0, 0.0))
        return p, lam

    # ================= #
    # PREDICT_MU METHOD #
    # ================= #
    def predict_mu(self, panel):
        """Expected donations per row, ``P(active) * (1 + lam)``.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to predict, with the spec covariates.

        Returns
        -------
        numpy.ndarray
            Expected value of ``y`` per row.
        """
        p, lam = self._parts(panel)
        return p * (1.0 + lam)          # E[y] = P(active) * E[y | active]

    # ============= #
    # SAMPLE METHOD #
    # ============= #
    def sample(self, panel, rng):
        """Draw activity from a Bernoulli, then ``1 + Poisson(lam)`` if active.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to simulate, with the spec covariates.
        rng : numpy.random.Generator
            Random generator used for the draws.

        Returns
        -------
        numpy.ndarray
            One simulated count per row: 0 if inactive, at least 1 if active.
        """
        p, lam = self._parts(panel)
        active = rng.random(len(p)) < p
        return np.where(active, 1 + rng.poisson(lam), 0)


# ===================== #
# LIGHTGBMPOISSON CLASS #
# ===================== #
class LightGBMPoisson(Model):
    """Gradient boosting with a Poisson objective.

    No imposed functional form: if it clearly beats the GLMs, the linear
    structure is missing something; if it does not, the linear structure is
    enough and is worth keeping because it is interpretable.

    Sampling uses the default Poisson draw of :meth:`Model.sample`.

    Attributes
    ----------
    params : dict
        Keyword arguments passed to ``lightgbm.LGBMRegressor``.
    res : lightgbm.LGBMRegressor
        Fitted regressor, available after :meth:`fit`.
    """

    name = "lightgbm"

    # =============== #
    # __INIT__ METHOD #
    # =============== #
    def __init__(self, spec=None, **kw):
        """Store the spec and the LightGBM hyper-parameters.

        Parameters
        ----------
        spec : features.DesignSpec, optional
            Covariate choice. If ``None``, a default spec is used.
        **kw
            Overrides of the default ``LGBMRegressor`` parameters (Poisson
            objective, 300 trees, deterministic row-wise training, ...).
        """
        super().__init__(spec)
        self.params = dict(objective="poisson", n_estimators=300,
                           learning_rate=0.06, num_leaves=31,
                           min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.9,
                           random_state=42, verbose=-1,
                           # Without these two, multithreaded histogram
                           # accumulation changes the last digit on every run.
                           deterministic=True, force_row_wise=True) | kw

    # ========== #
    # FIT METHOD #
    # ========== #
    def fit(self, panel):
        """Fit the gradient-boosting regressor on the training panel.

        ``lightgbm`` is imported here, so it is only required when this model
        is actually used.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year training panel with the spec covariates and ``y``.

        Returns
        -------
        LightGBMPoisson
            ``self``, fitted.
        """
        import lightgbm as lgb
        X, y, self.spec = design(panel, self.spec)
        self._record_cap(y)
        self.res = lgb.LGBMRegressor(**self.params).fit(X, y)
        return self

    # ================= #
    # PREDICT_MU METHOD #
    # ================= #
    def predict_mu(self, panel):
        """Expected number of donations per row, clipped to ``[0, mu_max]``.

        Parameters
        ----------
        panel : pandas.DataFrame
            Donor x year rows to predict, with the spec covariates.

        Returns
        -------
        numpy.ndarray
            Expected value of ``y`` per row.
        """
        return self._clip(self.res.predict(self.spec.build(panel)))


# ============== #
# CATALOG METHOD #
# ============== #
#: The candidates, with the spec variants worth distinguishing.
def catalog() -> dict[str, callable]:
    """The candidate models, with the spec variants worth distinguishing.

    Returns
    -------
    dict of str to callable
        Model name -> zero-argument function returning a new, unfitted
        model: ``"poisson"``, ``"poisson_senza_trend"``, ``"negbin"``,
        ``"hurdle"``, ``"hurdle_senza_trend"`` and ``"lightgbm"``. The
        ``*_senza_trend`` variants use ``DesignSpec(trend=False)``.
    """
    no_trend = lambda: DesignSpec(trend=False)
    return {
        "poisson":             lambda: PoissonGLM(),
        "poisson_senza_trend": lambda: PoissonGLM(no_trend()),
        "negbin":              lambda: NegBinGLM(),
        "hurdle":              lambda: Hurdle(),
        "hurdle_senza_trend":  lambda: Hurdle(no_trend()),
        "lightgbm":            lambda: LightGBMPoisson(),
    }
