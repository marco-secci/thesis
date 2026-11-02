# ================== #
# GLM_INFERENCE FILE #
# ================== #
"""Count regression model at the donor-year grain (Ch. 4).

`models.py` is predictive: it serves to choose a model with a backtest. This
module is **inferential**: it estimates how much a donor characteristic shifts
the expected frequency, and with how much uncertainty.

Three differences from the specification published in Ch. 5, all necessary:

1. **Grain.** The chapter estimates on event rows, with `num_donations`
   replicated on every row of the group (donor x year x type): each group
   enters the likelihood as many times as it has donations. Here estimation is
   on the donor-year panel with **materialised zeros**, where the sum of y over
   a year *is* that year's total. Deduplicating to one row per triple, as the
   chapter's note proposes, removes the replication but leaves the response
   truncated at y >= 1: dropout remains unobservable and the dispersion
   diagnosis remains flawed.
2. **Explicit reference categories.** The chapter inherits patsy's
   alphabetical ordering and ends up with `donor_class` referenced to a
   category with **a single observation**, which makes its CIs unusable. Here
   the reference is `P` (periodic), i.e. the modal category.
3. **Standard errors robust to donor clustering.** The same donor appears in
   many years. The sqrt(1.84) factor given in the chapter corrects replication
   within the group, not the correlation between years of the same donor:
   that has no closed form and must be handled with a cluster.

The donation-type covariate is **lagged** (the prevailing type in previous
years): at the donor-year grain the current year's type is an attribute of the
response, not a predictor, and using it would be leakage.

A point the chapter does not raise and that must be raised: **age, period and
entry cohort are not jointly identifiable**, because year = entry year + career
years. The chapter puts age, cohort and a spline on the year in a single
specification; the cohort coefficients that come out are identified only by
the five-year grouping, not by the data. Here a main model without cohort and
a secondary one with cohort are estimated, stating that its coefficients must
be read conditionally on the grouping.

Functions
---------
lagged_type
    Prevailing donation type of each donor up to the previous year. Used
    by :func:`sample` and ``tests/test_glm_inference.py``.
sample
    Builds the estimation sample from the panel. Used by :func:`main` and
    ``tests/test_glm_inference.py``.
estimate
    Fits a Poisson or NB GLM with donor-clustered standard errors. Used
    only inside this module, by :func:`main`.
alpha_mm
    NB dispersion by the method of moments. Used by :func:`main` and
    ``tests/test_glm_inference.py``.
irr_table
    IRRs with confidence intervals. Used only inside this module, by
    :func:`main`.
dispersion
    Variance/mean of y, truncated versus with zeros. Used by :func:`main`
    and ``tests/test_glm_inference.py``.
donor_class_regime
    Donor-class counts per year. Used only inside this module, by
    :func:`main`.
fig_trend
    Expected yearly frequency at a fixed profile. Used only inside this
    module, by :func:`main`.
fig_regime
    Donor-class counts on a log scale. Used only inside this module, by
    :func:`main`.
main
    Entry point. Run via ``python -m glm_inference`` from the ``Makefile``
    and ``scripts/run_pipeline.py``, and called by :mod:`figures_tesi`
    with the thesis image directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import viz
from config import (DONATIONS, DONOR_YEAR, PANEL_BURN_IN_YEAR, REPORTS,
                     YEAR_MAX)

#: Centring year of the time term, as in the chapter.
YEAR_CENTER = 2016

#: Width of the first-donation cohorts (five-year bands, as in the chapter).
COHORT_STEP = 5

#: Explicit references: the modal category, not the alphabetically first one.
REF_CLASS = "P"
REF_TYPE = "SANGUE"

#: Year of the administrative reclassification of `donor_class`: class O goes
#: from ~3,000 cases a year to fewer than ten and class D, previously
#: non-existent, appears. Comparing classes across this date means comparing
#: two coding schemes mounted on the same name.
REGIME_YEAR = 2016

#: Main model. `career_years` (years since entry) is the control missing from
#: the chapter: without it the entry cohort absorbs its effect and produces
#: IRRs up to 4.2 for recent cohorts, which do not measure a propensity to
#: donate but the fact that those donors are observed only in their first
#: years. `left_censored` marks those whose career started before the window,
#: for whom `career_years` is underestimated.
FORMULA = ("y ~ age + career_years + left_censored"
           " + C(sex, Treatment(reference='M'))"
           " + C(donor_class_at_entry, Treatment(reference='P'))"
           " + C(tipo_prev, Treatment(reference='SANGUE'))"
           " + year_c")

#: Secondary model, with the cohort. See the note on identification.
FORMULA_COHORT = FORMULA + " + C(coorte_ingresso)"


# ── Building the sample ──────────────────────────────────────────────────────

# ================== #
# LAGGED_TYPE METHOD #
# ================== #
def lagged_type(don: pd.DataFrame) -> pd.DataFrame:
    """Prevailing donation type of each donor **up to the previous year**.

    Causal by construction: when predicting year t only what was observed up
    to t-1 is used.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id``, ``donation_year`` and ``donation_type``.

    Returns
    -------
    pandas.DataFrame
        Columns ``donor_id``, ``tipo_prev`` (the type with most donations in
        all years before ``year``) and ``year``, one row per donor with some
        history, for every year from the second observed one to
        ``YEAR_MAX``.
    """
    counts = (don.groupby(["donor_id", "donation_year", "donation_type"],
                          observed=True).size().rename("n").reset_index())
    slices = []
    for year in range(counts["donation_year"].min() + 1, YEAR_MAX + 1):
        history = counts[counts["donation_year"] < year]
        if history.empty:
            continue
        tot = history.groupby(["donor_id", "donation_type"], observed=True)["n"].sum()
        dom = tot.reset_index().sort_values("n", ascending=False)
        dom = dom.drop_duplicates("donor_id")[["donor_id", "donation_type"]]
        dom["year"] = year
        slices.append(dom)
    out = pd.concat(slices, ignore_index=True)
    return out.rename(columns={"donation_type": "tipo_prev"})


# ============= #
# SAMPLE METHOD #
# ============= #
def sample(panel: pd.DataFrame, don: pd.DataFrame) -> pd.DataFrame:
    """Build the estimation sample from the panel.

    Keeps the years from ``PANEL_BURN_IN_YEAR``, attaches the lagged type
    (:func:`lagged_type`; donors with no history get ``REF_TYPE``), groups
    types with fewer than 500 rows into ``"ALTRO"``, adds the five-year
    entry cohort (dropping cohorts with fewer than 500 rows), the centred
    year and the entry regime.

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel (``DONOR_YEAR``).
    don : pandas.DataFrame
        Donations (``DONATIONS``).

    Returns
    -------
    pandas.DataFrame
        The panel rows used for estimation, with the extra columns
        ``tipo_prev``, ``coorte_ingresso``, ``year_c`` and ``regime``
        (``"pre"``/``"post"`` relative to ``REGIME_YEAR``), without rows
        missing ``y``, ``age``, ``sex`` or ``donor_class_at_entry``.
    """
    df = panel[panel["year"] >= PANEL_BURN_IN_YEAR].copy()
    df = df.merge(lagged_type(don), on=["donor_id", "year"], how="left")
    df["tipo_prev"] = df["tipo_prev"].fillna(REF_TYPE)

    # Types below threshold go into a residual category: cells that are too
    # small give very wide, uninformative CIs.
    counts = df["tipo_prev"].value_counts()
    minor = counts[counts < 500].index
    df.loc[df["tipo_prev"].isin(minor), "tipo_prev"] = "ALTRO"

    df["coorte_ingresso"] = (df["entry_year"] // COHORT_STEP * COHORT_STEP).astype(int)
    rare = df["coorte_ingresso"].value_counts()
    df = df[df["coorte_ingresso"].isin(rare[rare >= 500].index)]

    df["year_c"] = df["year"] - YEAR_CENTER
    df["regime"] = np.where(df["entry_year"] >= REGIME_YEAR, "post", "pre")
    return df.dropna(subset=["y", "age", "sex", "donor_class_at_entry"])


# ── Estimation ───────────────────────────────────────────────────────────────

# =============== #
# ESTIMATE METHOD #
# =============== #
def estimate(df: pd.DataFrame, family: str = "nb", alpha: float | None = None,
             formula: str = FORMULA):
    """Estimate with standard errors robust to donor clustering.

    Parameters
    ----------
    df : pandas.DataFrame
        Estimation sample, as returned by :func:`sample`.
    family : {"nb", "poisson"}, default "nb"
        GLM family; any value other than ``"poisson"`` gives the negative
        binomial.
    alpha : float, optional
        Negative binomial dispersion (e.g. from :func:`alpha_mm`). Falls
        back to 1.0 if ``None`` or 0. Ignored for Poisson.
    formula : str, default FORMULA
        Patsy formula; ``FORMULA_COHORT`` adds the entry cohort.

    Returns
    -------
    statsmodels.genmod.generalized_linear_model.GLMResults
        Fitted model with covariance clustered on ``donor_id``.
    """
    if family == "poisson":
        fam = sm.families.Poisson()
    else:
        fam = sm.families.NegativeBinomial(alpha=alpha if alpha else 1.0)
    mod = smf.glm(formula, data=df, family=fam)
    return mod.fit(cov_type="cluster", cov_kwds={"groups": df["donor_id"].values})


# =============== #
# ALPHA_MM METHOD #
# =============== #
def alpha_mm(df: pd.DataFrame) -> float:
    """Dispersion parameter by the method of moments: Var = mu + a*mu^2.

    Fits a Poisson GLM with ``FORMULA`` and averages
    ``((y - mu)^2 - mu) / mu^2`` over the rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Estimation sample, as returned by :func:`sample`.

    Returns
    -------
    float
        Estimated alpha, at least ``1e-4``.
    """
    base = smf.glm(FORMULA, data=df, family=sm.families.Poisson()).fit()
    mu = base.fittedvalues
    r = ((df["y"] - mu) ** 2 - mu) / np.maximum(mu ** 2, 1e-9)
    return float(max(r.mean(), 1e-4))


# ================ #
# IRR_TABLE METHOD #
# ================ #
def irr_table(res, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Incidence rate ratios with confidence intervals from a fitted GLM.

    Parameters
    ----------
    res : statsmodels GLMResults
        Fitted log-link model, as returned by :func:`estimate`.
    labels : dict of str to str, optional
        Renaming of term names for presentation.

    Returns
    -------
    pandas.DataFrame
        One row per coefficient with columns ``termine``, ``irr``
        (``exp(coef)``), ``ic_basso`` and ``ic_alto`` (exponentiated 95% CI
        bounds) and ``p`` (p-value).
    """
    ci = res.conf_int()
    out = pd.DataFrame({
        "termine": res.params.index,
        "irr": np.exp(res.params.values),
        "ic_basso": np.exp(ci[0].values),
        "ic_alto": np.exp(ci[1].values),
        "p": res.pvalues.values,
    })
    if labels:
        out["termine"] = out["termine"].replace(labels)
    return out


# ================= #
# DISPERSION METHOD #
# ================= #
def dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """Var/mean of the response at the truncated grain and with the zeros.

    It is the number that refutes sec:5.3: on the truncated data the response
    looks almost equidispersed; with the zeros it is clearly overdispersed.

    Parameters
    ----------
    df : pandas.DataFrame
        Donor x year panel or sample, with ``y``.

    Returns
    -------
    pandas.DataFrame
        Two rows (active years only; all rows) with columns ``grano``,
        ``n``, ``media``, ``varianza`` and ``var_su_media``.
    """
    active = df[df["y"] > 0]["y"]
    everyone = df["y"]
    return pd.DataFrame([
        {"grano": "donatore-anno, solo anni attivi (troncato a y>=1)",
         "n": len(active), "media": active.mean(), "varianza": active.var(),
         "var_su_media": active.var() / active.mean()},
        {"grano": "donatore-anno, zeri materializzati",
         "n": len(everyone), "media": everyone.mean(), "varianza": everyone.var(),
         "var_su_media": everyone.var() / everyone.mean()},
    ])


# ========================= #
# DONOR_CLASS_REGIME METHOD #
# ========================= #
def donor_class_regime(don: pd.DataFrame) -> pd.DataFrame:
    """Counts of `donor_class` per year: documents the coding break.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donation_year`` and ``donor_class``.

    Returns
    -------
    pandas.DataFrame
        One row per year: column ``donation_year`` plus one count column per
        donor class (0 where absent).
    """
    piv = (don.groupby(["donation_year", "donor_class"], observed=True)
              .size().unstack(fill_value=0))
    return piv.reset_index()


# ── Figure ───────────────────────────────────────────────────────────────────

# ================ #
# FIG_TREND METHOD #
# ================ #
def fig_trend(res, df: pd.DataFrame, path):
    """Expected frequency by year, at a fixed profile.

    The profile is a 40-year-old man, class ``REF_CLASS``, prevailing type
    ``REF_TYPE``, 8 career years, not left-censored. The observed mean of y
    per year is overlaid for comparison.

    Parameters
    ----------
    res : statsmodels GLMResults
        Fitted main model (``FORMULA``).
    df : pandas.DataFrame
        Estimation sample, for the observed means.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    profile = pd.DataFrame({
        "year": range(PANEL_BURN_IN_YEAR, YEAR_MAX + 1),
        "age": 40, "career_years": 8, "left_censored": 0,
        "sex": "M", "donor_class_at_entry": REF_CLASS, "tipo_prev": REF_TYPE,
    })
    profile["year_c"] = profile["year"] - YEAR_CENTER
    pr = res.get_prediction(profile)
    mu, ci = pr.predicted_mean, pr.conf_int()

    obs = df.groupby("year", observed=True)["y"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(profile["year"], ci[:, 0], ci[:, 1],
                    color=viz.SERIES[0], alpha=0.18, linewidth=0)
    ax.plot(profile["year"], mu, color=viz.SERIES[0],
            label="attesa a profilo fissato (M, P, sangue, 40 anni, 8 di carriera)")
    ax.plot(obs.index, obs.values, color=viz.INK_SOFT, linestyle="--",
            linewidth=1.4, label="media osservata sul panel")
    ax.legend(fontsize=8)
    viz.finish(ax, "Frequenza annua attesa di donazione", "anno",
               "donazioni per donatore-anno")
    viz.save(fig, path)


# ================= #
# FIG_REGIME METHOD #
# ================= #
def fig_regime(piv: pd.DataFrame, path):
    """Donations per year by donor class on a log scale, marking ``REGIME_YEAR``.

    Parameters
    ----------
    piv : pandas.DataFrame
        Counts per year and class, as returned by
        :func:`donor_class_regime`.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.0))
    classes = [c for c in piv.columns if c != "donation_year"]
    for i, c in enumerate(classes):
        v = piv[c].replace(0, np.nan)
        ax.plot(piv["donation_year"], v, marker="o",
                color=viz.SERIES[i % len(viz.SERIES)], label=str(c))
    ax.set_yscale("log")
    ax.axvline(REGIME_YEAR, color=viz.INK_SOFT, linestyle=":", linewidth=1.2)
    ax.annotate("riclassificazione", xy=(REGIME_YEAR, ax.get_ylim()[1]),
                xytext=(4, -10), textcoords="offset points",
                fontsize=8, color=viz.INK_SOFT)
    ax.legend(title="classe donatore", ncol=2, fontsize=8)
    viz.finish(ax, "Classe donatore per anno (scala logaritmica)", "anno",
               "donazioni")
    viz.save(fig, path)


# ── Entry point ──────────────────────────────────────────────────────────────

# =========== #
# MAIN METHOD #
# =========== #
def main(img_dir=None) -> None:
    """Entry point: estimate the inferential GLMs and save tables and figures.

    Writes ``glm_dispersione.csv``, ``glm_fit.csv`` (AIC/BIC of Poisson and
    negative binomial), ``glm_irr.csv`` (IRRs of the main and cohort
    models) and ``glm_donor_class_regime.csv`` to ``REPORTS``, plus the
    figures ``fig_glm_trend.png`` and ``fig_donor_class_regime.png``.

    Parameters
    ----------
    img_dir : pathlib.Path, optional
        Directory for the figures. Defaults to ``REPORTS``;
        :mod:`figures_tesi` passes the thesis image directory.

    Returns
    -------
    None
    """
    img = img_dir or REPORTS
    panel = pd.read_parquet(DONOR_YEAR)
    don = pd.read_parquet(DONATIONS)
    df = sample(panel, don)
    print(f"campione: {len(df):,} righe donatore-anno, "
          f"{df['donor_id'].nunique():,} donatori, "
          f"{100 * (df['y'] == 0).mean():.1f}% di zeri")

    disp = dispersion(panel)
    disp.to_csv(REPORTS / "glm_dispersione.csv", index=False)
    print(disp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    a = alpha_mm(df)
    print(f"\nalpha (metodo dei momenti): {a:.4f}")

    results = {}
    for fam in ("poisson", "nb"):
        res = estimate(df, fam, alpha=a)
        results[fam] = res
        print(f"  {fam:8s} AIC {res.aic:>14,.1f}  loglik {res.llf:>14,.1f}")

    cohort = estimate(df, "nb", alpha=a, formula=FORMULA_COHORT)

    fit = pd.DataFrame([
        {"modello": "Poisson", "aic": results["poisson"].aic,
         "bic": results["poisson"].bic_llf, "loglik": results["poisson"].llf},
        {"modello": "Binomiale negativo", "aic": results["nb"].aic,
         "bic": results["nb"].bic_llf, "loglik": results["nb"].llf},
    ])
    fit["delta_aic"] = fit["aic"] - fit["aic"].min()
    fit.to_csv(REPORTS / "glm_fit.csv", index=False)

    irr = irr_table(results["nb"])
    irr.insert(0, "modello", "principale")
    irr_c = irr_table(cohort)
    irr_c.insert(0, "modello", "con coorte")
    pd.concat([irr, irr_c], ignore_index=True).to_csv(
        REPORTS / "glm_irr.csv", index=False)
    print("\n=== modello principale ===")
    print(irr.drop(columns="modello").to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== con coorte (identificata solo dal raggruppamento) ===")
    print(irr_c[irr_c.termine.str.contains("coorte")].drop(columns="modello")
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    piv = donor_class_regime(don)
    piv.to_csv(REPORTS / "glm_donor_class_regime.csv", index=False)

    fig_trend(results["nb"], df, img / "fig_glm_trend.png")
    fig_regime(piv, img / "fig_donor_class_regime.png")


if __name__ == "__main__":
    main()
