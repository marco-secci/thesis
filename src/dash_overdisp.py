# ================== #
# DASH_OVERDISP FILE #
# ================== #
"""Reproduce table 6.1 (AIC comparison in the Regression tab) on the current data.

Replicates the specification of shiny_app/app.R:1295-1316: aggregation to one
row per (donor x sex x donation type) within the filter scope, response =
count, age = maximum observed in the group, covariates age + sex + type.

Importing the module reads ``DONATIONS`` into the module-level ``don``
DataFrame. The result is written to ``reports/dash_overdispersione.csv``.

Functions
---------
table
    Fits Poisson and negative binomial models for one year filter and
    returns the AIC comparison. Used only inside this module, by
    :func:`main`.
main
    Command-line entry point. Run via ``python -m dash_overdisp`` from the
    ``Makefile`` and ``scripts/run_pipeline.py``.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf

from config import DONATIONS, REPORTS

don = pd.read_parquet(DONATIONS)

# ============ #
# TABLE METHOD #
# ============ #
def table(year=None):
    """Compare Poisson and negative binomial AIC at dashboard grain.

    Aggregates the donations to one row per (donor, sex, donation type),
    with the number of donations as response and the maximum observed age,
    then fits ``total_donations ~ age + C(sex) + C(donation_type)`` with
    both families.

    Parameters
    ----------
    year : int, optional
        Restrict to donations of this ``donation_year``. If ``None``, all
        years are used.

    Returns
    -------
    dict
        ``n`` : int
            Number of aggregated rows used in the fit.
        ``aic_pois`` : float
            AIC of the Poisson GLM.
        ``aic_nb`` : float
            AIC of the negative binomial (NB2) model.
        ``delta`` : float
            ``aic_pois - aic_nb`` (positive = NB fits better).
        ``theta`` : float
            NB size parameter ``1 / alpha`` (``inf`` if ``alpha <= 0``).
    """
    d = don if year is None else don[don.donation_year == year]
    g = (d.groupby(['donor_id', 'sex', 'donation_type'], observed=True)
           .agg(total_donations=('age', 'size'), age=('age', 'max'))
           .reset_index()
           .dropna(subset=['sex', 'donation_type', 'age']))
    g = g[np.isfinite(g.age)]
    f = 'total_donations ~ age + C(sex) + C(donation_type)'
    pois = smf.glm(f, data=g, family=sm.families.Poisson()).fit()
    nb = sm.NegativeBinomial.from_formula(f, data=g).fit(disp=0, maxiter=200)
    alpha = float(nb.params.get('alpha', np.nan))
    return dict(n=len(g), aic_pois=pois.aic, aic_nb=nb.aic,
                delta=pois.aic - nb.aic, theta=1 / alpha if alpha > 0 else np.inf)

# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: build and save the AIC comparison table.

    Runs :func:`table` for the 2015 filter, the 2025 filter and all years,
    prints one line per filter and writes
    ``reports/dash_overdispersione.csv`` with columns ``filtro_anno``,
    ``n``, ``aic_pois``, ``aic_nb``, ``delta`` and ``theta``.

    Returns
    -------
    None
    """
    rows = []
    for label, year in [('2015', 2015), ('2025', 2025), ('Tutti', None)]:
        r = table(year)
        r['filtro_anno'] = label
        rows.append(r)
        print(f"{label:6} n={r['n']:7,}  AIC_Poisson={r['aic_pois']:12,.1f}  "
              f"AIC_NB={r['aic_nb']:12,.1f}  dAIC={r['delta']:+11,.1f}  "
              f"theta={r['theta']:,.4g}")
    out = pd.DataFrame(rows)[['filtro_anno', 'n', 'aic_pois', 'aic_nb',
                              'delta', 'theta']]
    out.to_csv(REPORTS / 'dash_overdispersione.csv', index=False)
    print(f"\nscritto {REPORTS / 'dash_overdispersione.csv'}")


if __name__ == '__main__':
    main()
