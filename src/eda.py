# ======== #
# EDA FILE #
# ======== #
"""Stage 4 — exploration, redone from scratch on the rebuilt data.

It takes no number from the original thesis documents: every figure and every
table here comes from `donations.parquet` and `donor_year.parquet`. The checks
for dependencies between columns (§ `exact_dependencies.csv`) are redone
independently precisely so as not to inherit unchecked conclusions.

It serves to decide how to model, so it looks at three things in particular:
the shape of the series to forecast, the dispersion of the response at the
chosen grain, and the traps in the covariates (coding breaks, columns that are
the target rewritten).

Functions
---------
exact_dependencies
    Finds pairs of columns in an exact functional relationship. Used only
    inside this module, by :func:`main`.
leakage_check
    Measures how much the control columns rewrite the target. Used only
    inside this module, by :func:`main`.
dispersion_table
    Variance/mean of the response with and without zeros. Used only inside
    this module, by :func:`main`.
fig_yearly
    Donations per year. Used by :func:`main` and :mod:`figures_tesi`.
fig_monthly
    Donations per month. Used by :func:`main` and :mod:`figures_tesi`.
fig_seasonality
    Monthly share of donations. Used by :func:`main` and
    :mod:`figures_tesi`.
fig_types
    Donations per year by type. Used by :func:`main` and
    :mod:`figures_tesi`.
fig_response_distribution
    Distribution of y with and without zeros. Used only by :func:`main`.
fig_recency
    Probability of donating by recency class. Used only by :func:`main`.
fig_age
    Mean y by age band. Used only by :func:`main`.
fig_donor_class_regime
    Donor class counts around the 2016 recoding. Used only by :func:`main`.
main
    Command-line entry point. Run via ``python -m eda`` from the
    ``Makefile`` and ``scripts/run_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import viz
from config import DONATIONS, DONOR_YEAR, REPORTS
from viz import SERIES, plt

TOP_TYPES = 3


# ── tables ───────────────────────────────────────────────────────────────────

# ========================= #
# EXACT_DEPENDENCIES METHOD #
# ========================= #
def exact_dependencies(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs of columns linked by an exact functional relationship.

    Two functionally related columns carry the same fact: putting both in the
    same formula makes the fit unidentifiable. The check is done here instead
    of being assumed.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned donations, with the candidate columns (``donation_year``,
        ``donation_type``, ``sex``, ``donor_class``, ``association``,
        ``association_desc``, ``collection_point``, ``age``,
        ``donor_class_regime``).

    Returns
    -------
    pandas.DataFrame
        One row per ordered pair (a, b) where each level of a maps to at most
        one level of b, with columns ``da``, ``a``, ``relazione``,
        ``livelli_da`` and ``livelli_a``. Empty if no such pair exists.
    """
    cand = ["donation_year", "donation_type", "sex", "donor_class", "association",
            "association_desc", "collection_point", "age", "donor_class_regime"]
    rows = []
    for a in cand:
        for b in cand:
            if a == b:
                continue
            g = df.groupby(a, observed=True)[b].nunique()
            if (g <= 1).all():
                rows.append({"da": a, "a": b, "relazione": "funzionale (1:1 o n:1)",
                             "livelli_da": int(df[a].nunique()),
                             "livelli_a": int(df[b].nunique())})
    return pd.DataFrame(rows)


# ==================== #
# LEAKAGE_CHECK METHOD #
# ==================== #
def leakage_check(df: pd.DataFrame) -> pd.DataFrame:
    """How much the 'control' columns are the target rewritten.

    `num_don_declared` is the count per (donor, year, type) copied onto every
    row of the group; `quantity_total_ml` is the sum of `quantity_ml` within the
    same group. Both are computed from the response: a model including them
    would report an almost perfect fit and it would mean nothing.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned donations with ``donor_id``, ``donation_year``,
        ``donation_type``, ``num_don_declared``, ``quantity_total_ml`` and
        ``quantity_ml``.

    Returns
    -------
    pandas.DataFrame
        Three rows (columns ``controllo``, ``gruppi``, ``coincide``,
        ``quota_%``): share of groups where ``num_don_declared`` equals the
        row count, share where ``quantity_total_ml`` equals the sum of
        ``quantity_ml``, and the correlation between declared and counted
        donations (in ``quota_%``).
    """
    g = (df.groupby(["donor_id", "donation_year", "donation_type"], observed=True)
           .agg(conteggio=("donor_id", "size"),
                dichiarato=("num_don_declared", "first"),
                volume_tot=("quantity_total_ml", "first"),
                volume_somma=("quantity_ml", "sum")))
    ok_cnt = (g.conteggio == g.dichiarato)
    v = g.dropna(subset=["volume_tot"])
    return pd.DataFrame([
        {"controllo": "num_don_declared == conteggio righe del gruppo",
         "gruppi": len(g), "coincide": int(ok_cnt.sum()),
         "quota_%": round(100 * float(ok_cnt.mean()), 2)},
        {"controllo": "quantity_total_ml == somma di quantity_ml nel gruppo",
         "gruppi": len(v), "coincide": int(np.isclose(v.volume_tot, v.volume_somma).sum()),
         "quota_%": round(100 * float(np.isclose(v.volume_tot, v.volume_somma).mean()), 2)},
        {"controllo": "correlazione num_don_declared / conteggio",
         "gruppi": len(g), "coincide": "",
         "quota_%": round(float(g.conteggio.corr(g.dichiarato)), 4)},
    ])


# ======================= #
# DISPERSION_TABLE METHOD #
# ======================= #
def dispersion_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean, variance and ratio of the response, with and without the zeros.

    The variance/mean ratio tells which count family is plausible. Computing it
    on the active rows only — i.e. on the truncated data — is what makes the
    data look *under*-dispersed and leads to wrong diagnoses.

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel with ``y`` and ``active``.

    Returns
    -------
    pandas.DataFrame
        Two rows (full panel; active rows only) with columns ``insieme``,
        ``n``, ``media``, ``varianza``, ``var/media`` and ``quota_zeri_%``.
    """
    out = []
    for label, s in [("panel completo (zeri inclusi)", panel.y),
                     ("solo anni attivi (y>=1)", panel.loc[panel.active == 1, "y"])]:
        out.append({"insieme": label, "n": len(s),
                    "media": round(float(s.mean()), 4),
                    "varianza": round(float(s.var()), 4),
                    "var/media": round(float(s.var() / s.mean()), 4),
                    "quota_zeri_%": round(100 * float((s == 0).mean()), 2)})
    return pd.DataFrame(out)


# ── figures ──────────────────────────────────────────────────────────────────

# ================= #
# FIG_YEARLY METHOD #
# ================= #
def fig_yearly(df, path):
    """Line chart of donations per year, labelling first and last value.

    Parameters
    ----------
    df : pandas.DataFrame
        Donations with ``donation_year``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    s = df.groupby("donation_year").size()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(s.index, s.values, color=SERIES[0], marker="o")
    ax.set_ylim(0, s.max() * 1.12)
    ax.annotate(f"{s.iloc[-1]:,}", (s.index[-1], s.iloc[-1]),
                textcoords="offset points", xytext=(6, 0), color=viz.INK, fontsize=9)
    ax.annotate(f"{s.iloc[0]:,}", (s.index[0], s.iloc[0]),
                textcoords="offset points", xytext=(-4, 8), color=viz.INK, fontsize=9)
    viz.finish(ax, "Donazioni per anno", None, "donazioni")
    viz.save(fig, path)


# ================== #
# FIG_MONTHLY METHOD #
# ================== #
def fig_monthly(df, path):
    """Line chart of donations per calendar month over the whole window.

    Parameters
    ----------
    df : pandas.DataFrame
        Donations with ``donation_year`` and ``donation_month``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    s = (df.groupby([df.donation_year, df.donation_month]).size()
           .rename("n").reset_index())
    s["t"] = pd.to_datetime(dict(year=s.donation_year, month=s.donation_month, day=1))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(s.t, s.n, color=SERIES[0], linewidth=1.3)
    viz.finish(ax, "Donazioni per mese", None, "donazioni")
    viz.save(fig, path)


MONTHS = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
        "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]


# ====================== #
# FIG_SEASONALITY METHOD #
# ====================== #
def fig_seasonality(df, path):
    """Monthly share of donations, with the uniformity line at 8.33%.

    The chapter argues in percentages (March 9.24%, August 7.60%) and compares
    them with the 100/12 expected in the absence of seasonality: the axis must
    be that one, not the absolute count, otherwise none of the figures quoted
    in the text can be read from the chart.

    Parameters
    ----------
    df : pandas.DataFrame
        Donations with ``donation_month``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    q = df.groupby("donation_month").size() / len(df) * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(q.index, q.values, color=SERIES[0], width=0.62)
    expected = 100 / 12
    ax.axhline(expected, linestyle="--", linewidth=1.1, color=viz.INK_SOFT)
    ax.annotate(f"uniforme: {expected:.2f}%".replace(".", ","),
                xy=(0.55, expected), xytext=(0, 4), textcoords="offset points",
                va="bottom", ha="left", fontsize=7.5, color=viz.INK_SOFT)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_ylim(0, max(q.max(), expected) * 1.12)
    viz.finish(ax, "Quota mensile delle donazioni, 2009-2025", None,
               "quota sul totale (%)")
    viz.save(fig, path)


# ================ #
# FIG_TYPES METHOD #
# ================ #
def fig_types(df, path):
    """Donations per year for the ``TOP_TYPES`` most common types plus "Altro".

    Parameters
    ----------
    df : pandas.DataFrame
        Donations with ``donation_year`` and ``donation_type``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    top = df.donation_type.value_counts().head(TOP_TYPES).index.tolist()
    t = df.assign(tipo=np.where(df.donation_type.isin(top), df.donation_type, "Altro"))
    piv = t.pivot_table(index="donation_year", columns="tipo", aggfunc="size", fill_value=0)
    piv = piv[[c for c in top if c in piv] + (["Altro"] if "Altro" in piv else [])]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, c in enumerate(piv.columns):
        ax.plot(piv.index, piv[c], color=SERIES[i], marker="o", label=c)
    ax.legend(loc="upper right", ncols=2)
    viz.finish(ax, "Donazioni per tipo", None, "donazioni")
    viz.save(fig, path)


# ================================ #
# FIG_RESPONSE_DISTRIBUTION METHOD #
# ================================ #
def fig_response_distribution(panel, path):
    """Bar chart of y on the full panel versus active rows only (y = 0..8).

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel with ``y`` and ``active``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    a = panel.y.value_counts().sort_index().head(9)
    b = panel.loc[panel.active == 1, "y"].value_counts().sort_index().head(9)
    x = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - 0.21, a.reindex(a.index).values, width=0.4,
           color=SERIES[0], label="panel completo (zeri inclusi)")
    ax.bar(x + 0.21, b.reindex(a.index, fill_value=0).values, width=0.4,
           color=SERIES[1], label="solo anni attivi")
    ax.set_xticks(x); ax.set_xticklabels(a.index)
    ax.legend(loc="upper right")
    viz.finish(ax, "Distribuzione della risposta (donazioni per donatore-anno)",
               "donazioni nell'anno", "donatori-anno")
    viz.save(fig, path)


# ================== #
# FIG_RECENCY METHOD #
# ================== #
def fig_recency(panel, path):
    """Share of active donor-years by recency class, entry to "6+".

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel with ``recency_class`` and ``active``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    order = ["entry", "1", "2", "3", "4", "5", "6+"]
    r = (panel.groupby("recency_class").active.mean().reindex(order) * 100)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(range(len(r)), r.values, color=SERIES[0], width=0.62)
    ax.set_xticks(range(len(r))); ax.set_xticklabels(order)
    for i, v in enumerate(r.values):
        ax.annotate(f"{v:.0f}%", (i, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, color=viz.INK)
    viz.finish(ax, "Probabilità di donare, per anni dall'ultima donazione",
               "anni dall'ultima donazione", "% attivi")
    viz.save(fig, path)


# ============== #
# FIG_AGE METHOD #
# ============== #
def fig_age(panel, path):
    """Mean y per donor-year by 5-year age band (ages 18-75).

    Parameters
    ----------
    panel : pandas.DataFrame
        Donor x year panel with ``age`` and ``y``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    b = panel[panel.age.between(18, 75)].copy()
    b["fascia"] = (b.age // 5) * 5
    m = b.groupby("fascia").y.mean()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(m.index, m.values, color=SERIES[0], marker="o")
    viz.finish(ax, "Donazioni attese per donatore-anno, per fascia d'età",
               "età (fasce di 5 anni)", "media di y")
    viz.save(fig, path)


# ============================= #
# FIG_DONOR_CLASS_REGIME METHOD #
# ============================= #
def fig_donor_class_regime(df, path):
    """Donations per year by donor class (P, O, D) on a log scale.

    A dashed line marks the 2016 administrative reclassification.

    Parameters
    ----------
    df : pandas.DataFrame
        Donations with ``donation_year`` and ``donor_class``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    piv = df.pivot_table(index="donation_year", columns="donor_class",
                         aggfunc="size", fill_value=0)
    keep = [c for c in ["P", "O", "D"] if c in piv]
    # Zeros become NaN: on a log scale a series that does not exist yet must
    # start when it appears, not plunge to the bottom of the chart.
    piv = piv[keep].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, c in enumerate(keep):
        ax.plot(piv.index, piv[c], color=SERIES[i], marker="o", label=f"classe {c}")
    ax.axvline(2016, color=viz.INK_SOFT, linestyle="--", linewidth=1)
    ax.annotate("riclassificazione\namministrativa", (2016, ax.get_ylim()[1] * 0.62),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                color=viz.INK_SOFT)
    ax.set_yscale("log")
    ax.legend(loc="center left")
    viz.finish(ax, "Classe del donatore: la codifica cambia nel 2016",
               None, "donazioni (scala log)")
    viz.save(fig, path)


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: write every exploratory figure and table.

    Writes the figures ``fig_*.png`` and the tables
    ``exact_dependencies.csv``, ``leakage_check.csv``, ``dispersion.csv``
    and ``yearly_summary.csv`` (donations, donors and donations per donor by
    year) to ``REPORTS``, printing each table.

    Returns
    -------
    None
    """
    df = pd.read_parquet(DONATIONS)
    panel = pd.read_parquet(DONOR_YEAR)
    print(f"donazioni {len(df):,} | panel {len(panel):,}\n")

    print("figure:")
    fig_yearly(df, REPORTS / "fig_donazioni_per_anno.png")
    fig_monthly(df, REPORTS / "fig_donazioni_per_mese.png")
    fig_seasonality(df, REPORTS / "fig_stagionalita.png")
    fig_types(df, REPORTS / "fig_tipi_donazione.png")
    fig_response_distribution(panel, REPORTS / "fig_distribuzione_risposta.png")
    fig_recency(panel, REPORTS / "fig_recency.png")
    fig_age(panel, REPORTS / "fig_eta.png")
    fig_donor_class_regime(df, REPORTS / "fig_donor_class_regime.png")

    print("\ntabelle:")
    dep = exact_dependencies(df)
    dep.to_csv(REPORTS / "exact_dependencies.csv", index=False)
    print(f"  dipendenze funzionali esatte trovate: {len(dep)}")
    if len(dep):
        print(dep.to_string(index=False))

    lk = leakage_check(df)
    lk.to_csv(REPORTS / "leakage_check.csv", index=False)
    print("\n  colonne che riscrivono il target:")
    print(lk.to_string(index=False))

    disp = dispersion_table(panel)
    disp.to_csv(REPORTS / "dispersion.csv", index=False)
    print("\n  dispersione della risposta:")
    print(disp.to_string(index=False))

    yearly = (df.groupby("donation_year")
                .agg(donazioni=("donor_id", "size"),
                     donatori=("donor_id", "nunique"))
                .assign(donazioni_per_donatore=lambda d:
                        (d.donazioni / d.donatori).round(4)))
    yearly.to_csv(REPORTS / "yearly_summary.csv")
    print("\n  serie annuale:")
    print(yearly.to_string())


if __name__ == "__main__":
    main()
