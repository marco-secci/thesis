# ============== #
# RETENTION FILE #
# ============== #
"""Retention of young donors (Ch. 3 of the thesis).

The original chapter defines a "young donor" as someone aged 18-25 **at the
first donation observed in the dataset**. With a third of donors already active
before 2009, that is not the first donation: the 2009 cohort turns out to be
half made of veterans caught mid-career, who by construction survive longer
than true recruits.

The raw data contains the true career start date (`first_activity_date`,
`dtinizatt_da` in the DIMT files), so the defect is measurable and fixable.

**How not to fix it.** It would be natural to use `first_activity_date` as the
entry year and recompute r from there. It does not work: for someone who
started in 2004 and is observed from 2009, r would count five years that are
not in the dataset. Retention would come out higher than what is observable for
a purely accounting reason.

**How to fix it.** Only **genuine entry cohorts** are kept: donors whose career
really starts inside the window, i.e. those for whom `first_activity_date` does
not precede the year of the first observed event. Those already active before
are excluded from the calculation, because they are left-truncated and have no
observable cohort. r stays measured on the data, not estimated.

Every measure is computed **twice**, with the thesis definition (`osservata`)
and with the corrected one (`ingresso`): the comparison between the two is the
content of the revision, not a change of numbers without explanation.

Functions
---------
donor_table
    One row per donor with observed and real entry year. Used by
    :func:`main`, :mod:`figures_tesi` and ``tests/test_retention.py``.
_r_class
    Private helper: retention years to class labels. Used only inside this
    module, by :func:`_base`.
_base
    Private helper: starting population for a definition. Used only inside
    this module.
young_cohort
    Young donors (18-25 at entry). Used by :func:`main` and
    ``tests/test_retention.py``.
distribution
    Distribution of retention over the published classes. Used by
    :func:`main` and ``tests/test_retention.py``.
survival
    Share of each cohort still active after k years. Used by :func:`main`
    and ``tests/test_retention.py``.
by_sex
    Retention statistics by sex. Used only inside this module, by
    :func:`main`.
by_age_band
    Young versus adult comparison. Used only inside this module, by
    :func:`main`.
cohort_composition
    Share of left-truncated donors per observed cohort. Used by
    :func:`main` and ``tests/test_retention.py``.
fig_young_retention
    Retention distribution and the 2009 cohort correction. Used only by
    :func:`main`.
fig_retention_by_age
    Survival by age band at entry. Used only by :func:`main`.
main
    Entry point. Run via ``python -m retention`` from the ``Makefile`` and
    ``scripts/run_pipeline.py``, and called by :mod:`figures_tesi` with the
    thesis image directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import viz
from config import DONATIONS, REPORTS, YEAR_MAX, YEAR_MIN

#: Window of the original chapter. The 2024-2025 cohorts have fewer than two
#: years of observation: including them would mechanically lower retention.
COHORT_MIN, COHORT_MAX = 2009, 2023

#: Last year of observation: the whole window. With cohorts stopping at 2023,
#: every cohort has at least two following years in which to return.
#: Truncating observation at 2023 (as in the first version of the chapter)
#: left the 2023 cohort without any following year: it came out 100%
#: "one-and-done" by construction and inflated the overall share by about
#: five points.
OBSERVATION_MAX = YEAR_MAX

#: Definition of "young": age at entry, bounds included.
YOUNG_AGE = (18, 25)

#: Age bands at entry for the comparison in sec:ret_comparison.
AGE_BANDS = [(18, 25, "18-25"), (26, 35, "26-35"),
             (36, 50, "36-50"), (51, 90, "51+")]

#: Cohorts with an observation horizon long enough for the curves.
SURVIVAL_COHORTS = range(2009, 2019)

#: Retention classes of the published tab:ret_dist.
R_CLASSES = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"),
             (4, 4, "4"), (5, 9, "5-9"), (10, 99, "10+")]


# ── Building the donor table ─────────────────────────────────────────────────

# ================== #
# DONOR_TABLE METHOD #
# ================== #
def donor_table(don: pd.DataFrame, through: int = OBSERVATION_MAX) -> pd.DataFrame:
    """One row per donor, with both definitions of entry year.

    - `anno_osservato`: first year present in the dataset (thesis definition)
    - `anno_reale`   : year of `first_activity_date`, i.e. the true start of
      the career; falls back on `anno_osservato` where the field is missing.

    Observation is truncated at `through` (default `OBSERVATION_MAX`, i.e. the
    whole window).

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id``, ``donation_year``, ``donation_datetime``,
        ``sex``, ``age`` and ``first_activity_date``.
    through : int, default OBSERVATION_MAX
        Last donation year considered.

    Returns
    -------
    pandas.DataFrame
        One row per donor with columns ``donor_id``, ``anno_osservato``,
        ``ultimo_anno``, ``donazioni``, ``sex``, ``anno_reale`` (never later
        than ``anno_osservato``), ``ha_data_inizio``,
        ``troncato_a_sinistra``, ``attivo_prima_2009``, ``eta_primo``,
        ``anno_primo``, ``eta_osservata`` (age at the first observed event)
        and ``eta_reale`` (age carried back to ``anno_reale``).
    """
    don = don[don["donation_year"] <= through]
    g = don.groupby("donor_id", observed=True)
    t = pd.DataFrame({
        "anno_osservato": g["donation_year"].min(),
        "ultimo_anno": g["donation_year"].max(),
        "donazioni": g.size(),
        "sex": g["sex"].first(),
    })
    start = g["first_activity_date"].min()
    t["anno_reale"] = start.dt.year
    t["ha_data_inizio"] = t["anno_reale"].notna()
    # A start date later than the first observed event is inconsistent: the
    # minimum of the two is kept.
    t["anno_reale"] = np.minimum(
        t["anno_reale"].fillna(t["anno_osservato"]), t["anno_osservato"]
    ).astype("int16")
    t["troncato_a_sinistra"] = (t["anno_reale"] < t["anno_osservato"]).astype("int8")
    t["attivo_prima_2009"] = (t["anno_reale"] < YEAR_MIN).astype("int8")

    # Age at entry: the one recorded on the oldest observed event, carried
    # back to the real entry year.
    first = (don.sort_values("donation_datetime")
                .groupby("donor_id", observed=True)
                .agg(eta_primo=("age", "first"), anno_primo=("donation_year", "first")))
    t = t.join(first)
    t["eta_osservata"] = t["eta_primo"]
    t["eta_reale"] = t["eta_primo"] - (t["anno_osservato"] - t["anno_reale"])
    return t.reset_index()


# =============== #
# _R_CLASS METHOD #
# =============== #
def _r_class(r: pd.Series) -> pd.Series:
    """Map retention years to the class labels of ``R_CLASSES``.

    Parameters
    ----------
    r : pandas.Series
        Retention in years (non-negative integers).

    Returns
    -------
    pandas.Series
        Same index as `r`, with labels ``"0"``..``"4"``, ``"5-9"`` or
        ``"10+"`` (empty string if `r` falls outside every class).
    """
    out = pd.Series("", index=r.index, dtype="object")
    for lo, hi, lab in R_CLASSES:
        out[(r >= lo) & (r <= hi)] = lab
    return out


# ============ #
# _BASE METHOD #
# ============ #
def _base(t: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Starting population according to `definition` ∈ {osservata, ingresso}.

    In both cases the cohort is the year of the **first observed event** and r
    is measured on the data. What changes is who enters the calculation:
    `ingresso` drops left-truncated donors, who are not entries of that cohort.

    Parameters
    ----------
    t : pandas.DataFrame
        Donor table, as returned by :func:`donor_table`.
    definition : {"osservata", "ingresso"}
        Thesis definition, or corrected one keeping only donors with a start
        date that is not left-truncated.

    Returns
    -------
    pandas.DataFrame
        Donors whose observed entry is in ``COHORT_MIN``-``COHORT_MAX``,
        with the extra columns ``coorte``, ``r`` (last year minus entry year)
        and ``classe_r``.
    """
    sel = t[t["anno_osservato"].between(COHORT_MIN, COHORT_MAX)]
    if definition == "ingresso":
        sel = sel[sel["ha_data_inizio"] & (sel["troncato_a_sinistra"] == 0)]
    sel = sel.copy()
    sel["coorte"] = sel["anno_osservato"]
    sel["r"] = (sel["ultimo_anno"] - sel["anno_osservato"]).clip(lower=0)
    sel["classe_r"] = _r_class(sel["r"])
    return sel


# =================== #
# YOUNG_COHORT METHOD #
# =================== #
def young_cohort(t: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Young donors (18-25 at entry) according to `definition`.

    Parameters
    ----------
    t : pandas.DataFrame
        Donor table, as returned by :func:`donor_table`.
    definition : {"osservata", "ingresso"}
        Entry definition, see :func:`_base`.

    Returns
    -------
    pandas.DataFrame
        Rows of :func:`_base` with ``eta_osservata`` within ``YOUNG_AGE``.
    """
    lo, hi = YOUNG_AGE
    sel = _base(t, definition)
    return sel[sel["eta_osservata"].between(lo, hi)].copy()


# ── Measures ─────────────────────────────────────────────────────────────────

# =================== #
# DISTRIBUTION METHOD #
# =================== #
def distribution(young: pd.DataFrame) -> pd.DataFrame:
    """Distribution of r over the classes of the published table.

    Parameters
    ----------
    young : pandas.DataFrame
        Cohort with ``classe_r``, e.g. from :func:`young_cohort`.

    Returns
    -------
    pandas.DataFrame
        One row per class of ``R_CLASSES`` (in order) with columns
        ``anni_retention``, ``donatori``, ``quota_pct`` and
        ``cumulata_pct``.
    """
    order = [lab for _, _, lab in R_CLASSES]
    n = young.groupby("classe_r", observed=True).size().reindex(order, fill_value=0)
    out = pd.DataFrame({"donatori": n})
    out["quota_pct"] = 100 * out["donatori"] / out["donatori"].sum()
    out["cumulata_pct"] = out["quota_pct"].cumsum()
    return out.reset_index().rename(columns={"classe_r": "anni_retention"})


# =============== #
# SURVIVAL METHOD #
# =============== #
def survival(don: pd.DataFrame, young: pd.DataFrame) -> pd.DataFrame:
    """Share of cohort c still active in year c+k.

    Only cohorts of ``SURVIVAL_COHORTS`` with at least 10 donors are used,
    for k from 1 to 10 (or up to ``OBSERVATION_MAX``).

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id`` and ``donation_year``.
    young : pandas.DataFrame
        Cohort with ``donor_id`` and ``coorte``, e.g. from
        :func:`young_cohort`.

    Returns
    -------
    pandas.DataFrame
        One row per (cohort, k) with columns ``coorte``, ``k``,
        ``n_coorte``, ``attivi`` (donors with a donation in year c+k) and
        ``quota_pct``.
    """
    don = don[don["donation_year"] <= OBSERVATION_MAX]
    active = don[["donor_id", "donation_year"]].drop_duplicates()
    active = active.merge(young[["donor_id", "coorte"]], on="donor_id")
    active["k"] = active["donation_year"] - active["coorte"]

    rows = []
    for c in SURVIVAL_COHORTS:
        base = young[young["coorte"] == c]
        if len(base) < 10:
            continue
        sub = active[active["coorte"] == c]
        for k in range(1, min(11, OBSERVATION_MAX - c + 1)):
            alive = sub.loc[sub["k"] == k, "donor_id"].nunique()
            rows.append({"coorte": c, "k": k, "n_coorte": len(base),
                         "attivi": alive, "quota_pct": 100 * alive / len(base)})
    return pd.DataFrame(rows)


# ============= #
# BY_SEX METHOD #
# ============= #
def by_sex(young: pd.DataFrame) -> pd.DataFrame:
    """Retention statistics by sex, plus a total row.

    Parameters
    ----------
    young : pandas.DataFrame
        Cohort with ``sex`` and ``r``, e.g. from :func:`young_cohort`.

    Returns
    -------
    pandas.DataFrame
        One row per sex and a ``"Totale"`` row, with columns ``sex``, ``n``,
        ``media``, ``ds``, ``mediana``, ``p75`` and ``one_timer_pct`` (share
        with ``r == 0``).
    """
    g = young.groupby("sex", observed=True)["r"]
    out = pd.DataFrame({
        "n": g.size(), "media": g.mean(), "ds": g.std(),
        "mediana": g.median(), "p75": g.quantile(0.75),
    })
    out["one_timer_pct"] = 100 * young.groupby("sex", observed=True)["r"].apply(
        lambda s: (s == 0).mean())
    tot = young["r"]
    out.loc["Totale"] = [len(tot), tot.mean(), tot.std(), tot.median(),
                         tot.quantile(0.75), 100 * (tot == 0).mean()]
    return out.reset_index().rename(columns={"index": "sex"})


# ================== #
# BY_AGE_BAND METHOD #
# ================== #
def by_age_band(t: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Young / adult comparison, sec:ret_comparison.

    Parameters
    ----------
    t : pandas.DataFrame
        Donor table, as returned by :func:`donor_table`.
    definition : {"osservata", "ingresso"}
        Entry definition, see :func:`_base`.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty band of ``AGE_BANDS`` with columns ``fascia``,
        ``donatori``, ``quota_pct``, ``one_timer_pct``, ``retention_media``
        and ``attivi_5_piu_pct`` (share with ``r >= 5``).
    """
    base = _base(t, definition)
    rows = []
    for lo, hi, lab in AGE_BANDS:
        sub = base[base["eta_osservata"].between(lo, hi)]
        if sub.empty:
            continue
        rows.append({"fascia": lab, "donatori": len(sub),
                     "quota_pct": 100 * len(sub) / len(base),
                     "one_timer_pct": 100 * (sub["r"] == 0).mean(),
                     "retention_media": sub["r"].mean(),
                     "attivi_5_piu_pct": 100 * (sub["r"] >= 5).mean()})
    return pd.DataFrame(rows)


# ========================= #
# COHORT_COMPOSITION METHOD #
# ========================= #
def cohort_composition(young_obs: pd.DataFrame) -> pd.DataFrame:
    """How much of each "observed" cohort is actually left-truncated.

    It is the number that refutes the 2009 cohort as the example of the
    chapter's survival curves.

    Parameters
    ----------
    young_obs : pandas.DataFrame
        Young cohort with the thesis definition (``"osservata"``), with
        ``coorte`` and ``troncato_a_sinistra``.

    Returns
    -------
    pandas.DataFrame
        One row per cohort with columns ``coorte``, ``donatori``,
        ``gia_attivi_prima`` and ``quota_pct``.
    """
    g = young_obs.groupby("coorte", observed=True)
    out = pd.DataFrame({
        "donatori": g.size(),
        "gia_attivi_prima": g["troncato_a_sinistra"].sum(),
    })
    out["quota_pct"] = 100 * out["gia_attivi_prima"] / out["donatori"]
    return out.reset_index()


# ── Figures ──────────────────────────────────────────────────────────────────

# ========================== #
# FIG_YOUNG_RETENTION METHOD #
# ========================== #
def fig_young_retention(dist, surv_entry, surv_observed, path):
    """Left: distribution of r. Right: what changes when correcting the cohort.

    The right panel is the chapter's correction in one figure: the 2009
    cohort, which the thesis uses as its main example, is half made of donors
    already active before 2009 and survives much longer than a true entry
    cohort.

    Parameters
    ----------
    dist : pandas.DataFrame
        Retention distribution, as returned by :func:`distribution`.
    surv_entry : pandas.DataFrame
        Survival with the corrected definition (:func:`survival`).
    surv_observed : pandas.DataFrame
        Survival with the thesis definition (:func:`survival`).
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))

    a1.bar(dist["anni_retention"], dist["quota_pct"], color=viz.SERIES[0], width=0.72)
    one = dist.loc[dist["anni_retention"] == "0", "quota_pct"].iloc[0]
    a1.text(0, one - 2.4, f"{one:.1f}%", ha="center", va="top",
            color="white", fontweight="bold")
    a1.set_ylim(0, max(dist["quota_pct"]) * 1.12)
    viz.finish(a1, "Anni fra prima e ultima donazione",
               "anni di retention", "% dei donatori giovani")

    others = surv_entry[surv_entry["coorte"] != 2009]
    for _, sub in others.groupby("coorte", observed=True):
        a2.plot(sub["k"], sub["quota_pct"], color=viz.GRID, linewidth=1.1, zorder=1)
    for name, surv, colour in (
            ("coorte 2009, definizione della tesi", surv_observed, viz.SERIES[1]),
            ("coorte 2009, soli ingressi genuini", surv_entry, viz.SERIES[0])):
        sub = surv[surv["coorte"] == 2009]
        a2.plot(sub["k"], sub["quota_pct"], color=colour, marker="o",
                label=name, zorder=3)
    a2.plot([], [], color=viz.GRID, linewidth=1.1, label="coorti 2010-2018")
    a2.legend(fontsize=8)
    viz.finish(a2, "Sopravvivenza della coorte 2009",
               "anni dall'ingresso", "% ancora attivi")
    fig.tight_layout()
    viz.save(fig, path)


# =========================== #
# FIG_RETENTION_BY_AGE METHOD #
# =========================== #
def fig_retention_by_age(don: pd.DataFrame, t: pd.DataFrame, path):
    """Survival curves by age band at real entry.

    Uses the corrected definition and cohorts up to 2018; at each horizon k
    only cohorts observable at c+k are counted.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id`` and ``donation_year``.
    t : pandas.DataFrame
        Donor table, as returned by :func:`donor_table`.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    don = don[don["donation_year"] <= OBSERVATION_MAX]
    active = don[["donor_id", "donation_year"]].drop_duplicates()
    base = _base(t, "ingresso")
    base = base[base["coorte"] <= 2018]
    active = active.merge(base[["donor_id", "coorte", "eta_osservata"]], on="donor_id")
    active["k"] = active["donation_year"] - active["coorte"]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (lo, hi, lab) in enumerate(AGE_BANDS):
        sel = base[base["eta_osservata"].between(lo, hi)]
        sub = active[active["eta_osservata"].between(lo, hi)]
        # At each horizon k only cohorts observable at c+k are included:
        # dividing by the whole band would lower the tail for an accounting
        # reason.
        shares = [100 * sub.loc[(sub["k"] == k)
                                & (sub["coorte"] + k <= OBSERVATION_MAX),
                                "donor_id"].nunique()
                  / (sel["coorte"] + k <= OBSERVATION_MAX).sum()
                  for k in range(1, 11)]
        ax.plot(range(1, 11), shares, marker="o", color=viz.SERIES[i], label=lab)
    ax.legend(title="età all'ingresso")
    viz.finish(ax, "Sopravvivenza per fascia d'età all'ingresso",
               "anni dall'ingresso", "% ancora attivi")
    viz.save(fig, path)


# ── Entry point ──────────────────────────────────────────────────────────────

# =========== #
# MAIN METHOD #
# =========== #
def main(img_dir=None) -> None:
    """Entry point: compute every retention table and figure.

    For both definitions (``"osservata"`` and ``"ingresso"``) computes the
    distribution, the age-band and sex comparisons and the survival, and
    writes them to ``reports/retention_*.csv``; also writes the cohort
    composition and the figures ``fig_young_retention.png`` and
    ``fig_retention_by_age.png``.

    Parameters
    ----------
    img_dir : pathlib.Path, optional
        Directory for the figures. Defaults to ``REPORTS``;
        :mod:`figures_tesi` passes the thesis image directory.

    Returns
    -------
    None
    """
    don = pd.read_parquet(DONATIONS)
    t = donor_table(don)
    img = img_dir or REPORTS

    print(f"donatori totali: {len(t):,}")
    print(f"  con data di inizio attività: {t['ha_data_inizio'].sum():,}")
    print(f"  con carriera iniziata prima del primo evento osservato: "
          f"{t['troncato_a_sinistra'].sum():,} "
          f"({100 * t['troncato_a_sinistra'].mean():.1f}%)")
    print(f"  già attivi prima del {YEAR_MIN}: {t['attivo_prima_2009'].sum():,} "
          f"({100 * t['attivo_prima_2009'].mean():.1f}%)")

    tables = {}
    for definition in ("osservata", "ingresso"):
        young = young_cohort(t, definition)
        d = distribution(young)
        d.insert(0, "definizione", definition)
        tables.setdefault("distribuzione", []).append(d)

        f = by_age_band(t, definition)
        f.insert(0, "definizione", definition)
        tables.setdefault("fasce", []).append(f)

        g = by_sex(young)
        g.insert(0, "definizione", definition)
        tables.setdefault("genere", []).append(g)

        s = survival(don, young)
        s.insert(0, "definizione", definition)
        tables.setdefault("sopravvivenza", []).append(s)

        print(f"\n[{definition}] giovani: {len(young):,}  "
              f"one-timer: {100 * (young['r'] == 0).mean():.1f}%  "
              f"attivi 5+: {100 * (young['r'] >= 5).mean():.1f}%")

    for name, pieces in tables.items():
        out = pd.concat(pieces, ignore_index=True)
        out.to_csv(REPORTS / f"retention_{name}.csv", index=False)
        print(f"  scritto retention_{name}.csv")

    comp = cohort_composition(young_cohort(t, "osservata"))
    comp.to_csv(REPORTS / "retention_composizione_coorti.csv", index=False)
    print("  scritto retention_composizione_coorti.csv")
    r2009 = comp[comp["coorte"] == 2009].iloc[0]
    print(f"\ncoorte 2009 (definizione della tesi): {int(r2009.donatori):,} donatori, "
          f"di cui {int(r2009.gia_attivi_prima):,} ({r2009.quota_pct:.1f}%) "
          f"già attivi prima del 2009")

    young = young_cohort(t, "ingresso")
    fig_young_retention(distribution(young), survival(don, young),
                        survival(don, young_cohort(t, "osservata")),
                        img / "fig_young_retention.png")
    fig_retention_by_age(don, t, img / "fig_retention_by_age.png")


if __name__ == "__main__":
    main()
