# =============== #
# POPULATION FILE #
# =============== #
"""Stage 5 — the denominator: how many potential donors there are.

**The gap this module fills.** All the rest of the work — and the whole
original thesis — counts donations and donors without ever dividing them by the
population that could donate. But the territory the data comes from (the
provinces of Trieste and Gorizia) is losing inhabitants and ageing: part of the
decline in donations is mechanical, not behavioural, and until the two are
separated every conclusion about "the system is eroding" remains ambiguous.
The question that matters to whoever plans collection is not how many
donations there were, but **whether the propensity to donate is declining** —
and that can only be seen with a denominator.

**Source of the main denominator: NUTS-3, Trieste and Gorizia.** Eurostat
`demo_r_pjanaggr3` (Population on 1 January by broad age group, sex and
NUTS 3 region), geo ``ITH43`` = Gorizia and ``ITH44`` = Trieste,
both sexes. Unlike `demo_r_pjangroup` (NUTS-2), this dataflow is not blocked by
network policy: the series used here was provided by the author locally and is
committed in a form reduced to the Friuli Venezia Giulia rows only
(`data/raw/istat/eurostat_nuts3_fvg_pjanaggr3.tsv`, ~4 KB). It is the fix for
the limitation stated earlier: before this series, the denominator used was the
population of the whole of Friuli Venezia Giulia (NUTS-2), while the donations
in the numerator come only from Trieste and Gorizia — roughly a fifth of the
regional catchment — which systematically underestimated the "per 1,000
residents" rates.

**An approximation that remains.** The NUTS-3 dataflow publishes only three
broad bands (``Y_LT15``, ``Y15-64``, ``Y_GE65``), not the
five-year bands available at NUTS-2 level. ``Y15-64`` is therefore used
as a proxy for the population of donating age (18-64), slightly wider than the
exact band; the effect is small (ages 15-17 are a minimal share of the band)
but it must be stated. For the same reason, the profile by five-year band
(figure ``fig:tassi_eta`` in the thesis) cannot be rebuilt for Trieste and Gorizia
with public Eurostat data: that figure is still computed on the population of
the whole of Friuli Venezia Giulia (function `load_population_regional`,
below), as a declared proxy for the shape of the age profile only, not for its
level.

**Source of the age profile: NUTS-2, whole of Friuli Venezia Giulia.** Eurostat
`demo_r_pjangroup` (Population on 1 January by age group, sex and NUTS 2
region), geo ITH4, five-year bands::

    curl "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/\\
    demo_r_pjangroup?format=JSON&geo=ITH4&sex=T&unit=NR" -o data/raw/istat/eurostat_fvg_pop.json

The downloaded file is committed: it is open data, weighs 13 KB, and having it
in the repo makes the pipeline runnable without network access. It is the same
ISTAT population register survey, republished with NUTS coding and with the
complete time series: the direct ISTAT API only exposes 2019 onwards in the
corresponding dataflow.

Functions
---------
load_population_ts_go
    Trieste + Gorizia population by broad band (NUTS-3). Used only inside
    this module, by :func:`main`.
load_population_regional
    FVG population by five-year band (NUTS-2). Used only inside this
    module, by :func:`main`.
eligible
    Population of donating age by year. Used only inside this module, by
    :func:`main`.
rates
    Donations and donors per 1,000 residents. Used only inside this module,
    by :func:`main`.
decomposition
    Splits the decline into population and propensity. Used only inside
    this module, by :func:`main`.
age_specific
    Donors per 1,000 residents by age band. Used only inside this module,
    by :func:`main`.
fig_rate
    Absolute and per-resident donations. Used only by :func:`main`.
fig_age_specific
    Age profile of donors per resident. Used only by :func:`main`.
main
    Entry point. Run via ``python -m population`` from the ``Makefile`` and
    ``scripts/run_pipeline.py``, and called by :mod:`figures_tesi` with the
    thesis image directory.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import viz
from config import DATA, DONATIONS, DONOR_YEAR, REPORTS, YEAR_MAX, YEAR_MIN
from viz import SERIES, plt

#: Regional population (NUTS-2, ITH4), five-year bands — used only for the
#: age profile (see "load_population_regional"): at NUTS-3 level Eurostat
#: does not publish bands finer than the three broad ones.
POP_JSON = DATA / "raw" / "istat" / "eurostat_fvg_pop.json"

#: Population of Trieste and Gorizia (NUTS-3, ITH43 + ITH44), broad bands —
#: main denominator of this module.
POP_NUTS3_TSV = DATA / "raw" / "istat" / "eurostat_nuts3_fvg_pjanaggr3.tsv"
NUTS3_TS_GO = ("ITH43", "ITH44")  # Gorizia, Trieste

#: Bands entirely within donating age (18-64 as a rule, up to 70 for regular
#: donors upon clinical assessment) — used only by the NUTS-2 age profile.
BANDS_CORE = ["Y20-24", "Y25-29", "Y30-34", "Y35-39", "Y40-44",
              "Y45-49", "Y50-54", "Y55-59", "Y60-64"]
#: Share of the 15-19 band aged 18 or 19.
SHARE_18_19 = 2 / 5


# ============================ #
# LOAD_POPULATION_TS_GO METHOD #
# ============================ #
def load_population_ts_go() -> pd.DataFrame:
    """Population of Trieste and Gorizia by broad band and year (NUTS-3).

    Reads ``POP_NUTS3_TSV`` (Eurostat TSV), keeps both sexes and the geo
    codes in ``NUTS3_TS_GO``, drops missing values (``:``) and sums the two
    provinces.

    Returns
    -------
    pandas.DataFrame
        Long table with columns ``year`` (``YEAR_MIN``-``YEAR_MAX``),
        ``age_band`` (Eurostat code, e.g. ``"Y15-64"``, ``"TOTAL"``) and
        ``population``.
    """
    raw = pd.read_csv(POP_NUTS3_TSV, sep="\t")
    key_col = raw.columns[0]
    year_cols = [c for c in raw.columns if c != key_col]
    key = raw[key_col].str.split(",", expand=True)
    key.columns = ["freq", "unit", "sex", "age_band", "geo"]
    wide = pd.concat([key[["sex", "age_band", "geo"]], raw[year_cols]], axis=1)
    wide = wide[(wide.sex == "T") & (wide.geo.isin(NUTS3_TS_GO))]

    long = wide.melt(id_vars=["age_band", "geo"], value_vars=year_cols,
                      var_name="year", value_name="population")
    long["year"] = long["year"].str.strip().astype(int)
    # Year columns without missing values are read as int64 by read_csv, the
    # others as strings (because of Eurostat's ':'): after the melt the column
    # is mixed, so it is forced to string before cleaning it.
    long["population"] = pd.to_numeric(
        long["population"].astype(str).str.strip(), errors="coerce")
    long = long.dropna(subset=["population"])
    long = long[long.year.between(YEAR_MIN, YEAR_MAX)]

    out = (long.groupby(["year", "age_band"], as_index=False).population.sum())
    return out


# =============================== #
# LOAD_POPULATION_REGIONAL METHOD #
# =============================== #
def load_population_regional() -> pd.DataFrame:
    """Population of the whole FVG by five-year band and year (NUTS-2).

    Used only for the age profile: at NUTS-3 Eurostat does not publish bands
    finer than Y_LT15 / Y15-64 / Y_GE65.

    Returns
    -------
    pandas.DataFrame
        Long table parsed from the JSON-stat file ``POP_JSON``, with columns
        ``year`` (``YEAR_MIN``-``YEAR_MAX``), ``age_band`` (e.g.
        ``"Y20-24"``) and ``population``.
    """
    d = json.loads(POP_JSON.read_text(encoding="utf-8"))
    ages = d["dimension"]["age"]["category"]["index"]
    times = d["dimension"]["time"]["category"]["index"]
    n_time = len(times)

    rows = []
    for age, ai in ages.items():
        for time, ti in times.items():
            # JSON-stat: flat index, the last two dimensions are age x time
            # (freq, unit, sex and geo all have size 1)
            key = str(ai * n_time + ti)
            if key in d["value"]:
                rows.append({"year": int(time), "age_band": age,
                             "population": d["value"][key]})
    df = pd.DataFrame(rows)
    return df[df.year.between(YEAR_MIN, YEAR_MAX)].reset_index(drop=True)


# =============== #
# ELIGIBLE METHOD #
# =============== #
def eligible(pop: pd.DataFrame) -> pd.DataFrame:
    """Population of Trieste and Gorizia of donating age, by year.

    ``Y15-64`` is the narrowest band available at NUTS-3: it is a proxy
    slightly wider than the 18-64 band used in the rest of the work.

    Parameters
    ----------
    pop : pandas.DataFrame
        Long population table, as returned by
        :func:`load_population_ts_go`.

    Returns
    -------
    pandas.DataFrame
        Indexed by year, with integer columns ``pop_15_64`` and
        ``pop_totale``.
    """
    wide = pop.pivot(index="year", columns="age_band", values="population")
    return pd.DataFrame({
        "pop_15_64": wide["Y15-64"],
        "pop_totale": wide["TOTAL"],
    }).round(0).astype(int)


# ============ #
# RATES METHOD #
# ============ #
def rates(don: pd.DataFrame, panel: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Donations and active donors per thousand residents of Trieste and Gorizia.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donation_year``.
    panel : pandas.DataFrame
        Donor x year panel with ``year`` and ``active``.
    elig : pandas.DataFrame
        Eligible population by year, as returned by :func:`eligible`.

    Returns
    -------
    pandas.DataFrame
        Indexed by year, with ``donazioni``, ``donatori_attivi``, the columns
        of `elig`, ``donazioni_per_1000`` and ``donatori_per_1000`` (per
        1,000 residents aged 15-64) and ``donazioni_per_donatore``.
    """
    out = pd.DataFrame({
        "donazioni": don.groupby("donation_year").size(),
        "donatori_attivi": panel[panel.active == 1].groupby("year").size(),
    }).join(elig)
    out["donazioni_per_1000"] = (1000 * out.donazioni / out.pop_15_64).round(2)
    out["donatori_per_1000"] = (1000 * out.donatori_attivi / out.pop_15_64).round(2)
    out["donazioni_per_donatore"] = (out.donazioni / out.donatori_attivi).round(3)
    return out


# ==================== #
# DECOMPOSITION METHOD #
# ==================== #
def decomposition(r: pd.DataFrame) -> pd.DataFrame:
    """How much of the decline is population and how much is propensity.

    Donations are the product of two factors: how many people could donate,
    and how often they do. In logarithms the total change is the exact sum of
    the two changes, so the decomposition has no residual and the two shares
    read directly as percentage points.

    Parameters
    ----------
    r : pandas.DataFrame
        Rates by year, as returned by :func:`rates`; the first and last year
        of its index are compared.

    Returns
    -------
    pandas.DataFrame
        Three rows (total, population, rate) with columns ``componente``,
        ``variazione_%`` (percent change) and ``quota_del_calo_%`` (share of
        the total log change).
    """
    a, b = r.index.min(), r.index.max()
    d_tot = np.log(r.donazioni[b] / r.donazioni[a])
    d_pop = np.log(r.pop_15_64[b] / r.pop_15_64[a])
    d_rate = np.log(r.donazioni_per_1000[b] / r.donazioni_per_1000[a])
    pct = lambda x: round(100 * (np.exp(x) - 1), 1)
    return pd.DataFrame([
        {"componente": f"donazioni {a} -> {b}", "variazione_%": pct(d_tot),
         "quota_del_calo_%": 100.0},
        {"componente": "di cui: popolazione 15-64 (Trieste e Gorizia)",
         "variazione_%": pct(d_pop), "quota_del_calo_%": round(100 * d_pop / d_tot, 1)},
        {"componente": "di cui: donazioni per 1.000 residenti", "variazione_%": pct(d_rate),
         "quota_del_calo_%": round(100 * d_rate / d_tot, 1)},
    ])


# =================== #
# AGE_SPECIFIC METHOD #
# =================== #
def age_specific(don: pd.DataFrame, pop_regional: pd.DataFrame) -> pd.DataFrame:
    """Distinct donors per thousand residents, by age band and year.

    Uses the *regional* population (NUTS-2) as a proxy: at NUTS-3
    Eurostat does not publish five-year bands for Trieste and Gorizia. It is
    therefore indicative of the *shape* of the age profile, not of its
    absolute level for the two provinces.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id``, ``donation_year`` and ``age``; only
        ages 20-69 are used.
    pop_regional : pandas.DataFrame
        Regional population, as returned by
        :func:`load_population_regional`.

    Returns
    -------
    pandas.DataFrame
        One row per year and five-year band with columns ``year``,
        ``age_band``, ``donatori``, ``population`` and ``per_1000``.
    """
    d = don[don.age.between(20, 69)].copy()
    d["age_band"] = "Y" + (d.age // 5 * 5).astype(str) + "-" + (d.age // 5 * 5 + 4).astype(str)
    donors = (d.groupby(["donation_year", "age_band"]).donor_id.nunique()
                 .rename("donatori").reset_index()
                 .rename(columns={"donation_year": "year"}))
    m = donors.merge(pop_regional, on=["year", "age_band"], how="left")
    m["per_1000"] = (1000 * m.donatori / m.population).round(2)
    return m


# ── figures ──────────────────────────────────────────────────────────────────

# =============== #
# FIG_RATE METHOD #
# =============== #
def fig_rate(r, path):
    """Two stacked panels: absolute donations and donations per 1,000 residents.

    Parameters
    ----------
    r : pandas.DataFrame
        Rates by year, as returned by :func:`rates`.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 5.4), sharex=True,
                                   gridspec_kw={"hspace": 0.28})
    ax1.plot(r.index, r.donazioni, color=SERIES[0], marker="o")
    ax1.set_ylim(0, r.donazioni.max() * 1.12)
    viz.finish(ax1, "Donazioni in valore assoluto", None, "donazioni")

    ax2.plot(r.index, r.donazioni_per_1000, color=SERIES[1], marker="o")
    ax2.set_ylim(0, r.donazioni_per_1000.max() * 1.12)
    viz.finish(ax2, "Donazioni ogni 1.000 residenti di 15-64 anni (Trieste e Gorizia)",
               None, "per 1.000")
    viz.save(fig, path)


# ======================= #
# FIG_AGE_SPECIFIC METHOD #
# ======================= #
def fig_age_specific(a, path):
    """Donors per 1,000 residents by age band, for 2009, 2015 and 2025.

    Parameters
    ----------
    a : pandas.DataFrame
        Age-specific rates, as returned by :func:`age_specific`.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    years = [YEAR_MIN, 2015, YEAR_MAX]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    order = ["Y20-24", "Y25-29", "Y30-34", "Y35-39", "Y40-44",
              "Y45-49", "Y50-54", "Y55-59", "Y60-64", "Y65-69"]
    x = np.arange(len(order))
    for i, y in enumerate(years):
        s = a[a.year == y].set_index("age_band").per_1000.reindex(order)
        ax.plot(x, s.values, color=SERIES[i], marker="o", label=str(y))
    ax.set_xticks(x)
    ax.set_xticklabels([o[1:] for o in order], fontsize=8.5)
    ax.legend(title=None, loc="upper right", ncols=3)
    viz.finish(ax, "Donatori ogni 1.000 residenti, per fascia d'età (proxy regionale)",
               "fascia d'età", "donatori per 1.000")
    viz.save(fig, path)


# =========== #
# MAIN METHOD #
# =========== #
def main(img_dir=None) -> None:
    """Entry point: compute rates, decomposition and age profile; save them.

    Writes ``population_rates.csv``, ``population_decomposition.csv`` and
    ``population_age_specific.csv`` to ``REPORTS`` and the figures
    ``fig_donazioni_per_1000.png`` and ``fig_tassi_per_eta.png``.

    Parameters
    ----------
    img_dir : pathlib.Path, optional
        Directory for the figures. Defaults to ``REPORTS``;
        :mod:`figures_tesi` passes the thesis image directory.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If one of the two population files is missing.
    """
    img = img_dir or REPORTS
    if not POP_NUTS3_TSV.exists():
        raise SystemExit(f"manca {POP_NUTS3_TSV}.")
    if not POP_JSON.exists():
        raise SystemExit(
            f"manca {POP_JSON}. Scaricalo con il comando nel docstring del modulo.")

    don = pd.read_parquet(DONATIONS, columns=["donor_id", "donation_year", "age"])
    panel = pd.read_parquet(DONOR_YEAR, columns=["year", "active"])

    pop_ts_go = load_population_ts_go()
    elig = eligible(pop_ts_go)

    r = rates(don, panel, elig)
    r.to_csv(REPORTS / "population_rates.csv")
    print("Donazioni, donatori e popolazione di Trieste e Gorizia in età donabile:")
    print(r[["donazioni", "donatori_attivi", "pop_15_64",
             "donazioni_per_1000", "donatori_per_1000"]].to_string())

    dec = decomposition(r)
    dec.to_csv(REPORTS / "population_decomposition.csv", index=False)
    print("\nScomposizione del calo:")
    print(dec.to_string(index=False))

    pop_regional = load_population_regional()
    a = age_specific(don, pop_regional)
    a.to_csv(REPORTS / "population_age_specific.csv", index=False)
    piv = a.pivot(index="age_band", columns="year", values="per_1000")
    cols = [c for c in (YEAR_MIN, 2013, 2017, 2021, YEAR_MAX) if c in piv.columns]
    print("\nDonatori ogni 1.000 residenti, per fascia d'età (proxy regionale):")
    print(piv[cols].to_string())
    print("\nvariazione della fascia più giovane e di quella più anziana:")
    for band in ("Y20-24", "Y55-59"):
        if band in piv.index:
            v0, v1 = piv.loc[band, YEAR_MIN], piv.loc[band, YEAR_MAX]
            print(f"  {band}: {v0:.2f} -> {v1:.2f}  ({100*(v1/v0-1):+.1f}%)")

    print("\nfigure:")
    fig_rate(r, img / "fig_donazioni_per_1000.png")
    fig_age_specific(a, img / "fig_tassi_per_eta.png")


if __name__ == "__main__":
    main()
