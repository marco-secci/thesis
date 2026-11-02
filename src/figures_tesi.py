# ================= #
# FIGURES_TESI FILE #
# ================= #
"""All the thesis figures, written directly into `thesis/img/`.

A single copy of each figure, as in the project's original convention: there
is no second image folder to keep in sync. The module recomputes nothing that
is already computed elsewhere — it calls `eda`, `retention`, `glm_inference`,
`forecast` and `population` — and adds here only the descriptive figures of
Ch. 2 that no other module produces.

Functions
---------
All functions below are used only inside this module, by :func:`main`.

fig_gender
    Distinct donors by sex.
fig_age_hist
    Histogram of age at donation.
fig_association
    Share of donors who are members of an association.
fig_types_over_time
    Yearly composition of donations by type.
fig_new_donors
    New donors per year, observed versus genuine entries.
fig_collection_points
    Busiest collection points.
fig_num_donations
    Total donations per donor.
fig_pipeline
    Diagram of the four pipeline stages (Ch. 1).
fig_null_heatmap
    Missing values by column and year in the raw data (Ch. 1).
fig_forecast_donors
    New and active donors per year (App. A).
thesis_numbers
    Every number quoted in bold in the thesis, with its source.
main
    Command-line entry point. Run via ``python -m figures_tesi`` from the
    ``Makefile`` and ``scripts/run_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import eda, forecast, glm_inference, ingest, population, retention, viz
from config import DONATIONS, DONOR_YEAR, EVENTS_RAW, ROOT, YEAR_MAX, YEAR_MIN

IMG = ROOT / "thesis" / "img"

#: Collection points to show in the geographic figure.
TOP_POINTS = 10

#: Donation types above this threshold get their own bar; the rest is "Altre".
TYPE_THRESHOLD = 0.004


# ── Descriptive figures for Ch. 2 ────────────────────────────────────────────

# ================= #
# FIG_GENDER METHOD #
# ================= #
def fig_gender(don, path):
    """Horizontal bars of distinct donors by sex, with counts and shares.

    Each donor is counted once, with the sex of their first row.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id`` and ``sex``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    per_don = don.groupby("donor_id", observed=True)["sex"].first()
    n = per_don.value_counts()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = {"M": "Uomini", "F": "Donne"}
    order = [s for s in ("M", "F") if s in n.index]
    ax.barh([labels[s] for s in order], [n[s] for s in order],
            color=[viz.SERIES[0], viz.SERIES[1]], height=0.55)
    for i, s in enumerate(order):
        ax.text(n[s] * 0.98, i, f"{n[s]:,}".replace(",", ".")
                + f"  ({100 * n[s] / n.sum():.1f}%)",
                va="center", ha="right", color="white", fontweight="bold")
    ax.invert_yaxis()
    viz.finish(ax, "Donatori distinti per sesso", "donatori", None)
    viz.save(fig, path)


# =================== #
# FIG_AGE_HIST METHOD #
# =================== #
def fig_age_hist(don, path):
    """Histogram of age at the time of donation.

    Not to be confused with `eda.fig_age`, which lives on the donor-year panel
    and shows the expected frequency by band: that one answers "how much does
    someone of this age donate", this one "how old are the donations". Ch. 2
    quotes mean, median and standard deviation, and they can only be read
    from here.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``age``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    a = don["age"].dropna().astype(int)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist(a, bins=range(int(a.min()), int(a.max()) + 2), color=viz.SERIES[0],
            edgecolor="white", linewidth=0.3)
    mean, median = a.mean(), a.median()
    ax.axvline(mean, linestyle="--", linewidth=1.1, color=viz.INK_SOFT)
    ax.annotate(f"media {mean:.1f}".replace(".", ","),
                xy=(mean, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=7.5, color=viz.INK_SOFT)
    n = f"{len(a):,}".replace(",", ".")
    ds = f"{a.std():.1f}".replace(".", ",")
    viz.finish(ax, "Età al momento della donazione", "età (anni)", "donazioni",
               note=f"n = {n} donazioni; mediana {median:.0f}, "
                    f"deviazione standard {ds}.")
    viz.save(fig, path)


# ====================== #
# FIG_ASSOCIATION METHOD #
# ====================== #
def fig_association(don, path):
    """Share of donors who are members of an association.

    A donor is a member if at least one of their donations reports the
    association, by code or by full description: the two fields are filled in
    independently and neither alone is reliable.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id``, ``association`` and
        ``association_desc``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    d = don.assign(ha=don["association"].notna() | don["association_desc"].notna())
    per_don = d.groupby("donor_id", observed=True)["ha"].any()
    n = per_don.map({True: "Associato", False: "Non associato"}).value_counts()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.barh(n.index.tolist(), n.values,
            color=[viz.SERIES[0], viz.SERIES[1]][:len(n)], height=0.55)
    for i, v in enumerate(n.values):
        ax.text(v * 0.98, i, f"{100 * v / n.sum():.1f}%", va="center",
                ha="right", color="white", fontweight="bold")
    ax.invert_yaxis()
    viz.finish(ax, "Adesione associativa dei donatori", "donatori", None)
    viz.save(fig, path)


# ========================== #
# FIG_TYPES_OVER_TIME METHOD #
# ========================== #
def fig_types_over_time(don, path):
    """Yearly share of each donation type, for types above ``TYPE_THRESHOLD``.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donation_year`` and ``donation_type``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    shares = (don.groupby(["donation_year", "donation_type"], observed=True)
                 .size().unstack(fill_value=0))
    shares = 100 * shares.div(shares.sum(axis=1), axis=0)
    kept = shares.mean().sort_values(ascending=False)
    kept = kept[kept > 100 * TYPE_THRESHOLD].index.tolist()
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, t in enumerate(kept):
        ax.plot(shares.index, shares[t], color=viz.SERIES[i], label=t)
        ax.annotate(t, xy=(shares.index[-1], shares[t].iloc[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=viz.SERIES[i], fontsize=8, va="center")
    ax.set_xlim(shares.index.min(), shares.index.max() + 2.4)
    viz.finish(ax, "Composizione delle donazioni per tipologia",
               "anno", "% delle donazioni dell'anno")
    viz.save(fig, path)


# ===================== #
# FIG_NEW_DONORS METHOD #
# ===================== #
def fig_new_donors(don, path):
    """New donors per year, with the share of true entries highlighted.

    2009 is not a year of entries but the edge of the window: two thirds of
    its "new" donors had already started donating before. The figure shows it
    instead of leaving it to a footnote.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations, as needed by :func:`retention.donor_table`.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    t = retention.donor_table(don, through=YEAR_MAX)
    n = t.groupby("anno_osservato").size()
    genuine = t[t["troncato_a_sinistra"] == 0].groupby("anno_osservato").size()
    genuine = genuine.reindex(n.index, fill_value=0)

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(n.index, n.values, color=viz.SERIES[0], alpha=0.30,
           label="primo evento osservato nell'anno")
    ax.bar(genuine.index, genuine.values, color=viz.SERIES[0],
           label="carriera iniziata davvero nell'anno")

    # The light 2009 bar is 12,957 and would squash everything else: the axis
    # is cut and this is stated, because the comparison that matters is
    # between the following years.
    ceiling = n.drop(index=n.index.min()).max() * 1.35
    ax.set_ylim(0, ceiling)
    a0 = n.index.min()
    ax.annotate(f"{n.iloc[0]:,}".replace(",", "."),
                xy=(a0, ceiling), xytext=(0, -12), textcoords="offset points",
                ha="center", fontsize=8, color=viz.SERIES[0], fontweight="bold")
    ax.annotate("fuori scala", xy=(a0, ceiling), xytext=(0, -24),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=viz.INK_SOFT)
    ax.set_xticks(range(int(n.index.min()), int(n.index.max()) + 1, 2))
    ax.legend(fontsize=8, loc="upper right")
    viz.finish(ax, "Nuovi donatori per anno", "anno", "donatori")
    viz.save(fig, path)


# ============================ #
# FIG_COLLECTION_POINTS METHOD #
# ============================ #
def fig_collection_points(don, path):
    """The ``TOP_POINTS`` busiest collection points, with their share.

    Shares are computed on the donations with a known collection point.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``collection_point``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    known = don[don["collection_point"].notna()]
    n = known["collection_point"].value_counts().head(TOP_POINTS)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.barh(n.index.tolist()[::-1], n.values[::-1], color=viz.SERIES[0], height=0.62)
    tot = len(known)
    for i, v in enumerate(n.values[::-1]):
        ax.text(v + tot * 0.004, i, f"{100 * v / tot:.1f}%", va="center",
                fontsize=8, color=viz.INK_SOFT)
    viz.finish(ax, f"Punti di raccolta più attivi", "donazioni", None,
               note=f"Quote calcolate sulle {tot:,} donazioni con punto di "
                    f"raccolta noto.".replace(",", "."))
    viz.save(fig, path)


# ======================== #
# FIG_NUM_DONATIONS METHOD #
# ======================== #
def fig_num_donations(don, path):
    """Histogram of total donations per donor, capped at 30.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations with ``donor_id``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    n = don.groupby("donor_id", observed=True).size().clip(upper=30)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.hist(n, bins=range(1, 32), color=viz.SERIES[0], rwidth=0.85)
    viz.finish(ax, "Donazioni totali per donatore", "donazioni (cappate a 30)",
               "donatori")
    viz.save(fig, path)


# ── Ch. 1: the pipeline and the nulls in the raw data ────────────────────────

# =================== #
# FIG_PIPELINE METHOD #
# =================== #
def fig_pipeline(path):
    """Pipeline diagram: four stages, each one writes a new file.

    The row counts shown in the boxes are written in the code, not computed.

    Parameters
    ----------
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stages = [
        ("27 file XLSX", "370.552 righe\nlette", "data/raw/XLSX/"),
        ("Unione", "353.707 eventi\n16.845 fusi", "events_raw.parquet"),
        ("Pulizia", "353.671 donazioni\n36 scartate", "donations.parquet"),
        ("Panel", "514.098 righe\ndonatore-anno", "donor_year.parquet"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    ax.set_xlim(0, len(stages)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (title, body, file) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.24), 0.88, 0.52,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    facecolor="#eef4fd" if i else "#f3f2ee",
                                    edgecolor=viz.SERIES[0] if i else viz.GRID,
                                    linewidth=1.1))
        ax.text(i + 0.5, 0.66, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=viz.INK)
        ax.text(i + 0.5, 0.44, body, ha="center", va="center",
                fontsize=8.5, color=viz.INK_SOFT, linespacing=1.45)
        ax.text(i + 0.5, 0.15, file, ha="center", va="center",
                fontsize=7.5, family="monospace", color=viz.SERIES[0])
        if i:
            ax.add_patch(FancyArrowPatch((i - 0.04, 0.5), (i + 0.04, 0.5),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color=viz.INK_SOFT, linewidth=1.1))
    ax.text(0.06, 0.02, "Ogni stadio scrive un file nuovo: nessuno riscrive il "
            "proprio input, quindi ogni comando è rieseguibile.",
            fontsize=7.5, color=viz.INK_SOFT)
    viz.save(fig, path)


# ======================= #
# FIG_NULL_HEATMAP METHOD #
# ======================= #
def fig_null_heatmap(path):
    """Missing values by column and by year, on the raw data.

    Two different pieces of information live in the same grid and must be kept
    apart: a column that *is not there* in that year's file and a column that
    is there but half filled in. The first is grey and has no number, because
    100% is a property of the file layout and not of the data; the second
    carries its percentage, and the blue scale is calibrated on the observed
    values only — otherwise the only real gradient (`first_activity_date`,
    from 20% to 11%) would be indistinguishable from white.

    Reads ``EVENTS_RAW`` and saves at 300 dpi.

    Parameters
    ----------
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    raw = pd.read_parquet(EVENTS_RAW)
    raw["_year"] = raw["donation_datetime"].dt.year
    years = list(range(YEAR_MIN, YEAR_MAX + 1))
    cols = [c for c in raw.columns if c not in ("_year", "source_file")]
    m = np.array([[100 * raw.loc[raw["_year"] == a, c].isna().mean()
                   for a in years] for c in cols])

    # A column entirely empty in a year is a column that was not exported that
    # year: these are removed from the colour scale, so the blue measures only
    # partial filling.
    absent = m > 99.999
    present = (~absent).sum(axis=1)
    order = np.lexsort((m.mean(axis=1), -present))
    cols = [cols[i] for i in order]
    m, absent, present = m[order], absent[order], present[order]

    vmax = max(5.0, np.ceil(m[~absent].max() / 5) * 5)
    fig, ax = plt.subplots(figsize=(7.0, 0.235 * len(cols) + 1.55))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#b8b5ad")
    ax.imshow(np.ma.masked_where(absent, m), aspect="auto", cmap=cmap,
              vmin=0, vmax=vmax)

    for i in range(len(cols)):
        for j in range(len(years)):
            if absent[i, j]:
                continue
            v = m[i, j]
            text = "0" if v == 0 else ("<1" if v < 1 else f"{v:.0f}")
            ax.text(j, i, text, ha="center", va="center", fontsize=6.2,
                    color="white" if v > 0.62 * vmax else
                          (viz.INK if v else "#9a978f"))
        ax.text(len(years) - 0.32, i, f"{present[i]}/{len(years)}", ha="left",
                va="center", fontsize=6.2, color=viz.INK_SOFT)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=6.4)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7, family="monospace")
    ax.set_xticks(np.arange(len(years) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=viz.SURFACE, linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(len(years) - 0.32, -0.85, "anni con\nla colonna", ha="left",
            va="center", fontsize=6, color=viz.INK_SOFT, linespacing=1.3)
    ax.set_title("Valori mancanti per colonna e per anno, nel dato grezzo",
                 loc="left", pad=16)
    ax.legend(handles=[Patch(facecolor="#b8b5ad",
                             label="colonna assente dal file di quell'anno"),
                       Patch(facecolor=plt.get_cmap("Blues")(0.62),
                             label=f"colonna presente: % di valori mancanti "
                                   f"(0–{vmax:.0f}%)")],
              loc="upper left", bbox_to_anchor=(0, -0.055), ncol=2,
              frameon=False, fontsize=6.6, handlelength=1.1, handleheight=0.9,
              columnspacing=1.4, borderpad=0)
    viz.save(fig, path, dpi=300)


# ========================== #
# FIG_FORECAST_DONORS METHOD #
# ========================== #
def fig_forecast_donors(don, panel, path):
    """New donors and active donors per year, side by side.

    Parameters
    ----------
    don : pandas.DataFrame
        Donations, as needed by :func:`retention.donor_table`; only genuine
        entries (not left-truncated) are counted as new donors.
    panel : pandas.DataFrame
        Donor x year panel with ``year`` and ``active``.
    path : pathlib.Path
        Output image file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    t = retention.donor_table(don, through=YEAR_MAX)
    new = t[t["troncato_a_sinistra"] == 0].groupby("anno_osservato").size()
    active = panel[panel["active"] == 1].groupby("year").size()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    a1.plot(new.index, new.values, color=viz.SERIES[0], marker="o")
    a1.set_ylim(0, new.max() * 1.12)
    viz.finish(a1, "Nuovi donatori (ingressi genuini)", "anno", "donatori")
    a2.plot(active.index, active.values, color=viz.SERIES[1], marker="o")
    a2.set_ylim(0, active.max() * 1.12)
    viz.finish(a2, "Donatori attivi nell'anno", "anno", "donatori")
    fig.tight_layout()
    viz.save(fig, path)


# ── The numbers quoted in the thesis ─────────────────────────────────────────

# ===================== #
# THESIS_NUMBERS METHOD #
# ===================== #
def thesis_numbers(don, panel) -> pd.DataFrame:
    """Every figure that appears in bold in the text, with its source.

    It serves one purpose only: being able to check by comparison that no
    number in the thesis has been orphaned from the pipeline that should
    produce it. It recomputes nothing new — it re-reads what the modules have
    already written to `reports/` and adds the direct counts.

    Numbers whose source CSV in ``reports/`` is missing are skipped.

    Parameters
    ----------
    don : pandas.DataFrame
        Cleaned donations.
    panel : pandas.DataFrame
        Donor x year panel.

    Returns
    -------
    pandas.DataFrame
        One row per number, with columns ``numero`` (description),
        ``valore`` and ``fonte`` (file it comes from).
    """
    from config import REPORTS

    def csv(name):
        p = REPORTS / name
        return pd.read_csv(p) if p.exists() else None

    rows = []
    add = lambda k, v, f: rows.append({"numero": k, "valore": v, "fonte": f})

    y = don.groupby("donation_year").size()
    add("eventi di donazione", len(don), "donations.parquet")
    add("donatori distinti", don.donor_id.nunique(), "donations.parquet")
    add("righe del panel", len(panel), "donor_year.parquet")
    add("quota di zeri nel panel (%)", round(100 * (panel.y == 0).mean(), 1),
        "donor_year.parquet")
    add("donazioni 2009", int(y.loc[YEAR_MIN]), "yearly_summary.csv")
    add("donazioni 2025", int(y.loc[YEAR_MAX]), "yearly_summary.csv")
    add("variazione 2009-2025 (%)", round(100 * (y.loc[YEAR_MAX] / y.loc[YEAR_MIN] - 1), 1),
        "yearly_summary.csv")

    per_don = don.groupby("donor_id", observed=True).sex.first()
    add("uomini (%)", round(100 * (per_don == "M").mean(), 1), "donations.parquet")
    add("eta media alla donazione", round(don.age.mean(), 2), "donations.parquet")

    known = don[don.collection_point.notna()]
    for site in ("Maggiore", "Sede Gorizia", "Monfalcone S. Polo"):
        add(f"quota {site} (%)",
            round(100 * (known.collection_point == site).mean(), 1),
            "donations.parquet")

    dec = csv("population_decomposition.csv")
    if dec is not None:
        for _, r in dec.iterrows():
            add(f"scomposizione: {r.componente}", r["quota_del_calo_%"],
                "population_decomposition.csv")

    dist = csv("retention_distribuzione.csv")
    if dist is not None:
        d = dist[(dist.definizione == "ingresso") & (dist.anni_retention == "0")]
        add("giovani mordi-e-fuggi (%)", round(float(d.quota_pct.iloc[0]), 1),
            "retention_distribuzione.csv")

    comp = csv("retention_composizione_coorti.csv")
    if comp is not None:
        add("coorte 2009 gia attiva prima (%)",
            round(float(comp.loc[comp.coorte == 2009, "quota_pct"].iloc[0]), 1),
            "retention_composizione_coorti.csv")

    mod = csv("backtest_modelli.csv")
    if mod is not None:
        for _, r in mod.head(3).iterrows():
            add(f"MAPE {r.modello} (%)", round(r.mape, 2), "backtest_modelli.csv")

    fc = csv("forecast_2026_2031.csv")
    if fc is not None:
        add("totale previsto 2026-2031", int(fc.donazioni_previste.sum()),
            "forecast_2026_2031.csv")

    irr = csv("glm_irr.csv")
    if irr is not None:
        pr = irr[irr.modello == "principale"]
        for key, label in (("sex", "IRR genere F"), ("age", "IRR eta"),
                           ("career_years", "IRR anni di carriera")):
            m = pr[pr.termine.str.contains(key, regex=False)]
            if len(m):
                add(label, round(float(m.irr.iloc[0]), 4), "glm_irr.csv")

    disp = csv("glm_dispersione.csv")
    if disp is not None:
        for _, r in disp.iterrows():
            add(f"var/media, {r.grano}", round(r.var_su_media, 2),
                "glm_dispersione.csv")

    return pd.DataFrame(rows)


# ── Entry point ──────────────────────────────────────────────────────────────

# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: regenerate every thesis figure.

    Writes the figures into ``thesis/img/`` chapter by chapter, calling the
    figure functions of this module and of :mod:`eda`, and the ``main`` of
    :mod:`population`, :mod:`retention`, :mod:`forecast` and
    :mod:`glm_inference` with ``img_dir=IMG``. The null heatmap is skipped
    if ``EVENTS_RAW`` is missing. Finally writes
    ``reports/numeri_tesi.csv`` from :func:`thesis_numbers`.

    Returns
    -------
    None
    """
    IMG.mkdir(parents=True, exist_ok=True)
    don = pd.read_parquet(DONATIONS)
    panel = pd.read_parquet(DONOR_YEAR)

    print("Cap. 1 — pipeline e qualità del dato")
    fig_pipeline(IMG / "fig_pipeline_diagram.png")
    if EVENTS_RAW.exists():
        fig_null_heatmap(IMG / "fig_null_heatmap.png")
    else:
        # Only donations.parquet available: the null heatmap needs the raw
        # events, so the one already versioned in thesis/img/ is kept.
        print(f"  saltata fig_null_heatmap: {EVENTS_RAW.name} assente")

    print("Cap. 2 — descrittive")
    fig_gender(don, IMG / "fig_gender.png")
    fig_age_hist(don, IMG / "fig_age_hist.png")
    fig_association(don, IMG / "fig_association.png")
    eda.fig_types(don, IMG / "fig_donation_types.png")
    fig_types_over_time(don, IMG / "fig_types_over_time.png")
    eda.fig_yearly(don, IMG / "fig_donations_per_year.png")
    fig_new_donors(don, IMG / "fig_new_donors.png")
    eda.fig_seasonality(don, IMG / "fig_monthly_seasonality.png")
    fig_collection_points(don, IMG / "fig_collection_points.png")
    fig_num_donations(don, IMG / "fig_num_donations.png")

    print("Cap. 2 — denominatore demografico")
    population.main(img_dir=IMG)

    print("Cap. 3 — fidelizzazione")
    retention.main(img_dir=IMG)

    print("App. A — proiezioni")
    eda.fig_monthly(don, IMG / "fig_forecast_monthly_history.png")
    forecast.main(img_dir=IMG)
    fig_forecast_donors(don, panel, IMG / "fig_forecast_donors.png")

    print("Cap. 4 — regressione")
    glm_inference.main(img_dir=IMG)

    from config import REPORTS
    n = thesis_numbers(don, panel)
    n.to_csv(REPORTS / "numeri_tesi.csv", index=False)
    print(f"\nfigure scritte in {IMG}")
    print(f"{len(n)} numeri citati in tesi in reports/numeri_tesi.csv")


if __name__ == "__main__":
    main()
