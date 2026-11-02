# ========== #
# PANEL FILE #
# ========== #
"""Stage 3 — from the donations dataset to the donor x year panel.

**Why this is the right grain.** The raw data is a list of events: a row exists
only if that donor donated. Modelling "how many donations a donor makes" on
those rows means conditioning on Y>=1, i.e. on a truncated response, and — if
the `num_don` column is used, which is a count per (donor, year, type)
replicated on every row of the group — it means letting each group in as many
times as its own value: the response becomes its own weight. Here the count is
obtained **by counting rows**, and years without donations become rows with y=0.

Materialising the zeros solves three things at once:

* truncation disappears, so no truncated likelihood is needed;
* dropout becomes estimable, which is the event we want to predict and which
  simply does not exist in the event format;
* the sum of y over all donors in a year **is** the annual total, so individual
  and aggregate forecasts cannot contradict each other.

**Once in, a donor stays in the panel.** The alternative — removing them after
N years of inactivity — would introduce an arbitrary parameter and lose the
returns, which are not rare. `recency_class` carries the information, with a
"6+" class acting as a reservoir.

**Every covariate looks only backwards.** Everything is computed from the
donor's history *strictly before* the row's year, so
`build_panel(df, through_year=O)` contains exactly what someone at the end of O
knew. This is what makes the panel usable inside a rolling-origin backtest
without leakage, and there is a test that checks it
(`tests/test_panel.py::test_covariates_are_causal`).

Functions
---------
load_donations
    Reads the donation columns needed by the panel, up to a given year.
    Used by :func:`main` and ``tests/test_panel.py``.
build_panel
    Builds the dense donor x year panel with backward-looking covariates.
    Used by :func:`main` and ``tests/test_panel.py``; :mod:`simulate`
    reproduces its recency coding and imports ``RECENCY_CAP`` from here.
summarise
    Size, share of zeros and dispersion of the panel. Used only inside this
    module, by :func:`main`.
main
    Command-line entry point (``--through YEAR``). Run via
    ``python -m panel`` from the ``Makefile`` and
    ``scripts/run_pipeline.py``; its output ``DONOR_YEAR`` is read by most
    later stages.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import DONATIONS, DONOR_YEAR, PROCESSED, REPORTS, YEAR_MAX

#: Recency classes 1..RECENCY_CAP, then a single "6+" class.
RECENCY_CAP = 5


# ===================== #
# LOAD_DONATIONS METHOD #
# ===================== #
def load_donations(through_year: int) -> pd.DataFrame:
    """Read the donations needed to build the panel, up to a given year.

    Parameters
    ----------
    through_year : int
        Last donation year to keep (inclusive).

    Returns
    -------
    pandas.DataFrame
        One row per donation with columns ``donor_id``, ``donation_year``,
        ``donation_type``, ``birth_year``, ``sex``, ``donor_class``,
        ``association``, ``collection_point`` and ``first_activity_date``.
    """
    df = pd.read_parquet(DONATIONS, columns=[
        "donor_id", "donation_year", "donation_type", "birth_year", "sex",
        "donor_class", "association", "collection_point", "first_activity_date",
    ])
    return df[df.donation_year <= through_year]


# ================== #
# BUILD_PANEL METHOD #
# ================== #
def build_panel(df: pd.DataFrame, through_year: int) -> pd.DataFrame:
    """One row per (donor, year), from the donor's entry to `through_year`.

    Years without donations are materialised with ``y = 0``. Donor
    attributes are taken from the first observed row, and every history
    covariate uses only years strictly before the row's year.

    Parameters
    ----------
    df : pandas.DataFrame
        Donations, one row per event, as returned by :func:`load_donations`.
        Should not contain years after `through_year`.
    through_year : int
        Last year of the grid for every donor.

    Returns
    -------
    pandas.DataFrame
        Panel sorted by ``donor_id`` and ``year``, with columns:

        - identifiers: ``donor_id``, ``year``;
        - response: ``y`` (donations in the year), ``active`` (``y > 0``),
          ``n_types`` (distinct donation types);
        - history: ``recency`` (years since last active year, NaN on the
          entry row), ``recency_class`` (``"entry"``, ``"1"``..``"5"``,
          ``"6+"``), ``is_entry_year``, ``prev_count``, ``prev_active``,
          ``prev_n_types``, ``cum_count``, ``log1p_cum``,
          ``cum_active_years``, ``tenure``, ``career_years``,
          ``left_censored``;
        - demographics and entry attributes: ``age``, ``birth_year``,
          ``sex``, ``donor_class_at_entry``, ``association_at_entry``,
          ``collection_point_at_entry``, ``first_activity_year``,
          ``entry_year``.
    """
    # ── observed activity, per donor and year ────────────────────────────────
    per = (df.groupby(["donor_id", "donation_year"], observed=True)
             .agg(y=("donation_type", "size"),
                  n_types=("donation_type", "nunique"))
             .reset_index())

    # ── donor attributes, taken at the FIRST observed row ────────────────────
    # Taking the last one would be future information; the first is not.
    first = (df.sort_values("donation_year")
               .drop_duplicates("donor_id", keep="first")
               .set_index("donor_id"))
    attrs = pd.DataFrame({
        "birth_year": first["birth_year"],
        "sex": first["sex"],
        "donor_class_at_entry": first["donor_class"],
        "association_at_entry": first["association"],
        "collection_point_at_entry": first["collection_point"],
        "first_activity_year": first["first_activity_date"].dt.year,
    })
    attrs["entry_year"] = per.groupby("donor_id").donation_year.min()

    # ── dense grid: for each donor, entry_year .. through_year ───────────────
    n_years = (through_year - attrs["entry_year"] + 1).astype(int)
    grid = pd.DataFrame({
        "donor_id": np.repeat(attrs.index.to_numpy(), n_years.to_numpy()),
        "year": (np.repeat(attrs["entry_year"].to_numpy(), n_years.to_numpy())
                 + np.concatenate([np.arange(k) for k in n_years.to_numpy()])),
    })

    grid = grid.merge(per, how="left",
                      left_on=["donor_id", "year"],
                      right_on=["donor_id", "donation_year"])
    grid["y"] = grid["y"].fillna(0).astype("int16")
    grid["n_types"] = grid["n_types"].fillna(0).astype("int8")
    grid = grid.drop(columns="donation_year")
    grid["active"] = (grid.y > 0).astype("int8")
    grid = grid.sort_values(["donor_id", "year"]).reset_index(drop=True)

    # ── history covariates, all backward-looking ─────────────────────────────
    # The grid is contiguous by construction, so shift(1) within a donor is
    # always the previous calendar year: no gaps to handle.
    g = grid.groupby("donor_id", sort=False)
    grid["prev_count"] = g["y"].shift(1).fillna(0).astype("int16")
    grid["prev_active"] = g["active"].shift(1).fillna(0).astype("int8")
    grid["prev_n_types"] = g["n_types"].shift(1).fillna(0).astype("int8")
    grid["cum_count"] = (g["y"].cumsum() - grid["y"]).astype("int32")
    grid["cum_active_years"] = (g["active"].cumsum() - grid["active"]).astype("int16")

    # recency = years elapsed since the last ACTIVE year strictly before
    act_year = grid["year"].where(grid.active == 1)
    last_act = (act_year.groupby(grid.donor_id, sort=False).shift(1)
                        .groupby(grid.donor_id, sort=False).ffill())
    grid["recency"] = grid["year"] - last_act          # NaN on the entry row
    grid["is_entry_year"] = grid["recency"].isna().astype("int8")
    grid["recency_class"] = np.where(
        grid.is_entry_year == 1, "entry",
        np.where(grid.recency <= RECENCY_CAP,
                 grid.recency.fillna(0).astype("Int64").astype(str), "6+"))

    # ── derived demographics ─────────────────────────────────────────────────
    grid = grid.merge(attrs.reset_index(), on="donor_id", how="left")
    grid["age"] = (grid["year"] - grid["birth_year"]).astype("int16")
    grid["tenure"] = (grid["year"] - grid["entry_year"]).astype("int16")
    grid["log1p_cum"] = np.log1p(grid["cum_count"])

    # True experience versus apparent experience. `tenure` counts years from
    # the first OBSERVED year, which for a third of donors is not the first
    # year of their career: 33% had already started before 2009 (median
    # 2004). Using `tenure` as a measure of experience makes veterans look like
    # newcomers, and since the share of misclassified veterans runs out as the
    # years pass, the variable drifts for a purely instrumental reason.
    # `first_activity_year` comes from the raw data and does not have this
    # problem; where it is missing (9% of donors) we fall back on entry_year.
    fa = grid["first_activity_year"].fillna(grid["entry_year"])
    grid["career_years"] = (grid["year"] - fa).clip(lower=0).astype("int16")
    grid["left_censored"] = (fa < grid["entry_year"]).astype("int8")

    cols = ["donor_id", "year", "y", "active", "n_types",
            "recency", "recency_class", "is_entry_year",
            "prev_count", "prev_active", "prev_n_types",
            "cum_count", "log1p_cum", "cum_active_years", "tenure",
            "career_years", "left_censored",
            "age", "birth_year", "sex", "donor_class_at_entry",
            "association_at_entry", "collection_point_at_entry",
            "first_activity_year", "entry_year"]
    return grid[cols]


# ================ #
# SUMMARISE METHOD #
# ================ #
def summarise(panel: pd.DataFrame) -> pd.DataFrame:
    """Descriptive summary of the panel: size, zeros and dispersion of y.

    Parameters
    ----------
    panel : pandas.DataFrame
        Panel as returned by :func:`build_panel` (needs ``donor_id``,
        ``year``, ``y`` and ``active``).

    Returns
    -------
    pandas.DataFrame
        Two columns, ``grandezza`` (quantity name, in Italian) and
        ``valore``: rows, distinct donors, years, zero rows and their share,
        total donations, and mean, variance and variance/mean of ``y`` on
        all rows and on active rows only.
    """
    y = panel.y
    rows = [
        ("righe del panel", len(panel)),
        ("donatori distinti", int(panel.donor_id.nunique())),
        ("anni coperti", int(panel.year.nunique())),
        ("righe con y=0", int((y == 0).sum())),
        ("quota di zeri (%)", round(100 * float((y == 0).mean()), 2)),
        ("donazioni totali", int(y.sum())),
        ("media y (tutte le righe)", round(float(y.mean()), 4)),
        ("varianza y (tutte le righe)", round(float(y.var()), 4)),
        ("var/media (tutte le righe)", round(float(y.var() / y.mean()), 4)),
        ("media y | attivo", round(float(y[panel.active == 1].mean()), 4)),
        ("varianza y | attivo", round(float(y[panel.active == 1].var()), 4)),
        ("var/media | attivo",
         round(float(y[panel.active == 1].var() / y[panel.active == 1].mean()), 4)),
    ]
    return pd.DataFrame(rows, columns=["grandezza", "valore"])


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: build, check and save the panel.

    Command-line option ``--through YEAR`` (default ``YEAR_MAX``) limits the
    data to that year. Checks that the panel accounts for every donation,
    writes it to ``DONOR_YEAR`` (or ``donor_year_through<YEAR>.parquet``
    when ``--through`` is earlier), writes the :func:`summarise` table to
    ``reports/`` and prints activity by recency class.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the total of ``y`` differs from the number of donations read.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--through", type=int, default=YEAR_MAX,
                    help="costruisce il panel usando solo i dati fino a quest'anno")
    args = ap.parse_args()

    df = load_donations(args.through)
    print(f"donazioni fino al {args.through}: {len(df):,} "
          f"({df.donor_id.nunique():,} donatori)")

    panel = build_panel(df, args.through)

    # Non-negotiable invariant: the panel must account for every donation.
    total = int(panel.y.sum())
    assert total == len(df), f"il panel ha {total:,} donazioni, la fonte {len(df):,}"
    print(f"panel: {len(panel):,} righe, {100 * (panel.y == 0).mean():.1f}% zeri")
    print("controllo: ogni donazione della finestra è contabilizzata  OK")

    out = (DONOR_YEAR if args.through == YEAR_MAX
           else PROCESSED / f"donor_year_through{args.through}.parquet")
    panel.to_parquet(out, index=False)

    s = summarise(panel)
    s.to_csv(REPORTS / f"{out.stem}_summary.csv", index=False)
    print("\n" + s.to_string(index=False))

    rc = (panel.groupby("recency_class")
               .agg(righe=("y", "size"), quota_attivi=("active", "mean"),
                    media_y_se_attivo=("y", lambda v: v[v > 0].mean()))
               .round(4))
    rc["quota_attivi"] = (100 * rc.quota_attivi).round(2)
    print("\nAttività per classe di recency:")
    print(rc.to_string())
    print(f"\nscritto {out}")


if __name__ == "__main__":
    main()
