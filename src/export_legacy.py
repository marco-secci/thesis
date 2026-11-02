# ================== #
# EXPORT_LEGACY FILE #
# ================== #
"""Compatibility bridge for the Shiny dashboard of App. B.

**It is not a pipeline artefact.** `app.R` was written against the schema of
the old `dataframe_cleaned.csv` (20 columns, English names) and against the
forecast CSVs exported by the notebooks. This module rewrites those same files
starting from the rebuilt data, so the dashboard runs again — on the complete
data and on the corrected forecast — without touching a line of R.

If one day the dashboard is rewritten on the parquet files, this module can be
deleted and leaves no dependencies.

Functions
---------
dataset
    Donations in the old ``dataframe_cleaned.csv`` schema. Used only inside
    this module, by :func:`main`.
forecasts
    Forecast tables in the format of the dashboard's *Forecast* tab. Used
    only inside this module, by :func:`main`.
main
    Command-line entry point. Run via ``python -m export_legacy`` from the
    ``Makefile`` and ``scripts/run_pipeline.py``; its outputs in
    ``data/FINAL/`` are read by ``shiny_app/app.R``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DONATIONS, DONOR_YEAR, DATA, REPORTS

FINAL = DATA / "FINAL"
FORECASTS = FINAL / "forecasts"

#: Schema expected by `app.R`: name in the rebuilt repo -> historical name.
COLUMNS = {
    "donor_id": "donor_id",
    "donor_class": "donor_class",
    "donation_type": "donation_type",
    "sex": "gender_label",
    "birth_year": "birth_year",
    "age": "age",
    "quantity_ml": "quantity_ml",
    "quantity_total_ml": "quantity_total_ml",
    "collection_point": "collection_point",
    "association": "association",
    "association_desc": "association_desc",
    "donation_year": "donation_year",
    "donation_month": "donation_month",
    "first_activity_date": "first_donation_date",
    "last_donation_date": "last_donation_date",
    "donation_datetime": "activity_date",
}


# ============== #
# DATASET METHOD #
# ============== #
def dataset(don: pd.DataFrame) -> pd.DataFrame:
    """Rewrite the donations in the schema of the old ``dataframe_cleaned.csv``.

    Parameters
    ----------
    don : pandas.DataFrame
        Cleaned donations (``DONATIONS``), with every key of ``COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        Rows sorted by ``donor_id`` and ``activity_date``, with the columns
        renamed through ``COLUMNS`` plus the derived ones expected by
        ``app.R``: ``gender`` (1 = M, 2 = otherwise), ``data_year``,
        ``num_donations`` (running count per donor), ``birth_cohort``
        (decade, e.g. ``"1970s"``), ``first_donation_year`` and
        ``first_donation_cohort`` (5-year bucket).
    """
    df = don.copy()
    out = df[list(COLUMNS)].rename(columns=COLUMNS)
    out["gender"] = np.where(out["gender_label"] == "M", 1, 2)
    out["data_year"] = out["donation_year"]
    # Running counter per donor: the `num_donations` column of the old schema.
    # Recomputed by counting rows, never read from `num_don`.
    out = out.sort_values(["donor_id", "activity_date"])
    out["num_donations"] = out.groupby("donor_id", observed=True).cumcount() + 1
    out["birth_cohort"] = (out["birth_year"] // 10 * 10).astype("Int64").astype(str) + "s"
    fy = out["first_donation_date"].dt.year
    out["first_donation_year"] = fy
    out["first_donation_cohort"] = (fy // 5 * 5).astype("Int64").astype(str)
    return out


# ================ #
# FORECASTS METHOD #
# ================ #
def forecasts() -> dict[str, pd.DataFrame]:
    """The CSVs the *Forecast* tab expects, from the selected model.

    Starts from ``reports/forecast_2026_2031.csv`` (written by
    :mod:`forecast`) and disaggregates the annual totals with empirical
    profiles: monthly volume shares, day-of-week weights and the shares of
    the four most common donation types. Donor counts are flat projections
    (mean of the last three observed years) with the mean retention rate of
    the last five years.

    Returns
    -------
    dict of str to pandas.DataFrame
        File stem -> table: ``monthly_forecast_2026_2031``,
        ``daily_forecast_2026_2031``, ``donation_type_forecast_2026_2031``,
        ``new_donors_forecast_2026_2031``, ``retention_forecast_2026_2031``,
        ``monthly_donors_forecast_2026_2031``, ``yearly_summary_2026_2031``
        and ``monthly_history_corrected``.
    """
    prev = pd.read_csv(REPORTS / "forecast_2026_2031.csv")
    don = pd.read_parquet(DONATIONS, columns=["donation_year", "donation_month",
                                              "donation_type", "donor_id",
                                              "donation_date"])
    panel = pd.read_parquet(DONOR_YEAR,
                            columns=["year", "active", "prev_active", "is_entry_year"])

    # Empirical monthly profile, to disaggregate the forecast annual total.
    monthly = don.groupby(["donation_year", "donation_month"]).size()
    share = (monthly.groupby("donation_month").sum()
             / monthly.groupby("donation_month").sum().sum())

    rows = []
    for _, r in prev.iterrows():
        for m in range(1, 13):
            rows.append({"month_start": f"{int(r.anno)}-{m:02d}-01",
                         "forecast": r.donazioni_previste * share[m],
                         "lower_90": r.lo * share[m],
                         "upper_90": r.hi * share[m]})
    fc_monthly = pd.DataFrame(rows).round(1)

    history = (don.groupby(["donation_year", "donation_month"]).size()
                  .rename("donations").reset_index())
    history["month_start"] = pd.to_datetime(
        dict(year=history.donation_year, month=history.donation_month, day=1))
    history = history[["month_start", "donations"]]

    days = pd.date_range(f"{int(prev.anno.min())}-01-01",
                         f"{int(prev.anno.max())}-12-31", freq="D")
    dd = pd.DataFrame({"date": days})
    dd["year"] = dd.date.dt.year
    per_year = prev.set_index("anno").donazioni_previste
    # Empirical day-of-week profile: Sundays are almost empty, Fridays the
    # busiest. Without this weight the dashboard calendar would show a uniform
    # month, which resembles no real month.
    weight = don.groupby(don.donation_date.dt.dayofweek).size()
    weight = weight / weight.mean()
    dd["weight"] = dd.date.dt.dayofweek.map(weight).fillna(0.0)
    dd["donations"] = (dd.groupby("year").weight.transform(lambda s: s / s.sum())
                       * dd.year.map(per_year)).round(1)
    fc_daily = dd[["date", "donations"]]

    type_shares = (don.groupby("donation_type").size()
                   / len(don)).sort_values(ascending=False).head(4)
    type_shares = type_shares / type_shares.sum()
    fc_type = pd.DataFrame([
        {"year": int(a), "donation_type": t, "donations": round(v * q)}
        for a, v in per_year.items() for t, q in type_shares.items()])

    active = panel[panel.active == 1].groupby("year").size()
    new = panel[panel.is_entry_year == 1].groupby("year").size()
    # Share of donors active the previous year who are active again: NOT the
    # ratio between the aggregate totals of two years (that measures pool
    # growth, not the individual probability of returning, and stays close to
    # 1 by construction).
    rate = float(panel.loc[panel.prev_active == 1]
                 .groupby("year").active.mean().tail(5).mean())
    base_active, base_new = int(active.tail(3).mean()), int(new.tail(3).mean())
    fc_donors = pd.DataFrame([
        {"year": int(a),
         "expected_new_donors": base_new,
         "expected_returning_donors": base_active - base_new,
         "retention_rate": round(rate, 4)}
        for a in per_year.index])

    summary = pd.DataFrame({
        "year": per_year.index.astype(int),
        "forecast_donations": per_year.values.round().astype(int),
        "expected_total_donors": base_active,
        "expected_new_donors": base_new,
        "retention_rate_pct": round(100 * rate, 1),
    })

    # Monthly profile of expected donors: one value per calendar month
    # (January..December), not one per (year, month). The annual total of
    # `fc_donors` is flat by construction (mean of the last observed years)
    # and identical across all six forecast years, so repeating it for each
    # of the six years would add no information: it would just give the same
    # 12-month profile repeated six times. The empirical seasonal profile
    # (`share`, the same historical monthly volume share used above for
    # `fc_monthly`) is therefore applied to the annual total only once.
    fc_donors_monthly = pd.DataFrame([
        {"month": m,
         "expected_new_donors": round(base_new * share[m]),
         "expected_returning_donors": round((base_active - base_new) * share[m])}
        for m in range(1, 13)
    ])

    return {
        "monthly_forecast_2026_2031": fc_monthly,
        "daily_forecast_2026_2031": fc_daily,
        "donation_type_forecast_2026_2031": fc_type,
        "new_donors_forecast_2026_2031": fc_donors[["year", "expected_new_donors"]],
        "retention_forecast_2026_2031": fc_donors,
        "monthly_donors_forecast_2026_2031": fc_donors_monthly,
        "yearly_summary_2026_2031": summary,
        "monthly_history_corrected": history,
    }


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: write every file read by the dashboard.

    Writes ``data/FINAL/dataframe_cleaned.csv`` from :func:`dataset` and
    one CSV per table of :func:`forecasts` into ``data/FINAL/forecasts/``.

    Returns
    -------
    None
    """
    FORECASTS.mkdir(parents=True, exist_ok=True)
    don = pd.read_parquet(DONATIONS)
    out = dataset(don)
    path = FINAL / "dataframe_cleaned.csv"
    out.to_csv(path, index=False)
    print(f"scritto {path.relative_to(DATA.parent)}  "
          f"({len(out):,} righe, {out.shape[1]} colonne)")

    for name, df in forecasts().items():
        p = FORECASTS / f"{name}.csv"
        df.to_csv(p, index=False)
        print(f"  scritto forecasts/{name}.csv  ({len(df):,} righe)")


if __name__ == "__main__":
    main()
