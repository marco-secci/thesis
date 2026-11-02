# ========== #
# CLEAN FILE #
# ========== #
"""Stage 2 — from raw events to the donations dataset.

Principles:

* **Every rule is a function** that takes a dataframe and returns
  `(dataframe, n_rows_touched)`. No rule is hidden inside another.
* **No in-place writes.** The stage reads `events_raw.parquet` and writes
  `donations.parquet`. Re-running it always gives the same result.
* **Every row that disappears shows up in the report.** `reports/cleaning_rules.csv`
  lists, rule by rule, how many rows were removed or modified, and the total
  must balance with the difference between input and output.

Non-obvious decisions, with reasons:

`validity` — the V/N column exists in 4 files out of 27 (2023 and 2025): 752
donations are marked invalid. Dropping them only in those years would introduce
an artificial step precisely in the final years, i.e. those that weigh most in a
forecast. The default behaviour is therefore to **keep them**, preserving the
flag; `--drop-invalid` produces the alternative variant to measure how much it
changes. The number is small (~2% where observable) but it must be decided, not
just accepted.

`donor_class` — the coding changes in 2016/2017: class `O` (occasional) drops
from ~3,000 cases a year to fewer than 10, and class `D` appears. It is an
administrative reclassification, not a change in donor behaviour. The column
stays in the dataset but **must not be used as a predictor without accounting
for it**: the `donor_class_regime` flag marks before and after.

Functions
---------
All ``rule_*`` functions take a DataFrame and return
``(DataFrame, n_rows_touched)``. They are used only inside this module: all
but :func:`rule_validity` through the ``RULES`` list, which :func:`main`
applies in order.

rule_derive_time
    Derives date, year and month from the donation instant.
rule_drop_out_of_range_years
    Drops events outside ``YEAR_MIN``-``YEAR_MAX``.
rule_drop_missing_birth_date
    Drops events without a birth date.
rule_age
    Computes age at donation and drops values outside
    ``AGE_MIN``-``AGE_MAX``.
rule_sex
    Recodes sex from 1/2 to M/F.
rule_collection_point
    Unifies the spellings of each collection point.
rule_association
    Normalises association codes, rare codes become ``ALTRO``.
rule_donor_class_regime
    Flags the 2016 change of ``donor_class`` coding.
rule_validity
    Flags (or drops) the donations marked invalid; called directly by
    :func:`main`.
main
    Command-line entry point. Run via ``python -m clean`` from the
    ``Makefile`` and ``scripts/run_pipeline.py``; its output
    ``DONATIONS`` is read by almost every later stage.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import (AGE_MAX, AGE_MIN, DONATIONS, EVENTS_RAW, REPORTS,
                             YEAR_MAX, YEAR_MIN)

# ── Normalisation of categorical levels ──────────────────────────────────────

#: The collection points are ~11 sites written in 18 different ways. The
#: blood-mobiles are mobile units: which of the three in Trieste went out on a
#: given day is not information about the donor, so they are unified per site.
COLLECTION_POINT_MAP = {
    "AutoEmoteca GO":         "Autoemoteca GO",
    "Autoemoteca-GO":         "Autoemoteca GO",
    "Seconda autoemoteca GO": "Autoemoteca GO",
    "Autoemoteca TS":         "Autoemoteca TS",
    "Autoemoteca 1 TS":       "Autoemoteca TS",
    "Autoemoteca 2 TS":       "Autoemoteca TS",
    "Autoemoteca 3 TS":       "Autoemoteca TS",
    "Autoemoteca-UD":         "Autoemoteca UD",
    "UD Autoemoteca AE1":     "Autoemoteca UD",
    "UD Autoemoteca AE2":     "Autoemoteca UD",
    "UD Autoemoteca AE3":     "Autoemoteca UD",
}

#: Association codes: lowercase ones are the same code, and codes with fewer
#: than 100 occurrences out of 353k are typos. They are collected into "ALTRO"
#: instead of surviving as spurious levels in every group-by.
ASSOCIATION_MIN_COUNT = 100

SEX_MAP = {"1": "M", "2": "F"}

#: Year from which the `donor_class` coding changes (see module docstring).
DONOR_CLASS_REGIME_YEAR = 2016


# ── Rules ────────────────────────────────────────────────────────────────────

# ======================= #
# RULE_DERIVE_TIME METHOD #
# ======================= #
def rule_derive_time(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Extract year, month and date (without time) from the donation instant.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with a datetime column ``donation_datetime``.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with ``donation_date`` (normalised to midnight),
        ``donation_year`` (int16) and ``donation_month`` (int8) added.
    touched : int
        Number of rows modified (all of them).
    """
    df = df.copy()
    df["donation_date"] = df["donation_datetime"].dt.normalize()
    df["donation_year"] = df["donation_datetime"].dt.year.astype("int16")
    df["donation_month"] = df["donation_datetime"].dt.month.astype("int8")
    return df, len(df)


# =================================== #
# RULE_DROP_OUT_OF_RANGE_YEARS METHOD #
# =================================== #
def rule_drop_out_of_range_years(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop events outside the window covered by the raw files.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``donation_year``.

    Returns
    -------
    df : pandas.DataFrame
        Rows with ``YEAR_MIN <= donation_year <= YEAR_MAX``.
    removed : int
        Number of rows dropped.
    """
    bad = ~df["donation_year"].between(YEAR_MIN, YEAR_MAX)
    return df.loc[~bad], int(bad.sum())


# =================================== #
# RULE_DROP_MISSING_BIRTH_DATE METHOD #
# =================================== #
def rule_drop_missing_birth_date(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Without a birth date age cannot be computed, and it is a central predictor.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``birth_date``.

    Returns
    -------
    df : pandas.DataFrame
        Rows with a non-missing ``birth_date``.
    removed : int
        Number of rows dropped.
    """
    bad = df["birth_date"].isna()
    return df.loc[~bad], int(bad.sum())


# =============== #
# RULE_AGE METHOD #
# =============== #
def rule_age(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Age at donation, and removal of impossible values.

    Age is computed on the actual date, not on the difference between years.
    Values outside 18-90 (observed min 1, max 122) are typing errors in the
    birth date, not real donors.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``birth_date`` and ``donation_datetime``.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with ``age`` (completed years, int16), restricted to
        ``AGE_MIN <= age <= AGE_MAX``.
    removed : int
        Number of rows dropped.
    """
    df = df.copy()
    born = df["birth_date"]
    when = df["donation_datetime"]
    age = when.dt.year - born.dt.year
    before_birthday = (
        (when.dt.month < born.dt.month)
        | ((when.dt.month == born.dt.month) & (when.dt.day < born.dt.day))
    )
    df["age"] = (age - before_birthday.astype(int)).astype("int16")
    bad = ~df["age"].between(AGE_MIN, AGE_MAX)
    return df.loc[~bad], int(bad.sum())


# =============== #
# RULE_SEX METHOD #
# =============== #
def rule_sex(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Recode sex from 1/2 to M/F.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``sex`` coded as the strings ``"1"``/``"2"``.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with ``sex`` as ``"M"``/``"F"`` (pandas string dtype);
        any other code becomes missing.
    touched : int
        Number of rows with a recoded (non-missing) sex.
    """
    df = df.copy()
    df["sex"] = df["sex"].map(SEX_MAP).astype("string")
    return df, int(df["sex"].notna().sum())


# ============================ #
# RULE_COLLECTION_POINT METHOD #
# ============================ #
def rule_collection_point(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Unify the different spellings of the same collection point.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``collection_point``.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with ``collection_point`` mapped through
        ``COLLECTION_POINT_MAP`` (pandas string dtype).
    touched : int
        Number of rows whose value was in the map.
    """
    df = df.copy()
    hit = df["collection_point"].isin(COLLECTION_POINT_MAP)
    df["collection_point"] = (df["collection_point"]
                              .replace(COLLECTION_POINT_MAP).astype("string"))
    return df, int(hit.sum())


# ======================= #
# RULE_ASSOCIATION METHOD #
# ======================= #
def rule_association(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalise association codes and collect typos into ALTRO.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``association``.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with ``association`` upper-cased and stripped; codes
        with fewer than ``ASSOCIATION_MIN_COUNT`` occurrences are replaced
        by ``"ALTRO"``.
    touched : int
        Number of rows recoded to ``"ALTRO"``.
    """
    df = df.copy()
    code = df["association"].str.upper().str.strip()
    counts = code.value_counts()
    rare = counts[counts < ASSOCIATION_MIN_COUNT].index
    hit = code.isin(rare)
    df["association"] = code.where(~hit, "ALTRO").astype("string")
    return df, int(hit.sum())


# ============================== #
# RULE_DONOR_CLASS_REGIME METHOD #
# ============================== #
def rule_donor_class_regime(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Mark the `donor_class` coding change that happened in 2016.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``donation_year``.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with ``donor_class_regime`` equal to ``"pre2016"``
        before ``DONOR_CLASS_REGIME_YEAR`` and ``"post2016"`` from then on.
    touched : int
        Number of rows modified (all of them).
    """
    df = df.copy()
    df["donor_class_regime"] = np.where(
        df["donation_year"] < DONOR_CLASS_REGIME_YEAR, "pre2016", "post2016")
    return df, len(df)


# ==================== #
# RULE_VALIDITY METHOD #
# ==================== #
def rule_validity(df: pd.DataFrame, drop_invalid: bool) -> tuple[pd.DataFrame, int]:
    """Donations marked invalid: kept by default, see the module docstring.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with ``validity`` (``"V"``, ``"N"`` or missing).
    drop_invalid : bool
        If true, rows with ``validity == "N"`` are dropped; otherwise they
        are only flagged.

    Returns
    -------
    df : pandas.DataFrame
        Copy of `df` with the boolean flag ``is_invalid``, without the
        invalid rows if `drop_invalid`.
    removed : int
        Number of rows dropped (0 when `drop_invalid` is false).
    """
    df = df.copy()
    df["is_invalid"] = df["validity"].eq("N").fillna(False)
    if not drop_invalid:
        return df, 0
    bad = df["is_invalid"]
    return df.loc[~bad], int(bad.sum())


RULES = [
    ("data/anno/mese derivati",              rule_derive_time,             "modifica"),
    ("anni fuori 2009-2025",                 rule_drop_out_of_range_years, "rimozione"),
    ("data di nascita mancante",             rule_drop_missing_birth_date, "rimozione"),
    (f"eta' fuori {AGE_MIN}-{AGE_MAX}",      rule_age,                     "rimozione"),
    ("sesso 1/2 -> M/F",                     rule_sex,                     "modifica"),
    ("punti di prelievo unificati",          rule_collection_point,        "modifica"),
    ("codici associazione normalizzati",     rule_association,             "modifica"),
    ("regime di donor_class marcato",        rule_donor_class_regime,      "modifica"),
]

#: Final column order. `num_don_declared` and `quantity_total_ml` remain as a
#: check but are listed in LEAKY_COLUMNS: no model may use them.
OUTPUT_COLUMNS = [
    "donor_id", "donation_date", "donation_datetime", "donation_year",
    "donation_month", "donation_type", "age", "sex", "birth_year",
    "donor_class", "donor_class_regime", "association", "association_desc",
    "collection_point", "first_activity_date", "last_donation_date",
    "quantity_ml", "is_invalid", "validity",
    "num_don_declared", "quantity_total_ml",       # check only
    "donor_id_anag", "source_file",
]


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: apply every rule and write the donations.

    Command-line options: ``--drop-invalid`` drops the donations marked
    invalid (sensitivity variant); ``--out PATH`` writes to an alternative
    file instead of ``DONATIONS``. Reads ``EVENTS_RAW``, applies ``RULES``
    in order and then :func:`rule_validity`, keeps ``OUTPUT_COLUMNS``, and
    writes the per-rule log to ``reports/cleaning_rules.csv``.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        With code 1 if input rows minus removed rows does not equal the
        output rows.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drop-invalid", action="store_true",
                    help="scarta le donazioni con validita='N' (variante di sensitivita')")
    ap.add_argument("--out", default=None, help="percorso alternativo di output")
    args = ap.parse_args()

    df = pd.read_parquet(EVENTS_RAW)
    n_in = len(df)
    print(f"eventi in ingresso: {n_in:,}\n")

    log = []
    for name, fn, kind in RULES:
        before = len(df)
        df, touched = fn(df)
        removed = before - len(df)
        log.append({"regola": name, "tipo": kind,
                    "righe_toccate": touched, "righe_rimosse": removed,
                    "righe_residue": len(df)})
        print(f"  {name:36s} {kind:10s} toccate={touched:>7,d} rimosse={removed:>5,d}"
              f" residue={len(df):>8,d}")

    before = len(df)
    df, touched = rule_validity(df, args.drop_invalid)
    log.append({"regola": "donazioni non valide", "righe_toccate": touched,
                "tipo": "rimozione" if args.drop_invalid else "solo flag",
                "righe_rimosse": before - len(df), "righe_residue": len(df)})
    print(f"  {'donazioni non valide':36s} "
          f"{'rimozione' if args.drop_invalid else 'solo flag':10s} "
          f"toccate={touched:>7,d} rimosse={before - len(df):>5,d} residue={len(df):>8,d}")

    # The birth date is only needed to compute age: the published dataset
    # keeps only the year, as agreed with ASUGI.
    df["birth_year"] = df["birth_date"].dt.year.astype("int16")
    df = df[OUTPUT_COLUMNS].sort_values(["donation_datetime", "donor_id"]).reset_index(drop=True)
    out = args.out or DONATIONS
    df.to_parquet(out, index=False)

    rec = pd.DataFrame(log)
    rec.to_csv(REPORTS / "cleaning_rules.csv", index=False)

    n_out = len(df)
    balance = n_in - rec["righe_rimosse"].sum() - n_out
    print(f"\n  eventi in ingresso  {n_in:>9,d}")
    print(f"  righe rimosse      -{rec['righe_rimosse'].sum():>9,d}")
    print(f"  donazioni scritte   {n_out:>9,d}")
    print(f"  bilancio            {balance:>9,d}   {'OK' if balance == 0 else 'NON QUADRA'}")
    print(f"\n  donatori distinti   {df.donor_id.nunique():>9,d}")
    print(f"  scritto {out}")
    if balance != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
