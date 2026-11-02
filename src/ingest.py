# =========== #
# INGEST FILE #
# =========== #
"""Stage 1 — from the 27 raw XLSX files to a single events table.

What it does, and why each step:

1. **Renames the columns** from the DIMT management-system names to canonical
   names. The mapping is explicit and complete: an unknown raw column is an
   error, it is not silently ignored.

2. **`donor_id` = `crd`, not `id_anag`.** `crd` appears in 27 files out of 27;
   `id_anag` in 23 (missing from the 2021 and 2022 files and from both 2023
   ones). Over the 41,225 distinct `crd` of the files that have both columns
   the correspondence is effectively 1:1 (3 `crd` with two `id_anag`, 0
   `id_anag` with two `crd`). Choosing `id_anag` — as the original pipeline did
   — forces reconstructing the identity of 2021-2023 donors with a partial
   mapping, which is the source of the "donors with truncated identity" and of
   the false dropouts in 2023.

3. **Merges rows describing the same event.** `2023 tot dimt new.xlsx` and
   `2023 ADS Emoteca new.xlsx` contain exactly the same set of 16,843 events
   and differ by a single column (`sez` in the first, `validita` in the
   second). Loading both would double 2023; loading only one would throw away
   a column. They are merged on the event key taking the first non-null value
   of each column, so both are kept.

4. **Reconciles.** `reports/ingest_reconciliation.csv` reports, for each file,
   how many rows were read; and at the end the global balance. No row can
   disappear without showing up there.

It does no cleaning: filters, normalisations and decisions about the target
live in `clean`, to keep "what was in the files" separate from "what we decided
to keep".

Functions
---------
_norm_id
    Private helper: normalises identifier codes to clean strings. Used only
    inside this module, by :func:`read_one`.
_parse_dt
    Private helper: parses day-first Italian dates. Used only inside this
    module, by :func:`read_one`.
read_one
    Reads one XLSX file into the canonical columns. Used only inside this
    module, by :func:`main`.
merge_duplicate_events
    Merges rows describing the same event. Used only inside this module, by
    :func:`main`.
main
    Command-line entry point. Run via ``python -m ingest`` from the
    ``Makefile`` and ``scripts/run_pipeline.py``; its output
    ``EVENTS_RAW`` is read by :mod:`clean`. (:mod:`figures_tesi` imports
    this module but calls none of its functions.)
"""

from __future__ import annotations

import sys

import pandas as pd

import config
from config import EVENT_KEY, EVENTS_RAW, RAW_XLSX, REPORTS

# ── Raw -> canonical name mapping ────────────────────────────────────────────
# The names on the left are those of the management system, abbreviated and in Italian.
COLUMN_MAP = {
    # present in all 27 files
    "crd":                      "donor_id",            # donor code
    "dtatt_da":                 "donation_datetime",   # donation date/time
    "descr_breve_emcdon":       "donation_type",       # SANGUE, PLASMAF, ...
    "dtnasc":                   "birth_date",
    "dtinizatt_da":             "first_activity_date", # start of donating activity
    "num_don":                  "num_don_declared",    # NOT the target: see the clean module
    "s":                        "sex",
    "class_don":                "donor_class",         # P = periodic, O = occasional
    # present in 23 files
    "id_anag":                  "donor_id_anag",       # kept only as a check
    "descrizione_associazione": "association_desc",
    "dt_ult_don":               "last_donation_date",
    "punto_di_prelievo":        "collection_point",
    "qta":                      "quantity_ml",
    "qta_tot":                  "quantity_total_ml",   # NOT a feature: see clean
    "emc":                      "emc",
    # present in 21 files
    "ass":                      "association",
    # present in 3-4 files
    "validita":                 "validity",            # V = valid, N = not valid
    "sez":                      "section_code",
    "descrizione_classdon":     "donor_class_desc",
    "descrizione_sezione":      "section_desc",
    "id_pprel":                 "collection_point_id",
    "mot_pres":                 "visit_reason",
    "validita_descr":           "validity_desc",
}

#: Columns on which, when merging, the first non-null value is kept.
_MERGE_EXCLUDE = set(EVENT_KEY) | {"source_file"}


# =============== #
# _NORM_ID METHOD #
# =============== #
def _norm_id(s: pd.Series) -> pd.Series:
    """Identifier codes as clean strings.

    Excel returns integers as floats (`70055441.0`) when there are empty cells
    in the same column: without normalisation the same donor would have two
    different identities depending on the file.

    Parameters
    ----------
    s : pandas.Series
        Raw identifier column (numbers, floats or strings).

    Returns
    -------
    pandas.Series
        Pandas string dtype, stripped, without a trailing ``".0"``; empty
        strings and ``"nan"`` become ``pd.NA``.
    """
    out = s.astype("string").str.strip()
    return out.str.replace(r"\.0$", "", regex=True).replace({"": pd.NA, "nan": pd.NA})


# ================ #
# _PARSE_DT METHOD #
# ================ #
def _parse_dt(s: pd.Series) -> pd.Series:
    """Italian dates, in two formats and sometimes already typed by Excel.

    The files mix `05/11/2009 09:43:00` and `05-11-2009`, both day-first, plus
    cells that openpyxl already returns as datetime. Parsing is attempted on
    the explicit formats and only the leftovers go to inference, so that pandas
    is not left guessing month/day at random.

    Parameters
    ----------
    s : pandas.Series
        Raw date column (strings, or already datetime).

    Returns
    -------
    pandas.Series
        ``datetime64[ns]`` series with the same index; values that cannot be
        parsed are ``NaT``. Returned unchanged if `s` is already datetime.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    txt = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        todo = out.isna() & txt.notna()
        if not todo.any():
            break
        out.loc[todo] = pd.to_datetime(txt[todo], format=fmt, errors="coerce")
    todo = out.isna() & txt.notna()
    if todo.any():
        out.loc[todo] = pd.to_datetime(txt[todo], dayfirst=True, errors="coerce")
    return out


# =============== #
# READ_ONE METHOD #
# =============== #
def read_one(path) -> pd.DataFrame:
    """Read one XLSX and return the canonical columns.

    Renames the columns with ``COLUMN_MAP``, normalises identifiers
    (:func:`_norm_id`), parses dates (:func:`_parse_dt`), strips text
    columns and converts numeric ones.

    Parameters
    ----------
    path : pathlib.Path
        Raw XLSX file; only its first sheet is read.

    Returns
    -------
    pandas.DataFrame
        One row per raw row, with the canonical columns present in that file
        plus ``source_file`` (the file name).

    Raises
    ------
    ValueError
        If the file has a column not listed in ``COLUMN_MAP``.
    """
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    unknown = set(df.columns) - set(COLUMN_MAP)
    if unknown:
        raise ValueError(
            f"{path.name}: colonne non mappate {sorted(unknown)}. "
            "Aggiungile a COLUMN_MAP invece di lasciarle cadere in silenzio."
        )

    df = df.rename(columns=COLUMN_MAP)
    for col in ("donor_id", "donor_id_anag"):
        if col in df:
            df[col] = _norm_id(df[col])
    for col in ("donation_datetime", "birth_date", "first_activity_date", "last_donation_date"):
        if col in df:
            df[col] = _parse_dt(df[col])
    for col in ("donation_type", "sex", "donor_class", "association", "association_desc",
                "collection_point", "validity", "section_code", "donor_class_desc",
                "section_desc", "collection_point_id", "visit_reason", "validity_desc"):
        if col in df:
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA})
    for col in ("num_don_declared", "quantity_ml", "quantity_total_ml", "emc"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_file"] = path.name
    return df


# ============================= #
# MERGE_DUPLICATE_EVENTS METHOD #
# ============================= #
def merge_duplicate_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge the rows that describe the same donation event.

    The same donor cannot donate the same type twice at the same instant: rows
    sharing the event key are the same donation seen from two different files.
    They are reduced to a single row taking, for each column, the first
    non-null value — so columns present in only one of the two files (`sez`
    from one, `validita` from the other) both survive.

    Parameters
    ----------
    df : pandas.DataFrame
        Concatenated events of all files, with the ``config.EVENT_KEY``
        columns and ``source_file``.

    Returns
    -------
    merged : pandas.DataFrame
        One row per distinct event key; ``source_file`` lists the files of
        the merged rows joined by ``" + "``.
    detail : pandas.DataFrame
        One row per combination of files that shared events, with columns
        ``files``, ``eventi`` (events merged) and ``righe`` (rows involved).
        Empty if there are no duplicates.
    """
    dup_mask = df.duplicated(subset=EVENT_KEY, keep=False)
    dups = df.loc[dup_mask]

    if dups.empty:
        detail = pd.DataFrame(columns=["files", "eventi", "righe"])
    else:
        pairs = (dups.groupby(EVENT_KEY, dropna=False, sort=False)["source_file"]
                     .agg(lambda s: " + ".join(sorted(set(s)))))
        detail = (pairs.value_counts().rename_axis("files").reset_index(name="eventi"))
        sizes = dups.groupby(EVENT_KEY, dropna=False, sort=False).size()
        detail["righe"] = detail["files"].map(
            pairs.to_frame("files").assign(n=sizes).groupby("files")["n"].sum())

    value_cols = [c for c in df.columns if c not in _MERGE_EXCLUDE]
    agg = {c: "first" for c in value_cols}          # `first` skips nulls
    agg["source_file"] = lambda s: " + ".join(sorted(set(s)))

    merged = (df.groupby(EVENT_KEY, dropna=False, sort=False, as_index=False)
                .agg(agg))
    return merged, detail


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: read every raw file and write the event table.

    Reads all files with :func:`read_one`, drops rows without an event
    date, merges duplicates with :func:`merge_duplicate_events` and writes
    ``EVENTS_RAW``. Writes the per-file reconciliation to
    ``reports/ingest_reconciliation.csv`` and the merge detail to
    ``reports/ingest_merged_events.csv``.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the raw files are missing, or (code 1) if the row balance does not
        close.
    """
    config.require_raw()
    files = config.raw_files()

    frames, per_file = [], []
    for path in files:
        df = read_one(path)
        frames.append(df)
        per_file.append({
            "file": path.name,
            "righe_lette": len(df),
            "colonne": df.shape[1] - 1,
            "senza_data_evento": int(df["donation_datetime"].isna().sum()),
            "anno_min": df["donation_datetime"].dt.year.min(),
            "anno_max": df["donation_datetime"].dt.year.max(),
        })
        print(f"  {path.name:44s} {len(df):7,d} righe", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    n_read = len(raw)

    # Rows without an event date: they cannot be placed in time and cannot be
    # part of the key. They are dropped here and counted in the report.
    no_dt = int(raw["donation_datetime"].isna().sum())
    raw = raw[raw["donation_datetime"].notna()]

    merged, dup_detail = merge_duplicate_events(raw)
    n_out = len(merged)

    merged = merged.sort_values(["donation_datetime", "donor_id"]).reset_index(drop=True)
    merged.to_parquet(EVENTS_RAW, index=False)

    # ── reconciliation ───────────────────────────────────────────────────────
    rec = pd.DataFrame(per_file)
    tot = pd.DataFrame([{
        "file": "TOTALE LETTO", "righe_lette": n_read,
        "colonne": raw.shape[1] - 1, "senza_data_evento": no_dt,
        "anno_min": merged["donation_datetime"].dt.year.min(),
        "anno_max": merged["donation_datetime"].dt.year.max(),
    }, {
        "file": "- scartate: nessuna data evento", "righe_lette": -no_dt,
    }, {
        "file": "- fuse: stesso evento in piu' file",
        "righe_lette": -(n_read - no_dt - n_out),
    }, {
        "file": "= EVENTI SCRITTI", "righe_lette": n_out,
    }])
    rec = pd.concat([rec, tot], ignore_index=True)
    rec.to_csv(REPORTS / "ingest_reconciliation.csv", index=False)
    dup_detail.to_csv(REPORTS / "ingest_merged_events.csv", index=False)

    balance = n_read - no_dt - (n_read - no_dt - n_out) - n_out
    print(f"\n  righe lette              {n_read:>8,d}")
    print(f"  senza data evento       -{no_dt:>8,d}")
    print(f"  fuse (stesso evento)    -{n_read - no_dt - n_out:>8,d}")
    print(f"  eventi scritti           {n_out:>8,d}")
    print(f"  bilancio                 {balance:>8,d}   {'OK' if balance == 0 else 'NON QUADRA'}")
    if not dup_detail.empty:
        print("\n  fusioni per coppia di file:")
        print(dup_detail.to_string(index=False))
    print(f"\n  scritto {EVENTS_RAW}")
    if balance != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
