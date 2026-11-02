# =========== #
# CONFIG FILE #
# =========== #
"""Project paths and constants.

Every path is derived from the location of THIS file, never written as an
absolute path: the repo works identically on the host, inside the container and
on another machine. (In the original repo an absolute sandbox path had ended up
inside `rebuild_2021_2023.py`, making the 2021-2023 fix irreproducible.)

Importing the module also creates the ``data/interim``, ``data/processed`` and
``reports`` directories if they do not exist yet.

The path and data-semantics constants (``DONATIONS``, ``DONOR_YEAR``,
``REPORTS``, ``LEAKY_COLUMNS``, ``PANEL_BURN_IN_YEAR``, ...) are imported by
almost every other module of the project.

Functions
---------
raw_files
    Lists the raw XLSX files, case-insensitively on the extension. Used by
    :mod:`manifest`, :mod:`ingest` and ``scripts/run_pipeline.py``.
require_raw
    Exits with a readable message when the raw XLSX files are missing. Used
    by :mod:`manifest` and :mod:`ingest`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW_XLSX = DATA / "raw" / "XLSX"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
MANIFEST = DATA / "manifest.sha256"

EVENTS_RAW = INTERIM / "events_raw.parquet"
DONATIONS = PROCESSED / "donations.parquet"
DONOR_YEAR = PROCESSED / "donor_year.parquet"

for _d in (INTERIM, PROCESSED, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


# ── Data semantics ───────────────────────────────────────────────────────────

#: Years covered by the raw files.
YEAR_MIN, YEAR_MAX = 2009, 2025

#: Key identifying a donation event. Used both to deduplicate and to join the
#: two 2023 files, which are the same set of events seen with two different
#: auxiliary columns (`sez` in one, `validita` in the other).
EVENT_KEY = ["donor_id", "donation_datetime", "donation_type"]

#: Columns that are the target rewritten in another form. They must not appear
#: in any design matrix:
#:
#: - `num_don`  = count of (donor, year, type), replicated on every row
#: - `qta_tot`  = sum of `qta` within the same group, hence ~ num_don * qta
#:
#: They are kept in the dataset only as control columns.
LEAKY_COLUMNS = ["num_don_declared", "quantity_total_ml"]

#: First year usable for estimating a model on the panel.
#: `recency` is not observable before the start of the window: in 2009 every
#: donor appears to be at their first donation, in 2010 79% appear as "donated
#: last year", and the "6+" class can only exist from 2015. Training on those
#: years means estimating coefficients on a composition that will never occur
#: again. Donations from the excluded years are still used to build donor
#: histories: only the estimation sample starts later.
PANEL_BURN_IN_YEAR = 2013

#: Age range considered admissible for a donor (Italian regulations: 18-65 for
#: the first donation, with extensions up to 70; above 90 they are birth date
#: errors). The original repo used 18-90: we keep the same interval so counts
#: can be reconciled, but the filter is parametric.
AGE_MIN, AGE_MAX = 18, 90


# ================ #
# RAW_FILES METHOD #
# ================ #
def raw_files() -> list[Path]:
    """The 27 XLSX files, sorted, regardless of how the extension is written.

    On Windows the filesystem is case-insensitive, on Linux it is not: an
    `.XLSX` would be seen by one and not by the other, and the pipeline would
    silently produce different results on the two machines.

    Returns
    -------
    list of pathlib.Path
        Sorted paths of the regular files in ``RAW_XLSX`` whose extension is
        ``.xlsx`` in any letter case. Empty if the directory does not exist
        or contains no such file.
    """
    if not RAW_XLSX.exists():
        return []
    return sorted(p for p in RAW_XLSX.iterdir()
                  if p.suffix.lower() == ".xlsx" and p.is_file())


# ================== #
# REQUIRE_RAW METHOD #
# ================== #
def require_raw() -> None:
    """Stop the command with a readable message if the raw files are missing.

    The 27 XLSX files contain pseudonymised data provided by DIMT of Trieste
    and Gorizia and are not versioned: whoever clones the repository does not
    have them. Without this guard the first pipeline command would die inside
    pandas, with a stack trace that does not say what the problem is.

    Returns
    -------
    None
        Returns normally when :func:`raw_files` finds at least one file.

    Raises
    ------
    SystemExit
        If no raw XLSX file is present; the message explains where the files
        should go and how to restart the pipeline from the processed data.
    """
    if raw_files():
        return
    raise SystemExit(
        f"\nNessun file .xlsx in {RAW_XLSX}.\n\n"
        "Gli XLSX originali di ASUGI non sono nel repository; il dataset pulito\n"
        "data/processed/donations.parquet invece si', e la pipeline riparte da\n"
        "li' con run_pipeline.sh / run_pipeline.ps1, saltando ingest e clean.\n"
        "Chi ha gli XLSX li copia in quella cartella; `python -m manifest --check`\n"
        "verifica che siano quelli attesi.\n"
    )
