# ============= #
# MANIFEST FILE #
# ============= #
"""sha256 manifest of the raw files.

The data is not committed (it is pseudonymised and not redistributable), but
the manifest is: it is the only way to verify that two runs, months apart or on
another machine, started from exactly the same files.

Run as ``python -m manifest`` to (re)write ``data/manifest.sha256``, or as
``python -m manifest --check`` to verify the present files against it.

Functions
---------
sha256
    Streaming SHA-256 digest of a file. Used only inside this module, by
    :func:`build` and :func:`verify`.
build
    Builds the manifest text for the current raw files. Used only inside
    this module, by :func:`main`.
verify
    Compares the present raw files with the manifest. Used by :func:`main`
    and by ``scripts/run_pipeline.py`` before a full run.
main
    Command-line entry point. Run via ``python -m manifest`` from the
    ``Makefile``.
"""

from __future__ import annotations

import hashlib
import sys

import config
from config import MANIFEST, RAW_XLSX


# ============= #
# SHA256 METHOD #
# ============= #
def sha256(path, chunk: int = 1 << 20) -> str:
    """Compute the SHA-256 digest of a file, reading it in chunks.

    Parameters
    ----------
    path : str or pathlib.Path
        File to hash.
    chunk : int, default 1 MiB
        Number of bytes read per iteration, so large files are never loaded
        into memory at once.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest (64 characters).
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ============ #
# BUILD METHOD #
# ============ #
def build() -> str:
    """Build the manifest text for the raw XLSX files currently present.

    Returns
    -------
    str
        One line per file, ``"<sha256>  <file name>  <size in bytes>"``
        (fields separated by two spaces), in the order of
        :func:`config.raw_files`, with a trailing newline.

    Raises
    ------
    SystemExit
        If no raw XLSX file is present.
    """
    files = config.raw_files()
    if not files:
        raise SystemExit(f"Nessun file .xlsx in {RAW_XLSX}")
    lines = [f"{sha256(f)}  {f.name}  {f.stat().st_size}" for f in files]
    return "\n".join(lines) + "\n"


# ============= #
# VERIFY METHOD #
# ============= #
def verify() -> list[str]:
    """Compare the files present with the manifest; return the discrepancies.

    Empty list = the same files, byte for byte, as those the thesis numbers
    were produced from.

    Returns
    -------
    list of str
        Human-readable problems (in Italian), in this order: files listed in
        the manifest but missing (``"manca: ..."``), files present but not
        listed (``"non nel manifest: ..."``), and files whose digest differs
        (``"sha256 diverso: ..."``).
    """
    expected = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name, _size = line.split("  ")
            expected[name] = digest
    present = {f.name: f for f in config.raw_files()}
    problems = [f"manca: {n}" for n in sorted(expected.keys() - present.keys())]
    problems += [f"non nel manifest: {n}" for n in sorted(present.keys() - expected.keys())]
    problems += [f"sha256 diverso: {n}" for n in sorted(expected.keys() & present.keys())
                 if sha256(present[n]) != expected[n]]
    return problems


# =========== #
# MAIN METHOD #
# =========== #
def main() -> None:
    """Command-line entry point: write or check the manifest.

    With ``--check`` among the command-line arguments, runs :func:`verify`
    and prints each discrepancy; otherwise rewrites ``MANIFEST`` with
    :func:`build` and prints its contents.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the raw files are missing, or if ``--check`` finds at least one
        discrepancy.
    """
    config.require_raw()
    if "--check" in sys.argv[1:]:
        problems = verify()
        for p in problems:
            print(f"  {p}")
        if problems:
            raise SystemExit(f"{len(problems)} discrepanze rispetto a {MANIFEST.name}")
        print(f"tutti i file corrispondono a {MANIFEST.name}")
        return
    text = build()
    MANIFEST.write_text(text, encoding="utf-8")
    n = text.count("\n")
    print(f"{n} file elencati in {MANIFEST.relative_to(MANIFEST.parents[1])}")
    print(text, end="")


if __name__ == "__main__":
    main()
