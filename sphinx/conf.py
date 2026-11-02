# ========= #
# CONF FILE #
# ========= #
"""Sphinx configuration for the API documentation of the pipeline.

Build from the repository root with::

    uv sync --extra dev --extra docs
    uv run sphinx-build -b html sphinx sphinx/_build/html

then open ``sphinx/_build/html/index.html``.

The modules are not an installed package: ``src/``, ``scripts/``,
``shiny_app/`` and ``tests/`` are put on ``sys.path`` and each file is
imported as a top-level module, exactly as the pipeline does with
``PYTHONPATH=src``. Importing them has the same side effects as running the
pipeline (``config`` creates the output directories, ``dash_overdisp``
reads ``data/processed/donations.parquet``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts", "shiny_app", "tests"):
    sys.path.insert(0, str(ROOT / sub))

# -- Project information -----------------------------------------------------

project = "Donazioni di sangue Trieste-Gorizia"
author = "Marco Secci"
copyright = "2026, Marco Secci"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",      # pull documentation from the docstrings
    "sphinx.ext.napoleon",     # parse the NumPy-style sections
    "sphinx.ext.viewcode",     # "[source]" links to the highlighted code
]

exclude_patterns = ["_build"]

# Single backticks (`name`) are tried as a cross-reference to a Python object;
# parameter names that match nothing are simply shown as code.
default_role = "py:obj"

# -- autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "members": True,            # documented functions, classes, `#:` constants
    "private-members": True,    # the documented `_helpers` too
    "member-order": "bysource",
}
autodoc_typehints = "description"
# Imported only inside a function, or only by the screenshot script: mocked so
# the docs build without them.
autodoc_mock_imports = ["lightgbm", "playwright"]

# -- napoleon (NumPy docstrings) ---------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# The module docstrings end with a list of their functions and of the modules
# that use them; rendered like a Parameters list.
napoleon_custom_sections = [
    ("Functions", "params_style"),
    ("Classes and functions", "params_style"),
]

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "Donazioni di sangue — documentazione del codice"
