# ======== #
# VIZ FILE #
# ======== #
"""Shared figure style.

Fixed-order categorical palette: the same subject has the same colour in every
figure, and colours are never recycled in a different order depending on how
many series there are. The first three hues are those that remain
distinguishable even with colour-vision deficiency: beyond three series we
switch to "Other" or to small multiples instead of adding hues.

Importing the module selects the non-interactive ``Agg`` backend and applies
``STYLE`` to ``matplotlib.pyplot.rcParams``; the palette constants
(``SERIES``, ``INK``, ...) and the configured ``plt`` are re-exported to the
figure-producing modules.

Functions
---------
finish
    Applies title, axis labels, horizontal-only grid and an optional note to
    an axis. Used by :mod:`eda`, :mod:`forecast`, :mod:`glm_inference`,
    :mod:`population`, :mod:`retention` and :mod:`figures_tesi`.
save
    Writes a figure to disk and closes it. Used by the same modules as
    :func:`finish`.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "text.color": INK, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 4.5, "figure.autolayout": False,
}

plt.rcParams.update(STYLE)


# ============= #
# FINISH METHOD #
# ============= #
def finish(ax, title=None, xlabel=None, ylabel=None, note=None):
    """Recurring finishing touches: title, axes, horizontal-only grid, note.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to finish; it is modified in place.
    title : str, optional
        Left-aligned axis title. Skipped if ``None`` or empty.
    xlabel : str, optional
        Label of the x axis. Skipped if ``None`` or empty.
    ylabel : str, optional
        Label of the y axis. Skipped if ``None`` or empty.
    note : str, optional
        Small grey text placed below the x-axis label (source, caveats).
        Skipped if ``None`` or empty.

    Returns
    -------
    matplotlib.axes.Axes
        The same `ax`, to allow chaining.
    """
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if note:
        # y=-0.04 in axes coordinates falls inside the tick-label band and the
        # note ends up on top of it: we go below the x-axis label instead.
        ax.figure.text(0.0, -0.26, note, fontsize=8, color=INK_SOFT,
                       ha="left", va="top", transform=ax.transAxes)
    return ax


# =========== #
# SAVE METHOD #
# =========== #
def save(fig, path, dpi=None):
    """Write the figure to disk, close it and print its file name.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write; it is closed afterwards to free memory.
    path : pathlib.Path
        Destination file; the format is inferred from its extension.
    dpi : int, optional
        Resolution override, for figures dense with small text, which must be
        rasterised finer to stay legible once printed. If ``None`` the
        ``savefig.dpi`` of ``STYLE`` is used.

    Returns
    -------
    None
    """
    fig.savefig(path, dpi=dpi) if dpi else fig.savefig(path)
    plt.close(fig)
    print(f"  scritto {path.name}")
