"""Shared plot styling for module figures.

Every figure in the guide is produced by a function in `figures.py`. The source
of that function is what gets embedded in the module's collapsible "Show the
code" block, so the code a reader sees is exactly the code that drew the image.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#111827"
MUTED = "#4b5563"
GRID = "#e5e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
AMBER = "#d97706"
PURPLE = "#7c3aed"
CYCLE = [BLUE, RED, GREEN, AMBER, PURPLE]


def style():
    """Applies the guide's figure styling: light, flat, high-contrast."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "white",
            "legend.fontsize": 10,
            "font.size": 11,
            # 2x the 832px content column, so figures stay sharp on retina screens.
            "figure.dpi": 280,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            # Keeps a suptitle clear of the panel titles beneath it.
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.06,
        }
    )


def fig(w=7.2, h=3.6, ncols=1):
    """A figure sized for the guide's 832px text column."""
    style()
    f, ax = plt.subplots(1, ncols, figsize=(w, h))
    return f, ax
