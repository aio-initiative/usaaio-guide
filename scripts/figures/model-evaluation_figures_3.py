"""Figures for the model-evaluation module that B does not draw.

`reliability-diagram` is A's own calibration figure, redrawn in A's ink through
`b_theme` from the drawing code that shipped in A's `2.3-ML6-model-evaluation.md`
(deterministic: `default_rng(15)`, 4,000 rows, so the curves are the same ones
A's lesson showed). The merged lesson introduces the diagram before any score
is defined, so the in-image title and the Brier values in the legend are gone;
the legend keeps the two model names.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from b_theme import EMPH, P, clean, render, run_all  # noqa: E402


@render("ml6-calibration", size="wide")
def reliability_diagram(ax):
    rng = np.random.default_rng(15)
    p_true = rng.uniform(0, 1, 4000)
    y = (rng.uniform(size=p_true.size) < p_true).astype(int)
    overconfident = np.clip(p_true + 0.28 * (p_true - 0.5) * 2, 0.001, 0.999)

    bins = np.linspace(0, 1, 11)
    for probs, style, name in [
        (p_true, dict(P.line(rank=0), marker="o", markersize=3.2), "calibrated"),
        (overconfident, dict(color=P.signal, lw=EMPH, marker="o",
                             markersize=3.2, solid_capstyle="round"), "overconfident"),
    ]:
        idx = np.digitize(probs, bins) - 1
        xs = [probs[idx == b].mean() for b in range(10) if np.any(idx == b)]
        ys = [y[idx == b].mean() for b in range(10) if np.any(idx == b)]
        ax.plot(xs, ys, label=name, **style)

    ax.plot([0, 1], [0, 1], **P.guide())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(**P.grid())
    clean(ax, spines="lb")


if __name__ == "__main__":
    run_all(globals())
