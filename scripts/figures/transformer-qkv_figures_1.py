"""Redrawn figures for the Query, Key and Value module (transformer-qkv).

Copy of site B's `tools/diagrams/gen_transformer_figures.py` for the two
figures this sub-lane owns, drawn through A's theme. Only the two figures the
module uses are here; the rest of B's script is left to its own sub-lanes.

What changed from B's script, and why: in B's `transformer-qkv` the numbers in
the boxed cells are wider than the cells (the widest, "−0.416", is about 1.25
cell widths), so the minus sign and the last digit touch or cross the cell
outline. B has the same flaw on its own page. The cells are widened here, and
the figure is drawn on a wider canvas, so every number has clear space on both
sides. Nothing else about the layout is different: same tokens, same numbers,
same weights tables, same arrows, same labels, same colours.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import b_theme  # noqa: E402

# A canvas proportion for this figure only. B draws it at 6.0 x 3.2 in, which
# is what leaves the cells too narrow for their numbers; 8.0 x 4.25 in keeps
# the original proportions while giving every cell the room its text needs.
b_theme.SIZES["wide-qkv"] = (8.0, 4.25)

from b_theme import P, arrow, block, label, render, run_all  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

DASH = (0, (4, 2.5))


def fmt(v, digits=3):
    """A number as the pages print it: fixed decimals and a real minus sign."""
    if isinstance(v, str):
        return v
    return f"{v:.{digits}f}".replace("-", "−")


def cells(ax, x0, y0, rows, cw=1.0, ch=0.5, *, on=True, digits=3, muted=()):
    """A grid of numbers with its top-left row first. Copied from B's script.

    `muted` lists (row, col) cells drawn in the receding ink, for values that
    are switched off.
    """
    n = len(rows)
    for i, row in enumerate(rows):
        y = y0 + (n - 1 - i) * ch
        for j, v in enumerate(row):
            quiet = (i, j) in muted or not on
            ax.add_patch(Rectangle(
                (x0 + j * cw, y), cw, ch, facecolor="none",
                edgecolor=P.series[2] if quiet else P.ink,
                lw=b_theme.HAIR if quiet else b_theme.THIN,
                zorder=2 if quiet else 3))
            label(ax, x0 + (j + 0.5) * cw, y + ch / 2, fmt(v, digits),
                  rank="small", ha="center", va="center",
                  color=P.mute if quiet else P.ink)


@render("transformer-qkv", size="wide-qkv")
def transformer_qkv(ax):
    h = [0.909, -0.416, 1.020, 2.000]
    outs = [("query", "$W_Q$", [1.020, 2.000, -0.416, 2.000], 3.6),
            ("key", "$W_K$", [-0.416, 1.020, -0.416, 2.000], 2.0),
            ("value", "$W_V$", [0.909, -0.416, 1.020, 2.000], 0.4)]
    # B: cw, ch, gx = 1.3, 0.55, 9.3. cw is wider so each number fits its cell.
    cw, ch, gx = 1.6, 0.55, 9.9
    cells(ax, 0.2, 2.0, [h], cw, ch)
    label(ax, 0.2 + 2 * cw, 2.9, "vector of “sat”, from Step 3", rank="tick",
          ha="center")
    for name, weights, vals, y in outs:
        arrow(ax, (0.3 + 4 * cw, 2.0 + ch / 2), (gx - 1.7, y + ch / 2))
        label(ax, gx - 1.45, y + ch / 2, weights, rank="tick", ha="left",
              va="center", color=P.ink)
        cells(ax, gx, y, [vals[:2]], cw, ch)
        cells(ax, gx + 2 * cw + 0.3, y, [vals[2:]], cw, ch)
        label(ax, gx + 4 * cw + 0.55, y + ch / 2, name, rank="label",
              ha="left", va="center")
    label(ax, gx + cw, 4.45, "head 1", rank="small", ha="center")
    label(ax, gx + 3 * cw + 0.3, 4.45, "head 2", rank="small", ha="center")
    ax.set_xlim(0, gx + 4 * cw + 1.9)
    ax.set_ylim(0.1, 4.8)
    ax.axis("off")


@render("transformer-problem-qkv", size="banner")
def transformer_problem_qkv(ax):
    w, h, y = 1.6, 0.75, 1.0
    xs = [0.6, 4.4, 8.2]
    for x, tok in zip(xs, ["the", "cat", "sat"]):
        block(ax, x, y, w, h, tok, emph=tok != "the")
    arrow(ax, (xs[1] + w / 2, y + h + 0.05), (xs[2] + w / 2, y + h + 0.05),
          emph=True, curve=-0.45)
    label(ax, (xs[1] + xs[2]) / 2 + w / 2, y + h + 1.15,
          "3. hands over information", rank="tick", ha="center", color=P.ink)
    label(ax, xs[2] + w / 2, y - 0.3, "1. says what it\nis looking for",
          rank="small", ha="center", va="top")
    label(ax, xs[1] + w / 2, y - 0.3, "2. shows what\nit offers", rank="small",
          ha="center", va="top")
    label(ax, xs[0] + w / 2, y - 0.3, "not what “sat”\nis looking for",
          rank="small", ha="center", va="top")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.3, 3.2)
    ax.axis("off")


if __name__ == "__main__":
    run_all(globals())
