"""Redrawn figure for the training-loop module (training-loop-in-pytorch).

Copy of site B's `tools/diagrams/gen_training_figures.py` for the one figure
this sub-lane owns that needed a fix, drawn through A's theme. The other
figure, `training-curves`, comes straight from B's script through
`figures.py` and is not duplicated here.

What changed from B's script, and why: B draws the "evaluate on validation"
box at the same 3.0 unit width as the five step boxes, but the label is 3.31
units wide at the label text rank, so it runs past both sides of its own box.
B's original is just as tight on its own page. Here that box is widened to
3.9 units, the width of the "for each batch" box that already wraps the five
steps, so the two boxes line up and the label has clear space on both sides.
Nothing else about the layout moved: same five steps, same order, same two
dashed loop boxes, same labels, same colours, same canvas.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import b_theme  # noqa: E402

from b_theme import P, arrow, block, label, render, run_all  # noqa: E402


@render("training-loop-flow", size="tall")
def training_loop_flow(ax):
    """Two nested loops drawn top to bottom: an epoch is a pass over the
    batches, and every batch gets the same five steps."""
    cx = 3.6
    steps = ["forward pass", "loss", "zero_grad()", "backward()", "step()"]
    w, h, gap = 3.0, 0.62, 0.3
    y_top = 8.9
    y = y_top
    for i, text in enumerate(steps):
        block(ax, cx - w / 2, y - h, w, h, f"{i + 1}. {text}", rank="label")
        if i < len(steps) - 1:
            arrow(ax, (cx, y - h), (cx, y - h - gap))
        y -= h + gap
    y_bottom = y + gap
    # The batch loop wraps the five steps, and the epoch loop wraps that plus
    # an evaluation pass.
    inner = (cx - w / 2 - 0.45, y_bottom - 0.35, w + 0.9, y_top - y_bottom + 0.9)
    block(ax, *inner, "", rank="small", linewidth=0.8, linestyle=(0, (4, 3)))
    label(ax, inner[0] + 0.15, inner[1] + inner[3] - 0.12, "for each batch",
          rank="tick", ha="left", va="top")
    ev_y = inner[1] - 0.9
    arrow(ax, (cx, inner[1]), (cx, ev_y + h))
    # B draws this box at the step width `w`; the label is wider than that.
    # `ew` matches the batch box, so the two line up and the label clears.
    ew = w + 0.9
    block(ax, cx - ew / 2, ev_y, ew, h, "evaluate on validation", rank="label")
    outer = (inner[0] - 0.45, ev_y - 0.4, inner[2] + 0.9, inner[1] + inner[3] + 0.9 - (ev_y - 0.4))
    block(ax, *outer, "", rank="small", linewidth=0.8, linestyle=(0, (4, 3)))
    label(ax, outer[0] + 0.15, outer[1] + outer[3] - 0.12, "for each epoch",
          rank="tick", ha="left", va="top")
    ax.set_xlim(outer[0] - 0.2, outer[0] + outer[2] + 0.2)
    ax.set_ylim(outer[1] - 0.2, outer[1] + outer[3] + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    run_all(globals())
