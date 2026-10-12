"""Patch for B's `gen_unsupervised_figures.py`: one colour, one meaning.

`kmeans-assumptions` draws its centroids with `facecolor=P.signal` (A's
terracotta), while `kmeans` from `gen_ml_figures_2.py` draws the same object
with `color=P.ink` (A's near-black). Both figures sit in one lesson, so the
same object should be the same colour on the page, and terracotta is A's
single-highlight colour rather than a third cluster mark. This recolours the
centres to ink and keeps every other line of the generator as B wrote it.

Only the centroid scatter is touched. The figure still draws four panels,
calls `KMeans` with the same seeds, and writes the same file name.
"""

import matplotlib.axes

import b_theme


def apply(module):
    run = module.kmeans_assumptions

    def patched():
        # Recolour only the centroid scatter: it is the one `marker="X"` call
        # in the figure, drawn after the per-cluster `P.dots` scatters.
        original = matplotlib.axes.Axes.scatter

        def scatter(self, x, y, **kw):
            if kw.get("marker") == "X":
                kw["facecolor"] = b_theme.P.ink
                kw["edgecolor"] = b_theme.P.ink
            return original(self, x, y, **kw)

        matplotlib.axes.Axes.scatter = scatter
        try:
            return run()
        finally:
            matplotlib.axes.Axes.scatter = original

    patched.figure_name = run.figure_name
    module.kmeans_assumptions = patched
