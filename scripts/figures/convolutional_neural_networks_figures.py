"""Figures for Convolutional Neural Networks (content/1_Round_1/1.30-CV7-...).

    python scripts/figures/convolutional_neural_networks_figures.py   # every figure
    python scripts/figures/convolutional_neural_networks_figures.py cnn-receptive-field

Writes assets/module-media/cnn-<name>.png, raster at 2x so dense plots stay sharp.
One width, one set of font sizes and one meaning per colour, as in mlp_figures.py:

    blue   (BLUE)   the baseline design
    amber  (AMBER)  the wider kernel
    red    (BRAND)  the point the caption is about

Text is Inter, the reader's font; maths is Computer Modern, matching the KaTeX in
the lesson text.

The receptive-field plot is carried over from the retired A module 1.22, whose
own figure script drew the pooled curve as 2**n. The closed form for the design
the legend names (one 3x3 conv plus one 2x2 stride-2 pool per layer) is
r = 3 * 2**n - 2, which is what is drawn here; the axis label also reads
"global receptive field" rather than "global view". Both come from the same
review (adv-237) that cleared the figure for this module.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import figlib  # noqa: E402
from figlib import AMBER, BLUE, GRID, INK, RED  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "assets" / "module-media"

W = 7.2                 # every figure is this wide, in inches
PX = 740                # ...and shows this wide, in CSS px: one scale for every figure
SCALE = PX / (W * 72)   # CSS px per point, so 10 pt text is about 15 px on the page

INTER = Path("/usr/share/texmf-dist/fonts/opentype/public/inter")


def setup():
    figlib.style()
    for face in ("Regular", "Italic", "Medium", "SemiBold"):
        f = INTER / f"Inter-{face}.otf"
        if f.exists():
            fm.fontManager.addfont(str(f))
    plt.rcParams.update(
        {
            "font.family": "Inter" if (INTER / "Inter-Regular.otf").exists() else "DejaVu Sans",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "svg.fonttype": "path",
            "svg.hashsalt": "cnn",
            "figure.constrained_layout.use": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def plot_axes(ax):
    """A light plot frame: left and bottom spines only, a faint grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color="#9ca3af")


def receptive_field():
    """Receptive field against depth for three stacks."""
    layers = np.arange(1, 13)
    k3 = 1 + 2 * layers                # stacked 3x3 convs, stride 1
    k5 = 1 + 4 * layers                # stacked 5x5 convs, stride 1
    pooled = 3 * 2 ** layers - 2       # one 3x3 conv plus one 2x2 stride-2 pool per layer

    f, ax = plt.subplots(figsize=(W, 3.1))
    ax.plot(layers, k3, "-o", color=BLUE, lw=2.2, ms=4.5, label="$3\\times3$ convs")
    ax.plot(layers, k5, "-o", color=AMBER, lw=2.2, ms=4.5, label="$5\\times5$ convs")
    ax.plot(layers, pooled, "-o", color=RED, lw=2.2, ms=4.5,
            label="$3\\times3$ + stride-2 pooling")
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xlabel("layers")
    ax.set_ylabel("global receptive field (pixels, log scale)")
    ax.set_title("Downsampling is what buys a global receptive field cheaply", color=INK)
    ax.legend(frameon=True, loc="upper left")
    plot_axes(ax)
    return f


FIGURES = {
    "cnn-receptive-field": (receptive_field, "png"),
}


def main(names):
    setup()
    for name in names or FIGURES:
        fn, ext = FIGURES[name]
        f = fn()
        path = MEDIA / f"{name}.{ext}"
        # 2x the display scale, so the plot is sharp on retina screens and its
        # text matches the SVGs once the page halves it.
        f.savefig(path, facecolor="white", dpi=2 * SCALE * 72)
        plt.close(f)
        from PIL import Image

        w, h = (v / 2 for v in Image.open(path).size)
        print(f"{path.name:26s} {w:4.0f} x {h:3.0f} CSS px  {path.stat().st_size / 1024:4.0f} KB")


if __name__ == "__main__":
    main(sys.argv[1:])
