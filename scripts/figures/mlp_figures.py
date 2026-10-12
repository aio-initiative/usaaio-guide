"""Figures for The Multilayer Perceptron (content/1_Round_1/1.18-DL1-...).

    python scripts/figures/mlp_figures.py            # every figure
    python scripts/figures/mlp_figures.py neuron     # one figure, by name

Writes assets/module-media/<B's figure name>.svg (diagrams and plots, vector)
and mnist-digits.png (the MNIST digits, raster at 2x). Every figure shares one width,
one set of font sizes and one meaning per colour, so text reads at the same
size across the page:

    grey   (MUTED)  the input
    blue   (BLUE)   hidden units and the curves being explained
    clay   (BRAND)  the output, and the one point the caption talks about

Text is Inter, the reader's font; maths is Computer Modern so that symbols look
like the KaTeX in the lesson text.

The digits figure needs MNIST. It reads torchvision's copy from $MNIST_ROOT
(the folder that holds MNIST/raw) and otherwise $MNIST_NPZ, an .npz with
x_train and y_train in MNIST's original order. No data is kept in this repo.
"""
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
import figlib  # noqa: E402
from figlib import BLUE, GRID, INK, MUTED  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "assets" / "module-media"

BRAND = "#b84e30"       # the site's terracotta accent (src/styles/main.css --brand)
BLUE_SOFT = "#dbeafe"   # Tailwind blue-100, the light tint of BLUE (kept visible on white)
BRAND_SOFT = "#f6e9e4"  # a light tint of BRAND
WIRE = "#cbd5e1"        # slate-300, connections between neurons
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
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "svg.fonttype": "path",      # text as outlines: identical in every browser
            "svg.hashsalt": "mlp",       # stable ids, so re-rendering gives the same file
            "figure.constrained_layout.use": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def tent(x):
    return 2 * relu(x) - 4 * relu(x - 0.5)


def plot_axes(ax):
    """A light plot frame: left and bottom spines only, a faint grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color="#9ca3af")


def node(ax, x, y, r, color, text="", fill="white", lw=1.6, size=12, z=3):
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=color, lw=lw, zorder=z))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, zorder=z + 1)


def arrow(ax, p, q, color=MUTED, lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=2))


def digits_data():
    root = os.environ.get("MNIST_ROOT")
    if root:
        from torchvision import datasets

        train = datasets.MNIST(root, train=True, download=False)
        x, y = train.data.numpy(), train.targets.numpy()
    else:
        d = np.load(os.environ["MNIST_NPZ"])
        x, y = d["x_train"], d["y_train"]
    return np.stack([x[[np.where(y == k)[0][row] for k in range(10)]] for row in range(2)])


# ----------------------------------------------------------------- figures ---
def digits():
    """The first two training examples of each digit."""
    imgs = digits_data()
    ink = LinearSegmentedColormap.from_list("ink", ["#ffffff", INK])
    f, axes = plt.subplots(2, 10, figsize=(W, 1.72))
    f.subplots_adjust(left=0, right=1, top=0.86, bottom=0, wspace=0.08, hspace=0.06)
    for r in range(2):
        for k in range(10):
            ax = axes[r, k]
            ax.imshow(imgs[r, k], cmap=ink, vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
                s.set_linewidth(0.8)
            if r == 0:
                ax.set_title(str(k), fontsize=10, color=MUTED, fontweight="medium", pad=4)
    return f


def neuron():
    """One neuron, with the worked numbers from the text beside the symbols."""
    xlim, ylim = (-0.1, 8.75), (-0.7, 3.3)
    unit = 0.62   # inches per diagram unit: the circles stay small next to 10 pt text
    f = plt.figure(figsize=(unit * (xlim[1] - xlim[0]), unit * (ylim[1] - ylim[0])))
    ax = f.add_axes([0, 0, 1, 1])
    ys = [2.3, 1.3, 0.3]
    x_in, x_sum, x_act, x_out = 0.9, 3.9, 5.9, 7.35
    vals, ws = ["1.0", "0.5", "0.0"], ["3", "-2", "1"]
    for i, (y, v, w) in enumerate(zip(ys, vals, ws), start=1):
        node(ax, x_in, y, 0.3, MUTED, f"$x_{i}$")
        ax.text(x_in - 0.45, y, v, ha="right", va="center", fontsize=10, color=MUTED)
        end = (x_sum - 0.46 * np.cos(np.arctan2(1.3 - y, x_sum - x_in)),
               1.3 - 0.46 * np.sin(np.arctan2(1.3 - y, x_sum - x_in)))
        arrow(ax, (x_in + 0.3, y), end)
        t = 0.42
        ax.text(x_in + 0.3 + t * (end[0] - x_in - 0.3), y + t * (end[1] - y) + 0.13,
                f"$w_{i} = {w.replace('-', '{-}')}$", ha="center", va="bottom", fontsize=11, color=INK)
    node(ax, x_sum, 1.3, 0.46, BLUE, r"$\Sigma$", fill=BLUE_SOFT, size=13)
    arrow(ax, (x_sum, -0.55), (x_sum, 1.3 - 0.46))
    ax.text(x_sum + 0.14, -0.42, "$b = {-}1$", ha="left", va="bottom", fontsize=11, color=INK)
    arrow(ax, (x_sum + 0.46, 1.3), (x_act - 0.46, 1.3))
    ax.text((x_sum + x_act) / 2, 1.42, "$z = 1$", ha="center", va="bottom", fontsize=11, color=INK)
    node(ax, x_act, 1.3, 0.46, BLUE, r"$\sigma$", fill=BLUE_SOFT, size=14)
    arrow(ax, (x_act + 0.46, 1.3), (x_out - 0.12, 1.3))
    ax.text(x_out, 1.3, "$a = 0.73$", ha="left", va="center", fontsize=11, color=BRAND)
    for x, text in [(x_in, "inputs"), (x_sum, "weighted sum"), (x_act, "activation"), (x_out + 0.55, "output")]:
        ax.text(x, 2.95, text, ha="center", va="bottom", fontsize=10, color=MUTED)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return f


def sigmoid_curve():
    f, ax = plt.subplots(figsize=(W, 2.7))
    z = np.linspace(-8, 8, 600)
    ax.plot(z, sigmoid(z), color=BLUE, lw=2.4, solid_capstyle="round", zorder=3)
    s1 = sigmoid(1)
    ax.plot([1, 1], [0, s1], color=BRAND, lw=1, ls=(0, (2, 2)), zorder=2)
    ax.plot([-8, 1], [s1, s1], color=BRAND, lw=1, ls=(0, (2, 2)), zorder=2)
    ax.scatter([1], [s1], s=34, color=BRAND, zorder=5, edgecolor="white", linewidth=1.2)
    ax.text(1.45, s1 - 0.045, r"$\sigma(1) = 0.73$", ha="left", va="top", fontsize=11, color=BRAND)
    ax.scatter([0], [0.5], s=22, color=BRAND, zorder=5, edgecolor="white", linewidth=1)
    ax.text(-0.4, 0.53, r"$\sigma(0) = 0.5$", ha="right", va="bottom", fontsize=11, color=BRAND)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.08, 1.1)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"weighted sum $z$")
    ax.set_ylabel(r"output $\sigma(z)$")
    plot_axes(ax)
    return f


def activations():
    z = np.linspace(-4, 4, 600)
    curves = [
        ("sigmoid", sigmoid(z), (-0.15, 1.15), [0, 0.5, 1]),
        ("tanh", np.tanh(z), (-1.2, 1.2), [-1, 0, 1]),
        ("ReLU", relu(z), (-0.6, 4.2), [0, 2, 4]),
    ]
    f, axes = plt.subplots(1, 3, figsize=(W, 2.35))
    f.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, wspace=0.32)
    for ax, (name, y, lim, ticks) in zip(axes, curves):
        ax.axhline(0, color="#9ca3af", lw=0.8)
        ax.axvline(0, color="#9ca3af", lw=0.8)
        ax.plot(z, y, color=BLUE, lw=2.4, solid_capstyle="round", zorder=3)
        ax.set_title(name, color=INK)
        ax.set_xlim(-4, 4)
        ax.set_ylim(*lim)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks(ticks)
        ax.set_xlabel("$z$")
        plot_axes(ax)
    return f


def network():
    """784-16-16-10, with the input drawn as the digit it came from."""
    xlim, ylim = (-0.15, 8.0), (-2.45, 1.85)
    f = plt.figure(figsize=(W, W * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
    ax = f.add_axes([0, 0, 1, 1])
    xs = [1.9, 3.85, 5.8, 7.6]
    gap = 0.205

    def column(n, x):
        return [(x, (n - 1) / 2 * gap - i * gap) for i in range(n)]

    # Inputs: 784 of them, drawn as 7 dots, a gap, and 7 more.
    inp = [(xs[0], 1.6 - i * gap) for i in range(7)] + [(xs[0], -0.37 - i * gap) for i in range(7)]
    h1, h2, out = column(16, xs[1]), column(16, xs[2]), column(10, xs[3])
    wires = [[p, q] for left, right in [(inp, h1), (h1, h2), (h2, out)] for p in left for q in right]
    ax.add_collection(LineCollection(wires, colors=WIRE, linewidths=0.35, alpha=0.75, zorder=1))
    for x, y in inp:
        node(ax, x, y, 0.062, MUTED, lw=1.1)
    ax.text(xs[0], 0.02, r"$\vdots$", ha="center", va="center", fontsize=13, color=MUTED)
    for x, y in h1 + h2:
        node(ax, x, y, 0.07, BLUE, fill=BLUE_SOFT, lw=1.2)
    for k, (x, y) in enumerate(out):
        node(ax, x, y, 0.08, BRAND, fill=BRAND_SOFT, lw=1.2)
        ax.text(x + 0.2, y, str(k), ha="left", va="center", fontsize=9, color=BRAND)

    # The digit, and an arrow into the input column.
    img = digits_data()[0, 7]
    ink = LinearSegmentedColormap.from_list("ink", ["#ffffff", INK])
    ax.imshow(img, cmap=ink, vmin=0, vmax=255, extent=(-0.05, 1.05, -0.55, 0.55),
              interpolation="nearest", zorder=2)
    ax.add_patch(plt.Rectangle((-0.05, -0.55), 1.1, 1.1, fill=False, ec=GRID, lw=0.8, zorder=3))
    arrow(ax, (1.2, 0), (xs[0] - 0.2, 0))
    ax.text(0.5, -0.72, r"$28 \times 28$", ha="center", va="top", fontsize=10, color=MUTED)

    labels = [("input $\\mathbf{x}$", "784 pixels", MUTED), ("hidden $\\mathbf{a}_1$", "16", BLUE),
              ("hidden $\\mathbf{a}_2$", "16", BLUE), ("scores $\\mathbf{s}$", "10", BRAND)]
    for x, (top, count, c) in zip(xs, labels):
        ax.text(x, -1.8, top, ha="center", va="top", fontsize=10, color=INK)
        ax.text(x, -2.12, count, ha="center", va="top", fontsize=10, color=c)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return f


def regions():
    """Five ReLU units on two inputs: 16 regions, a flat piece of plane in each."""
    rng = np.random.default_rng(27)
    Wt = rng.standard_normal((5, 2))
    b = rng.uniform(-0.6, 0.6, 5)
    c = rng.standard_normal(5)
    x1 = np.linspace(-1, 1, 601)
    x2 = np.linspace(-1, 1, 601)
    X1, X2 = np.meshgrid(x1, x2)
    Z = X1[..., None] * Wt[:, 0] + X2[..., None] * Wt[:, 1] + b
    out = relu(Z) @ c
    on = Z > 0
    patterns = np.unique(on.reshape(-1, 5), axis=0)
    parity = on.sum(-1) % 2    # neighbouring regions differ in exactly one unit

    f, ax = plt.subplots(figsize=(4.6, 4.6))
    # Raster fill: its edges sit under the heavy lines, so a pixel grid never shows.
    ax.imshow(parity, extent=(x1[0], x1[-1], x2[0], x2[-1]), origin="lower", interpolation="nearest",
              cmap=matplotlib.colors.ListedColormap(["white", BLUE_SOFT]), zorder=0, aspect="auto")
    ax.contour(X1, X2, out, levels=18, colors="#9ca3af", linewidths=0.6, linestyles="solid", zorder=1)
    for (w1, w2), bi in zip(Wt, b):
        if abs(w2) > abs(w1):
            ax.plot(x1, -(w1 * x1 + bi) / w2, color=BLUE, lw=2.0, zorder=3)
        else:
            ax.plot(-(w2 * x2 + bi) / w1, x2, color=BLUE, lw=2.0, zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$", rotation=0, labelpad=10)
    for s in ax.spines.values():
        s.set_color("#9ca3af")
    ax.tick_params(length=3, color="#9ca3af")
    print(f"    regions: {len(patterns)} activation patterns in view")
    return f


def sawtooth():
    x = np.linspace(0, 1, 4097)
    f, axes = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
    f.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2, wspace=0.12)
    y = x
    for k, ax in enumerate(axes, start=1):
        y = tent(y)
        ax.plot(x, y, color=BLUE, lw=2.0, solid_joinstyle="miter", zorder=3)
        knots = np.arange(2 ** k + 1) / 2 ** k
        ax.scatter(knots, tent_k(knots, k), s=14, color=BRAND, zorder=4, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{k} layer{'s' if k > 1 else ''}, {2 ** k} pieces", color=INK)
        ax.set_xlabel("$x$")
        ax.set_xticks([0, 0.5, 1], ["0", r"$\frac{1}{2}$", "1"])
        ax.set_yticks([0, 1])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.08, 1.15)
        plot_axes(ax)
    axes[0].set_ylabel(r"$t^{(k)}(x)$")
    return f


def tent_k(x, k):
    for _ in range(k):
        x = tent(x)
    return x


FIGURES = {
    "mnist-digits": (digits, "png"),
    "neuron": (neuron, "svg"),
    "sigmoid": (sigmoid_curve, "svg"),
    "activation-functions": (activations, "svg"),
    "mlp-network": (network, "svg"),
    "relu-regions": (regions, "svg"),
    "relu-sawtooth": (sawtooth, "svg"),
}


def size_svg(path):
    """Writes the display size into the SVG in CSS px (matplotlib writes pt).

    The page shows a figure at its own width and height, never larger, so every
    figure gets the same px-per-point scale and its text the same size.
    """
    text = path.read_text()
    m = re.search(r'<svg[^>]*?width="([\d.]+)pt" height="([\d.]+)pt"', text)
    w, h = float(m.group(1)) * SCALE, float(m.group(2)) * SCALE
    text = text.replace(m.group(0), m.group(0).replace(f'width="{m.group(1)}pt"', f'width="{w:.0f}"')
                        .replace(f'height="{m.group(2)}pt"', f'height="{h:.0f}"'), 1)
    path.write_text(text)
    return w, h


def main(names):
    setup()
    out_dir = Path(os.environ.get("MLP_FIG_OUT", MEDIA))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names or FIGURES:
        fn, ext = FIGURES[name]
        f = fn()
        path = out_dir / f"{name}.{ext}"
        if ext == "svg":
            f.savefig(path, facecolor="white", metadata={"Date": None})
            w, h = size_svg(path)
        else:
            # 2x the display scale, so it is sharp on retina screens and its
            # text matches the SVGs once the page halves it.
            f.savefig(path, facecolor="white", dpi=2 * SCALE * 72)
            from PIL import Image

            w, h = (v / 2 for v in Image.open(path).size)
        plt.close(f)
        print(f"{path.name:22s} {w:4.0f} x {h:3.0f} px  {path.stat().st_size / 1024:4.0f} KB")


if __name__ == "__main__":
    main(sys.argv[1:])
