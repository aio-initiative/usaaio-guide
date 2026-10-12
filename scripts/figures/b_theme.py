"""Site A's look, as a drop-in replacement for site B's `usaaio_theme`.

Site B draws every figure through one shared style module, `tools/diagrams/
usaaio_theme.py`, and every generator imports it and nothing else for colour,
type, stroke or size. That means a rollout of the other 93 modules only needs
this file: same names, same signatures, A's look behind them.

    # in a B generator, unchanged:
    from usaaio_theme import P, arrow, clean, label, render, run_all

The runner (`scripts/import-b/figures.py`) puts this module into
`sys.modules["usaaio_theme"]` before importing a B generator, so the generator
never sees B's own theme and B is never written to.

What changes, and why:

  Light only. B saves each figure twice, light and `-dark`, on a transparent
  background so it can sit on either surface. A has one light theme
  (PLAN decision 7), so this module draws one pass on white and never writes a
  `-dark` file.

  Colour. B is monochrome by design: a greyscale ladder plus one teal `signal`
  for the single mark a figure is about. A's own figures (`figlib.py`,
  `mlp_figures.py`) use a blue primary for the thing being explained, grey for
  what recedes, and the terracotta brand colour for the point the caption talks
  about. The mapping below carries B's ladder onto A's, so a B figure comes out
  reading like an A figure without anyone redrawing it.

  Type. B sets Geist at 12/10.5/9.5/8.5 pt for figures displayed about 640 px
  wide. A's lessons set Inter and show figures 740 px wide, so this module uses
  Inter and normalises every figure to one display width (`PX` below), which
  keeps text the same size from one figure to the next the way B intended.

  Format. B writes transparent PNG at 200 DPI. A serves SVG for diagrams and
  2x PNG for dense plots. This module picks per figure, at save time, by
  looking at what the figure actually contains.

  Output path. B writes to `static/diagrams/` inside its own repo. This module
  writes to `$AIO_FIG_OUT` (the runner sets it), or `assets/module-media/` by
  default, and never touches B.
"""

from __future__ import annotations

import functools
import os
import pathlib
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[2]
MEDIA = ROOT / "assets" / "module-media"
INTER = pathlib.Path("/usr/share/texmf-dist/fonts/opentype/public/inter")

# --------------------------------------------------------------------- type

# Points. B's four steps kept; the gaps between them are what reads as
# hierarchy at the 740 px the figure is shown at on A.
TITLE = 12.0    # the figure's own heading, when it has one
LABEL = 10.5    # axis labels, series names, the thing being pointed at
TICK = 9.5      # tick numbers, units, anything you read only if you look
SMALL = 8.5     # dense in-figure annotation (matrix cells, grid indices)

# ------------------------------------------------------------------ strokes

# Points. B's ladder kept as-is; it already matches A's own figure weights.
HAIR = 0.8      # construction: grid, guides, gridlines inside a schematic
THIN = 1.2      # secondary data, connectors between boxes, box outlines
LINE = 1.7      # a data series
EMPH = 2.4      # the one line the figure is about

# ------------------------------------------------------------------ sizes

# Inches, as in B. The runner normalises the display width afterwards, so a
# generator still picks a proportion here and not a pixel count.
SIZES = {
    "square": (4.4, 4.4),
    "wide": (6.0, 3.2),
    "banner": (7.2, 2.4),
    "tall": (4.0, 5.0),
    "tower": (4.6, 7.6),
    "block": (6.4, 4.8),
    "panel": (7.2, 3.2),
}

DPI = 200
PAD = 0.12

PX = int(os.environ.get("AIO_FIG_PX", "740"))

# A's own figures are all 7.2 in wide and show at `PX` CSS px, so there is one
# pt-to-CSS-px scale on the page and a 10 pt label is the same size in every
# figure. B's generators pick several nominal inch sizes (a 4.4 in "square"
# next to a 7.2 in "banner"), so this module keeps that one scale rather than
# stretching each figure to `PX` wide: a narrower figure is narrower on the
# page, and its text still matches the figure above and below it.
#
# A figure wider than the column is scaled to `MAX_PX`, and its labels shrink
# with it; any that fall below `MIN_LABEL_PX` are listed in the inventory for a
# module manager to look at.
REF_IN = 7.2
PX_PER_PT = PX / (REF_IN * 72)
PX2 = PX * 2    # raster figures are saved at 2x so they stay sharp on retina
MAX_PX = 760
SMALLEST_TEXT_PT = SMALL    # 8.5 pt: matrix cells, grid indices
MIN_LABEL_PX = 11           # the floor L0 set; below this a label is hard to read

# Category by shape first, then lightness: B's reason (greyscale, print,
# colour deficiencies) still holds in A's palette.
MARKERS = ["o", "^", "s", "D"]
DASHES = ["-", (0, (5, 2.2)), (0, (1.4, 1.8)), (0, (6, 2, 1.4, 2))]


# ------------------------------------------------------------------ palette

# One colour per role, in every figure. Two figures on one page must look like
# one set, which is a rule about roles before it is a rule about hex values:
#
#   blue      #2563eb   the main objects and the curves being explained.
#                       `P.emph()` and `P.line(rank=0)`, the heaviest marks.
#   terracotta #b84e30  the one highlighted result or point in a figure.
#                       `P.signal`. A figure that uses it twice is using it
#                       wrong, as B's own theme puts it.
#   dark grey #4b5563   neutral structure: arrows, box outlines, the second
#                       data series, body text. `series[1]`, `arrow()`, `box()`.
#   light grey #e5e7eb / #d8d8dd / #9ca3af
#                       gridlines, dashed construction guides, then the frame
#                       (spines and axes lines). Three steps, lightest first,
#                       and all behind the data.
#
# Type and strokes are fixed in the constants above and never re-scaled: every
# figure is drawn at one pt-to-CSS-px scale (`PX_PER_PT`), so a 10.5 pt label
# is the same size on the page in a 4.4 in square as in a 7.2 in banner.

class _Palette:
    """A's ink, in the shape of B's so a B generator needs no edit.

    B keeps a light and a dark instance and swaps them between passes. A has
    one light pass, so there is one instance and `P` points straight at it.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def line(self, rank=0, **kw):
        """A data series. `rank` 0 is the primary one, 1 and 2 recede."""
        out = dict(color=self.series[rank], lw=LINE, solid_capstyle="round")
        out.update(kw)
        return out

    def emph(self, **kw):
        """The one line the figure exists to show.

        B drew this in ink; A draws the explained curve in its blue primary
        (see `mlp_figures.py`, where every curve the lesson is about is blue
        and the heaviest line on the page is that blue).
        """
        out = dict(color=self.primary, lw=EMPH, solid_capstyle="round")
        out.update(kw)
        return out

    def guide(self, **kw):
        """Construction geometry: margins, projections, reference levels.

        Light grey, like the grid, and one step darker so a dashed guide is
        still findable; it is construction, never load-bearing.
        """
        out = dict(color=self.guide_colour, lw=THIN, ls=(0, (4, 3)))
        out.update(kw)
        return out

    def grid(self, **kw):
        out = dict(color=self.faint, lw=HAIR)
        out.update(kw)
        return out

    def trace(self, rank=0, **kw):
        """One line among several on the same axes.

        Shape and lightness separate the series, as in B, so the dash ladder
        comes along unchanged.
        """
        out = dict(color=self.series[rank], ls=DASHES[rank], lw=LINE,
                   solid_capstyle="round", dash_capstyle="round")
        out.update(kw)
        return out

    def wire(self, **kw):
        """A connector that is content, not construction.

        Lighter than B's third series step, to match the wiring weight A's own
        network figure uses; the lattice of edges should sit behind the nodes.
        """
        out = dict(color=self.wire_colour, lw=HAIR, alpha=0.8, zorder=1)
        out.update(kw)
        return out

    def dots(self, rank=0, **kw):
        """A scatter series. Shape carries the category, lightness reinforces."""
        out = dict(
            s=26,
            facecolor=self.series[rank] if rank == 0 else "none",
            edgecolor=self.series[rank],
            linewidths=THIN,
            marker=MARKERS[rank],
            zorder=3,
        )
        out.update(kw)
        return out

    def box(self, **kw):
        """A schematic block: outline only, never a filled slab."""
        out = dict(facecolor="none", edgecolor=self.series[1], linewidth=THIN)
        out.update(kw)
        return out


# B's greyscale ladder mapped onto A's. Each line says which B role it carries.
A = _Palette(
    ink="#111827",          # B ink #121212  -> A's --ink-2 family; all text
    body="#4b5563",         # B body #444444 -> A's --muted family; body text
    mute="#4b5563",         # B mute -> A's body grey, so small annotations read at page size
    faint="#e5e7eb",        # gridlines: A's lightest grey
    guide_colour="#d8d8dd", # construction dashes: A's --line, one step darker
    rule="#9ca3af",         # spines and axes lines, the frame only
    primary="#2563eb",      # A's blue: the thing the figure is explaining
    series=["#2563eb", "#4b5563", "#9ca3af"],   # B's 3-step ladder, in A
    signal="#b84e30",       # B signal #0F7363 -> A's --brand; the one answer
    soft="#dbeafe",         # A's blue-100: tints behind a region or a band
    wire_colour="#94a3b8",  # the lattice of edges in a network schematic
    surface="#ffffff",      # A's --paper
)

P = A


# -------------------------------------------------------------------- fonts

@functools.lru_cache(maxsize=1)
def _font() -> str | None:
    """Inter, the face A's lessons and figures are set in. None if absent."""
    if not INTER.exists():
        return None
    import matplotlib.font_manager as fm

    for face in ("Regular", "Italic", "Medium", "SemiBold", "Bold"):
        f = INTER / f"Inter-{face}.otf"
        if f.exists():
            fm.fontManager.addfont(str(f))
    return "Inter"


@functools.lru_cache(maxsize=1)
def _init() -> None:
    family = ([_font()] if _font() else []) + [
        "Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans",
    ]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family,
        "font.size": LABEL,
        "axes.titlesize": TITLE,
        "axes.titleweight": "normal",
        "axes.labelsize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": TICK,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": HAIR,
        "xtick.major.width": HAIR,
        "ytick.major.width": HAIR,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.antialiased": True,
        "patch.antialiased": True,
        "axes.unicode_minus": True,
        "svg.fonttype": "path",       # outlines: identical in every browser
        "svg.hashsalt": "aio",        # stable ids, so a re-render is a no-op
        "mathtext.fontset": "cm",     # symbols match the page's KaTeX
        "figure.facecolor": P.surface,
        "axes.facecolor": P.surface,
        "savefig.facecolor": P.surface,
        "savefig.bbox": "tight",
        "savefig.pad_inches": PAD,
    })


# ------------------------------------------------------------------ helpers

def clean(ax, spines: str = "none") -> None:
    """Strip an axes to its content. B's signature, A's frame weights."""
    keep = {"lb": {"left", "bottom"}}.get(spines, set())
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
        spine.set_color(P.rule)
    if not keep:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(colors=P.mute, labelsize=TICK)
    ax.xaxis.label.set_color(P.body)
    ax.yaxis.label.set_color(P.body)
    ax.title.set_color(P.ink)


def arrow(ax, start, end, *, emph: bool = False, curve: float = 0.0, **kw):
    """The only arrow in B's codebase, kept exactly as it is."""
    style = dict(
        arrowstyle="-|>,head_length=5,head_width=2.6",
        color=P.primary if emph else P.series[1],
        lw=EMPH if emph else THIN,
        shrinkA=2,
        shrinkB=2,
        joinstyle="round",
        capstyle="round",
        connectionstyle=f"arc3,rad={curve}",
        zorder=4,
    )
    style.update(kw)
    patch = FancyArrowPatch(start, end, **style)
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, rank: str = "label", halo: bool = False, **kw):
    """In-figure text. `rank` is one of title/label/tick/small."""
    size, color = {
        "title": (TITLE, P.ink),
        "label": (LABEL, P.ink),
        "tick": (TICK, P.body),
        "small": (SMALL, P.mute),
    }[rank]
    kw.setdefault("color", color)
    kw.setdefault("fontsize", size)
    kw.setdefault("fontweight", "normal")
    if halo:
        from matplotlib import patheffects

        kw.setdefault("path_effects",
                      [patheffects.withStroke(linewidth=3.2, foreground=P.surface)])
        kw.setdefault("zorder", 6)
    return ax.text(x, y, text, **kw)


def block(ax, x, y, w, h, text, *, emph: bool = False, rank: str = "tick", **kw):
    """A labelled box in a schematic. Outline and a label, never a tint."""
    style = dict(
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor="none",
        edgecolor=P.primary if emph else P.series[1],
        linewidth=LINE if emph else THIN,
        zorder=3,
    )
    style.update(kw)
    ax.add_patch(FancyBboxPatch((x, y), w, h, **style))
    label(ax, x + w / 2, y + h / 2, text, rank=rank, ha="center", va="center",
          color=P.ink, zorder=4)


def classes(ax, X, y, *, size: float = 7):
    """A two-class scatter that stays out of the way of a boundary on it."""
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=size, marker="o", facecolor=P.ink,
               edgecolor="none", zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=size + 1, marker="s", facecolor="none",
               edgecolor=P.series[1], linewidths=0.7, zorder=3)


def region(ax, xx, yy, Z, *, level: float = 0.5):
    """Shade where a classifier predicts class 1 and draw its boundary.

    B shaded in its third grey step; A's own decision figures shade in a light
    blue tint, so the class region reads as the same colour as the boundary.
    """
    side = (Z > level).astype(float)
    ax.contourf(xx, yy, side, levels=[0.5, 1.5], colors=[P.soft], alpha=1.0, zorder=1)
    ax.contour(xx, yy, Z, levels=[level], colors=P.ink, linewidths=THIN, zorder=2)


def ink_cmap():
    """Page colour at 0 and full ink at 1, so a digit needs no border box."""
    return LinearSegmentedColormap.from_list("ink", [P.surface, P.ink])


def _dense(fig) -> bool:
    """Does this figure hold photographic or dense raster content?

    The split between SVG and 2x PNG is made here, from what the figure drew,
    instead of from a hand-kept list. A two-colour mask stays vector (A's own
    region figure is); anything with a real colour ramp, or a grid big enough
    that the vector file costs more than it is worth, is rasterised.
    """
    for ax in fig.axes:
        for im in ax.get_images():
            arr = np.asarray(im.get_array())
            pixels = int(np.prod(arr.shape[:2])) if arr.ndim >= 2 else 1
            ncolours = getattr(im.get_cmap(), "N", 256)
            if ncolours > 2 or pixels >= 250_000:
                return True
    return False


# ----------------------------------------------------------------- writing

def out_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("AIO_FIG_OUT", str(MEDIA)))
    d.mkdir(parents=True, exist_ok=True)
    return d


# What the last `save` wrote, keyed by file name: the display size, the scale
# applied to fit the column, and the smallest label in CSS px. A runner reads
# this to report per figure; nothing on the page does.
last: dict[str, dict] = {}


def _fit(css_w: float, css_h: float) -> tuple[int, int, float]:
    """CSS px already at the one scale, shrunk only past the column.

    The caller converts what was drawn to CSS px once: an SVG's viewBox in
    points times `PX_PER_PT`, or a PNG's pixels divided by two (the page's own
    rule). Returns the size written and the scale applied, which is 1.0 unless
    the figure was too wide and everything in it, text included, shrank.
    """
    w = max(1, round(css_w))
    h = max(1, round(css_h))
    shrink = 1.0
    if w > MAX_PX:
        shrink = MAX_PX / w
        w, h = MAX_PX, max(1, round(css_h * shrink))
    return w, h, shrink


def _size_svg(path: pathlib.Path) -> tuple[int, int, float]:
    """Write the display size into the SVG in CSS px, from what was drawn.

    matplotlib writes `width="432pt"` from the *nominal* figure size, but
    `bbox_inches="tight"` crops the canvas to the content, so two figures of
    the same nominal size can draw very different amounts. Sizing from the
    nominal value would scale one up and the other down, and the text with
    them. The viewBox is what was actually drawn, so the width comes from it.

    Returns the display size written, and the scale factor applied to the
    drawing (1.0 unless the figure had to be shrunk to fit the column).
    """
    import re

    text = path.read_text()
    m = re.search(r'<svg[^>]*?width="([\d.]+)pt" height="([\d.]+)pt"', text)
    if not m:
        # Already in px or a shape this module did not write; leave it alone.
        n = re.search(r'<svg[^>]*?width="(\d+)" height="(\d+)"', text)
        if not n:
            return 0, 0, 1.0
        return int(n.group(1)), int(n.group(2)), 1.0
    drawn_w_pt, drawn_h_pt = float(m.group(1)), float(m.group(2))
    w, h, shrink = _fit(drawn_w_pt * PX_PER_PT, drawn_h_pt * PX_PER_PT)
    repl = (m.group(0)
            .replace(f'width="{m.group(1)}pt"', f'width="{w}"')
            .replace(f'height="{m.group(2)}pt"', f'height="{h}"'))
    path.write_text(text.replace(m.group(0), repl, 1))
    return w, h, shrink


def _fit_png(path: pathlib.Path) -> tuple[int, int, float]:
    """A raster figure shows at 2x the CSS px of the equivalent SVG, so the
    page's "PNG pixels divided by 2" rule gives the same width for both.

    `bbox_inches="tight"` trims a little more or less than expected, which is
    why the saved size is corrected instead of assumed. Oversized figures are
    shrunk to the column like the SVG path, and the same scale is returned so
    the caller can report what happened to the text.
    """
    try:
        from PIL import Image
    except ImportError:
        return 0, 0, 1.0
    im = Image.open(path)
    px_w, px_h = im.size
    # The page shows a PNG at pixels / 2, so that is the CSS px directly.
    w, h, shrink = _fit(px_w / 2, px_h / 2)
    if (px_w, px_h) != (w * 2, h * 2):
        im = im.resize((w * 2, h * 2), Image.LANCZOS)
        im.save(path)
    return w, h, shrink


def save(fig, name: str, suffix: str = "") -> pathlib.Path:
    """One figure to disk, in A's look. `suffix` is accepted and ignored.

    B's signature takes a suffix because it writes a light and a dark variant.
    A has one light pass, so the parameter stays for compatibility and the
    dark file is never written.
    """
    prefix = os.environ.get("AIO_FIG_PREFIX", "")
    if _dense(fig):
        path = out_dir() / f"{prefix}{name}.png"
        fig.savefig(path, facecolor=P.surface, dpi=PX2 / REF_IN)
        w, h, shrink = _fit_png(path)
    else:
        path = out_dir() / f"{prefix}{name}.svg"
        fig.savefig(path, facecolor=P.surface, metadata={"Date": None})
        w, h, shrink = _size_svg(path)
    smallest = SMALLEST_TEXT_PT * PX_PER_PT * shrink
    last[path.name] = {"width": w, "height": h, "shrink": round(shrink, 3),
                       "smallest_label_px": round(smallest, 1)}
    plt.close(fig)
    return path


def render(name: str, size: str = "wide", *, axes: bool = True):
    """Decorate a draw function the way B's `usaaio_theme.render` does.

    The decorated function receives an `ax` (or a `fig` when `axes=False`) and
    draws. It must not save, must not close, and must not name a colour that
    is not on `P`.
    """

    def decorate(draw):
        @functools.wraps(draw)
        def run():
            _init()
            if axes:
                fig, ax = plt.subplots(figsize=SIZES[size])
                draw(ax)
            else:
                fig = plt.figure(figsize=SIZES[size])
                draw(fig)
            path = save(fig, name)
            print(f"  {path.name}")
            return path

        run.figure_name = name
        return run

    return decorate


def run_all(namespace: dict) -> list[pathlib.Path]:
    """Call every `@render`-decorated function in a module's globals.

    `gen_step_figures.py` passes a filtered dict; both shapes work because only
    the `figure_name` marker is used, exactly as in B.
    """
    figures = [v for v in namespace.values()
               if callable(v) and hasattr(v, "figure_name")]
    paths = []
    for fn in figures:
        paths.append(fn())
        print(f"  {fn.figure_name}")
    print(f"{len(figures)} figure(s) -> {out_dir()}")
    return paths


if __name__ == "__main__":
    # A quick smoke test: draw one schematic and one plot in A's look.
    import numpy as np

    @render("b-theme-smoke-plot", size="wide")
    def _plot(ax):
        z = np.linspace(-8, 8, 400)
        ax.plot(z, 1 / (1 + np.exp(-z)), **P.emph())
        ax.scatter([1], [1 / (1 + np.exp(-1))], s=42, color=P.signal, zorder=5)
        ax.set_xlabel("weighted sum  $z$")
        clean(ax, spines="lb")

    @render("b-theme-smoke-diagram", size="banner")
    def _diagram(ax):
        block(ax, 0.05, 0.35, 0.2, 0.3, "input")
        arrow(ax, (0.27, 0.5), (0.45, 0.5))
        block(ax, 0.47, 0.35, 0.2, 0.3, "layer", emph=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    run_all(globals())
