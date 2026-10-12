"""Shared style for the module pictures (owner, 2026-10-12): the textbook figure look with no text.

White card, crisp ink lines, light grey axes and guides, one terracotta accent with a soft terracotta fill, and
a muted teal used sparingly as a second series. No text, numbers or letters anywhere: the diagram must speak alone.
Every curve and position is computed from the real formula of the module's idea.
Each module has scripts/module-illustrations-v2/<section>/<id>.py that calls write(svg_body) to produce
assets/module-illustrations/<section>/<id>.svg. Run it with python3 from the repo root.
"""
import math, os, sys

W, H = 600, 400                 # 3:2 canvas, shown at about 260 x 173 on module cards
BG = "#ffffff"
FRAME = "#e3e5e8"               # 1px card border
INK = "#1f2328"                 # main lines and points
GREY = "#8b929a"                # secondary lines
GUIDE = "#d4d8dd"               # axes, grids, dashed guides
ACCENT = "#b84e30"              # terracotta, the one accent
ACCENT_SOFT = "#f6e4dd"         # region fills
SECOND = "#2f7f78"              # muted teal, a second series when two are needed
SECOND_SOFT = "#e2f0ee"
LINE = 2.4                      # main stroke
THIN = 1.4                      # guides and axes


def path(pts):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def line(x1, y1, x2, y2, color=INK, sw=LINE, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"{d}/>'


def poly(pts, color=INK, sw=LINE, fill="none", dash=None, close=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{path(pts)}{" Z" if close else ""}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{d}/>'


def dot(x, y, r=7, fill=INK, stroke="none", sw=0):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def ring(x, y, r=7, color=INK, sw=2.2):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{BG}" stroke="{color}" stroke-width="{sw}"/>'


def rect(x, y, w, h, fill="none", stroke=INK, sw=LINE, rx=4):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2, color=INK, sw=LINE, size=None):
    size = size or sw * 4 + 5
    a = math.atan2(y2 - y1, x2 - x1)
    sx, sy = x2 - size * 0.7 * math.cos(a), y2 - size * 0.7 * math.sin(a)
    p1 = (x2 - size * math.cos(a - 0.42), y2 - size * math.sin(a - 0.42))
    p2 = (x2 - size * math.cos(a + 0.42), y2 - size * math.sin(a + 0.42))
    return (line(x1, y1, sx, sy, color, sw) +
            f'<path d="M{x2:.1f},{y2:.1f} L{p1[0]:.1f},{p1[1]:.1f} L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{color}"/>')


def axes(x0, y0, x1, y1, ticks=0):
    """Plain L-shaped axes from (x0, y1) bottom-left to x1 and up to y0, optional tick marks, no labels."""
    out = line(x0, y1, x1, y1, GUIDE, THIN) + line(x0, y1, x0, y0, GUIDE, THIN)
    for i in range(1, ticks + 1):
        tx = x0 + (x1 - x0) * i / (ticks + 1)
        out += line(tx, y1, tx, y1 + 6, GUIDE, THIN)
    return out


def write(body, defs=""):
    """Write the SVG for the calling script: scripts/module-illustrations-v2/<section>/<id>.py -> assets/..."""
    here = os.path.abspath(sys.argv[0])
    sec, mid = os.path.basename(os.path.dirname(here)), os.path.splitext(os.path.basename(here))[0]
    root = os.path.abspath(os.path.join(os.path.dirname(here), "..", "..", ".."))
    out = os.path.join(root, "assets", "module-illustrations", sec, mid + ".svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<defs>{defs}</defs><rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="10" fill="{BG}" stroke="{FRAME}"/>'
           f'{body}</svg>\n')
    if "<text" in svg:
        raise SystemExit("No text allowed in module pictures")
    open(out, "w").write(svg)
    print(out)
