"""Mixture of experts: one token, a router, six experts, only the top two run.

The router's scores for one token pass through a real softmax; each expert's probability is drawn as a bar on its
own row. The two largest (terracotta) are kept, their weights renormalized to sum to one, and only those two expert
boxes light up and carry the token. The other four experts stay stored but idle (pale, dashed). The two outputs
merge into the layer output with stroke widths set by the renormalized gate weights.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

SCORES = [0.3, 1.9, -0.4, 0.6, 1.3, -0.9]
e = [math.exp(s) for s in SCORES]
g = [x / sum(e) for x in e]
top = sorted(range(len(g)), key=lambda i: -g[i])[:2]
wsum = sum(g[i] for i in top)
w = {i: g[i] / wsum for i in top}

N = len(g)
RH, GAP, Y0 = 50, 12, 30
rows = [Y0 + i * (RH + GAP) + RH / 2 for i in range(N)]
CY = (rows[0] + rows[-1]) / 2

body = ""
# Token.
TX, TS = 40, 76
body += rect(TX, CY - TS / 2, TS, TS, ACCENT_SOFT, ACCENT, 2.6, 10)
for k, h in enumerate((0.55, 0.9, 0.35, 0.7)):
    bx = TX + 14 + k * 13
    body += line(bx, CY + 22, bx, CY + 22 - 40 * h, ACCENT, 7)

# Router: a spine, then one softmax bar per expert row.
RX, BMAX = 160, 120
body += line(TX + TS, CY, RX - 14, CY, INK, LINE)
body += line(RX - 14, rows[0], RX - 14, rows[-1], GUIDE, THIN)
for i, y in enumerate(rows):
    on = i in top
    body += line(RX - 14, y, RX, y, GUIDE, THIN)
    body += rect(RX, y - 13, BMAX, 26, "#f1f2f4", "none", 0, 3)
    body += rect(RX, y - 13, max(6, BMAX * g[i] / max(g)), 26, ACCENT if on else GREY, "none", 0, 3)

# Experts: each a tiny two-layer network. Chosen ones are lit, the rest idle.
EX, EW = 330, 112
for i, y in enumerate(rows):
    on = i in top
    col = ACCENT if on else GUIDE
    box = rect(EX, y - RH / 2, EW, RH, ACCENT_SOFT if on else BG, col, 2.6 if on else 2.0, 8)
    if not on:
        box = box.replace("/>", ' stroke-dasharray="6 5"/>')
    body += box
    L = [(EX + 22, y - 12 + 12 * k) for k in range(3)]
    M = [(EX + EW / 2, y - 16 + 10.7 * k) for k in range(4)]
    R = [(EX + EW - 22, y - 12 + 12 * k) for k in range(3)]
    ink = INK if on else GUIDE
    for a in L:
        for b in M:
            body += line(*a, *b, ink, 1.0)
    for a in M:
        for b in R:
            body += line(*a, *b, ink, 1.0)
    for p in L + M + R:
        body += dot(*p, 3.2, ink)
    if on:
        body += arrow(RX + BMAX * g[i] / max(g) + 8, y, EX - 4, y, ACCENT, 2.6, 13)

# Output: the two chosen outputs merge, width by renormalized gate weight.
OX, OS = 500, 64
for i in top:
    y = rows[i]
    x0 = EX + EW
    d = f"M{x0:.1f},{y:.1f} C{x0 + 40:.1f},{y:.1f} {OX - 40:.1f},{CY:.1f} {OX:.1f},{CY:.1f}"
    body += f'<path d="{d}" fill="none" stroke="{ACCENT}" stroke-width="{2 + 12 * w[i]:.1f}" stroke-linecap="round"/>'
body += rect(OX, CY - OS / 2, OS, OS, BG, INK, 2.6, 10)
for k, h in enumerate((0.75, 0.4, 0.85)):
    bx = OX + 16 + k * 16
    body += line(bx, CY + 18, bx, CY + 18 - 36 * h, INK, 7)
write(body)
