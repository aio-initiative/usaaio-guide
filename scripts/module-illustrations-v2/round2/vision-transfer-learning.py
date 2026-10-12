"""The running example's linear probe. Three small images (vertical line, horizontal line, cross) go through the
kept backbone, drawn grey with a padlock because it is frozen: its two 3 x 3 filters (vertical and horizontal) and
their two totals. Only the new layer is trained: its 2 x 3 = 6 weights, learned here by the lesson's 100 softmax
steps, are drawn as edges whose width is the weight's size, terracotta when positive and dashed teal when negative,
to three outputs marked with the class shapes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
features = np.array([[12.0, 0.0], [0.0, 12.0], [10.0, 10.0]])
Wt = np.zeros((2, 3))
for _ in range(100):
    s = features @ Wt
    p = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    p[np.arange(3), np.arange(3)] -= 1
    Wt -= 0.01 * features.T @ p / 3

def glyph(cx, cy, kind, r, color=INK, sw=3):
    out = ""
    if kind in (0, 2): out += line(cx, cy - r, cx, cy + r, color, sw)
    if kind in (1, 2): out += line(cx - r, cy, cx + r, cy, color, sw)
    return out

body = ""
ys = [100, 200, 300]
# input images: framed squares with the line pattern
for k, y in enumerate(ys):
    body += rect(42, y - 26, 52, 52, BG, INK, 2, 3) + glyph(68, y, k, 18, INK, 5)
# frozen backbone
BX0, BX1 = 132, 300
body += rect(BX0, 52, BX1 - BX0, 296, "#f1f2f4", GREY, 2, 10)
for y in ys:
    body += arrow(100, y, BX0 - 6, y, GREY, 2)
# two 3 x 3 filters inside, vertical [-1, 2, -1] columns and its transpose
def kernel(x0, y0, vertical, c=20):
    out = ""
    for i in range(3):
        for j in range(3):
            v = 2 if (j == 1 if vertical else i == 1) else -1
            fill = INK if v > 0 else "#c9cdd2"
            out += f'<rect x="{x0 + j * c:.1f}" y="{y0 + i * c:.1f}" width="{c}" height="{c}" fill="{fill}" stroke="#f1f2f4" stroke-width="2"/>'
    return out
FY = [150, 250]
for k, fy in enumerate(FY):
    body += kernel(158, fy - 30, k == 0)
# padlock at the top of the backbone
lx, ly = (BX0 + BX1) / 2, 86
body += f'<path d="M{lx - 11},{ly} L{lx - 11},{ly - 9} A11,11 0 0 1 {lx + 11},{ly - 9} L{lx + 11},{ly}" fill="none" stroke="{GREY}" stroke-width="3.2"/>'
body += rect(lx - 17, ly - 1, 34, 26, GREY, "none", 0, 4) + dot(lx, ly + 10, 3.5, "#f1f2f4")
# the two totals (feature nodes), at the backbone's right edge
NX = 262
for k, fy in enumerate(FY):
    body += line(222, fy, NX - 14, fy, GREY, 2)
    body += dot(NX, fy, 14, BG, GREY, 2.6) + glyph(NX, fy, k, 7, GREY, 2.6)
# the new layer: six weights as edges
OX = 470
mx = np.abs(Wt).max()
for i, fy in enumerate(FY):
    for c, oy in enumerate(ys):
        w = Wt[i, c]
        sw = 1.5 + 6.5 * abs(w) / mx
        if w >= 0: body += line(NX + 14, fy, OX - 20, oy, ACCENT, sw)
        else: body += line(NX + 14, fy, OX - 20, oy, SECOND, sw, dash="9 7")
for c, oy in enumerate(ys):
    body += dot(OX, oy, 20, ACCENT_SOFT, ACCENT, 2.8) + glyph(OX, oy, c, 10, INK, 3.2)
# softmax probabilities of each image's own class, as small bars to the right
s = features @ Wt
P = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
for c, oy in enumerate(ys):
    body += rect(506, oy - 8, 54, 16, "#eef0f2", "none", 0, 3) + rect(506, oy - 8, 54 * P[c, c], 16, ACCENT, "none", 0, 3)
write(body)
