"""The module's running example: the vertical feature map (-3, 6, -3, 0 in every row; teal for negative,
terracotta for positive), ReLU turning every negative cell to 0 (a true max(0, x) hinge drawn over the arrow),
then 2 x 2 max pooling to the 2 x 2 map (6, 0 in each row). The top-left block and the cell it pools to are
outlined, with guide lines between them."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

fmap = [[-3.0, 6.0, -3.0, 0.0] for _ in range(4)]
relu = [[max(0.0, v) for v in r] for r in fmap]
pooled = [[max(relu[2 * i + a][2 * j + c] for a in range(2) for c in range(2)) for j in range(2)] for i in range(2)]
top = max(abs(v) for r in fmap for v in r)

def grid(x0, y0, c, vals):
    out = ""
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            x, y = x0 + j * c, y0 + i * c
            out += rect(x, y, c, c, BG, GUIDE, THIN, 0)
            if v: out += f'<rect x="{x+2:.1f}" y="{y+2:.1f}" width="{c-4:.1f}" height="{c-4:.1f}" fill="{ACCENT if v > 0 else SECOND}" fill-opacity="{abs(v)/top:.2f}"/>'
    return out + rect(x0, y0, len(vals) * c, len(vals) * c, "none", INK, 2, 0)

C = 42                                # same cell size throughout, so pooling visibly halves width and height
GAP = (W - 80 - 10 * C) / 2
x1 = 40
x2 = x1 + 4 * C + GAP
x3 = x2 + 4 * C + GAP
y0, ym = H / 2 - 2 * C, H / 2
yp = ym - C
b = grid(x1, y0, C, fmap) + grid(x2, y0, C, relu) + grid(x3, yp, C, pooled)
# 2 x 2 block boundaries on the ReLU map
b += line(x2 + 2 * C, y0, x2 + 2 * C, y0 + 4 * C, INK, 2.6) + line(x2, y0 + 2 * C, x2 + 4 * C, y0 + 2 * C, INK, 2.6)
# top-left block -> the pooled cell it becomes
b += line(x2 + 2 * C, y0, x3, yp, ACCENT, 1.8, "6 5") + line(x2 + 2 * C, y0 + 2 * C, x3, yp + C, ACCENT, 1.8, "6 5")
b += f'<rect x="{x2:.1f}" y="{y0:.1f}" width="{2*C:.1f}" height="{2*C:.1f}" fill="none" stroke="{ACCENT}" stroke-width="4" rx="2"/>'
b += f'<rect x="{x3:.1f}" y="{yp:.1f}" width="{C:.1f}" height="{C:.1f}" fill="none" stroke="{ACCENT}" stroke-width="4" rx="2"/>'
# ReLU step: arrow with a true max(0, x) hinge on small axes above it, the dashed band from block to cell is the pooling step
b += arrow(x1 + 4 * C + 8, ym, x2 - 8, ym, GREY, 2.4)
gx, gy, gs = x1 + 4 * C + GAP / 2, y0 - 50, 30
b += line(gx - gs, gy, gx + gs, gy, GUIDE, THIN) + line(gx, gy - gs, gx, gy + gs, GUIDE, THIN)
b += poly([(gx + u, gy - max(0.0, u)) for u in (-gs, 0, gs)], ACCENT, 3.6)
b += line(gx, gy + gs + 6, gx, ym - 14, GUIDE, THIN, "4 4")
write(b)
