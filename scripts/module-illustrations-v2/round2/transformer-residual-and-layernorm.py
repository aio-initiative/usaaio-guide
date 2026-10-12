"""Add and normalize for "sat": the input H skips around the attention block on a bold terracotta residual arc
into the sum, the sum H + M rides high above the zero line (its mean dashed), and LayerNorm recentres it to Y,
mean zero and spread one (dashed +-1 guides)."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

H = [0.909, -0.416, 1.020, 2.000]
M = [0.872, 1.135, 0.893, 1.771]
S = [h + m for h, m in zip(H, M)]
mu = sum(S) / 4
sd = math.sqrt(sum((s - mu) ** 2 for s in S) / 4)
Yn = [(s - mu) / sd for s in S]

BASE, K, BW, GAP = 262, 46, 18, 6          # zero line, px per unit, bar width, gap


def bars(x0, vals, fill, scale=K):
    out = line(x0 - 8, BASE, x0 + 4 * BW + 3 * GAP + 8, BASE, GUIDE, THIN)
    for i, v in enumerate(vals):
        x = x0 + i * (BW + GAP)
        y0, y1 = sorted((BASE, BASE - v * scale))
        out += rect(x, y0, BW, max(y1 - y0, 1), fill, fill, 1, 2)
    return out


WV = 4 * BW + 3 * GAP
XH, XS, XY = 40, 338, 470
body = ""
# Input H.
body += bars(XH, H, INK)
# Attention block: a box with three queries mixing three keys inside.
BX, BY, BWD, BHT = 162, BASE - 64, 104, 128
body += rect(BX, BY, BWD, BHT, BG, INK, 2.4, 8)
ly = [BY + 30, BY + 62, BY + 94]
wts = [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.115, 0.709, 0.175]]
for a in range(3):
    for b in range(3):
        body += line(BX + 26, ly[a], BX + BWD - 26, ly[b], ACCENT if a == 2 and b == 1 else GREY,
                     1 + 5 * wts[a][b])
for yy in ly:
    body += dot(BX + 26, yy, 5.5, INK) + dot(BX + BWD - 26, yy, 5.5, INK)
body += arrow(XH + WV + 12, BASE, BX - 4, BASE, GREY, 2.2)
# Plus node.
PX = 300
body += arrow(BX + BWD + 4, BASE, PX - 17, BASE, GREY, 2.2)
body += ring(PX, BASE, 15, INK, 2.4)
body += line(PX - 8, BASE, PX + 8, BASE, INK, 2.4) + line(PX, BASE - 8, PX, BASE + 8, INK, 2.4)
# Residual arc: from the top of H, high over the block, down into the plus.
x0, ytop = XH + WV / 2, BASE - 2.0 * K - 14
pts = []
for t in [i / 60 for i in range(61)]:
    x = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * x0 + 3 * (1 - t) * t ** 2 * PX + t ** 3 * PX
    y = (1 - t) ** 3 * ytop + 3 * (1 - t) ** 2 * t * 18 + 3 * (1 - t) * t ** 2 * 18 + t ** 3 * (BASE - 19)
    pts.append((x, y))
body += poly(pts[:-3], ACCENT, 3.4)
body += arrow(pts[-6][0], pts[-6][1], PX, BASE - 16, ACCENT, 3.4)
# Sum H + M, with its mean dashed.
body += arrow(PX + 17, BASE, XS - 12, BASE, GREY, 2.2)
body += bars(XS, S, GREY)
body += line(XS - 8, BASE - mu * K, XS + WV + 8, BASE - mu * K, ACCENT, 1.8, "6 4")
# LayerNorm: recentred and rescaled, with dashed +-1 spread guides.
body += arrow(XS + WV + 14, BASE, XY - 14, BASE, ACCENT, 2.4)
for s in (1, -1):
    body += line(XY - 8, BASE - s * K, XY + WV + 8, BASE - s * K, GUIDE, 1.6, "5 4")
body += bars(XY, Yn, ACCENT)
write(body)
