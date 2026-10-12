"""KV cache: decoding "on" in head 1 of the running example.

The cache is two columns, keys and values, one row per earlier token, each cell drawn as its real 2-number vector.
The new token's query (used once, not stored) links to every cached key with a width set by its real softmax
weight; its own key and value join the cache as the new terracotta row, and the dashed slot below waits for the
next token. The weighted sum of the values leaves on the right as the output vector.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

KEYS = [(1.000, 0.000), (1.540, 1.010), (-0.416, 1.020), (-1.990, 0.030)]
VALS = [(1.000, 1.000), (0.841, 1.540), (0.909, -0.416), (-0.859, -1.990)]
Q = (0.030, 2.000)
s = [(k[0] * Q[0] + k[1] * Q[1]) / math.sqrt(2) for k in KEYS]
e = [math.exp(x) for x in s]
w = [x / sum(e) for x in e]
OUT = (sum(wi * v[0] for wi, v in zip(w, VALS)), sum(wi * v[1] for wi, v in zip(w, VALS)))

C = 72                     # cell size
GAP = 6
Y0 = 30
KX, VX = 238, 238 + C + 12
rows = [Y0 + i * (C + GAP) for i in range(5)]
SC = 15.5                  # px per unit of a vector


def vec(cx, cy, v, color, sw=3.0):
    return arrow(cx, cy, cx + v[0] * SC, cy - v[1] * SC, color, sw, 12)


body = ""
# Query box on the left.
QX, QY, QS = 40, 190 - 60, 120
qc = (QX + QS / 2, QY + QS / 2)
# Links from the query to every cached key, width by softmax weight.
for i, (k, wi) in enumerate(zip(KEYS, w)):
    ky = rows[i] + C / 2
    x0, y0 = QX + QS, qc[1]
    d = f"M{x0:.1f},{y0:.1f} C{x0 + 60:.1f},{y0:.1f} {KX - 60:.1f},{ky:.1f} {KX:.1f},{ky:.1f}"
    body += f'<path d="{d}" fill="none" stroke="{ACCENT if i == 3 else INK}" stroke-width="{1.2 + 15 * wi:.1f}" stroke-linecap="round" opacity="{0.85}"/>'

# Weighted values flowing to the output.
OX, OS = 450, 110
oc = (OX + OS / 2, 190)
for i, wi in enumerate(w):
    vy = rows[i] + C / 2
    x0 = VX + C
    d = f"M{x0:.1f},{vy:.1f} C{x0 + 45:.1f},{vy:.1f} {OX - 45:.1f},{oc[1]:.1f} {OX:.1f},{oc[1]:.1f}"
    body += f'<path d="{d}" fill="none" stroke="{GREY}" stroke-width="{1.2 + 15 * wi:.1f}" stroke-linecap="round"/>'

body += rect(QX, QY, QS, QS, ACCENT_SOFT, ACCENT, 2.6, 10)
body += vec(qc[0] - Q[0] * SC * 1.5 / 2, qc[1] + Q[1] * SC * 1.5 / 2, (Q[0] * 1.5, Q[1] * 1.5), ACCENT, 4.2)

# The cache: keys and values, one row per token; the new row in terracotta, an empty slot for the next token.
for i in range(4):
    y = rows[i]
    new = i == 3
    for X, v in ((KX, KEYS[i]), (VX, VALS[i])):
        body += rect(X, y, C, C, ACCENT_SOFT if new else BG, ACCENT if new else INK, 2.4, 6)
        body += vec(X + C / 2, y + C / 2, v, ACCENT if new else (SECOND if X == KX else INK))
for X in (KX, VX):
    body += rect(X, rows[4] + 4, C, 30, BG, GUIDE, 2.2, 6).replace("/>", ' stroke-dasharray="7 6"/>')
# Brace-like guide around the cache.
body += rect(KX - 9, Y0 - 9, 2 * C + 12 + 18, 4 * (C + GAP) + 12, "none", GUIDE, THIN, 12)

# Output box.
body += rect(OX, oc[1] - OS / 2, OS, OS, BG, INK, 2.6, 10)
body += vec(oc[0] - OUT[0] * SC * 3.2 / 2, oc[1] + OUT[1] * SC * 3.2 / 2, (OUT[0] * 3.2, OUT[1] * 3.2), INK, 4.2)
write(body)
