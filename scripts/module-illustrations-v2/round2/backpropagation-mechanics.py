"""The lesson's backward pass by hand, on the forward-propagation network (x = (1, 2), true class 0).
Left: the network, forward edges faint, with the errors flowing back as terracotta dashed arrows. The output
units hold delta2 = yhat - y = (-0.88, +0.88) (teal negative, terracotta positive); back through W2 the first
hidden unit is blocked by its ReLU, the third receives -0.88 + 0.88 = 0, and only the second gets delta1 = 0.88.
Right: dL/dW2 = delta2 a1^T as an outer-product grid. Its rows are headed by the output units themselves and its
columns by bars of the hidden activations a1 = (0, 2, 1); each cell's square has area |delta_i a_j|, so the
dead unit's column is empty."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

x = [1, 2]
W1 = [[1, -1], [0, 1], [1, 1]]
b1 = [0, 0, -2]
W2 = [[1, 0, 1], [0, 1, 1]]
z1 = [sum(W1[i][j] * x[j] for j in range(2)) + b1[i] for i in range(3)]
a1 = [max(0, v) for v in z1]
z2 = [sum(W2[i][j] * a1[j] for j in range(3)) for i in range(2)]
e = [math.exp(v) for v in z2]
yhat = [v / sum(e) for v in e]
d2 = [yhat[0] - 1, yhat[1]]
d1 = [sum(W2[i][j] * d2[i] for i in range(2)) * (1 if z1[j] > 0 else 0) for j in range(3)]
dW2 = [[d2[i] * a1[j] for j in range(3)] for i in range(2)]

R = 23
CELL = 62
GX, GY = 382, 206                        # top-left of the 2x3 gradient grid
rowc = [GY + CELL * (i + 0.5) for i in range(2)]
pos = [[(66, 160), (66, 285)], [(196, 95), (196, 215), (196, 335)], [(322, rowc[0]), (322, rowc[1])]]

body, defs = "", ""
for k, W in enumerate([W1, W2]):
    for i, row in enumerate(W):
        for j, w in enumerate(row):
            if w:
                (sx, sy), (tx, ty) = pos[k][j], pos[k + 1][i]
                body += line(sx, sy, tx, ty, GUIDE, 1.8)

# errors flowing back from the outputs along W2
for i in range(2):
    for j in range(3):
        if W2[i][j] == 0:
            continue
        (sx, sy), (tx, ty) = pos[2][i], pos[1][j]
        d = math.hypot(tx - sx, ty - sy)
        ux, uy = (tx - sx) / d, (ty - sy) / d
        blocked = z1[j] <= 0
        col = GREY if blocked else (SECOND if d2[i] < 0 else ACCENT)
        if blocked:
            body += line(sx + ux * R, sy + uy * R, tx - ux * (R + 4), ty - uy * (R + 4), col, 2, "6 6")
        else:
            body += line(sx + ux * R, sy + uy * R, tx - ux * (R + 14), ty - uy * (R + 14), col, 2.6, "7 6")
            body += arrow(tx - ux * (R + 15), ty - uy * (R + 15), tx - ux * (R + 3), ty - uy * (R + 3), col, 2.6)


def gauge(cx, cy, frac, col, cid):
    global defs
    out = dot(cx, cy, R, BG)
    if frac > 0:
        defs += f'<clipPath id="{cid}"><circle cx="{cx}" cy="{cy}" r="{R}"/></clipPath>'
        out += f'<rect x="{cx - R}" y="{cy + R - 2 * R * frac:.1f}" width="{2 * R}" height="{2 * R * frac:.1f}" fill="{col}" clip-path="url(#{cid})"/>'
    return out + f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{INK}" stroke-width="2.4"/>'


sign = lambda v: SECOND if v < 0 else ACCENT
for j, (cx, cy) in enumerate(pos[0]):
    body += gauge(cx, cy, 0, INK, f"i{j}")
for j, (cx, cy) in enumerate(pos[1]):
    body += gauge(cx, cy, abs(d1[j]), sign(d1[j]), f"h{j}")
    if z1[j] <= 0:                       # the ReLU is off: a small bar across the unit
        body += line(cx - R * 0.55, cy, cx + R * 0.55, cy, GREY, 2.6)
for i, (cx, cy) in enumerate(pos[2]):
    body += gauge(cx, cy, abs(d2[i]), sign(d2[i]), f"o{i}")
    body += line(cx + R + 4, cy, GX - 6, cy, GUIDE, 1.6, "4 5")

# column heads: bars of the hidden activations
for j in range(3):
    cx = GX + CELL * (j + 0.5)
    h = 44 * a1[j]
    if h:
        body += rect(cx - 13, GY - 12 - h, 26, h, INK, INK, 0, 2)
    else:
        body += line(cx - 13, GY - 12, cx + 13, GY - 12, GREY, 2.4)

# the outer-product grid
top = max(abs(v) for row in dW2 for v in row)
for i in range(2):
    for j in range(3):
        x0, y0 = GX + CELL * j, GY + CELL * i
        body += rect(x0, y0, CELL, CELL, "none", GUIDE, 1.6, 0)
        v = dW2[i][j]
        if v:
            s = (CELL - 10) * math.sqrt(abs(v) / top)
            body += rect(x0 + (CELL - s) / 2, y0 + (CELL - s) / 2, s, s, sign(v), sign(v), 0, 2)
body += rect(GX, GY, 3 * CELL, 2 * CELL, "none", INK, 2.2, 3)

write(body, defs)
