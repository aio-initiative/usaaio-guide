"""The lesson's forward pass by hand: x = (1, 2) through W1 = [[1,-1],[0,1],[1,1]], b1 = (0,0,-2), ReLU, then
W2 = [[1,0,1],[0,1,1]] and a softmax. Each unit is a gauge filled to the value it holds: inputs (teal) 1 and 2,
hidden activations 0, 2, 1 (the first unit's sum is -1, so ReLU leaves it empty), and output probabilities
0.12 and 0.88. Edges are the nonzero weights, width by size, dashed when negative; edges that carry a nonzero
product w * a are ink, the rest grey. A terracotta arrow underneath gives the direction of the pass."""
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
y = [v / sum(e) for v in e]

R = 30
cols = [110, 300, 490]
rows = [[145, 255], [90, 200, 310], [145, 255]]
pos = [[(cols[k], yy) for yy in rows[k]] for k in range(3)]
vals = [[v / 2 for v in x], [v / 2 for v in a1], y]          # gauge fill fractions

body, defs = "", ""
for k, (W, src) in enumerate([(W1, x), (W2, a1)]):
    for i, row in enumerate(W):
        for j, w in enumerate(row):
            if w == 0:
                continue
            (sx, sy), (tx, ty) = pos[k][j], pos[k + 1][i]
            d = math.hypot(tx - sx, ty - sy)
            ux, uy = (tx - sx) / d, (ty - sy) / d
            live = w * src[j] != 0 and (k == 1 or a1[i] > 0)
            body += line(sx + ux * R, sy + uy * R, tx - ux * R, ty - uy * R, INK if live else GUIDE,
                         1.4 + 1.2 * abs(w), "7 6" if w < 0 else None)

for k in range(3):
    for i, (cx, cy) in enumerate(pos[k]):
        f = vals[k][i]
        cid = f"u{k}{i}"
        defs += f'<clipPath id="{cid}"><circle cx="{cx}" cy="{cy}" r="{R}"/></clipPath>'
        body += dot(cx, cy, R, BG)
        if f > 0:
            col = SECOND if k == 0 else ACCENT
            body += f'<rect x="{cx - R}" y="{cy + R - 2 * R * f:.1f}" width="{2 * R}" height="{2 * R * f:.1f}" fill="{col}" clip-path="url(#{cid})"/>'
        body += ring(cx, cy, R, INK, 2.6).replace(f'fill="{BG}"', 'fill="none"')

body += arrow(cols[0] - 20, 370, cols[2] + 50, 370, ACCENT, 3)
write(body, defs)
