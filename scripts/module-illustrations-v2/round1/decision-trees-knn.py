"""The module's example: eight points (teal triangles class 0, terracotta dots class 1) and the query (4, 2).
The dashed circle reaches its three nearest neighbors (one dot, two triangles, so k = 3 votes class 0 while k = 1
says class 1). The vertical line at x1 = 3.5 is the Gini-best first split, its two halves shaded by majority class."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

c0 = [(4, 5), (5, 1), (6, 4), (6, 2)]
c1 = [(1, 3), (4, 1), (1, 4), (3, 6)]
q = (4, 2)
X0, X1, Y0, Y1 = 40, 560, 36, 364
ylo, yhi = -0.55, 6.55                    # equal scale on both axes so the circle is a true distance circle
unit = (Y1 - Y0) / (yhi - ylo)
xlo, xhi = 3.5 - (X1 - X0) / 2 / unit, 3.5 + (X1 - X0) / 2 / unit
def sx(v): return X0 + (v - xlo) * unit
def sy(v): return Y1 - (v - ylo) * unit
pts = [(p, 0) for p in c0] + [(p, 1) for p in c1]
d = sorted(math.dist(p, q) for p, _ in pts)
r3 = d[2]
# Gini-best threshold on x1
def gini(g):
    if not g: return 0
    f = sum(g) / len(g); return 2 * f * (1 - f)
best = None
xs = sorted({p[0] for p, _ in pts})
for a, c in zip(xs, xs[1:]):
    t = (a + c) / 2
    L = [l for p, l in pts if p[0] <= t]; R = [l for p, l in pts if p[0] > t]
    g = round((len(L) * gini(L) + len(R) * gini(R)) / len(pts), 12)   # 3.5 and 4.5 tie; keep the first, as the lesson does
    if best is None or g < best[0]: best = (g, t)
t = best[1]
b = rect(X0, Y0, sx(t) - X0, Y1 - Y0, ACCENT_SOFT, "none", 0, 0)
b += rect(sx(t), Y0, X1 - sx(t), Y1 - Y0, SECOND_SOFT, "none", 0, 0)
b += line(sx(t), Y0, sx(t), Y1, INK, 2.8)
b += f'<circle cx="{sx(q[0]):.1f}" cy="{sy(q[1]):.1f}" r="{r3 * unit + 14:.1f}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="7 6"/>'
def tri(x, y, s=12):
    return poly([(x, y - s), (x + s * 0.95, y + s * 0.7), (x - s * 0.95, y + s * 0.7)], SECOND, 1, fill=SECOND, close=True)
for p in c0: b += tri(sx(p[0]), sy(p[1]))
for p in c1: b += dot(sx(p[0]), sy(p[1]), 10, ACCENT)
x, y, s = sx(q[0]), sy(q[1]), 10
b += line(x - s, y - s, x + s, y + s, INK, 3.4) + line(x - s, y + s, x + s, y - s, INK, 3.4)
write(b)
