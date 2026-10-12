"""The gradient of a mean squared error: tilted elliptical loss contours of fitting y = t1 + t2 t to the points
(0, 1), (1, 3), (2, 4), the minimum at (7/6, 3/2), and minus the gradient at (1, 1), (4/3, 2), drawn
perpendicular to the contour through that point, with its tangent line and a right-angle mark."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

data = [(0, 1), (1, 3), (2, 4)]
L = lambda a, b: sum((y - a - b * t) ** 2 for t, y in data) / len(data)
grad = lambda a, b: (-2 / 3 * sum((y - a - b * t) for t, y in data), -2 / 3 * sum((y - a - b * t) * t for t, y in data))
opt = (7 / 6, 1.5)
Lmin = L(*opt)
A = [[1, 1], [1, 5 / 3]]           # L = Lmin + d^T A d with d = theta - opt

S = 150                            # px per unit, same on both axes so right angles look right
OX, OY = 300, 200                  # canvas position of the minimum
P = lambda a, b: (OX + (a - opt[0]) * S, OY - (b - opt[1]) * S)

# eigen-directions of A give the ellipse axes
tr, det = A[0][0] + A[1][1], A[0][0] * A[1][1] - A[0][1] ** 2
lams = [(tr + s * math.sqrt(tr * tr - 4 * det)) / 2 for s in (1, -1)]
vecs = []
for lam in lams:
    v = (A[0][1], lam - A[0][0])
    n = math.hypot(*v)
    vecs.append((v[0] / n, v[1] / n))


def contour(level):
    c = level - Lmin
    pts = []
    for k in range(181):
        u = 2 * math.pi * k / 180
        r0, r1 = math.sqrt(c / lams[0]) * math.cos(u), math.sqrt(c / lams[1]) * math.sin(u)
        pts.append(P(opt[0] + r0 * vecs[0][0] + r1 * vecs[1][0], opt[1] + r0 * vecs[0][1] + r1 * vecs[1][1]))
    return pts


start = (1, 1)
lv = L(*start)                     # 2/3
defs = f'<clipPath id="plot"><rect x="40" y="36" width="520" height="328"/></clipPath>'
body = f'<g clip-path="url(#plot)">'
for lev, col, sw in [(0.12, GREY, 1.8), (0.3, GREY, 1.8), (lv, INK, 2.6), (1.15, GREY, 1.8), (1.9, GUIDE, 1.8)]:
    body += poly(contour(lev), col, sw, close=True)
body += "</g>"
sx, sy = P(*start)
g = grad(*start)
d = (-g[0], -g[1])                 # (4/3, 2), the steepest-descent direction
n = math.hypot(*d)
u = (d[0] / n * S, -d[1] / n * S)  # unit direction in canvas px (y flipped)
ux, uy = u[0] / S, u[1] / S
# tangent line to the contour at the start: perpendicular to the gradient
body += line(sx - uy * 175, sy + ux * 175, sx + uy * 175, sy - ux * 175, ACCENT, 2.2, "8 7")
# right-angle mark
m = 20
body += poly([(sx + ux * m, sy + uy * m), (sx + ux * m + uy * m, sy + uy * m - ux * m), (sx + uy * m, sy - ux * m)], INK, 2.2)
body += arrow(sx, sy, sx + ux * 150, sy + uy * 150, ACCENT, 3.4)
body += dot(*P(*opt), 6, INK)
body += dot(sx, sy, 8, ACCENT)

write(body, defs)
