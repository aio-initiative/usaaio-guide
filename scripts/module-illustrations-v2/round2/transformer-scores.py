"""One query against three keys: each key projected onto the query direction (the dot product), and the scores as bars,
the raw dot product outlined and the score after dividing by the square root of the head size filled."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

q = (1.020, 2.000)                                   # head-1 query of "sat"
keys = [(1.000, 0.000), (1.540, 1.010), (-0.416, 1.020)]   # the, cat, sat
d = 2
dots_ = [q[0] * k[0] + q[1] * k[1] for k in keys]
scores = [s / math.sqrt(d) for s in dots_]
best = max(range(3), key=lambda i: scores[i])

# Plane: origin and scale, equal on both axes so projections are true.
ox, oy, sc = 150, 330, 118
P = lambda v: (ox + v[0] * sc, oy - v[1] * sc)
body = line(ox - 100, oy, ox + 220, oy, GUIDE, THIN) + line(ox, oy + 30, ox, 40, GUIDE, THIN)
qn = math.hypot(*q)
u = (q[0] / qn, q[1] / qn)
# Query line extended.
body += line(*P((-0.2 * u[0], -0.2 * u[1])), *P((1.15 * qn * u[0], 1.15 * qn * u[1])), GUIDE, 1.4, dash="6 5")
for i, k in enumerate(keys):
    t = k[0] * u[0] + k[1] * u[1]                    # scalar projection onto the query direction
    foot = (t * u[0], t * u[1])
    c = ACCENT if i == best else GREY
    body += line(*P(k), *P(foot), c, 1.6, dash="4 4")
    body += dot(*P(foot), 4.5, c)
for i, k in enumerate(keys):
    c = ACCENT if i == best else INK
    body += arrow(*P((0, 0)), *P(k), c, 2.6 if i == best else 2.2)
body += arrow(*P((0, 0)), *P(q), SECOND, 3.4)
body += dot(ox, oy, 5, INK)

# Scores as bars: raw dot product outlined, scaled score filled.
bx, bmax = 390, 3.6
bscale = 160 / bmax
for i in range(3):
    y = 120 + i * 80
    body += rect(bx, y - 20, dots_[i] * bscale, 40, "none", GREY, 1.6, 3)
    body += rect(bx, y - 20, scores[i] * bscale, 40, ACCENT if i == best else "#c3c8ce", "none", 0, 3)
    body += line(bx + dots_[i] * bscale, y - 26, bx + dots_[i] * bscale, y + 26, GUIDE, 1.2)
body += line(bx, 70, bx, 330, INK, 2)
write(body)
