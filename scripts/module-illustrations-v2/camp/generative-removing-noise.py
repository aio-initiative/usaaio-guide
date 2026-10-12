"""Generation on the running example, run with the lesson's exact best-possible network: the estimate of the
top-left pixel as the step counts down from 1000 to 0, for start A (terracotta) and start B (teal). Both begin as
the grey blend 0.5 and fork, A to 1 (LEFT) and B to 0 (RIGHT). The two pure-noise starts sit at the left as 2 x 2
tiles, and the images each one ends at sit at the right."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

T = 1000
alpha_bar, a = [], 1.0
for i in range(T):
    a *= 1 - (1e-4 + (0.02 - 1e-4) * i / (T - 1))
    alpha_bar.append(a)
abar = lambda t: 1.0 if t == 0 else alpha_bar[t - 1]
LEFT, RIGHT = [1.0, 0, 1, 0], [0, 1.0, 0, 1]

def estimate(x, t):
    d = [sum((xi - math.sqrt(abar(t)) * mi) ** 2 for xi, mi in zip(x, m)) / (2 * (1 - abar(t))) for m in (LEFT, RIGHT)]
    w = [math.exp(-(di - min(d))) for di in d]
    return [(w[0] * l + w[1] * r) / sum(w) for l, r in zip(LEFT, RIGHT)]

def run(x):
    pts = []
    for t in range(T, 0, -100):
        e0 = estimate(x, t)
        pts.append((t, e0[0]))
        eps = [(xi - math.sqrt(abar(t)) * ei) / math.sqrt(1 - abar(t)) for xi, ei in zip(x, e0)]
        x = [math.sqrt(abar(t - 100)) * ei + math.sqrt(1 - abar(t - 100)) * pi for ei, pi in zip(e0, eps)]
    return pts + [(0, x[0])], x

startA, startB = [0.19, -0.52, -0.41, -2.44], [1.05, 1.78, -2.55, -0.14]
pa, endA = run(startA)
pb, endB = run(startB)

def tile(x, y, c, vals, frame=INK, noise=False):
    out = ""
    for k, v in enumerate(vals):
        px, py = x + (k % 2) * c, y + (k // 2) * c
        out += rect(px, py, c, c, BG, GUIDE, THIN, 0)
        s = max(0.0, min(1.0, (v + 2.6) / 5.2 if noise else v))    # noise spans about -2.6..2.6
        if s > 0.01: out += f'<rect x="{px:.1f}" y="{py:.1f}" width="{c:.1f}" height="{c:.1f}" fill="{INK}" fill-opacity="{s:.2f}"/>'
    return out + rect(x, y, 2 * c, 2 * c, "none", frame, 2.6, 0)

C = 26
PX0, PX1, PY0, PY1 = 150, 448, 70, 330      # time runs left to right from step 1000 down to 0
tx = lambda t: PX0 + (PX1 - PX0) * (T - t) / T
vy = lambda v: PY1 - (PY1 - PY0) * v

b = axes(PX0 - 10, PY0 - 20, PX1 + 10, PY1 + 20)
for v in (0, 0.5, 1):
    b += line(PX0, vy(v), PX1, vy(v), GUIDE, THIN, "4 5")

# pure-noise starts on the left, feeding the shared grey blend
for vals, ty, col in ((startA, vy(0.5) - 2 * C - 34, ACCENT), (startB, vy(0.5) + 34, SECOND)):
    b += tile(40, ty, C, vals, col, noise=True)
    b += arrow(40 + 2 * C + 8, ty + C, PX0 - 14, vy(0.5) + (-10 if col == ACCENT else 10), col, 2.2)

for pts, col in ((pa, ACCENT), (pb, SECOND)):
    b += poly([(tx(t), vy(v)) for t, v in pts], col, 3.4)
    for t, v in pts[:-1]:
        b += dot(tx(t), vy(v), 5.5, col)
b += dot(tx(T), vy(0.5), 8, BG, INK, 2.6)

# where each start ends: LEFT and RIGHT
for end, col in ((endA, ACCENT), (endB, SECOND)):
    yv = vy(end[0])
    b += arrow(PX1 + 14, yv, W - 40 - 2 * C - 12, yv, col, 2.2)
    b += tile(W - 40 - 2 * C, yv - C, C, end, col)
write(b)
