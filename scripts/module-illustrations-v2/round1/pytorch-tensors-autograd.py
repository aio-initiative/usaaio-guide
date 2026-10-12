"""Autograd on the lesson's example y = (x ** 2).sum() with x = [1, 2, 3].

Forward (ink, left to right): three leaf tensors, each squared, then summed into one scalar. Every leaf box shows
its value as an ink bar. Backward (terracotta, right to left): the seed gradient leaves the scalar, passes through
the sum unchanged (three equal thin arrows) and is multiplied by 2x at each square, so the arrows into the leaves
widen with 2, 4, 6. Each leaf's .grad is the terracotta bar beside its value bar, exactly twice as tall.
Square nodes carry a small true parabola; the sum node stacks the squares 1, 4, 9 into one bar.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

x = [1.0, 2.0, 3.0]
sq = [v * v for v in x]
y = sum(sq)
grad = [2 * v for v in x]               # dy/dx_i, what x.grad holds after y.backward()

LX, LY = 92, [82, 200, 318]             # leaf centres
SX = 268                                 # square nodes
PX, PY = 432, 200                        # sum node
OX = 548                                 # output scalar
LW, LH, R = 74, 82, 28
U = 10.5                                 # px per unit in the leaf bars
OFF = 9                                  # forward above, backward below each edge


def edge(x1, y1, x2, y2, r1, r2, fwd_w, back_w):
    """Forward arrow along the edge shifted to one side, backward arrow shifted to the other."""
    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    nx, ny = uy, -ux
    a = (x1 + ux * r1, y1 + uy * r1)
    b = (x2 - ux * r2, y2 - uy * r2)
    s = arrow(a[0] + nx * OFF, a[1] + ny * OFF, b[0] + nx * OFF, b[1] + ny * OFF, INK, fwd_w, 13)
    s += arrow(b[0] - nx * OFF, b[1] - ny * OFF, a[0] - nx * OFF, a[1] - ny * OFF, ACCENT, back_w,
               back_w * 1.6 + 9)
    return s


body = ""
# Edges first so nodes sit on top.
for i in range(3):
    body += edge(LX, LY[i], SX, LY[i], LW / 2 + 6, R + 6, 2.4, 1.6 * grad[i])
    body += edge(SX, LY[i], PX, PY, R + 6, 40, 2.4, 2.2)
body += edge(PX, PY, OX, PY, 40, 18, 2.4, 2.2)

# Leaves: value bar (ink) and gradient bar (terracotta).
for i in range(3):
    cx, cy = LX, LY[i]
    body += rect(cx - LW / 2, cy - LH / 2, LW, LH, BG, INK, 2.6, 10)
    base = cy + LH / 2 - 10
    body += rect(cx - 19, base - U * x[i], 15, U * x[i], INK, "none", 0, 2)
    body += rect(cx + 4, base - U * grad[i], 15, U * grad[i], ACCENT, "none", 0, 2)

# Square nodes with a true y = x^2 glyph and a dot at this node's input.
for i in range(3):
    cx, cy = SX, LY[i]
    body += ring(cx, cy, R, INK, 2.6)
    pts = [(cx + 15 * t, cy + 11 - 22 * t * t) for t in [k / 20 - 1 for k in range(41)]]
    body += poly(pts, GREY, 2.2)
    t = x[i] / 3
    body += dot(cx + 15 * t, cy + 11 - 22 * t * t, 4.2, INK)

# Sum node: the three squares stacked into one bar.
body += rect(PX - 30, PY - 44, 60, 88, ACCENT_SOFT, INK, 2.6, 12)
k = 70 / y
yy = PY + 35
for v, c in zip(sq, (INK, GREY, INK)):
    body += rect(PX - 17, yy - k * v + 1.5, 34, k * v - 3, c, "none", 0, 2)
    yy -= k * v

# The scalar output, and the seed gradient around it.
body += dot(OX, PY, 11, INK)
body += ring(OX, PY, 19, ACCENT, 3).replace(f'fill="{BG}"', 'fill="none"')
write(body)
