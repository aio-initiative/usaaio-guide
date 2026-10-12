"""The lesson's worked example as convergence curves: f(x) = (x1^2 + 25 x2^2)/2 from (1, 1), loss on a log scale
against iterations. Gradient descent with eta = 0.09 diverges (grey dashed, rising), with eta = 0.04 it falls as a
straight line of slope 0.96^2 and crosses the dashed 1e-6 target at step 161 (ink), Nesterov with beta = 2/3
crosses at step 40 (teal), and Newton solves the quadratic in one step, dropping straight to the floor (terracotta)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

f = lambda x: (x[0] ** 2 + 25 * x[1] ** 2) / 2
grad = lambda x: (x[0], 25 * x[1])
T = 175


def gd(eta, steps):
    x, out = (1.0, 1.0), [f((1.0, 1.0))]
    for _ in range(steps):
        g = grad(x)
        x = (x[0] - eta * g[0], x[1] - eta * g[1])
        out.append(f(x))
    return out


def nesterov(eta, beta, steps):
    x = prev = (1.0, 1.0)
    out = [f(x)]
    for _ in range(steps):
        y = (x[0] + beta * (x[0] - prev[0]), x[1] + beta * (x[1] - prev[1]))
        g = grad(y)
        prev, x = x, (y[0] - eta * g[0], y[1] - eta * g[1])
        out.append(max(f(x), 1e-300))
    return out


X0, X1, Y0, Y1 = 52, 562, 40, 352
LTOP, LBOT = 3.0, -8.0                  # log10 range shown
tx = lambda t: X0 + 8 + t / T * (X1 - X0 - 8)
ly = lambda v: Y0 + (LTOP - math.log10(v)) / (LTOP - LBOT) * (Y1 - Y0)
target = 1e-6

defs = f'<clipPath id="pl"><rect x="{X0}" y="{Y0 - 6}" width="{X1 - X0 + 6}" height="{Y1 - Y0 + 6}"/></clipPath>'
body = axes(X0, Y0, X1, Y1)
# decade guides
for d in range(-7, 3, 3):
    body += line(X0, ly(10 ** d), X1, ly(10 ** d), GUIDE, 1.0, "2 6")
body += line(X0, ly(target), X1, ly(target), INK, 1.8, "9 7")

body += f'<g clip-path="url(#pl)">'
div = gd(0.09, 40)
body += poly([(tx(t), ly(v)) for t, v in enumerate(div)], GREY, 2.6, dash="10 8")
slow = gd(0.04, T)
body += poly([(tx(t), ly(v)) for t, v in enumerate(slow)], INK, 3.0)
fast = nesterov(0.04, 2 / 3, 70)
body += poly([(tx(t), ly(max(v, 10 ** (LBOT - 1)))) for t, v in enumerate(fast)], SECOND, 3.0)
body += "</g>"
# Newton: one step from f = 13 to exactly 0, straight down to the floor
body += arrow(tx(0), ly(13), tx(1), Y1 - 4, ACCENT, 3.2)

# where each method first reaches the target
for series, col in [(slow, INK), (fast, SECOND)]:
    k = next(t for t, v in enumerate(series) if v < target)
    body += dot(tx(k), ly(target), 7.5, col, BG, 2.5)
body += dot(tx(0), ly(13), 7.5, ACCENT)

write(body, defs)
