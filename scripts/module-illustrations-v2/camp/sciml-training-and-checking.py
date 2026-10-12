"""Training, then checking, for the lesson's 13-weight tanh network on dy/dt = -y, y(0) = 1, collocation points
t = 0, 0.25, ..., 2. Left: the physics loss on a log scale against L-BFGS iteration (SciPy, from rng 0's starting
weights), falling from 23.5 to about 2e-7 over 100 iterations, with ink dots at iterations 0, 1, 5, 20, 50 and 100.
Right: the trained network's error |y(t) - e^(-t)| on a linear scale for t from 0 to 4, the training range 0 to 2
shaded teal; the terracotta error stays flat on the floor inside the range and climbs outside it, with rings at the
checking points t = 0, 0.5, 1, 1.5, 2 (grey) and t = 3, 4 (terracotta)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
from scipy.optimize import minimize

T = np.linspace(0, 2, 9)
def net(p, t):
    w, b, v, c = p[0:4], p[4:8], p[8:12], p[12]
    h = np.tanh(np.multiply.outer(t, w) + b)
    return h @ v + c, (1 - h ** 2) @ (v * w)
def loss(p):
    y, dy = net(p, T)
    return np.mean((dy + y) ** 2) + (net(p, 0.0)[0] - 1) ** 2
p0 = np.random.default_rng(0).normal(0, 1, 13)
hist = [loss(p0)]
p = minimize(loss, p0, method="L-BFGS-B", callback=lambda x: hist.append(loss(x))).x

# left: loss against iteration, log scale
LX0, LX1, LY0, LY1 = 56, 272, 56, 344
L0, L1 = -7.2, 1.6
n = len(hist) - 1
lx = lambda i: LX0 + (LX1 - LX0) * i / n
ly = lambda l: LY1 - (LY1 - LY0) * (math.log10(l) - L0) / (L1 - L0)
b = axes(LX0 - 10, LY0 - 16, LX1 + 14, LY1 + 10)
for e in range(-7, 2, 2):
    b += line(LX0, ly(10 ** e), LX1, ly(10 ** e), GUIDE, 1.0, dash="3 6")
b += poly([(lx(i), ly(l)) for i, l in enumerate(hist)], INK, 3.2)
for i in (0, 1, 5, 20, 50, n):
    b += dot(lx(i), ly(hist[i]), 6.5, INK)

# right: error against t, training range shaded
RX0, RX1, RY0, RY1 = 330, 560, 56, 344
EMAX = 0.0034
rx = lambda t: RX0 + (RX1 - RX0) * t / 4
ry = lambda e: RY1 - (RY1 - RY0) * e / EMAX
b += rect(rx(0), RY0 - 16, rx(2) - rx(0), RY1 - RY0 + 16, fill=SECOND_SOFT, stroke="none", rx=0)
b += line(rx(2), RY0 - 16, rx(2), RY1, SECOND, 2.0, dash="6 6")
b += axes(RX0 - 10, RY0 - 16, RX1 + 14, RY1 + 10)
ts = np.linspace(0, 4, 400)
err = np.abs(net(p, ts)[0] - np.exp(-ts))
b += poly([(rx(t), ry(e)) for t, e in zip(ts, err)], ACCENT, 4.0)
for t in (0, 0.5, 1, 1.5, 2):
    e = abs(net(p, np.array([t]))[0][0] - math.exp(-t))
    b += ring(rx(t), ry(e), 7.5, GREY, 2.6)
for t in (3, 4):
    e = abs(net(p, np.array([t]))[0][0] - math.exp(-t))
    b += line(rx(t), RY1, rx(t), ry(e), ACCENT, 1.6, dash="4 5")
    b += ring(rx(t), ry(e), 8.5, ACCENT, 3.0)
write(b)
