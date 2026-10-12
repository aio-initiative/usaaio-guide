"""Measured slope against exact slope. Left: e^(-t) near t = 1 with the chain-rule tangent there (terracotta, slope
-0.368) and the finite-difference secant through t = 1 and t + h = 1.8 (grey dashed, slope -0.253, a wider step than
the lesson's 0.5 so the gap shows at card size) cutting across the curve. Right: for the lesson's untrained 13-weight tanh network (rng 0) at t = 1, the error of the forward difference
(y(t+h) - y(t)) / h against step h on log-log axes (h from 1e-15 to 1, error from 1e-18 to 1): it falls as h shrinks,
then rises jaggedly once rounding takes over. The terracotta line at the bottom is the chain-rule (automatic
differentiation) slope's error, eps |dy/dt|, which has no step at all; the grey dot marks the best step."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np

# left: tangent vs secant on e^(-t)
LX0, LX1, LY0, LY1 = 50, 270, 60, 340
T0, T1, Y0_, Y1_ = 0.0, 2.2, 0.0, 1.05
lx = lambda t: LX0 + (LX1 - LX0) * (t - T0) / (T1 - T0)
ly = lambda y: LY1 - (LY1 - LY0) * (y - Y0_) / (Y1_ - Y0_)
b = axes(LX0 - 8, LY0 - 10, LX1 + 10, LY1)
ts = np.linspace(T0, T1, 200)
b += poly([(lx(t), ly(math.exp(-t))) for t in ts], INK, 3.0)
t0, h = 1.0, 0.8
y0, y1 = math.exp(-t0), math.exp(-t0 - h)
sec = (y1 - y0) / h
ta, tb = 0.3, 1.98
b += line(lx(ta), ly(y0 + sec * (ta - t0)), lx(tb), ly(y0 + sec * (tb - t0)), GREY, 2.6, dash="8 7")
b += line(lx(ta), ly(y0 - y0 * (ta - t0)), lx(tb), ly(y0 - y0 * (tb - t0)), ACCENT, 3.6)
b += line(lx(t0), ly(y0), lx(t0), LY1, GUIDE, THIN, dash="4 5")
b += line(lx(t0 + h), ly(y1), lx(t0 + h), LY1, GUIDE, THIN, dash="4 5")
b += ring(lx(t0 + h), ly(y1), 7, GREY, 2.6)
b += dot(lx(t0), ly(y0), 8, ACCENT)

# right: forward-difference error against h for the untrained network
p = np.random.default_rng(0).normal(0, 1, 13)
w, bb, v, c = p[0:4], p[4:8], p[8:12], p[12]
net = lambda t: np.tanh(w * t + bb) @ v + c
slope = (w * (1 - np.tanh(w + bb) ** 2)) @ v
RX0, RX1, RY0, RY1 = 330, 560, 50, 350
E0, E1 = -18.0, 0.0          # log10 error range
H0, H1 = -15.0, 0.0          # log10 step range
rx = lambda e: RX0 + (RX1 - RX0) * (e - H0) / (H1 - H0)
ry = lambda e: RY1 - (RY1 - RY0) * (e - E0) / (E1 - E0)
b += axes(RX0 - 8, RY0 - 10, RX1 + 12, RY1)
pts = []
for e in np.linspace(H0, H1, 110):
    hh = 10 ** e
    err = abs((net(1.0 + hh) - net(1.0)) / hh - slope)
    pts.append((rx(e), ry(math.log10(max(err, 1e-18)))))
b += poly(pts, GREY, 2.4)
ad = math.log10(np.finfo(float).eps * abs(slope))
b += line(RX0, ry(ad), RX1, ry(ad), ACCENT, 4.2)
best = max(pts, key=lambda q: q[1])
b += dot(best[0], best[1], 6, GREY)
write(b)
