"""A small tanh network as a function of time: the input t (left) fans out to four hidden units, each box showing its
own smooth step tanh(w t + b) over the plotted time range (same steepness w = 3, steps placed at t = -0.1, 0.45, 1.0
and 1.6), and the edges' weights v carry them to the output. On the right, the output c + sum v_i tanh(w_i t + b_i)
with v and c fitted by least squares to e^(-t) on [0, 2]: grey partial sums after one, two and three units, the heavy
terracotta full sum, and ink rings at the exact e^(-t) values t = 0, 0.5, 1, 1.5, 2 sitting on it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np

W_ = np.full(4, 3.0)
S = np.array([-0.1, 0.45, 1.0, 1.6])          # where each step sits
TMIN, TMAX = 0.0, 2.0
tt = np.linspace(TMIN, TMAX, 240)
A = np.column_stack([np.tanh(np.outer(tt, W_) - W_ * S), np.ones_like(tt)])
p, *_ = np.linalg.lstsq(A, np.exp(-tt), rcond=None)
V, C = p[:4], p[4]

# right plot
PX0, PX1, PY0, PY1 = 330, 560, 50, 350
YLO, YHI = 0.0, 1.15
sx = lambda t: PX0 + (PX1 - PX0) * (t - TMIN) / (TMAX - TMIN)
sy = lambda y: PY1 - (PY1 - PY0) * (y - YLO) / (YHI - YLO)
b = axes(PX0 - 8, PY0 - 10, PX1 + 12, PY1)

units = np.tanh(np.outer(tt, W_) - W_ * S)
# partial sums of the first k units, each shifted so it ends where the full curve ends (they meet at t = 2)
for k in (1, 2, 3):
    ys = units[:, :k] @ V[:k]
    ys = ys - ys[-1] + math.exp(-TMAX)
    b += poly([(sx(t), sy(y)) for t, y in zip(tt, ys)], GREY, 1.8, dash="6 6")
full = A @ p
b += poly([(sx(t), sy(y)) for t, y in zip(tt, full)], ACCENT, 4.4)
for t in (0, 0.5, 1, 1.5, 2):
    b += ring(sx(t), sy(math.exp(-t)), 7, INK, 2.6)

# network on the left
IX, IY = 52, 200
HX, BW, BH = 120, 104, 58
OX, OY = 290, 200
ys_h = [62, 154, 246, 338]
for i, hy in enumerate(ys_h):
    b += line(IX, IY, HX, hy, GREY, 2.0)
    sw = 1.6 + 9 * abs(V[i]) / abs(V).max()
    b += line(HX + BW, hy, OX, OY, ACCENT if i == 0 else GREY, sw)
for i, hy in enumerate(ys_h):
    x0, y0 = HX, hy - BH / 2
    b += rect(x0, y0, BW, BH, fill=BG, stroke=INK, sw=2.2, rx=8)
    gx = lambda t: x0 + 10 + (BW - 20) * (t - TMIN) / (TMAX - TMIN)
    gy = lambda u: hy - u * (BH / 2 - 9)
    b += line(x0 + 8, hy, x0 + BW - 8, hy, GUIDE, THIN)
    b += poly([(gx(t), gy(u)) for t, u in zip(tt, units[:, i])], SECOND, 3.0)
b += dot(IX, IY, 11, INK)
b += dot(OX, OY, 11, ACCENT)
write(b)
