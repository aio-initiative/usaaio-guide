"""KL is not a distance: a two-bump p (grey fill), the single Gaussian that minimizes the forward KL(p || q)
(teal, mass-covering: it matches p's mean and variance and smears across the gap), and the one that minimizes
the reverse KL(q || p) (terracotta, mode-seeking: it sits narrow on one bump). Both fits are computed: the forward
one by moment matching, the reverse one by a grid search over the mean and standard deviation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np

def gauss(x, m, s): return np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))

xs = np.linspace(-6, 6, 1201); dx = xs[1] - xs[0]
p = 0.5 * gauss(xs, -2.0, 0.65) + 0.5 * gauss(xs, 2.0, 0.65)

# forward KL(p || q) over Gaussians q: match the mean and variance of p
mf = float((xs * p).sum() * dx); sf = math.sqrt(float(((xs - mf) ** 2 * p).sum() * dx))

# reverse KL(q || p): grid search
best = None
for m in np.linspace(0.5, 3.5, 61):
    for s in np.linspace(0.3, 2.5, 89):
        q = gauss(xs, m, s)
        kl = float((q * (np.log(q + 1e-300) - np.log(p + 1e-300))).sum() * dx)
        if best is None or kl < best[0]: best = (kl, m, s)
_, mr, sr = best

X0, X1, Y0, Y1 = 44, 556, 40, 352
ymax = gauss(0, 0, sr) * 1.08
def sx(v): return X0 + (v + 6) / 12 * (X1 - X0)
def sy(v): return Y1 - v / ymax * (Y1 - Y0)

body = axes(X0, Y0, X1, Y1)
body += line(sx(0), Y1, sx(0), Y0 + 20, GUIDE, THIN, dash="5 6")
pts = [(sx(x), sy(v)) for x, v in zip(xs, p)]
qr = gauss(xs, mr, sr)
qpts = [(sx(x), sy(v)) for x, v in zip(xs, qr)]
body += poly([(sx(xs[0]), Y1)] + qpts + [(sx(xs[-1]), Y1)], "none", 0, fill=ACCENT_SOFT, close=True)
body += f'<path d="{path([(sx(xs[0]), Y1)] + pts + [(sx(xs[-1]), Y1)])} Z" fill="{GREY}" fill-opacity="0.16"/>'
body += poly(pts, GREY, 2.6)
body += poly([(sx(x), sy(v)) for x, v in zip(xs, gauss(xs, mf, sf))], SECOND, 3.2)
body += poly(qpts, ACCENT, 3.2)
body += dot(sx(mf), Y1, 7, SECOND) + dot(sx(mr), Y1, 7, ACCENT)
write(body)
