"""Where each term of the heat-equation PINN loss is checked: the rod's position x across and time t upwards, with
grey contour arches of the true solution u = e^(-a pi^2 t) sin(pi x) (a = 0.5) cooling as t grows. Ink rings scattered
inside are the collocation points for the residual u_t - a u_xx, teal squares on the two ends are where u must be 0,
and terracotta triangles along the bottom are where u must equal the starting sine profile, drawn as the
terracotta hump under the domain."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np

A, TMAX = 0.5, 0.4
X0, X1, Y0, Y1 = 70, 530, 36, 268          # the (x, t) domain; t grows upwards
B0 = 360                                    # baseline of the starting-profile hump below
def sx(x): return X0 + x * (X1 - X0)
def sy(t): return Y1 - t / TMAX * (Y1 - Y0)

body = rect(X0, Y0, X1 - X0, Y1 - Y0, fill=ACCENT_SOFT, stroke="none", rx=0)
# level sets u = c: sin(pi x) = c e^(a pi^2 t), drawn as arches
ts = np.linspace(0, TMAX, 400)
for c in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
    s = c * np.exp(A * math.pi ** 2 * ts)
    ok = s <= 1
    tt = ts[ok]; xl = np.arcsin(s[ok]) / math.pi
    pts = [(sx(x), sy(t)) for x, t in zip(xl, tt)] + [(sx(1 - x), sy(t)) for x, t in zip(xl[::-1], tt[::-1])]
    body += poly(pts, "#ffffff", 2.0, close=False)
body += rect(X0, Y0, X1 - X0, Y1 - Y0, stroke=INK, sw=LINE, rx=0)

# interior collocation points (fixed pseudo-random)
rng = np.random.default_rng(3)
pts = []
while len(pts) < 26:
    p = rng.uniform([0.07, 0.06], [0.93, 0.95])
    if all((p[0] - q[0]) ** 2 + ((p[1] - q[1]) * 0.55) ** 2 > 0.012 for q in pts):
        pts.append(p)
for x, f in pts:
    body += ring(sx(x), sy(f * TMAX), 7, INK, 2.4)

# boundary squares on both ends
for i in range(6):
    t = (i + 0.5) / 6 * TMAX
    for x in (0, 1):
        body += rect(sx(x) - 8, sy(t) - 8, 16, 16, fill=SECOND, stroke="none", rx=2)

# starting-profile triangles along the bottom, and the sine hump below
xs = np.linspace(0, 1, 200)
hump = [(sx(x), B0 - 72 * math.sin(math.pi * x)) for x in xs]
body += line(X0, B0, X1, B0, GUIDE, THIN)
body += poly([(X0, B0)] + hump + [(X1, B0)], "none", 0, fill=ACCENT_SOFT, close=True)
body += poly(hump, ACCENT, 3.0)
for i in range(1, 8):
    x = i / 8
    cx, cy = sx(x), Y1
    body += poly([(cx, cy - 10), (cx - 9, cy + 7), (cx + 9, cy + 7)], "none", 0, fill=ACCENT, close=True)
    body += line(cx, cy + 12, cx, B0 - 72 * math.sin(math.pi * x) - 6, GREY, THIN, dash="3 5")
write(body)
