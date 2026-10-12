"""The forward process on the running example: LEFT (1, 0, 1, 0) as a 2 x 2 tile at steps 0, 50, 200, 500 and 1000,
noised with the lesson's draw eps = (0.13, -0.13, 0.64, 0.10) through the one-jump formula under DDPM's schedule.
Below, the two weights of that formula over t: sqrt(alpha_bar) on the image (terracotta, falling) and
sqrt(1 - alpha_bar) on the noise (teal, rising). Dashed guides tie each tile to its step on the time axis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

T = 1000
abar, a = [1.0], 1.0
for i in range(T):
    a *= 1 - (1e-4 + (0.02 - 1e-4) * i / (T - 1))
    abar.append(a)

x0 = [1.0, 0.0, 1.0, 0.0]
eps = [0.13, -0.13, 0.64, 0.10]
steps = [0, 50, 200, 500, 1000]

def tile(cx, y, c, vals, frame=INK):
    out, x = "", cx - c
    for k, v in enumerate(vals):
        px, py = x + (k % 2) * c, y + (k // 2) * c
        out += rect(px, py, c, c, BG, GUIDE, THIN, 0)
        s = max(0.0, min(1.0, v))
        if s > 0.01: out += f'<rect x="{px:.1f}" y="{py:.1f}" width="{c:.1f}" height="{c:.1f}" fill="{INK}" fill-opacity="{s:.2f}"/>'
    return out + rect(x, y, 2 * c, 2 * c, "none", frame, 2.2, 0)

# plot area
PX0, PX1, PY0, PY1 = 56, 556, 196, 356
tx = lambda t: PX0 + (PX1 - PX0) * t / T
vy = lambda v: PY1 - (PY1 - PY0) * v

b = axes(PX0 - 8, PY0 - 10, PX1 + 12, PY1)
b += line(PX0 - 8, vy(1), PX1 + 12, vy(1), GUIDE, THIN, "4 5")

# tiles, evenly spaced across the top
C, TY = 30, 40
for i, t in enumerate(steps):
    cx = 40 + C + i * (W - 80 - 2 * C) / (len(steps) - 1)
    s, n = math.sqrt(abar[t]), math.sqrt(1 - abar[t])
    xt = [s * p + n * e for p, e in zip(x0, eps)]
    b += line(cx, TY + 2 * C + 8, tx(t), PY0 - 18, GUIDE, THIN, "5 5")
    b += line(tx(t), PY0 - 18, tx(t), PY1, GUIDE, THIN, "2 5")
    b += tile(cx, TY, C, xt, ACCENT if i == 0 else INK)
    if i < len(steps) - 1:
        nx = 40 + C + (i + 1) * (W - 80 - 2 * C) / (len(steps) - 1)
        b += arrow(cx + C + 10, TY + C, nx - C - 10, TY + C, GREY, 2)

sig = [(tx(t), vy(math.sqrt(abar[t]))) for t in range(0, T + 1, 5)]
noi = [(tx(t), vy(math.sqrt(1 - abar[t]))) for t in range(0, T + 1, 5)]
b += poly(sig + [(tx(T), vy(0)), (tx(0), vy(0))], "none", 0, ACCENT_SOFT, close=True)
b += poly(noi, SECOND, 3)
b += poly(sig, ACCENT, 3.2)
for t in steps:
    b += dot(tx(t), vy(math.sqrt(abar[t])), 5.5, ACCENT)
    b += dot(tx(t), vy(math.sqrt(1 - abar[t])), 5.5, SECOND)
write(b)
