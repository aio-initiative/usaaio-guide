"""Least squares as geometry: b dropped perpendicularly onto the plane spanned by the columns of A."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

a1, a2, b = (1.0, 0.0, 0.18), (0.25, 1.0, 0.05), (0.95, 0.75, 1.05)
dotp = lambda u, v: sum(p * q for p, q in zip(u, v))
# normal equations A^T A x = A^T b for the two columns
g11, g12, g22 = dotp(a1, a1), dotp(a1, a2), dotp(a2, a2)
r1, r2 = dotp(a1, b), dotp(a2, b)
det = g11 * g22 - g12 * g12
c = (g22 * r1 - g12 * r2) / det
m = (g11 * r2 - g12 * r1) / det
p = tuple(c * u + m * v for u, v in zip(a1, a2))
res = tuple(q - s for q, s in zip(b, p))

az, el, S, O = math.radians(-32), math.radians(34), 1.0, (0, 0)
def P(v):
    x, y, z = v
    xr = x * math.cos(az) - y * math.sin(az)
    yr = x * math.sin(az) + y * math.cos(az)
    return (O[0] + S * xr, O[1] - S * (z * math.cos(el) + yr * math.sin(el)))

def comb(s, t):
    return tuple(s * u + t * v for u, v in zip(a1, a2))

s0, s1, t0, t1 = -0.35, 1.35, -0.3, 1.2
raw = [P(comb(u, v)) for u in (s0, s1) for v in (t0, t1)] + [P(b), P((0, 0, 0))]
x_lo, x_hi = min(q[0] for q in raw), max(q[0] for q in raw)
y_lo, y_hi = min(q[1] for q in raw), max(q[1] for q in raw)
fit = min((W - 90) / (x_hi - x_lo), (H - 80) / (y_hi - y_lo))
S, O = fit, (W / 2 - fit * (x_lo + x_hi) / 2, H / 2 - fit * (y_lo + y_hi) / 2)
body = poly([P(comb(s0, t0)), P(comb(s1, t0)), P(comb(s1, t1)), P(comb(s0, t1))], SECOND, THIN, SECOND_SOFT, close=True)
for k in range(1, 6):
    s = s0 + (s1 - s0) * k / 6
    body += line(*P(comb(s, t0)), *P(comb(s, t1)), GUIDE, 1.1)
for k in range(1, 5):
    t = t0 + (t1 - t0) * k / 5
    body += line(*P(comb(s0, t)), *P(comb(s1, t)), GUIDE, 1.1)
o = P((0, 0, 0))
body += arrow(*o, *P(a1), SECOND, 3)
body += arrow(*o, *P(a2), SECOND, 3)
body += arrow(*o, *P(p), ACCENT, 3.4)
# right-angle marker where the residual meets the plane
n = math.sqrt(dotp(res, res)); rh = tuple(q / n for q in res)
d = a2; dn = math.sqrt(dotp(d, d)); dh = tuple(q / dn for q in d)
e = 0.11
q1 = tuple(pp + e * u for pp, u in zip(p, dh)); q2 = tuple(x + e * u for x, u in zip(q1, rh)); q3 = tuple(pp + e * u for pp, u in zip(p, rh))
body += poly([P(q1), P(q2), P(q3)], INK, 1.6)
body += line(*P(p), *P(b), ACCENT, 2.6, dash="7 6")
body += arrow(*o, *P(b), INK, 3.2)
body += dot(*P(b), 6.5, INK)
body += dot(*P(p), 6.5, ACCENT)
body += dot(*o, 6, INK)
write(body)
