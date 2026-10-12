"""The GAN figure: random inputs z (bottom line) mapped by the generator to points x (upper line), whose density
(terracotta) is chasing the data density (ink). Above them, the best discriminator
D*(x) = p_data / (p_data + p_g) in teal: high where real data dominates, low where the generator piles up."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

x0, x1 = 44, W - 44
X0, X1 = -4.0, 4.0


def px(v):
    return x0 + (v - X0) / (X1 - X0) * (x1 - x0)


def gauss(v, m, s):
    return math.exp(-0.5 * ((v - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


md, sd = 0.9, 0.75          # data
mg, sg = -0.8, 0.9          # generator, part way through training
base = 250                  # x-axis of the density plot
amp = 330                   # pixels per unit density
dtop, dbot = 46, 196        # D* is drawn in [dtop, dbot]: D=1 at dtop, D=0 at dbot

xs = [X0 + (X1 - X0) * i / 240 for i in range(241)]
pdata = [(px(v), base - amp * gauss(v, md, sd)) for v in xs]
pg = [(px(v), base - amp * gauss(v, mg, sg)) for v in xs]
dstar = [gauss(v, md, sd) / (gauss(v, md, sd) + gauss(v, mg, sg)) for v in xs]

body = line(x0, base, x1, base, GUIDE, THIN)
# half-height guide for D* = 1/2, the value at equilibrium
body += line(x0, (dtop + dbot) / 2, x1, (dtop + dbot) / 2, GUIDE, THIN, dash="5 6")
body += poly(pg + [(px(X1), base), (px(X0), base)], "none", 0, fill=ACCENT_SOFT, close=True)
body += poly(pg, ACCENT, 3)
body += poly(pdata, INK, 3.2, dash="3 7")
body += poly([(px(v), dbot - (dbot - dtop) * d) for v, d in zip(xs, dstar)], SECOND, 3)

# generator mapping: evenly spaced z on the lower line map to x = mg + sg * Phi^-1(u)
zl, xl = 362, base


def inv_phi(p):
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


body += line(x0, zl, x1, zl, GUIDE, THIN)
n = 11
for i in range(n):
    u = (i + 0.5) / n
    zx = px(X0 + 0.6 + (X1 - X0 - 1.2) * u)
    gx = px(mg + sg * inv_phi(u))
    body += arrow(zx, zl - 6, gx, xl + 8, GREY, 2.2, size=12)
    body += dot(zx, zl, 5, INK)
write(body)
