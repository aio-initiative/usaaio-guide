"""REINFORCE with a baseline on a continuous action: a Gaussian policy (grey) samples actions, each scored by the
reward curve (dashed ink). Samples above the baseline (dashed guide) have positive advantage (filled terracotta),
those below negative (open rings). The update mu += lr * mean(A (a - mu) / sigma^2) moves the policy (terracotta)
toward the actions that beat the baseline."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

x0, x1 = 44, W - 44
X0, X1 = -3.4, 3.4
base, top = 350, 46


def px(v):
    return x0 + (v - X0) / (X1 - X0) * (x1 - x0)


def gauss(v, m, s):
    return math.exp(-0.5 * ((v - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def reward(a):
    return math.exp(-0.5 * ((a - 1.5) / 1.0) ** 2)


def ry(r):                      # reward in [0, 1] drawn in the upper band
    return 178 - (178 - top) * r


def inv_phi(p):
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p else (lo, mid)
    return (lo + hi) / 2


mu, sd = -0.8, 0.95
n = 9
acts = [mu + sd * inv_phi((i + 0.5) / n) for i in range(n)]
rs = [reward(a) for a in acts]
b = sum(rs) / n                                   # baseline: mean return of the batch
grad = sum((r - b) * (a - mu) / sd ** 2 for a, r in zip(acts, rs)) / n
mu2 = mu + 3.0 * grad                           # one REINFORCE step, lr = 3
amp = 116 / gauss(0, 0, sd)

xs = [X0 + (X1 - X0) * i / 240 for i in range(241)]
old = [(px(v), base - amp * gauss(v, mu, sd)) for v in xs]
new = [(px(v), base - amp * gauss(v, mu2, sd)) for v in xs]

body = line(x0, base, x1, base, GUIDE, THIN)
body += line(x0, ry(b), x1, ry(b), GUIDE, 2, dash="7 7")
body += poly([(px(v), ry(reward(v))) for v in xs], INK, 2.6, dash="3 7")
body += poly(new + [(px(X1), base), (px(X0), base)], "none", 0, fill=ACCENT_SOFT, close=True)
body += poly(old, GREY, 2.6)
body += poly(new, ACCENT, 3.2)
for a, r in zip(acts, rs):
    x = px(a)
    body += line(x, base, x, ry(r), GUIDE, THIN)
    body += dot(x, base, 4.5, GREY)
    body += dot(x, ry(r), 8, ACCENT) if r > b else ring(x, ry(r), 7.5, INK, 2.4)
peak = base - amp * gauss(0, 0, sd) - 16
body += arrow(px(mu), peak, px(mu2) - 4, peak, ACCENT, 3.6, size=18)
write(body)
