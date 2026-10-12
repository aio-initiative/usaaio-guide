"""The module's focus problem: 87 of 100 against a baseline of 80%. The null distribution of the accuracy
(normal, mean 0.80, SE 0.04) with the one-sided p-value tail beyond the observed 0.87 shaded, and above it the
95% confidence interval 0.87 +- 1.96 * 0.0336 around the observed score, which just misses 0.80."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

lo, hi = 0.67, 0.95
X0, X1, Y0, Y1 = 50, 550, 50, 340
def sx(p): return X0 + (p - lo) / (hi - lo) * (X1 - X0)
mu, se = 0.80, 0.04
peak = 1 / (se * math.sqrt(2 * math.pi))
def sy(d): return Y1 - d / peak * 225
def pdf(p): return math.exp(-0.5 * ((p - mu) / se) ** 2) / (se * math.sqrt(2 * math.pi))
obs = 0.87
ps = [lo + (hi - lo) * i / 400 for i in range(401)]
curve = [(sx(p), sy(pdf(p))) for p in ps]
tail = [(sx(obs), Y1)] + [(sx(p), sy(pdf(p))) for p in ps if p >= obs] + [(sx(hi), Y1)]
b = axes(X0, Y0, X1, Y1)
b += f'<path d="{path(tail)} Z" fill="{ACCENT}" fill-opacity="0.35"/>'   # the one-sided p-value
b += line(sx(mu), Y1, sx(mu), sy(peak) - 8, GREY, THIN, dash="6 5")
b += poly(curve, INK, 2.8)
b += line(sx(obs), Y1, sx(obs), 104, ACCENT, 2.6)
# 95% confidence interval around the observed score
seh = math.sqrt(obs * (1 - obs) / 100)
ci = (obs - 1.96 * seh, obs + 1.96 * seh)
cy = 92
b += line(sx(ci[0]), cy, sx(ci[1]), cy, SECOND, 3.2)
b += line(sx(ci[0]), cy - 13, sx(ci[0]), cy + 13, SECOND, 3.2)
b += line(sx(ci[1]), cy - 13, sx(ci[1]), cy + 13, SECOND, 3.2)
b += dot(sx(obs), cy, 8, ACCENT)
b += dot(sx(mu), Y1, 6, GREY)
b += dot(sx(obs), Y1, 7, ACCENT)
write(b)
