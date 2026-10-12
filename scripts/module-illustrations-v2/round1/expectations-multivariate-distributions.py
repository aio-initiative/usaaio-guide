"""How much an ensemble helps: the variance of an average of n errors with variance 4 and pairwise correlation 0.5,
4 (0.5 + 0.5 / n), levels off at the floor 2 (dashed), while independent errors, 4 / n, fall toward 0.
The five-model ensemble from the worked example (variance 2.4) is ringed."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

s2, rho, N = 4.0, 0.5, 12
corr = lambda n: s2 * (rho + (1 - rho) / n)
ind = lambda n: s2 / n

x0, y0, x1, y1 = 60, 40, 560, 356
X = lambda n: x0 + 26 + (x1 - x0 - 40) * (n - 1) / (N - 1)
Y = lambda v: y1 - (y1 - y0 - 16) * v / s2

body = axes(x0, y0, x1, y1, ticks=0)
for n in range(1, N + 1):
    body += line(X(n), y1, X(n), y1 + 6, GUIDE, THIN)
body += line(x0, Y(s2), x1, Y(s2), GUIDE, THIN, "4 6")                 # a single model's variance
body += line(x0, Y(rho * s2), x1, Y(rho * s2), ACCENT, 2, "9 7")       # the floor rho sigma^2
body += poly([(X(n), Y(ind(n))) for n in range(1, N + 1)], SECOND, 2.4)
body += poly([(X(n), Y(corr(n))) for n in range(1, N + 1)], ACCENT, 2.6)
for n in range(1, N + 1):
    body += dot(X(n), Y(ind(n)), 5.5, SECOND)
    body += dot(X(n), Y(corr(n)), 6.5, ACCENT)
body += ring(X(5), Y(corr(5)), 12, INK, 2.4)
body += dot(X(5), Y(corr(5)), 6.5, ACCENT)

write(body)
