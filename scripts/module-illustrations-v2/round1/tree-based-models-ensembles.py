"""Bagging: many bootstrap regression trees, each a jagged step function, average into one smooth curve near the truth."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(7)
X0, X1, Y0, Y1 = 50, 560, 40, 350


def truth(x):
    return math.sin(2 * math.pi * x) * 0.8 + 0.3 * x


# Noisy training sample on [0, 1].
xs = sorted(random.random() for _ in range(40))
data = [(x, truth(x) + random.gauss(0, 0.28)) for x in xs]


def fit_tree(pts, depth):
    """Greedy squared-error regression tree; returns a predict function."""
    ys = [y for _, y in pts]
    mean = sum(ys) / len(ys)
    if depth == 0 or len(pts) < 4:
        return lambda x: mean
    pts = sorted(pts)
    best = None
    for i in range(2, len(pts) - 1):
        L, R = pts[:i], pts[i:]
        mL = sum(y for _, y in L) / len(L)
        mR = sum(y for _, y in R) / len(R)
        sse = sum((y - mL) ** 2 for _, y in L) + sum((y - mR) ** 2 for _, y in R)
        if best is None or sse < best[0]:
            best = (sse, (pts[i - 1][0] + pts[i][0]) / 2, L, R)
    _, t, L, R = best
    fl, fr = fit_tree(L, depth - 1), fit_tree(R, depth - 1)
    return lambda x: fl(x) if x < t else fr(x)


trees = []
for _ in range(14):
    boot = [random.choice(data) for _ in data]
    trees.append(fit_tree(boot, 3))

grid = [i / 600 for i in range(601)]
lo, hi = -1.2, 1.5


def S(x, y):
    return X0 + (X1 - X0) * x, Y1 - (Y1 - Y0) * (y - lo) / (hi - lo)


body = axes(X0 - 10, Y0 - 10, X1 + 10, Y1 + 10)
for f in trees:
    body += poly([S(x, f(x)) for x in grid], GUIDE, 1.6)
body += poly([S(x, truth(x)) for x in grid], GREY, 2.0, dash="7 6")
avg = [S(x, sum(f(x) for f in trees) / len(trees)) for x in grid]
body += poly(avg, ACCENT, 4.0)
for x, y in data:
    body += ring(*S(x, y), 5.5, INK, 2.0)
write(body)
