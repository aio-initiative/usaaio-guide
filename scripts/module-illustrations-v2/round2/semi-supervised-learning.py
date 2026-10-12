"""Label propagation on two moons: 300 points, a 6-nearest-neighbour graph (light edges), and only two labeled
points (large rings, one per moon). After 25 propagation steps the labels have flowed along the graph from each
seed through its own crescent; points the labels have not reached yet are still grey."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
from sklearn.datasets import make_moons
X, _ = make_moons(300, noise=0.06, random_state=2)
n, K = len(X), 6
D = ((X[:, None] - X[None]) ** 2).sum(-1)
nb = np.argsort(D, 1)[:, 1:K + 1]
Wm = np.zeros((n, n))
for i in range(n):
    Wm[i, nb[i]] = Wm[nb[i], i] = 1
seeds = [int(np.argmin(X[:, 0] - 3 * X[:, 1] * 0)), int(np.argmax(X[:, 0]))]
T = Wm / Wm.sum(1, keepdims=True)
F = np.zeros((n, 2)); F[seeds[0], 0] = F[seeds[1], 1] = 1
for _ in range(25):
    F = T @ F
    F[seeds] = [[1, 0], [0, 1]]
lo, hi = X.min(0), X.max(0)
s = min(510 / (hi[0] - lo[0]), 310 / (hi[1] - lo[1]))
c = (lo + hi) / 2
P = lambda p: (300 + (p[0] - c[0]) * s, 200 - (p[1] - c[1]) * s)
body = ""
for i in range(n):
    for j in nb[i]:
        body += line(*P(X[i]), *P(X[j]), GUIDE, 1.1)
for i in range(n):
    if F[i].sum() < 1e-3:
        body += dot(*P(X[i]), 5, GUIDE, GREY, 1)
    else:
        body += dot(*P(X[i]), 5.5, ACCENT if F[i, 0] > F[i, 1] else SECOND)
for k, i in enumerate(seeds):
    body += dot(*P(X[i]), 13, BG, ACCENT if k == 0 else SECOND, 3.5) + dot(*P(X[i]), 6.5, ACCENT if k == 0 else SECOND)
write(body)
