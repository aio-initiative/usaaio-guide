"""PCA on a correlated 2-D cloud: centre the points, take the eigenvectors of their covariance matrix.
The long terracotta arrow is PC1 (direction of most variance), the short teal arrow PC2; every point drops a
grey perpendicular onto the PC1 line, and its projection (its 1-D score) sits on that line."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
rng = np.random.default_rng(7)
X = rng.multivariate_normal([0, 0], [[3.0, 1.5], [1.5, 1.6]], 36)
X -= X.mean(0)
vals, vecs = np.linalg.eigh(np.cov(X.T))
v1, v2 = vecs[:, 1], vecs[:, 0]
if v1[0] < 0: v1 = -v1
if v2[1] < 0: v2 = -v2
S, CX, CY = 52, 300, 200
P = lambda p: (CX + p[0] * S, CY - p[1] * S)
body = ""
t = 5.6
body += line(*P(-t * v1), *P(t * v1), GUIDE, THIN, dash="6 6")
for p in X:
    q = (p @ v1) * v1
    body += line(*P(p), *P(q), GUIDE, 1.6)
for p in X:
    body += dot(*P((p @ v1) * v1), 4.5, ACCENT)
for p in X:
    body += dot(*P(p), 6.5, INK)
body += arrow(*P((0, 0)), *P(2.6 * np.sqrt(vals[1]) * v1), ACCENT, 4)
body += arrow(*P((0, 0)), *P(2.6 * np.sqrt(vals[0]) * v2), SECOND, 4)
body += dot(*P((0, 0)), 7, BG, INK, 2.5)
write(body)
