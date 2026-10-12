"""The handwritten digits 1, 4 and 7 (64 pixels each) taken to two dimensions twice: the linear PCA projection
on the left leaves the three classes as overlapping clouds, t-SNE (perplexity 30) on the right keeps each point's
neighbours together and pulls the classes into three separate groups."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
d = load_digits()
keep = np.isin(d.target, [1, 4, 7])
X, y = d.data[keep], d.target[keep]
idx = np.random.default_rng(0).permutation(len(y))[:270]
X, y = X[idx], y[idx]
emb = [PCA(2).fit_transform(X), TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(X)]
COL = {1: ACCENT, 4: SECOND, 7: INK}
body = ""
boxes = [(40, 50, 230, 300), (330, 50, 230, 300)]
for (bx, by, bw, bh), E in zip(boxes, emb):
    body += rect(bx, by, bw, bh, "none", GUIDE, THIN, 8)
    lo, hi = E.min(0), E.max(0)
    s = min((bw - 36) / (hi[0] - lo[0]), (bh - 36) / (hi[1] - lo[1]))
    c = (lo + hi) / 2
    for i in np.random.default_rng(1).permutation(len(y)):
        body += dot(bx + bw / 2 + (E[i, 0] - c[0]) * s, by + bh / 2 - (E[i, 1] - c[1]) * s, 4, COL[y[i]])
body += arrow(282, 200, 318, 200, GREY, 3)
write(body)
