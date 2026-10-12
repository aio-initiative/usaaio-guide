"""Self-attention over the lesson's eight-token sentence (the cat that ate the fish was full): every query token on
the top row connects to every key token below, line width set by its softmax weight from the lesson's own scores.
The query "was" is in terracotta; its heaviest line reaches back past "fish" to its subject "cat"."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
import numpy as np

n = 8
rng = np.random.default_rng(21)
scores = rng.normal(0, 0.5, (n, n))
scores[6, 1] += 3.2
scores[7, 1] += 2.1
scores[5, 3] += 2.4
scores = scores - scores.max(1, keepdims=True)
attn = np.exp(scores)
attn /= attn.sum(1, keepdims=True)

X = lambda i: 70 + i * (460 / (n - 1))
YQ, YK, S = 70, 330, 34
q = 6
body = ""
for i in range(n):
    if i == q:
        continue
    for j in range(n):
        body += line(X(i), YQ + S / 2, X(j), YK - S / 2, GUIDE, 0.5 + 5 * attn[i, j])
for j in np.argsort(attn[q]):
    w = attn[q, j]
    body += line(X(q), YQ + S / 2, X(j), YK - S / 2, ACCENT, 1.2 + 9 * w)
for i in range(n):
    top = i == q
    body += rect(X(i) - S / 2, YQ - S / 2, S, S, ACCENT_SOFT if top else BG, ACCENT if top else INK, 2.4)
    hit = i == int(np.argmax(attn[q]))
    body += rect(X(i) - S / 2, YK - S / 2, S, S, ACCENT_SOFT if hit else BG, ACCENT if hit else INK, 2.4)
write(body)
