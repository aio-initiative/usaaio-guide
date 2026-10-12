"""Skip-gram with negative sampling. Top: a sentence split into subword tokens (pieces of one word sit close together);
the center token (terracotta) predicts the tokens in its window (teal arcs). Bottom: the 2-D embedding plane after
real SGNS gradient steps: the context vectors (teal) have rotated toward the center vector, the negative samples
(grey) have been pushed away; rings mark where each started."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

# Sentence: piece widths per word (a word of two pieces is a subword split).
words = [[46], [40, 30], [54], [58], [44, 34], [50], [36, 30]]
GAP_WORD, GAP_PIECE, TH = 18, 4, 36
total = sum(sum(w) + GAP_PIECE * (len(w) - 1) for w in words) + GAP_WORD * (len(words) - 1)
x, ty = (W - total) / 2, 92
tiles = []
for w in words:
    for k, pw in enumerate(w):
        tiles.append((x, pw))
        x += pw + (GAP_PIECE if k < len(w) - 1 else 0)
    x += GAP_WORD
C, WIN = 4, 2                                     # center token index (a whole word), window of 2 each side
ctx = [i for i in range(C - WIN, C + WIN + 1) if i != C]

body = ""
mid = lambda i: tiles[i][0] + tiles[i][1] / 2
for i in ctx:                                     # arcs from the center token up and over to each context token
    x0, x1 = mid(C), mid(i)
    h = 22 + 13 * abs(i - C)
    pts = [(x0 + (x1 - x0) * t, ty - 4 - h * 4 * t * (1 - t)) for t in [j / 40 for j in range(41)]]
    body += poly(pts, SECOND, 2.4)
    body += dot(x1, ty - 4, 4, SECOND)
for i, (tx, pw) in enumerate(tiles):
    if i == C:
        body += rect(tx, ty, pw, TH, ACCENT_SOFT, ACCENT, 3, rx=6) + dot(mid(C), ty - 4, 5, ACCENT)
    elif i in ctx:
        body += rect(tx, ty, pw, TH, SECOND_SOFT, SECOND, 2.4, rx=6)
    else:
        body += rect(tx, ty, pw, TH, "#f3f4f6", GREY, 1.8, rx=6)

# Embedding plane: SGNS updates on 2-D vectors.
sig = lambda z: 1 / (1 + math.exp(-z))
ang = lambda a, r: [r * math.cos(math.radians(a)), r * math.sin(math.radians(a))]
v = ang(62, 1.0)                                  # center word's input vector
u0 = {"c": [ang(a, 0.9) for a in (165, -30, 12)], "n": [ang(a, 0.9) for a in (105, 0, 220)]}
u = {k: [p[:] for p in vs] for k, vs in u0.items()}
lr = 0.18
for _ in range(14):
    gv = [0.0, 0.0]
    for kind, label in (("c", 1), ("n", 0)):
        for p in u[kind]:
            g = sig(p[0] * v[0] + p[1] * v[1]) - label        # d loss / d (u . v)
            gv[0] += g * p[0]; gv[1] += g * p[1]
            p[0] -= lr * g * v[0]; p[1] -= lr * g * v[1]
    v[0] -= lr * gv[0] * 0.25; v[1] -= lr * gv[1] * 0.25
    for kind in u:                                # keep lengths comparable so direction carries the meaning
        for p in u[kind]:
            n = math.hypot(*p); p[0] *= 0.9 / n; p[1] *= 0.9 / n
nv = math.hypot(*v); v = [v[0] / nv, v[1] / nv]

OX, OY, S = 300, 258, 90
P = lambda p: (OX + p[0] * S, OY - p[1] * S)
body += line(OX - 180, OY, OX + 180, OY, GUIDE, THIN) + line(OX, OY + 100, OX, OY - 100, GUIDE, THIN)
for kind, col in (("n", GREY), ("c", SECOND)):
    for p0, p in zip(u0[kind], u[kind]):
        a0, a1 = math.atan2(p0[1], p0[0]), math.atan2(p[1], p[0])
        if a1 - a0 > math.pi: a1 -= 2 * math.pi
        if a0 - a1 > math.pi: a1 += 2 * math.pi
        arc = [P(ang(math.degrees(a0 + (a1 - a0) * t), 1.12 if kind == "c" else 1.26)) for t in [j / 30 for j in range(31)]]
        body += poly(arc[:-2], col, 2) + arrow(*arc[-4], *arc[-1], col, 2, size=11)
        body += ring(*arc[0], 6, col, 2.2)
for kind, col in (("n", GREY), ("c", SECOND)):
    for p in u[kind]:
        body += arrow(OX, OY, *P(p), col, 2.6)
body += arrow(OX, OY, *P([v[0] * 1.0, v[1] * 1.0]), ACCENT, 4)
body += dot(OX, OY, 5, INK)
write(body)
