"""Linear attention: the same product bracketed the other way, each laid out in the textbook multiplication scheme
(left factor on the left, right factor above, product where their rows and columns meet). Left, softmax attention:
Q (n x d) times K^T (d x n) makes the full n x n table of scores, shaded by its real softmax weights. Right, linear
attention: phi(K)^T (d x n) times V (n x d) makes only the d x d running state (terracotta), whatever the length;
the n x n square its factors span stays empty (dashed).
n = 12 tokens, d = 2, as in the lesson's head size of 2.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, D = 12, 2
U = 17                       # one token row / column
DW = 16                      # one feature column
GAP = 10
n, d = N * U, D * DW
PANEL = d + GAP + n
Y = (H - PANEL) / 2
LX, RX = 40, W - 40 - PANEL

# Real numbers: query and key vectors on a slow rotation, scores q . k / sqrt(d), causal softmax rows.
q = [(math.cos(0.55 * i), math.sin(0.55 * i) + 0.4) for i in range(N)]
k = [(math.cos(0.55 * i + 0.3) * 1.6, math.sin(0.55 * i + 0.3) * 1.6) for i in range(N)]
w = []
for i in range(N):
    sc = [(q[i][0] * k[j][0] + q[i][1] * k[j][1]) / math.sqrt(D) for j in range(i + 1)]
    m = max(sc)
    e = [math.exp(x - m) for x in sc]
    w.append([x / sum(e) for x in e])


def mix(a, b, t):
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


def tall(x, y, fill, stroke):          # an n x d matrix: one row per token
    out = rect(x, y, d, n, fill, stroke, 2.2, 2)
    for r in range(1, N):
        out += line(x, y + r * U, x + d, y + r * U, stroke, 0.8)
    out += line(x + DW, y, x + DW, y + n, stroke, 0.8)
    return out


def wide(x, y, fill, stroke):          # a d x n matrix: one column per token
    out = rect(x, y, n, d, fill, stroke, 2.2, 2)
    for c in range(1, N):
        out += line(x + c * U, y, x + c * U, y + d, stroke, 0.8)
    out += line(x, y + DW, x + n, y + DW, stroke, 0.8)
    return out


body = ""
# Left: Q at left, K^T above, the n x n table where they meet.
ox, oy = LX + d + GAP, Y + d + GAP
body += tall(LX, oy, "#eef0f2", INK)
body += wide(ox, Y, "#eef0f2", INK)
for i in range(N):
    for j in range(N):
        x, y = ox + j * U, oy + i * U
        if j <= i:
            body += rect(x + 1, y + 1, U - 2, U - 2, mix(BG, GREY, min(1, 0.15 + 1.2 * w[i][j])), "none", 0, 1)
body += rect(ox, oy, n, n, "none", GREY, 2.2, 2)

# Right: phi(K)^T at left, V above, the d x d state where they meet. The sum runs over the n tokens, so V is n
# tall and phi(K)^T is n wide, and the square they would span is left empty: no n x n table is ever made.
sx, sy = RX + n + GAP, Y + n + GAP
body += wide(RX, sy, SECOND_SOFT, SECOND)
body += tall(sx, Y, SECOND_SOFT, SECOND)
body += rect(RX, Y, n, n, "none", GUIDE, THIN, 2).replace("/>", ' stroke-dasharray="6 6"/>')
for r in range(D):
    for c in range(D):
        body += rect(sx + c * DW + 1, sy + r * DW + 1, DW - 2, DW - 2, ACCENT, "none", 0, 1)
body += rect(sx, sy, d, d, "none", ACCENT, 2.4, 2)

# divider
body += line(W / 2, Y, W / 2, Y + PANEL, GUIDE, THIN)
write(body)
