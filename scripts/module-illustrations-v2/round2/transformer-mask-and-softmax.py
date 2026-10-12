"""Causal mask and row softmax: a score matrix with every later position hatched out, and each row turned into
weights that fill exactly one full bar."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(7)
T, d = 5, 4
Q = [[random.gauss(0, 1) for _ in range(d)] for _ in range(T)]
K = [[random.gauss(0, 1) for _ in range(d)] for _ in range(T)]
S = [[sum(a * b for a, b in zip(Q[i], K[j])) / math.sqrt(d) for j in range(T)] for i in range(T)]
A = []
for i in range(T):
    e = [math.exp(S[i][j]) if j <= i else 0.0 for j in range(T)]
    A.append([x / sum(e) for x in e])


def mix(t):
    """White to accent, t in [0, 1]."""
    a, b = (255, 255, 255), (0xb8, 0x4e, 0x30)
    return "#%02x%02x%02x" % tuple(round(p + (q - p) * t) for p, q in zip(a, b))


defs = (f'<pattern id="hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<rect width="9" height="9" fill="#f1f2f4"/><line x1="0" y1="0" x2="0" y2="9" stroke="{GUIDE}" stroke-width="3"/></pattern>')
c, gx, gy = 52, 44, 70
lo = min(S[i][j] for i in range(T) for j in range(i + 1))
hi = max(S[i][j] for i in range(T) for j in range(i + 1))
body = ""
for i in range(T):
    for j in range(T):
        x, y = gx + j * c, gy + i * c
        fill = "url(#hatch)" if j > i else mix(0.08 + 0.8 * (S[i][j] - lo) / (hi - lo))
        body += rect(x, y, c, c, fill, BG, 2, 0)
body += rect(gx, gy, T * c, T * c, "none", INK, 2, 2)
# Staircase edge of the mask.
stair = [(gx, gy)]
for i in range(T):
    stair += [(gx + (i + 1) * c, gy + i * c), (gx + (i + 1) * c, gy + (i + 1) * c)]
body += poly(stair[1:], INK, 2.4)
body += arrow(gx + T * c + 18, 200, gx + T * c + 64, 200, GREY, 2.6)

# Softmax rows: stacked segments, each row exactly the full bar width.
bx, bw = 392, 166
for i in range(T):
    y = gy + i * c + 9
    x = bx
    m = max(A[i])
    for j in range(i + 1):
        w = A[i][j] * bw
        body += rect(x, y, w, c - 18, ACCENT if A[i][j] == m else mix(0.18 + 0.5 * A[i][j]), BG, 2, 0)
        x += w
    body += rect(bx, y, bw, c - 18, "none", INK, 1.8, 2)
body += line(bx + bw, gy - 8, bx + bw, gy + T * c + 8, GREY, 1.4, dash="4 4")
write(body, defs)
