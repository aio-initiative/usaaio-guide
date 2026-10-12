"""One sentence with a hidden word: context arcs reach the blank from both sides, and below it the softmax over the vocabulary puts most weight on the true word."""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(3)
N, MASKED = 9, 5                    # "the cat sat on the [MASK] and fell asleep"
LENS = [0.45, 0.55, 0.55, 0.35, 0.45, 0, 0.5, 0.6, 0.8]   # word-length placeholders
BW, BH, Y = 50, 40, 186
X0, GAP = 40, (520 - N * 50) / (N - 1)
xs = [X0 + i * (BW + GAP) for i in range(N)]
cx = [x + BW / 2 for x in xs]
mx = cx[MASKED]

body = ""
# Context arcs above the row: every visible word feeds the blank, from the left and from the right.
for i in range(N):
    if i == MASKED:
        continue
    d = abs(i - MASKED)
    h = 30 + 24 * d
    x1, x2 = cx[i], mx + (-1 if i < MASKED else 1) * (2 + 4.5 * d)
    pts = []
    for k in range(41):
        t = k / 40
        x = x1 + (x2 - x1) * t
        y = Y - 4 - h * 4 * t * (1 - t)
        pts.append((x, y))
    col = INK if d == 1 else GREY
    body += poly(pts[:-2], col, 2.0 if d == 1 else 1.6)
    (ax, ay), (bx, by) = pts[-4], pts[-1]
    body += arrow(ax, ay, bx, by, col, 2.0 if d == 1 else 1.6, 10)

# The token row.
for i in range(N):
    if i == MASKED:
        body += rect(xs[i], Y, BW, BH, ACCENT_SOFT, ACCENT, 2.6, 6)
        body += rect(xs[i] + 8, Y + 8, BW - 16, BH - 16, "none", ACCENT, 1.6, 3).replace("/>", ' stroke-dasharray="4 3"/>')
    else:
        body += rect(xs[i], Y, BW, BH, BG, INK, 2.2, 6)
        body += line(xs[i] + 10, Y + BH / 2, xs[i] + 10 + (BW - 20) * LENS[i], Y + BH / 2, GREY, 4)

# Prediction at the blank: logits -> softmax over a slice of the vocabulary.
logits = [math.log(p) for p in (0.02, 0.1, 0.6, 0.04, 0.3, 0.015, 0.03, 0.02)]
z = sum(math.exp(v) for v in logits)
probs = [math.exp(v) / z for v in logits]
top = max(range(len(probs)), key=lambda k: probs[k])
body += arrow(mx, Y + BH + 6, mx, Y + BH + 40, ACCENT, 2.4)
nb, bw, bg = len(probs), 22, 10
cw = nb * bw + (nb - 1) * bg
bx0, base, hmax = mx - cw / 2, 362, 98
body += line(bx0 - 12, base, bx0 + cw + 12, base, GUIDE, THIN)
for k, p in enumerate(probs):
    h = hmax * p / probs[top]
    x = bx0 + k * (bw + bg)
    body += rect(x, base - h, bw, h, ACCENT if k == top else GUIDE, ACCENT if k == top else GREY, 1.2 if k != top else 0, 2)
write(body)
