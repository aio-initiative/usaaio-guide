"""Next-token prediction: above each token sits its softmax over the vocabulary, and the winning bar becomes the next token, so every position predicts the one after it and the newest token is appended at the end."""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(11)
N = 6                                 # tokens so far; the last one was just generated
SLOTS = N + 1                         # plus the empty slot the next token goes into
BW, BH, Y = 52, 40, 282
X0 = 40
STEP = (520 - BW) / (SLOTS - 1)
cx = [X0 + BW / 2 + i * STEP for i in range(SLOTS)]
LENS = [0.45, 0.55, 0.5, 0.35, 0.6, 0.5]
V, bw, bg, base, hmax = 5, 8, 3, 222, 215

body = ""
for i in range(N):
    # Softmax of real logits at position i, temperature 1; the peak sharpens later in the sentence.
    logits = [random.gauss(0, 1) for _ in range(V)]
    win = random.randrange(V)
    logits[win] = max(logits) + 1.0 + 0.25 * i
    z = sum(math.exp(v) for v in logits)
    p = [math.exp(v) / z for v in logits]
    cw = V * bw + (V - 1) * bg
    x0 = cx[i] - cw / 2
    body += line(x0 - 5, base, x0 + cw + 5, base, GUIDE, THIN)
    for k in range(V):
        h = hmax * p[k]
        hot = k == win
        body += rect(x0 + k * (bw + bg), base - h, bw, h, ACCENT if hot else GUIDE, GREY, 0 if hot else 1, 1.5)
    # Token up into its own prediction, then the prediction down into the next slot.
    body += line(cx[i], Y - 6, cx[i], base + 8, GREY, 1.8)
    wx = x0 + win * (bw + bg) + bw / 2
    tx, ty = cx[i + 1] - 8, Y - 8
    pts = [(wx, base + 4)]
    for k in range(1, 31):
        t = k / 30
        # Quadratic Bezier from the winning bar down to the top of the next box.
        c = (wx + 30, base + 6)
        pts.append(((1 - t) ** 2 * wx + 2 * t * (1 - t) * c[0] + t * t * tx,
                    (1 - t) ** 2 * (base + 4) + 2 * t * (1 - t) * c[1] + t * t * ty))
    col = ACCENT if i == N - 1 else INK
    body += poly(pts[:-3], col, 2.0)
    body += arrow(*pts[-4], *pts[-1], col, 2.0, 10)

for i in range(N):
    x = cx[i] - BW / 2
    if i == N - 1:
        body += rect(x, Y, BW, BH, ACCENT_SOFT, ACCENT, 2.6, 6)
        body += line(x + 10, Y + BH / 2, x + 10 + (BW - 20) * LENS[i], Y + BH / 2, ACCENT, 4)
    else:
        body += rect(x, Y, BW, BH, BG, INK, 2.2, 6)
        body += line(x + 10, Y + BH / 2, x + 10 + (BW - 20) * LENS[i], Y + BH / 2, GREY, 4)
x = cx[N] - BW / 2
body += rect(x, Y, BW, BH, "none", ACCENT, 2.0, 6).replace("/>", ' stroke-dasharray="6 4"/>')
# Causal direction: the context only ever grows to the right.
body += arrow(40, 356, 560, 356, GUIDE, 2.0, 12)
write(body)
