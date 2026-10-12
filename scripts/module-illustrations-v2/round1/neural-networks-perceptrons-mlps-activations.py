"""A handwritten digit as pixels feeds a fully connected network: input, two ReLU hidden layers, ten outputs with one winner."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(5)
# An 8 x 10 pixel handwritten "3" with a faint anti-aliased rim.
DIGIT = ["00000000", "00111100", "01100110", "00000110", "00011100", "00000110",
         "00000011", "01100011", "00111110", "00000000"]
R_, C_ = len(DIGIT), len(DIGIT[0])
CELL, GX, GY = 13, 40, 200 - 5 * 13
body = ""
for r, row in enumerate(DIGIT):
    for c, ch in enumerate(row):
        v = 1.0 if ch == "1" else 0.0
        if not v:
            nb = sum(DIGIT[rr][cc] == "1" for rr in (r - 1, r, r + 1) for cc in (c - 1, c, c + 1)
                     if 0 <= rr < R_ and 0 <= cc < C_)
            v = min(0.07 * nb, 0.25)
        g = int(255 - (255 - 31) * min(v, 1))
        body += rect(GX + c * CELL, GY + r * CELL, CELL, CELL, f"#{g:02x}{g:02x}{g:02x}", GUIDE, 0.6, 0)
body += rect(GX, GY, C_ * CELL, R_ * CELL, "none", GREY, 1.4, 0)

sizes = [7, 6, 6, 10]
xs = [205, 315, 425, 540]


def ys(n, top=48, bot=352):
    if n == 10:
        return [top + (bot - top) * i / 9 for i in range(10)]
    span = (n - 1) * 46
    return [200 - span / 2 + 46 * i for i in range(n)]


pos = [[(x, y) for y in ys(n)] for x, n in zip(xs, sizes)]
# Real forward pass with random weights, input = mean of pixel rows grouped into seven bands.
pix = [[1.0 if ch == "1" else 0.0 for ch in row] for row in DIGIT]
a = [sum(pix[r][c] for r in range(R_)) / R_ for c in range(C_)][:7]
Ws = [[[random.gauss(0, 1) for _ in range(sizes[l])] for _ in range(sizes[l + 1])] for l in range(3)]
acts = [a]
for l, Wm in enumerate(Ws):
    z = [sum(w * x for w, x in zip(row, acts[-1])) for row in Wm]
    acts.append([max(0, v) for v in z] if l < 2 else z)
win = max(range(10), key=lambda k: acts[-1][k])

# Pixels to input column: a single bracket of arrows.
body += arrow(GX + C_ * CELL + 10, 200, xs[0] - 26, 200, GREY, 2.2)
for l, Wm in enumerate(Ws):
    for j, row in enumerate(Wm):
        for i, w in enumerate(row):
            (x1, y1), (x2, y2) = pos[l][i], pos[l + 1][j]
            hot = l == 2 and j == win and w * acts[l][i] > 0
            body += line(x1, y1, x2, y2, ACCENT if hot else GUIDE, 2.2 if hot else 0.6 + 0.5 * abs(w))
for l, col in enumerate(pos):
    for k, (x, y) in enumerate(col):
        if l == 3:
            body += dot(x, y, 10, ACCENT, ACCENT, 2) if k == win else ring(x, y, 10, INK, 2)
        elif l == 0:
            body += ring(x, y, 13, INK, 2.2)
        else:
            body += ring(x, y, 15, INK, 2.2)
            # ReLU glyph inside the hidden unit.
            body += poly([(x - 8, y + 4), (x, y + 4), (x + 7, y - 6)], ACCENT, 2.0)
write(body)
