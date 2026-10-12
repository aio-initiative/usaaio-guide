"""Patches as tokens, on the lesson's running example: the 6 x 6 image with a vertical line in column 2, cut into
four 3 x 3 patches; the patches become a sequence of tokens after a class token, each with its position embedding
added (the teal strip); and the top-left patch's attention to the four patches, softmax(patch . patch / sqrt 9),
drawn as arcs and bars to scale: 0.366 to itself and the bottom-left patch, 0.134 to the two empty ones."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np

image = np.zeros((6, 6)); image[:, 2] = 1.0
corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
patches = np.array([image[i:i + 3, j:j + 3].ravel() for i, j in corners])
scaled = patches @ patches[0] / 3.0
att = np.exp(scaled) / np.exp(scaled).sum()

body = ""
# the image, cut into four patches
c, gap, ix, iy = 32, 14, 44, 101
def cell_xy(r, k): return ix + k * c + (gap if k >= 3 else 0), iy + r * c + (gap if r >= 3 else 0)
for r in range(6):
    for k in range(6):
        x, y = cell_xy(r, k)
        body += rect(x, y, c, c, fill=INK if image[r, k] else BG, stroke=GUIDE, sw=1.2, rx=0)
for n, (i, j) in enumerate(corners):
    x, y = cell_xy(i, j)
    body += rect(x, y, 3 * c, 3 * c, stroke=ACCENT if n == 0 else INK, sw=3.2 if n == 0 else 2.2, rx=2)

# the token sequence: class token, then the four patches, each with a position embedding strip
tx, ts, tg, ty = 318, 46, 16, 40
tok_y = [ty + n * (ts + tg) for n in range(5)]
body += arrow(ix + 6 * c + gap + 10, 200, tx - 40, 200, GREY, 2.4)
body += rect(tx, tok_y[0], ts, ts, fill=ACCENT_SOFT, stroke=ACCENT, sw=2.4, rx=6)
body += dot(tx + ts / 2, tok_y[0] + ts / 2, 8, ACCENT)
for n, (i, j) in enumerate(corners):
    y0, m = tok_y[n + 1], ts / 3
    for r in range(3):
        for k in range(3):
            body += rect(tx + k * m, y0 + r * m, m, m, fill=INK if image[i + r, j + k] else BG, stroke=GUIDE, sw=1, rx=0)
    body += rect(tx, y0, ts, ts, stroke=ACCENT if n == 0 else INK, sw=3 if n == 0 else 2, rx=2)
for n in range(5):
    y0 = tok_y[n]
    body += rect(tx - 22, y0 + 4, 10, ts - 8, fill=SECOND_SOFT, stroke=SECOND, sw=1.8, rx=3)
    body += dot(tx - 17, y0 + 8 + (ts - 16) * n / 4, 3.6, SECOND)

# attention of the top-left patch: arcs whose width follows the weight, ending in bars drawn to scale
qx, qy = tx + ts, tok_y[1] + ts / 2
bx, bmax = 462, 94 / att.max()
for n in range(4):
    yk = tok_y[n + 1] + ts / 2
    col = ACCENT if att[n] > 0.25 else GREY
    sw = 1.5 + 10 * att[n]
    if n == 0:
        body += line(qx + 8, qy, bx - 6, yk, col, sw)
    else:
        body += (f'<path d="M{qx + 8:.1f},{qy:.1f} C{qx + 70:.1f},{qy:.1f} {bx - 100:.1f},{yk:.1f} {bx - 6:.1f},{yk:.1f}" '
                 f'fill="none" stroke="{col}" stroke-width="{sw:.1f}" stroke-linecap="round"/>')
    body += rect(bx, yk - 13, 94, 26, fill="#eceef0", stroke="none", rx=3)
    body += rect(bx, yk - 13, att[n] * bmax, 26, fill=ACCENT if att[n] > 0.25 else GREY, stroke="none", rx=3)
body += line(bx, tok_y[1] + 4, bx, tok_y[4] + ts - 4, INK, 2)
write(body)
