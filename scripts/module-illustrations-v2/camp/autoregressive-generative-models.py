"""Raster-order sampling, p(x) = prod p(x_i | x_<i). A 7 x 8 image of a soft disk is generated one pixel at a
time: the pixels already sampled are shaded and outlined as the context (every earlier pixel, the causal mask), the
current pixel in the last row is framed in terracotta with its left and upper neighbours in teal, and the pixels
still to come are empty dashed cells. On the right, the model's distribution over eight gray levels for the current pixel, a softmax
centred on a weighted mean of its upper and left neighbours; the sampled level is the terracotta bar that feeds the pixel."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

R, N, C = 7, 8, 40
GX, GY = 44, 40                   # grid top-left, 320 x 280
LEVELS = 8
img = [[round((LEVELS - 1) / (1 + math.exp(((r - 3.0) ** 2 + (c - 3.6) ** 2) ** 0.5 * 1.6 - 4.6)))
        for c in range(N)] for r in range(R)]
cur = (6, 3)                      # current pixel (row, col): raster index 51 of 56
idx = lambda r, c: r * N + c
k_cur = idx(*cur)

def shade(v):
    t = v / (LEVELS - 1)
    a, b = (255, 255, 255), (31, 35, 40)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))

body = ""
for r in range(R):
    for c in range(N):
        x, y = GX + c * C, GY + r * C
        k = idx(r, c)
        if k < k_cur:
            body += rect(x, y, C, C, shade(img[r][c]), BG, 1.6, rx=0)
        elif k > k_cur:
            body += rect(x + 3, y + 3, C - 6, C - 6, "none", GREY, 1.6, rx=2).replace("/>", ' stroke-dasharray="4 4"/>')
# context outline: the staircase around all earlier pixels
cr, cc = cur
ctx = [(GX, GY), (GX + N * C, GY), (GX + N * C, GY + cr * C), (GX + cc * C, GY + cr * C),
       (GX + cc * C, GY + (cr + 1) * C), (GX, GY + (cr + 1) * C)]
body += poly(ctx, INK, 2.2, close=True)
# the current pixel
px, py = GX + cc * C, GY + cr * C
left, up = img[cr][cc - 1], img[cr - 1][cc]
mean = 0.6 * up + 0.4 * left
logits = [-((v - mean) ** 2) / 1.6 for v in range(LEVELS)]
z = sum(math.exp(l) for l in logits)
probs = [math.exp(l) / z for l in logits]
sample = max(range(LEVELS), key=lambda v: probs[v])   # the drawn level
body += rect(px, py, C, C, ACCENT_SOFT, ACCENT, 3.4, rx=1)
for (nr, nc) in ((cr, cc - 1), (cr - 1, cc)):
    body += rect(GX + nc * C + 3, GY + nr * C + 3, C - 6, C - 6, "none", SECOND, 2.6, rx=2)

# distribution bars on the right
BX0, BX1, BY = 394, 560, 290
bw = (BX1 - BX0) / LEVELS
body += line(BX0 - 6, BY, BX1 + 4, BY, GUIDE, THIN)
pmax = max(probs)
for v, p in enumerate(probs):
    h = p / pmax * 220
    x = BX0 + v * bw + 3
    col = ACCENT if v == sample else GREY
    body += rect(x, BY - h, bw - 6, h, col if v == sample else "#e6e8eb", col, 2.0, rx=1)
    body += rect(x, BY + 10, bw - 6, bw - 6, shade(v), GUIDE, 1.2, rx=1)
# sampled level feeds the current pixel
sxb = BX0 + sample * bw + bw / 2
RY = 356
body += poly([(sxb, BY + bw + 8), (sxb, RY), (px + C / 2, RY)], ACCENT, 2.6)
body += arrow(px + C / 2, RY, px + C / 2, py + C + 6, ACCENT, 2.6)
write(body)
