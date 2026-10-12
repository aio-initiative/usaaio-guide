"""One token's vector (bars) multiplied by three weight matrices (grids with their copied entries marked) into its query, key and value."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

h = [0.909, -0.416, 1.020, 2.000]                     # the token "sat" from the lesson
# Each output column copies one dimension of h (the lesson's W_Q, W_K, W_V); columns 1-2 head 1, 3-4 head 2.
picks = {"Q": [2, 3, 1, 3], "K": [1, 2, 1, 3], "V": [0, 1, 2, 3]}
cols = {"Q": (ACCENT, ACCENT_SOFT), "K": (SECOND, SECOND_SOFT), "V": (INK, "#eceef1")}
S = 2.000                                             # value that fills a full bar half
C = 22                                                # matrix cell size

body = ""
# Input token: four vertical-cell column with horizontal bars from a centre line.
ix, iw, cell = 70, 120, 34
iy0 = 200 - 2 * cell
cx = ix + iw / 2
body += rect(ix, iy0, iw, 4 * cell, BG, GREY, 1.6)
for i, v in enumerate(h):
    y = iy0 + i * cell
    if i:
        body += line(ix, y, ix + iw, y, GUIDE, 1.2)
    w = v / S * (iw / 2 - 8)
    body += rect(min(cx, cx + w), y + 8, abs(w), cell - 16, INK, "none", 0, 2)
body += line(cx, iy0, cx, iy0 + 4 * cell, GREY, 1.2)

for k, (name, ys) in enumerate(zip("QKV", (88, 200, 312))):
    col, soft = cols[name]
    # Weight matrix: 4 x 4 grid, filled cell where the column copies that row's dimension.
    mx, my = 262, ys - 2 * C
    for j, r in enumerate(picks[name]):
        body += rect(mx + j * C, my + r * C, C, C, soft, "none", 0, 0)
        body += dot(mx + j * C + C / 2, my + r * C + C / 2, 5.5, col)
    for t in range(5):
        body += line(mx + t * C, my, mx + t * C, my + 4 * C, GUIDE, 1.1)
        body += line(mx, my + t * C, mx + 4 * C, my + t * C, GUIDE, 1.1)
    body += rect(mx, my, 4 * C, 4 * C, "none", col, 2.2, 2)
    body += poly([(ix + iw + 8, 200), (ix + iw + 34, 200), (mx - 34, ys), (mx - 10, ys)], GREY, 2)
    body += arrow(mx - 30, ys, mx - 8, ys, GREY, 2)
    # Output vector: h times the matrix, drawn as four vertical bars around a baseline, heads split by a gap.
    out = [h[r] for r in picks[name]]
    ox, base, bw = 420, ys + 8, 22
    body += arrow(mx + 4 * C + 10, ys, ox - 14, ys, GREY, 2)
    for j, v in enumerate(out):
        x = ox + j * (bw + 8) + (14 if j >= 2 else 0)
        hgt = v / S * 40
        body += rect(x, min(base, base - hgt), bw, abs(hgt), col, "none", 0, 2)
    body += line(ox - 6, base, ox + 4 * (bw + 8) + 14, base, GUIDE, 1.4)
    body += line(ox + 2 * (bw + 8) + 3, base - 44, ox + 2 * (bw + 8) + 3, base + 24, GUIDE, 1.2, dash="3 4")
write(body)
