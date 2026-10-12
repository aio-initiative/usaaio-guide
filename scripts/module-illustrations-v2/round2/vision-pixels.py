"""Grid versus list: the running 6 x 6 image (bright third column) kept as a grid, then flattened row by row into
one list. The pixels at row 1 and row 2 of column 3 touch in the grid (teal outline) but sit 6 places apart in the
list, joined by an arc. Rows 1 and 2 are tinted so the reader can follow them into the list."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N = 6
img = [[1.0 if c == 2 else 0.0 for c in range(N)] for r in range(N)]   # image[:, 2] = 1
flat = [v for row in img for v in row]

C = 30                                   # grid cell
G = N * C
gx, gy = (W - G) / 2, 34
L = 32                                   # list cell
SHOW = 15                                # cells of the list drawn, then an ellipsis
lx, ly = 40, 320
pair = (2, 8)                            # list indices of (row 1, col 3) and (row 2, col 3)
rowfill = [SECOND_SOFT, "#e8eaed"]       # rows 1 and 2, to follow them into the list


def cell(x, y, s, v, tint):
    return rect(x, y, s, s, fill=ACCENT if v else tint, stroke=GUIDE, sw=THIN, rx=0)


body = ""
for r in range(N):
    for c in range(N):
        body += cell(gx + c * C, gy + r * C, C, img[r][c], rowfill[r] if r < 2 else BG)
body += rect(gx, gy, G, G, stroke=INK, sw=LINE, rx=3)
body += rect(gx + 2 * C, gy, C, 2 * C, stroke=SECOND, sw=3.2, rx=2)          # the touching pair

for i in range(SHOW):
    body += cell(lx + i * L, ly, L, flat[i], rowfill[i // N] if i < 2 * N else BG)
body += rect(lx, ly, SHOW * L, L, stroke=INK, sw=LINE, rx=3)
for k in range(3):
    body += dot(lx + SHOW * L + 14 + k * 12, ly + L / 2, 2.6, GREY)
for i in pair:
    body += rect(lx + i * L, ly, L, L, stroke=SECOND, sw=3.2, rx=2)
# arc joining the same two pixels, now 6 places apart
a, b = lx + (pair[0] + 0.5) * L, lx + (pair[1] + 0.5) * L
body += poly([(a + (b - a) * t / 40, ly - 6 - 38 * math.sin(math.pi * t / 40)) for t in range(41)], SECOND, 2.2, dash="6 5")
# flatten: arrow from the grid down to the list
body += arrow(W / 2, gy + G + 14, W / 2, ly - 52, GREY, 2.6)
write(body)
