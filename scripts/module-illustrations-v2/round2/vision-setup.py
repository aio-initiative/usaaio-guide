"""The same object in two places: a 2 x 2 bright block in the top-left of one 6 x 6 image and the bottom-right of
another. A dashed ghost in the second marks where it used to be, so the two images share no bright pixel."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, C = 6, 38                     # grid size, cell size
G = N * C                        # 228
gy = (H - G) / 2
gx = [42, W - 42 - G]            # two grids
blocks = [(0, 0), (4, 4)]        # top-left corner (row, col) of the 2 x 2 object in each image


def grid(x0, r0, c0, fill, edge):
    out = ""
    for r in range(N):
        for c in range(N):
            on = r0 <= r < r0 + 2 and c0 <= c < c0 + 2
            out += rect(x0 + c * C, gy + r * C, C, C, fill=fill if on else BG, stroke=GUIDE, sw=THIN, rx=0)
    out += rect(x0 + c0 * C, gy + r0 * C, 2 * C, 2 * C, stroke=edge, sw=LINE, rx=2)
    out += rect(x0, gy, G, G, stroke=INK, sw=LINE, rx=3)
    return out


body = grid(gx[0], *blocks[0], ACCENT, ACCENT)
body += grid(gx[1], *blocks[1], ACCENT, ACCENT)
# ghost: where the object sat in the first image, dashed, and none of its cells is bright here
body += rect(gx[1] + 3, gy + 3, 2 * C - 6, 2 * C - 6, stroke=GREY, sw=2.2, rx=2).replace("/>", ' stroke-dasharray="7 5"/>')
# the move between the two images
ym = gy + G / 2
body += arrow(gx[0] + G + 12, ym, gx[1] - 12, ym, GREY, 2.6)
write(body)
