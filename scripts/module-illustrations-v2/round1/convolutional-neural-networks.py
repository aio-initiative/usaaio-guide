"""A small CNN as a shrinking pyramid, with one deep unit's receptive field traced back to the input.
A 16x16 mountain image goes through a real 3x3 edge convolution (padding 1) and a 2x2 max pool to an 8x8 map,
then a 3x3 summing convolution and another pool give a 4x4 map; each map is shaded by its actual values. One unit of the 4x4 map (terracotta) sees a
4x4 block of the 8x8 map and a 10x10 patch of the input, the receptive field r = 1 + 2 + 1 + 2*2 + 1*2 = 10."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N = 16
ridge = lambda c: 11.5 - 9 * math.exp(-((c - 5.5) / 3.2) ** 2) - 5 * math.exp(-((c - 11.5) / 2.2) ** 2)
img = [[1.0 if r >= ridge(c) else 0.0 for c in range(N)] for r in range(N)]

KX = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
KY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]


def conv(x, k):
    n = len(x)
    at = lambda r, c: x[min(max(r, 0), n - 1)][min(max(c, 0), n - 1)]   # replicate the border
    return [[sum(at(r + u - 1, c + v - 1) * k[u][v] for u in range(3) for v in range(3)) for c in range(n)] for r in range(n)]


def edges(x):
    gx, gy = conv(x, KX), conv(x, KY)
    return [[math.hypot(gx[r][c], gy[r][c]) for c in range(len(x))] for r in range(len(x))]


def pool(x):
    n = len(x) // 2
    return [[max(x[2 * r + a][2 * c + b] for a in (0, 1) for b in (0, 1)) for c in range(n)] for r in range(n)]


m1 = pool(edges(img))           # 8x8
m2 = pool(conv(m1, [[1] * 3] * 3))   # 4x4, a 3x3 sum of the edge map


def mix(a, b, t):
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(ca[i] + (cb[i] - ca[i]) * t):02x}" for i in range(3))


CELL = 16
GAP = 40
widths = [N * CELL, 8 * CELL, 4 * CELL]
x0 = (W - sum(widths) - 2 * GAP) / 2
xs = [x0, x0 + widths[0] + GAP, x0 + widths[0] + widths[1] + 2 * GAP]
ys = [(H - w) / 2 for w in widths]

body = ""
for k, grid in enumerate([img, m1, m2]):
    n = len(grid)
    top = max(max(row) for row in grid) or 1
    for r in range(n):
        for c in range(n):
            v = grid[r][c] / top
            fill = mix(BG, GREY, 0.55) if k == 0 and v > 0 else mix(BG, INK, 0.85 * v) if k else BG
            body += f'<rect x="{xs[k] + c * CELL:.1f}" y="{ys[k] + r * CELL:.1f}" width="{CELL}" height="{CELL}" fill="{fill}" stroke="{GUIDE}" stroke-width="0.8"/>'
    body += rect(xs[k], ys[k], widths[k], widths[k], "none", INK, 2, 1)

# receptive field of unit (1, 2) of the 4x4 map: pool -> rows 2i..2i+1, conv -> one more on each side
unit = (1, 2)
f8 = [(2 * i - 1, 2 * i + 2) for i in unit]            # rows/cols in the 8x8 map, inclusive
f16 = [(2 * a - 1, 2 * b + 2) for a, b in f8]          # rows/cols in the input, inclusive
f16 = [(max(a, 0), min(b, N - 1)) for a, b in f16]
regions = [
    (xs[0] + f16[1][0] * CELL, ys[0] + f16[0][0] * CELL, (f16[1][1] - f16[1][0] + 1) * CELL, (f16[0][1] - f16[0][0] + 1) * CELL),
    (xs[1] + f8[1][0] * CELL, ys[1] + f8[0][0] * CELL, (f8[1][1] - f8[1][0] + 1) * CELL, (f8[0][1] - f8[0][0] + 1) * CELL),
    (xs[2] + unit[1] * CELL, ys[2] + unit[0] * CELL, CELL, CELL),
]
for (ax, ay, aw, ah), (bx, by, bw, bh) in zip(regions, regions[1:]):
    body += line(ax + aw, ay, bx, by, ACCENT, 1.8, "6 5")
    body += line(ax + aw, ay + ah, bx, by + bh, ACCENT, 1.8, "6 5")
for i, (rx, ry, rw, rh) in enumerate(regions):
    if i == 2:
        body += rect(rx, ry, rw, rh, ACCENT, ACCENT, 2, 1)
    else:
        body += rect(rx, ry, rw, rh, "none", ACCENT, 3.2, 2)

write(body)
