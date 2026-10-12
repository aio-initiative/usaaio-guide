"""FlashAttention: the causal score table is cut into 4 x 4 blocks and walked one block at a time. Blocks above the
diagonal are skipped (dashed), the bottom query block's row is in progress: two blocks done (soft terracotta), one in
fast memory now (terracotta), one to come. On the right, one query of that row: its scores arrive block by block
(grey dots), the running largest score climbs as a staircase (terracotta) and the running softmax total (teal) is
shrunk by e^(m_old - m_new) whenever the largest grows, then adds e^(s - m) for each new key.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

NB, NC = 4, 4                      # blocks per side, cells per block
GX, GY, SZ = 40, 60, 280
B = SZ / NB
C = B / NC
ROW, CUR = 3, 2                    # query block in progress, key block in fast memory now

body = ""
for bi in range(NB):
    for bj in range(NB):
        x, y = GX + bj * B, GY + bi * B
        if bj > bi:
            body += rect(x + 5, y + 5, B - 10, B - 10, "none", GUIDE, THIN, 3).replace("/>", ' stroke-dasharray="5 5"/>')
            continue
        if bi == ROW and bj == CUR:
            fill, stroke = ACCENT, ACCENT
        elif bi == ROW and bj < CUR:
            fill, stroke = ACCENT_SOFT, ACCENT
        elif bi == ROW:
            fill, stroke = BG, INK
        else:
            fill, stroke = "#eef0f2", GREY
        body += rect(x + 5, y + 5, B - 10, B - 10, fill, stroke, 2.2, 3)
        # the cells inside each block; on the diagonal only the causal half
        cell = (B - 10) / NC
        lc = BG if (bi == ROW and bj == CUR) else (GUIDE if stroke == GREY else ACCENT if fill == ACCENT_SOFT else GUIDE)
        for k in range(1, NC):
            body += line(x + 5 + k * cell, y + 5, x + 5 + k * cell, y + B - 5, lc, 1)
            body += line(x + 5, y + 5 + k * cell, x + B - 5, y + 5 + k * cell, lc, 1)
        if bi == bj:
            tri = [(x + 5, y + 5)]
            for k in range(NC):
                tri += [(x + 5 + (k + 1) * cell, y + 5 + k * cell), (x + 5 + (k + 1) * cell, y + 5 + (k + 1) * cell)]
            tri += [(x + B - 5, y + 5)]
            body += poly(tri, "none", 0, BG, close=True)
            body += poly(tri[:-1], stroke, 1.4)
# the sweep along the row in progress
ay = GY + ROW * B + B / 2
body += arrow(GX + 10, GY + SZ + 22, GX + (CUR + 1) * B - 10, GY + SZ + 22, ACCENT, 2.6)

# Right: one query of that row, 16 keys in 4 blocks.
s = [0.3, -0.4, 0.8, 0.1, 1.2, 0.5, 1.6, 0.9, 0.7, 2.4, 1.1, 1.8, 1.3, 0.6, 2.0, 1.0]
PX0, PX1 = 365, 560
step = (PX1 - PX0) / len(s)
xs = [PX0 + (i + 0.5) * step for i in range(len(s))]

# top plot: scores and the running largest
T0, T1 = GY, GY + 125
lo, hi = -0.8, 2.7
sy = lambda v: T1 - (v - lo) / (hi - lo) * (T1 - T0)
# bottom plot: running total
L0, L1 = GY + 155, GY + SZ
m, tot, mx, tots = -1e9, 0.0, [], []
for b in range(NB):
    blk = s[b * NC:(b + 1) * NC]
    nm = max(m, max(blk))
    shrunk = tot * math.exp(m - nm) if tot else 0.0
    tots.append(("shrink", b, tot, shrunk))
    tot, m = shrunk, nm
    for v in blk:
        tot += math.exp(v - m)
        tots.append(("add", b, tot, None))
    mx.append(m)
tmax = max(t[2] for t in tots) * 1.08
ty = lambda v: L1 - v / tmax * (L1 - L0)

for b in range(NB):
    bx0, bx1 = PX0 + b * NC * step, PX0 + (b + 1) * NC * step
    if b == CUR:
        body += rect(bx0, T0 - 6, bx1 - bx0, L1 - T0 + 6, ACCENT_SOFT, "none", 0, 3)
    if b:
        body += line(bx0, T0 - 6, bx0, L1, GUIDE, THIN, dash="4 5")
body += axes(PX0, T0, PX1, T1) + axes(PX0, L0, PX1, L1)
for i, (x, v) in enumerate(zip(xs, s)):
    body += dot(x, sy(v), 4.6, INK if i // NC == CUR else GREY)
stair = []
for b in range(NB):
    bx0, bx1 = PX0 + b * NC * step, PX0 + (b + 1) * NC * step
    stair += [(bx0, sy(mx[b])), (bx1, sy(mx[b]))]
body += poly(stair, ACCENT, 3)
for b in range(1, NB):
    if mx[b] > mx[b - 1]:
        body += dot(PX0 + b * NC * step, sy(mx[b]), 5, ACCENT)

pts, i = [(PX0, ty(0))], 0
for kind, b, a, c in tots:
    bx0 = PX0 + b * NC * step
    if kind == "shrink":
        if b:
            pts += [(bx0, ty(a)), (bx0, ty(c))]
    else:
        pts += [(xs[i], ty(a))]
        i += 1
body += poly(pts, SECOND, 2.8)
write(body)
