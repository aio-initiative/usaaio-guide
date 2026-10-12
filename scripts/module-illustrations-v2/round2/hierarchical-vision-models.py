"""Swin's shifted windows. Left: an 8 x 8 patch map split into four M x M windows (M = 4); attention runs only inside
a window, so the four terracotta patches, one in each window, cannot see each other. Right: the
next block shifts the partition by floor(M/2) = 2 patches (old boundaries faint), which leaves nine ragged windows;
the centre one (teal) now holds all four patches, and the terracotta lines are the attention they can now share."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, M = 8, 4
S = M // 2                              # the shift, floor(M / 2)
C = 30                                  # patch size; an 8 x 8 map is 240 px
GY = (H - N * C) / 2
AX, BX = 40, W - 40 - N * C

def patches(x0):
    o = ""
    for k in range(1, N):
        o += line(x0 + k * C, GY, x0 + k * C, GY + N * C, GUIDE, 1.1) + line(x0, GY + k * C, x0 + N * C, GY + k * C, GUIDE, 1.1)
    return o

def windows(x0, offset, color, sw, dash=None):
    """Window boundaries every M patches, starting at the given offset, plus the outer frame."""
    o = rect(x0, GY, N * C, N * C, "none", color, sw, 0) if dash is None else ""
    for k in range(offset, N, M):
        if k == 0:
            continue
        o += line(x0 + k * C, GY, x0 + k * C, GY + N * C, color, sw, dash) + line(x0, GY + k * C, x0 + N * C, GY + k * C, color, sw, dash)
    return o

# Four patches, one in each original window, at the corners of the shifted centre window.
pts = [(2, 2), (2, 5), (5, 2), (5, 5)]
win = lambda i, j, off: ((i - off) % N // M, (j - off) % N // M) if off == 0 else ((i + M - off) // M, (j + M - off) // M)
assert len({win(i, j, 0) for i, j in pts}) == 4 and len({win(i, j, S) for i, j in pts}) == 1
centre = lambda x0, i, j: (x0 + (j + 0.5) * C, GY + (i + 0.5) * C)

body = ""
# left: regular partition, each window faintly tinted in turn so the four are distinct
for wi in range(N // M):
    for wj in range(N // M):
        if (wi + wj) % 2 == 0:
            body += rect(AX + wj * M * C, GY + wi * M * C, M * C, M * C, "#f3f4f6", "none", 0, 0)
body += patches(AX) + windows(AX, 0, INK, 3.4)
for i, j in pts:
    body += dot(*centre(AX, i, j), 8, ACCENT)

# right: shifted partition, centre window shaded teal, old boundaries dashed for reference
body += rect(BX + S * C, GY + S * C, M * C, M * C, SECOND_SOFT, "none", 0, 0)
body += patches(BX) + windows(BX, 0, GREY, 1.8, "6 5")
body += rect(BX, GY, N * C, N * C, "none", SECOND, 3.4, 0) + windows(BX, S, SECOND, 3.4)
cs = [centre(BX, i, j) for i, j in pts]
for a in range(4):
    for b in range(a + 1, 4):
        body += line(*cs[a], *cs[b], ACCENT, 2.2)
for x, y in cs:
    body += dot(x, y, 8, ACCENT)

body += arrow(AX + N * C + 8, H / 2, BX - 8, H / 2, GREY, 2.6)
write(body)
