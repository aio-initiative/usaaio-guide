"""The embedding lookup of "the cat sat": token IDs 1, 2, 3 pick three rows of the lesson's 8 x 4 table E, which form X.
Cells are shaded by value: terracotta for +1, teal for -1, white for 0."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

E = [[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
     [-1, -1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
ids = [1, 2, 3]
C = 34                                   # cell size
fill = lambda v: ACCENT if v > 0 else SECOND if v < 0 else BG

def grid(x0, y0, rows, outline=GREY):
    out = ""
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            out += rect(x0 + c * C, y0 + r * C, C, C, fill(v), GUIDE, 1.4, rx=0)
    out += rect(x0, y0, C * 4, C * len(rows), "none", outline, 2.2, rx=2)
    return out

# the table E in the middle
EX, EY = 216, 64
body = grid(EX, EY, E, INK)
# the three input tokens on the left, one marker per token, each joined to its row
TX = 72
for k, i in enumerate(ids):
    ty = 128 + k * 84
    ry = EY + i * C + C / 2
    body += rect(TX - 24, ty - 20, 48, 40, ACCENT_SOFT, ACCENT, 2.4, rx=6)
    body += dot(TX, ty, 6.5, ACCENT)
    body += poly([(TX + 24, ty), (TX + 52, ty), (EX - 50, ry), (EX - 14, ry)], GREY, 2.2)
    body += arrow(EX - 16, ry, EX - 5, ry, GREY, 2.2)
# highlight the looked-up rows
body += rect(EX - 4, EY + C - 4, C * 4 + 8, C * 3 + 8, "none", ACCENT, 3.2, rx=4)
for i in ids[1:]:
    body += line(EX - 4, EY + i * C, EX + C * 4 + 4, EY + i * C, ACCENT, 2.4)
# X on the right: the same three rows carried straight out
XX, XY = 424, EY + C
body += rect(XX - 10, XY - 10, C * 4 + 20, C * 3 + 20, ACCENT_SOFT, "none", 0, rx=8)
body += grid(XX, XY, [E[i] for i in ids], INK)
for k in range(3):
    y = XY + k * C + C / 2
    body += arrow(EX + C * 4 + 10, y, XX - 16, y, ACCENT, 2.6)
write(body)
