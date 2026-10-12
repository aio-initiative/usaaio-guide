"""The three costs of running a model. Left, repeated work: six generation runs as rows of tokens, each run redoing
every earlier token (grey) to add one new one (terracotta). Middle, memory: a memory bank joined to the chip by one
narrow pipe carrying a single byte, while the chip holds 12 x 13 = 156 units of compute, the operations an A100 can do
in the time it reads that byte. Right, long texts: the table of scores for 4 tokens (terracotta) inside the table for
8 tokens; twice the tokens along each edge, four times the area."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

body = ""
cy = H / 2

# 1. repeated work: run k redoes k - 1 old tokens and adds one
c, rh, gap = 27, 34, 15
px = 44
top = cy - (6 * rh + 5 * gap) / 2
for k in range(1, 7):
    y = top + (k - 1) * (rh + gap)
    for j in range(k):
        x = px + j * (c + 2)
        new = j == k - 1
        body += rect(x, y, c - 2, rh, fill=ACCENT if new else "#e6e8eb", stroke=ACCENT if new else GREY, sw=1.6, rx=3)

# 2. memory: bank -> narrow pipe -> chip
mx, mw = 240, 124
my, mh = top, 58
body += rect(mx, my, mw, mh, fill="#f3f4f6", stroke=INK, sw=LINE, rx=5)
for i in range(1, 5):
    body += line(mx + 10, my + i * mh / 5, mx + mw - 10, my + i * mh / 5, GUIDE, THIN)
chip_y = top + 6 * rh + 5 * gap - mw
pipe_x = mx + mw / 2
body += line(pipe_x, my + mh, pipe_x, chip_y, INK, 2.2)
body += rect(pipe_x - 7, (my + mh + chip_y) / 2 - 7, 14, 14, fill=ACCENT, stroke="none", sw=0, rx=2)
body += rect(mx, chip_y, mw, mw, fill=BG, stroke=INK, sw=LINE, rx=5)
cols, rows = 12, 13
sx, sy = (mw - 20) / (cols - 1), (mw - 20) / (rows - 1)
for r in range(rows):
    for q in range(cols):
        body += dot(mx + 10 + q * sx, chip_y + 10 + r * sy, 2.6, SECOND)

# 3. long texts: the 4-token table inside the 8-token table
G = 160
qx = W - 44 - G
qy = top + 6 * rh + 5 * gap - G
n = 8
cs = G / n
body += rect(qx, qy, G / 2, G / 2, fill=ACCENT_SOFT, stroke="none", sw=0, rx=0)
for i in range(1, n):
    body += line(qx + i * cs, qy, qx + i * cs, qy + G, GUIDE, THIN)
    body += line(qx, qy + i * cs, qx + G, qy + i * cs, GUIDE, THIN)
body += rect(qx, qy, G / 2, G / 2, stroke=ACCENT, sw=2.8, rx=0)
body += rect(qx, qy, G, G, stroke=INK, sw=LINE, rx=2)
# the tokens along the top edge
for i in range(n):
    body += dot(qx + (i + 0.5) * cs, qy - 16, 5, ACCENT if i < n // 2 else INK)
write(body)
