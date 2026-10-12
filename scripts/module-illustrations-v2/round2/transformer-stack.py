"""Stacking blocks: the same-width row of token vectors passes through two identical blocks. Block 1's
previous-token head marks the token after the first "Harry"; block 2 lets the second "Harry" find that mark and copy
"Potter", the two-round induction step one block cannot do, so the top predicts the blank slot."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

# Harry Potter went home . Later , Harry [blank]
N = 9
xs = [60 + 60 * i for i in range(N)]
rows = [342, 212, 82]               # H0 (input), H1 (after block 1), H2 (after block 2)
HARRY, POTTER, LAST = (0, 7), 1, 7

body = ""
# two identical block frames, one between each pair of rows
for ya, yb in ((rows[0], rows[1]), (rows[1], rows[2])):
    top, bot = yb + 28, ya - 28
    body += f'<rect x="36" y="{top}" width="494" height="{bot - top}" rx="12" fill="{BG}" stroke="{GREY}" stroke-width="1.6" stroke-dasharray="7 6"/>'

# residual stream: every token keeps its own column all the way up
for i in range(N - 1):
    body += line(xs[i], rows[0], xs[i], rows[2], GUIDE, THIN)

# block 1: a previous-token head, each token looks one place back
for i in range(1, N - 1):
    c, w = (SECOND, 2.8) if i == POTTER else (GREY, 1.6)
    body += line(xs[i - 1], rows[0], xs[i], rows[1], c, w)

# block 2: the last "Harry" finds the token whose previous token was "Harry" and copies it
x0, y0, x1, y1 = xs[POTTER], rows[1], xs[LAST], rows[2]
cx, cy = 250, 168                   # arrive from below so the curve crosses no other token
bez = [((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1, (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1)
       for t in [k / 40 for k in range(41)]]
stop = [p for p in bez if math.hypot(p[0] - x1, p[1] - y1) > 13]
body += poly(stop[:-2], ACCENT, 3.4)
body += arrow(stop[-3][0], stop[-3][1], stop[-1][0], stop[-1][1], ACCENT, 3.4)

# the token dots on each row
for r, y in enumerate(rows):
    for i in range(N - 1):
        if i in HARRY and not (r == 2 and i == LAST):
            body += dot(xs[i], y, 9, SECOND)
        elif i == POTTER:
            body += dot(xs[i], y, 9, ACCENT)
            if r >= 1:
                body += f'<circle cx="{xs[i]}" cy="{y}" r="14" fill="none" stroke="{SECOND}" stroke-width="2.6"/>'
        elif r == 2 and i == LAST:
            body += dot(xs[i], y, 9, ACCENT)
            body += f'<circle cx="{xs[i]}" cy="{y}" r="14" fill="none" stroke="{SECOND}" stroke-width="2.6"/>'
        else:
            body += dot(xs[i], y, 7, INK)

# the blank slot: empty below, filled with the copied token at the top
for y in rows[:2]:
    body += f'<circle cx="{xs[8]}" cy="{y}" r="11" fill="{BG}" stroke="{GREY}" stroke-width="2" stroke-dasharray="4 4"/>'
body += arrow(xs[LAST] + 16, rows[2], xs[8] - 15, rows[2], ACCENT, 2.6)
body += dot(xs[8], rows[2], 12, ACCENT_SOFT, ACCENT, 2.8)

write(body)
