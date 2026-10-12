"""Broadcasting: a (3, 1) column plus a (1, 4) row, each stretched along its size-1 axis, fill a (3, 4) grid."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

col = [1.0, 3.0, 2.0]               # shape (3, 1)
row = [0.0, 2.0, 1.0, 3.0]          # shape (1, 4)
out = [[a + b for b in row] for a in col]   # col[:, None] + row[None, :]

gx, gy, cw, ch = 220, 140, 82, 70   # result grid
vx, hy = 56, 40                     # column vector x, row vector y
cell = lambda x, y, w, h, fill, op, stroke, sw=2.2, dash=None: (
    f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="4" fill="{fill}" fill-opacity="{op:.2f}" '
    f'stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
body = ""
for i, a in enumerate(col):         # the column, and its stretch to the right
    y = gy + i * ch
    body += cell(vx, y + 4, 72, ch - 8, ACCENT, 0.2 + 0.8 * a / 3, ACCENT)
    body += arrow(vx + 82, y + ch / 2, gx - 12, y + ch / 2, ACCENT, 2.4)
for j, b in enumerate(row):         # the row, and its stretch downward
    x = gx + j * cw
    body += cell(x + 4, hy, cw - 8, 62, SECOND, 0.2 + 0.8 * b / 3, SECOND)
    body += arrow(x + cw / 2, hy + 70, x + cw / 2, gy - 10, SECOND, 2.4)
for i, r in enumerate(out):         # the result: each cell carries its row's column entry and its column's row entry
    for j, v in enumerate(r):
        x, y = gx + j * cw, gy + i * ch
        a, b = col[i], row[j]
        x0, y0, x1, y1 = x + 4, y + 4, x + cw - 4, y + ch - 4
        body += f'<path d="M{x0},{y0} L{x0},{y1} L{x1},{y1} Z" fill="{ACCENT}" fill-opacity="{0.2 + 0.8 * a / 3:.2f}"/>'
        body += f'<path d="M{x0},{y0} L{x1},{y0} L{x1},{y1} Z" fill="{SECOND}" fill-opacity="{0.2 + 0.8 * b / 3:.2f}"/>'
        body += cell(x0, y0, cw - 8, ch - 8, "none", 0, INK, 1.8)
write(body)
