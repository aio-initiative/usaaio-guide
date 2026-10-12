"""Byte-pair encoding of the unseen word "lowest": its six characters are joined by the learned merges in order
(e+s, es+t, l+o, lo+w), one level per merge, leaving the two tokens low and est, which are then looked up as rows
of the fixed vocabulary (their IDs)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

chars = list("lowest")
merges = [("e", "s"), ("es", "t"), ("l", "o"), ("lo", "w")]     # the lesson's joins 1-4, in order

CX0, CW, CG = 48, 50, 10            # character cells
ROW0, RH, STEP = 318, 40, 58        # bottom row y, piece height, rise per merge
cell_x = [CX0 + i * (CW + CG) for i in range(len(chars))]

# Apply the merges for real and record the span (first, last character) and level of each piece.
pieces = [(c, i, i, 0) for i, c in enumerate(chars)]           # (string, first, last, level)
nodes = []
for lvl, (a, b) in enumerate(merges, 1):
    for k in range(len(pieces) - 1):
        if pieces[k][0] == a and pieces[k + 1][0] == b:
            pa, pb = pieces[k], pieces[k + 1]
            new = (a + b, pa[1], pb[2], lvl)
            nodes.append((new, pa, pb))
            pieces[k:k + 2] = [new]
            break
assert [p[0] for p in pieces] == ["low", "est"]


def box(p):
    s, i, j, lvl = p
    x0, x1 = cell_x[i], cell_x[j] + CW
    y = ROW0 - lvl * STEP
    return x0, y, x1 - x0, RH


final = {pieces[0][0]: (ACCENT, ACCENT_SOFT), pieces[1][0]: (SECOND, SECOND_SOFT)}
body = ""
# Connectors from each merged piece down to its two parts.
for new, pa, pb in nodes:
    nx, ny, nw, nh = box(new)
    for part in (pa, pb):
        px, py, pw, ph = box(part)
        cx = px + pw / 2
        body += line(cx, ny + nh, cx, py, GREY, 1.8)
# Pieces: characters as plain cells, intermediate joins grey, the two final tokens coloured.
for p in [(c, i, i, 0) for i, c in enumerate(chars)] + [n[0] for n in nodes]:
    x, y, w, h = box(p)
    if p[0] in final:
        stroke, fill = final[p[0]]
        body += rect(x, y, w, h, fill=fill, stroke=stroke, sw=3.0, rx=7)
    elif p[3] == 0:
        body += rect(x, y, w, h, fill=BG, stroke=INK, sw=2.4, rx=6)
        body += dot(x + w / 2, y + h / 2, 4.5, GREY)
    else:
        body += rect(x, y, w, h, fill="#eef0f2", stroke=GREY, sw=2.2, rx=7)

# The fixed vocabulary as a column of rows; each final token points at its own row.
# Row pitch 2 * STEP / 4 lines each token up with its own row, so both arrows run straight across.
VX, VW, VN, VH = 488, 64, 11, 22
VP = 2 * STEP / 4
VG = VP - VH
VY0 = ROW0 - 4 * STEP + RH / 2 - 2 * VP - VH / 2
rows = {"low": 2, "est": 6}
for r in range(VN):
    y = VY0 + r * (VH + VG)
    tok = next((t for t, rr in rows.items() if rr == r), None)
    if tok:
        stroke, fill = final[tok]
        body += rect(VX, y, VW, VH, fill=fill, stroke=stroke, sw=2.6, rx=4)
    else:
        body += rect(VX, y, VW, VH, fill=BG, stroke=GUIDE, sw=1.8, rx=4)
for tok, r in rows.items():
    p = next(n[0] for n in nodes if n[0][0] == tok)
    x, y, w, h = box(p)
    ty = VY0 + r * (VH + VG) + VH / 2
    sx, sy = x + w + 8, y + h / 2
    body += arrow(sx, sy, VX - 8, ty, final[tok][0], 2.8)

write(body)
