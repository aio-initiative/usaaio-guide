"""The module's example: five passengers and a test row become the matrix a model receives. Left, the raw table:
age and fare as bars, the port as a shape (circle S, triangle C, square Q), one age missing (dashed terracotta cell),
the test row set apart in teal. Right, the matrix: a ones column, age and fare standardized with the training mean
and standard deviation (signed bars about a centre line, the imputed median age landing exactly on zero), and the
port one-hot encoded with the first category, C, dropped, so the triangle row has two empty cells."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

train = [(20, 10, "S"), (30, 16, "C"), (None, 18, "S"), (40, 22, "Q"), (50, 34, "S")]
test = (45, 36, "Q")
ages = sorted(a for a, _, _ in train if a is not None)
med = (ages[1] + ages[2]) / 2
filled = [(a if a is not None else med, f, p) for a, f, p in train]
def stats(v):
    m = sum(v) / len(v)
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
am, asd = stats([a for a, _, _ in filled])
fm, fsd = stats([f for _, f, _ in filled])

RH, Y0, GAP = 44, 56, 22
rows = [(r, False, train[i][0] is None) for i, r in enumerate(filled)] + [(test, True, False)]
def ry(i): return Y0 + i * RH + (GAP if i == len(train) else 0)

def shape(kind, x, y, color, s=10):
    if kind == "S": return dot(x, y, s, color)
    if kind == "C": return poly([(x, y - s * 1.1), (x + s, y + s * 0.8), (x - s, y + s * 0.8)], color, 1, fill=color, close=True)
    return rect(x - s * 0.9, y - s * 0.9, s * 1.8, s * 1.8, color, "none", 0, 2)

b = ""
# raw table
LX, LW = [40, 116, 192], [76, 76, 46]
vmax = 50
for i, ((a, f, p), is_test, missing) in enumerate(rows):
    y = ry(i); c = SECOND if is_test else INK
    for x, w in zip(LX, LW):
        b += rect(x, y, w, RH, "none", GUIDE, THIN, 0)
    if missing:
        b += rect(LX[0] + 7, y + 7, LW[0] - 14, RH - 14, "none", ACCENT, 2.2, 3).replace("/>", ' stroke-dasharray="6 4"/>')
    else:
        b += rect(LX[0] + 8, y + RH / 2 - 7, (LW[0] - 16) * a / vmax, 14, c, "none", 0, 2)
    b += rect(LX[1] + 8, y + RH / 2 - 7, (LW[1] - 16) * f / vmax, 14, c, "none", 0, 2)
    b += shape(p, LX[2] + LW[2] / 2, y + RH / 2, c)
yc = (Y0 + ry(len(rows) - 1) + RH) / 2
b += arrow(252, yc, 290, yc, GREY, 3)
# model matrix
cols = [(302, 32), (334, 86), (420, 86), (506, 28), (534, 28)]   # ones, age z, fare z, port Q, port S
half = 86 / 2 - 6                                                 # bar length for |z| = 2
for i, ((a, f, p), is_test, missing) in enumerate(rows):
    y = ry(i); c = SECOND if is_test else INK; yy = y + RH / 2
    for x, w in cols:
        b += rect(x, y, w, RH, "none", GUIDE, THIN, 0)
    b += dot(cols[0][0] + cols[0][1] / 2, yy, 5, GREY if not is_test else SECOND)
    for (x, w), z in ((cols[1], (a - am) / asd), (cols[2], (f - fm) / fsd)):
        cx = x + w / 2
        b += line(cx, y + 6, cx, y + RH - 6, GUIDE, THIN)
        if missing and x == cols[1][0]:
            b += dot(cx, yy, 7, ACCENT)
        else:
            L = half * z / 2.0
            b += rect(min(cx, cx + L), yy - 7, abs(L), 14, c, "none", 0, 2)
    if p == "Q": b += shape("Q", cols[3][0] + cols[3][1] / 2, yy, c, 8)
    if p == "S": b += shape("S", cols[4][0] + cols[4][1] / 2, yy, c, 8)
write(b)
