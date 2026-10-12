"""The classification layer of the running example: two pooled 2 x 2 maps flattened into 8 numbers, weighted into
two class scores, and a softmax turning the scores 3 and 0 into probabilities 0.953 and 0.047."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

VERT = [[6.0, 0.0], [6.0, 0.0]]                 # vertical pooled map
HORI = [[0.0, 0.0], [0.0, 0.0]]                 # horizontal pooled map
flat = [v for row in VERT for v in row] + [v for row in HORI for v in row]
W_ = [[0.25] * 4 + [0.0] * 4, [0.0] * 4 + [0.25] * 4]   # one weight per number, per class
scores = [sum(w * x for w, x in zip(ws, flat)) for ws in W_]
ex = [math.exp(s) for s in scores]
probs = [e / sum(ex) for e in ex]               # 0.953, 0.047
COL = [ACCENT, SECOND]
SOFT = [ACCENT_SOFT, SECOND_SOFT]


def fill(v, k):
    return COL[k] if v > 0 else BG


body = ""
# The two pooled maps, one per filter.
MC, MX = 40, 44
for k, (m, y0) in enumerate([(VERT, 56), (HORI, 264)]):
    for r in range(2):
        for c in range(2):
            body += rect(MX + c * MC, y0 + r * MC, MC, MC, fill(m[r][c], k), COL[k], 2.0, 0)
    body += rect(MX, y0, 2 * MC, 2 * MC, "none", COL[k], 2.8, 2)

# Flattened list of 8 numbers.
FX, FW, FH, FY = 196, 36, 34, 64
cells = []
for i, v in enumerate(flat):
    k = 0 if i < 4 else 1
    y = FY + i * FH
    body += rect(FX, y, FW, FH, fill(v, k), COL[k], 2.0, 0)
    cells.append((FX + FW, y + FH / 2))
body += rect(FX, FY, FW, 8 * FH, "none", INK, 2.4, 2)
body += arrow(MX + 2 * MC + 12, 96, FX - 12, FY + 2 * FH, GREY, 2.2)
body += arrow(MX + 2 * MC + 12, 304, FX - 12, FY + 6 * FH, GREY, 2.2)

# Weights into the two class scores; a line carries signal where weight x input > 0.
NX, NY, NR = 352, [140, 260], 20
lines = ""
for k in range(2):
    for i, (cx, cy) in enumerate(cells):
        w = W_[k][i]
        if w == 0:
            lines += line(cx, cy, NX - NR, NY[k], GUIDE, 1.0)
        else:
            live = w * flat[i] > 0
            lines += line(cx, cy, NX - NR, NY[k], COL[k], 3.2 if live else 1.6)
body += lines
for k in range(2):
    body += dot(NX, NY[k], NR, SOFT[k], COL[k], 2.6)
    # Score as a filled share of the node: score / 3 of the radius.
    body += dot(NX, NY[k], NR * scores[k] / max(scores) * 0.62, COL[k]) if scores[k] > 0 else ""

# Softmax into probability bars that share one track length, so they add up to the whole.
BX, BW, BH = 420, 140, 34
body += line(BX, 96, BX, 300, INK, 2.2)
for k in range(2):
    y = NY[k] - BH / 2
    body += arrow(NX + NR + 8, NY[k], BX - 10, NY[k], GREY, 2.0)
    body += rect(BX, y, BW, BH, "#eef0f2", "none", 0, 0)
    body += rect(BX, y, BW * probs[k], BH, COL[k], "none", 0, 0)
# The two probabilities stacked end to end fill exactly one track.
SY = 318
body += rect(BX, SY, BW * probs[0], 14, COL[0], "none", 0, 0)
body += rect(BX + BW * probs[0], SY, BW * probs[1], 14, COL[1], "none", 0, 0)
body += line(BX + BW, SY - 6, BX + BW, SY + 20, INK, 2.0)
write(body)
