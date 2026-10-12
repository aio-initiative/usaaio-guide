"""A batch of activations (rows = examples, columns = features): BatchNorm normalizes a column, LayerNorm a row, dropout zeroes cells and scales the rest."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(2)
N, Dm = 7, 5
X = [[random.gauss(0, 1) for _ in range(Dm)] for _ in range(N)]
CW, CH, TOP = 29, 38, 67
lefts = [42, 228, 414]


def shade(v):
    """Activation magnitude as a grey: larger |v| is darker."""
    t = min(abs(v) / 2.2, 1)
    g = int(246 - (246 - 120) * t)
    return f"#{g:02x}{g + 2:02x}{g + 4:02x}"


def tint(v, soft, strong):
    """A normalized value as a shade between the soft and strong form of a series colour."""
    t = min(abs(v) / 1.8, 1)
    a, b = [int(soft[i:i + 2], 16) for i in (1, 3, 5)], [int(strong[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(x + (y - x) * (0.15 + 0.85 * t)):02x}" for x, y in zip(a, b))


def grid(x0, cells, fills=None):
    out = ""
    for r in range(N):
        for c in range(Dm):
            f = fills[r][c] if fills else shade(cells[r][c])
            out += rect(x0 + c * CW, TOP + r * CH, CW, CH, f, BG, 2, 2)
    return out


body = ""
# BatchNorm: the statistics of one column (a feature across the batch).
bc = 2
col = [X[r][bc] for r in range(N)]
mu = sum(col) / N
sd = math.sqrt(sum((v - mu) ** 2 for v in col) / N)
bn = [row[:] for row in X]
for r in range(N):
    bn[r][bc] = (X[r][bc] - mu) / sd
f0 = [[shade(v) for v in row] for row in X]
for r in range(N):
    f0[r][bc] = tint(bn[r][bc], ACCENT_SOFT, ACCENT)
body += grid(lefts[0], bn, f0)
body += rect(lefts[0] + bc * CW - 3, TOP - 3, CW + 6, N * CH + 6, "none", ACCENT, 3.2, 5)

# LayerNorm: the statistics of one row (one example across its features).
lr = 3
ln = [row[:] for row in X]
m = sum(X[lr]) / Dm
s = math.sqrt(sum((v - m) ** 2 for v in X[lr]) / Dm)
ln[lr] = [(v - m) / s for v in X[lr]]
f1 = [[shade(v) for v in row] for row in X]
f1[lr] = [tint(v, SECOND_SOFT, SECOND) for v in ln[lr]]
body += grid(lefts[1], ln, f1)
body += rect(lefts[1] - 3, TOP + lr * CH - 3, Dm * CW + 6, CH + 6, "none", SECOND, 3.2, 5)

# Dropout p = 0.4: dropped cells become empty, survivors are scaled by 1 / (1 - p) and so look darker.
p = 0.4
fills = []
for r in range(N):
    row = []
    for c in range(Dm):
        if random.random() < p:
            row.append(None)
        else:
            row.append(shade(X[r][c] / (1 - p)))
    fills.append(row)
x0 = lefts[2]
for r in range(N):
    for c in range(Dm):
        if fills[r][c] is None:
            body += (f'<rect x="{x0 + c * CW + 3:.1f}" y="{TOP + r * CH + 3:.1f}" width="{CW - 6}" height="{CH - 6}" rx="2" '
                     f'fill="{BG}" stroke="{GREY}" stroke-width="1.6" stroke-dasharray="4 3"/>')
        else:
            body += rect(x0 + c * CW, TOP + r * CH, CW, CH, fills[r][c], BG, 2, 2)
write(body)
