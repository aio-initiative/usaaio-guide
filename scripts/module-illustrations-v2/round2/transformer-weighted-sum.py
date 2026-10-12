"""The weighted sum: three attention weights as bars, and the output as the weighted average of the three value
vectors, a point inside their triangle pulled toward the heaviest value."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

w = [0.115, 0.709, 0.175]                                    # head-1 weights of "sat" over the, cat, sat
V = [(1.000, 1.000), (0.841, 1.540), (0.909, -0.416)]        # head-1 values
o = (sum(a * v[0] for a, v in zip(w, V)), sum(a * v[1] for a, v in zip(w, V)))
cols = [GREY, ACCENT, SECOND]

# Weight bars on the left.
body = ""
base = 330
for i, a in enumerate(w):
    x = 50 + i * 52
    body += rect(x, base - a * 340, 38, a * 340, cols[i], "none", 0, 3)
body += line(40, base, 208, base, GUIDE, 1.6)

# Value plane on the right (an affine map: weighted averages are preserved).
X = lambda x: 270 + (x - 0.80) / 0.24 * 290
Y = lambda y: 345 - (y + 0.55) / 2.2 * 300
P = lambda v: (X(v[0]), Y(v[1]))
body += poly([P(v) for v in V], GUIDE, 1.6, fill="#f4f5f7", close=True)
for i, v in enumerate(V):
    body += line(*P(o), *P(v), cols[i], 1.2 + 7 * w[i])
for i, v in enumerate(V):
    body += dot(*P(v), 9, BG, cols[i], 3)
    body += dot(*P(v), 4 + 8 * math.sqrt(w[i]), cols[i])
body += dot(*P(o), 11, INK, BG, 3)
write(body)
