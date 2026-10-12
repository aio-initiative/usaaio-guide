"""Predict: the final vector of "sat" is dotted with every token's embedding (weight tying) and the softmax gives
eight probabilities, with the true next token "on" at 0.741. A dashed guide carries that height across to the
cross-entropy curve, loss = -log p, drawn on the same probability scale, where it reads off the small loss."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

h = [-0.639, -0.891, -0.136, 1.667]                       # final vector of "sat" (Step 10)
E = [[0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
     [-1, -1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
logits = [sum(a * b for a, b in zip(h, e)) for e in E]
z = sum(math.exp(v) for v in logits)
p = [math.exp(v) / z for v in logits]
TRUE = 4                                                    # "on"

base, top = 352, 48
Y = lambda q: base - (base - top) * q

body = ""
# the final vector: four cells, each a signed bar about its centre line
vx, vy, cw, ch = 40, 120, 60, 40
body += line(vx + cw / 2, vy - 6, vx + cw / 2, vy + 4 * ch + 6, GUIDE, THIN)
for k, v in enumerate(h):
    y = vy + k * ch
    body += rect(vx, y, cw, ch, BG, INK, 2, 2)
    w = 24 * v / max(abs(t) for t in h)
    body += f'<rect x="{min(vx + cw / 2, vx + cw / 2 + w):.1f}" y="{y + 11}" width="{abs(w):.1f}" height="{ch - 22}" fill="{SECOND}"/>'
body += arrow(vx + cw + 10, vy + 2 * ch, 146, vy + 2 * ch, INK, 2.4)

# the eight probabilities
bx, slot, bw = 160, 25, 18
body += line(bx - 8, base, bx + 8 * slot - 1, base, GUIDE, THIN)
for i, q in enumerate(p):
    x = bx + i * slot
    c = ACCENT if i == TRUE else GREY
    body += f'<rect x="{x}" y="{Y(q):.1f}" width="{bw}" height="{base - Y(q):.1f}" fill="{c}"/>'

# the cross-entropy curve on the same vertical probability scale: loss runs to the right
cx0, cx1, LMAX = 380, 562, 3.0
X = lambda loss: cx0 + (cx1 - cx0) * loss / LMAX
body += axes(cx0, top - 4, cx1, base)
qs = [math.exp(-LMAX * k / 120) for k in range(121)]
body += poly([(X(-math.log(q)), Y(q)) for q in qs], INK, 2.6)

# a uniform guess among the eight tokens, for comparison
pu, lu = 1 / 8, math.log(8)
body += line(bx - 8, Y(pu), X(lu) - 7, Y(pu), GREY, 1.4, "3 5")
body += dot(X(lu), Y(pu), 5, GREY)

# carry p(on) across to the curve and down to its loss
pt, lt = p[TRUE], -math.log(p[TRUE])
body += line(bx + TRUE * slot + bw + 4, Y(pt), X(lt) - 9, Y(pt), ACCENT, 1.8, "6 5")
body += line(X(lt), Y(pt) + 9, X(lt), base, ACCENT, 1.8, "6 5")
body += ring(X(lt), Y(pt), 8, ACCENT, 3)

write(body)
