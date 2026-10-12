"""S = [[2, 1], [1, 2]] sends the unit circle to an ellipse whose axes are its perpendicular eigenvectors."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

S = ((2.0, 1.0), (1.0, 2.0))
Sv = lambda v: (S[0][0] * v[0] + S[0][1] * v[1], S[1][0] * v[0] + S[1][1] * v[1])
cx, cy, k = 300, 200, 62
X = lambda v: (cx + k * v[0], cy - k * v[1])
r2 = 1 / math.sqrt(2)
q1, q2 = (r2, r2), (r2, -r2)                    # eigenvalues 3 and 1

body = ""
for g in range(-5, 6):                          # light grid, kept inside the margin
    if 40 <= cx + g * k <= 560:
        body += line(cx + g * k, 30, cx + g * k, 370, GUIDE, 0.9)
    if 30 <= cy + g * k <= 370:
        body += line(40, cy + g * k, 560, cy + g * k, GUIDE, 0.9)
circ = [(math.cos(2 * math.pi * i / 200), math.sin(2 * math.pi * i / 200)) for i in range(201)]
ell = [Sv(v) for v in circ]
body += poly([X(v) for v in ell], ACCENT, 2.8, ACCENT_SOFT, close=True)
for v, L in ((q1, 3.0), (q2, 3.0)):             # eigen-directions
    body += line(*X((-L * v[0], -L * v[1])), *X((L * v[0], L * v[1])), GREY, 1.5, dash="6 6")
body += poly([X(v) for v in circ], INK, 2.0, BG, close=True)
for i in range(16):                             # each point of the circle and where it lands
    v = (math.cos(2 * math.pi * i / 16), math.sin(2 * math.pi * i / 16))
    body += line(*X(v), *X(Sv(v)), GREY, 1.1)
    body += dot(*X(Sv(v)), 4.2, ACCENT)
    body += dot(*X(v), 3.6, INK)
o = X((0, 0))
e = (1.0, 0.0)                                  # a vector that turns
body += arrow(*o, *X(Sv(e)), GREY, 2.4)
body += arrow(*o, *X(e), GREY, 2.4)
body += arrow(*o, *X(Sv(q1)), ACCENT, 3.4)      # stretched by 3, same direction
body += arrow(*o, *X(q1), INK, 3.4)
body += arrow(*o, *X(q2), INK, 3.4)             # stretched by 1: lands on itself
body += dot(*o, 5.5, INK)
write(body)
