"""The dual view, on the worked example: A = -0.5, B = C = 1, delta = 1, input an impulse.

Top: the recurrent form. An impulse enters on the left and a chain of state cells passes it along, each step the
same multiplication by A_bar = exp(-0.5); the terracotta level inside each cell is the true h_t
(0.7869, 0.4773, 0.2895, ...). Bottom: the convolutional form, the kernel K_bar = (C B_bar, C A_bar B_bar, ...)
as stems over the true decaying exponential (dashed grey). Dashed guides drop from every cell to its stem: the
two forms give the same numbers, one step at a time or all at once.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

A, B, C_, DT = -0.5, 1.0, 1.0, 1.0
A_bar = math.exp(DT * A)
B_bar = (A_bar - 1.0) / A * B
T = 6
K = [C_ * A_bar ** i * B_bar for i in range(T)]
h, hs = 0.0, []
for t in range(T):
    h = A_bar * h + B_bar * (1.0 if t == 0 else 0.0)
    hs.append(h)

X0, DX = 130, 82
xs = [X0 + t * DX for t in range(T)]

body = ""
# Bottom: the kernel as stems over the continuous exponential.
BASE, KS = 362, 150 / K[0]
body += axes(70, 200, 565, BASE)
curve = [(X0 + u * DX, BASE - KS * B_bar * math.exp(A * DT * u)) for u in [i / 20 for i in range(int(5.3 * 20) + 1)]]
body += poly(curve, GREY, 1.8, dash="7 6")

# Top: the recurrent chain.
BOX, BY = 52, 64
for t in range(T):
    x = xs[t]
    bx = x - BOX / 2
    lvl = (BOX - 8) * hs[t] / hs[0]
    body += line(x, BY + BOX + 6, x, BASE - KS * K[t] - 12, GUIDE, THIN, "5 5")
    body += rect(bx, BY, BOX, BOX, BG, INK, 2.4, 6)
    body += rect(bx + 4, BY + BOX - 4 - lvl, BOX - 8, lvl, ACCENT, "none", 0, 3)
    if t + 1 < T:
        body += arrow(bx + BOX + 4, BY + BOX / 2, xs[t + 1] - BOX / 2 - 4, BY + BOX / 2, INK, 2.4, 13)
# The impulse entering on the left.
IX = 52
body += line(IX - 14, BY + BOX + 4, IX + 14, BY + BOX + 4, GUIDE, THIN)
body += line(IX, BY + BOX + 4, IX, BY + 4, INK, 3.0) + dot(IX, BY + 4, 7)
body += arrow(IX + 8, BY + BOX / 2, xs[0] - BOX / 2 - 4, BY + BOX / 2, INK, 2.4, 13)

for t in range(T):
    y = BASE - KS * K[t]
    body += line(xs[t], BASE, xs[t], y, ACCENT, 3.2) + dot(xs[t], y, 7, ACCENT)
write(body)
