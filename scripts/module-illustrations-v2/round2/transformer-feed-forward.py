"""The lesson's feed-forward network for "sat": 4 inputs expand through the real W1 wiring to 8 hidden values,
the ReLU keeps only the two positive ones (terracotta) and drops the negative parts (dashed), and W2 shrinks back to 4."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

Y = [-0.241, -1.208, -0.121, 1.571]
W1 = [[1, 0, 0, -1, 0, 1, 0, 0],
      [0, 1, 0, 0, -1, 0, 1, 0],
      [0, 0, 1, 0, 0, -1, 0, 1],
      [-1, 0, 0, 1, 0, 0, -1, 0]]
Z = [sum(Y[i] * W1[i][j] for i in range(4)) for j in range(8)]
A = [max(0.0, z) for z in Z]
W2 = [[0.5 * ((j == k) - (j == k + 4)) for k in range(4)] for j in range(8)]
OUT = [sum(A[j] * W2[j][k] for j in range(8)) for k in range(4)]

yin = [200 + 62 * (i - 1.5) for i in range(4)]
yh = [200 + 42 * (j - 3.5) for j in range(8)]
AX_IN, AX_H, AX_OUT = 98, 300, 500
IN_R, H_L, H_R, OUT_L = 165, 212, 388, 440
S_IN, S_H, S_OUT = 40, 30, 56
BAR = 18

body = ""
# Expand: only the nonzero entries of W1 are wires; negative weights dashed.
for i in range(4):
    for j in range(8):
        w = W1[i][j]
        if not w:
            continue
        hot = A[j] > 0
        body += line(IN_R, yin[i], H_L, yh[j], ACCENT if hot else GREY,
                     2.6 if hot else 1.4, None if w > 0 else "5 4")
# Shrink: hidden k adds to output k, hidden k+4 subtracts from it.
for j in range(8):
    for k in range(4):
        w = W2[j][k]
        if not w:
            continue
        hot = A[j] > 0
        body += line(H_R, yh[j], OUT_L, yin[k], ACCENT if hot else GUIDE,
                     2.6 if hot else 1.4, None if w > 0 else "5 4")


def hbar(ax, y, v, scale, fill, stroke, dash=None):
    x0, x1 = sorted((ax, ax + v * scale))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x0:.1f}" y="{y - BAR / 2:.1f}" width="{max(x1 - x0, 1):.1f}" height="{BAR}" rx="2" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{d}/>')


# Input column.
body += line(AX_IN, yin[0] - 26, AX_IN, yin[-1] + 26, GUIDE, THIN)
for i in range(4):
    body += hbar(AX_IN, yin[i], Y[i], S_IN, INK, INK)
    body += dot(IN_R, yin[i], 5, INK)
# Hidden column: the pre-ReLU value as a dashed outline, the surviving positive part filled.
body += line(AX_H, yh[0] - 22, AX_H, yh[-1] + 22, GUIDE, THIN)
for j in range(8):
    if A[j] > 0:
        body += hbar(AX_H, yh[j], A[j], S_H, ACCENT, ACCENT)
    else:
        body += hbar(AX_H, yh[j], Z[j], S_H, "none", GREY, "4 3")
    on = A[j] > 0
    body += dot(H_L, yh[j], 5, ACCENT) if on else ring(H_L, yh[j], 4.5, GREY, 1.8)
    body += dot(H_R, yh[j], 5, ACCENT) if on else ring(H_R, yh[j], 4.5, GREY, 1.8)
# Output column.
body += line(AX_OUT, yin[0] - 26, AX_OUT, yin[-1] + 26, GUIDE, THIN)
for k in range(4):
    body += dot(OUT_L, yin[k], 5, INK)
    if abs(OUT[k]) > 1e-9:
        body += hbar(AX_OUT, yin[k], OUT[k], S_OUT, SECOND, SECOND)
    else:
        body += line(AX_OUT - 8, yin[k], AX_OUT + 8, yin[k], SECOND, 2.4)
write(body)
