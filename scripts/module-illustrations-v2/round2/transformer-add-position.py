"""Sinusoidal position encodings. Left: sin(w p) for four speeds w = 10000^(-2i/d), each sampled at the whole positions,
with one position picked out across all of them. Right: the fast pair (sin p, cos p) at positions 0 to 6 on the unit
circle, every step the same 1-radian turn."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

# left panel: four waves, fastest on top
L0, L1, T, B = 44, 312, 52, 348
P = 24                                    # positions 0 .. P
speeds = [1.0, 0.42, 0.18, 0.07]          # geometric, like 10000^(-2i/d), compressed so every band visibly moves
band = (B - T) / 4
X = lambda p: L0 + p * (L1 - L0) / P
pick = 7
body = line(X(pick), T - 6, X(pick), B + 4, ACCENT_SOFT, 18)
for k, w in enumerate(speeds):
    cy = T + band * (k + 0.5); amp = band * 0.3
    Y = lambda v: cy - amp * v
    body += line(L0, cy, L1, cy, GUIDE, THIN)
    body += poly([(X(t / 10), Y(math.sin(w * t / 10))) for t in range(P * 10 + 1)], GREY if k else INK, 2.4)
    for p in range(P + 1):
        if p != pick:
            body += dot(X(p), Y(math.sin(w * p)), 3.2, INK)
    body += dot(X(pick), Y(math.sin(w * pick)), 7.5, ACCENT, INK, 1.6)

# right panel: the fast pair on the unit circle
CX, CY, R = 458, 200, 108
body += line(CX - R - 14, CY, CX + R + 14, CY, GUIDE, THIN) + line(CX, CY - R - 14, CX, CY + R + 14, GUIDE, THIN)
body += f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="none" stroke="{GUIDE}" stroke-width="2"/>'
pt = lambda a, r=R: (CX + r * math.sin(a), CY - r * math.cos(a))
for p in range(6):                         # equal turns, drawn as arcs just inside the circle
    arc = [pt(p + 0.12 + j * 0.76 / 20, R - 22) for j in range(21)]
    body += poly(arc, ACCENT, 2.6)
    a = p + 0.88; h = pt(a, R - 22); tdir = (math.cos(a), math.sin(a))
    body += arrow(h[0] - 3 * tdir[0], h[1] - 3 * tdir[1], h[0] + 2 * tdir[0], h[1] + 2 * tdir[1], ACCENT, 2.6, 13)
x0, y0 = pt(0)
body += line(CX, CY, x0, y0, GREY, 2, dash="5 4") + line(CX, CY, *pt(1), GREY, 2, dash="5 4")
for p in range(7):
    x, y = pt(p)
    body += dot(x, y, 8.5 if p == 0 else 7, ACCENT if p == 0 else INK)
write(body)
